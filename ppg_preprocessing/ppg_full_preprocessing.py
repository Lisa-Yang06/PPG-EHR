"""
Filter MIMIC PPG segments: keep only valid 10s windows.
Multiprocessing version.

For each patient's ppg.parquet:

  SEGMENT LEVEL
    Step 1: drop if NaN fraction > NAN_FRAC_MAX (0.05); otherwise interpolate
    Step 2: drop if the RAW segment is flatline
            (_is_flatline: max-min < FLATLINE_ATOL=1e-6, or fewer than
             MIN_UNIQUE=10 distinct values)
    Step 3: preprocess_one_ppg_signal (PaPaGei pyPPG-based pipeline)
    Step 4: drop if the PROCESSED segment is flatline
    Step 5: drop if shorter than one WIN_S=10 s window

  WINDOW LEVEL (non-overlapping 10 s)
    Step 6: skip the window if it is flatline
    Step 7: skip the window if biobss.sqatools.detect_flatline_segments marks
            more than 25% of it flat (min flat run 0.5 s)
    Step 8: z-score normalise, then resample to TARGET_FS=125 Hz

Output: per-patient parquet, one row per surviving 10s window.

WHAT IS *NOT* CHECKED -- read before treating these windows as "clean"
---------------------------------------------------------------------
Every morphology test above detects a DEAD signal (flat, constant, NaN).
Nothing here scores pulse morphology, so motion artefact, low perfusion and
clipping pass through as long as the trace is not flat. A window that survives
is "not dead", not "good".

This docstring used to describe a Step 4 running
nk.ppg_quality(method='templatematch') and keeping windows whose 15th
percentile exceeded 0.9. THAT STEP IS NOT EXECUTED: `ppg_quality` appears
nowhere in the code below. It evidently existed once -- the output row still
carries a commented-out `q15` field where the score was written -- and was
removed without updating this header, so every window in the current parquets
was admitted on the flatline tests alone.

Consequences for anything downstream: the parquet stores only
(patient_id, record_id, segment, segment_start_time, window_idx,
window_start_time, fs, signal) and no quality column, so a per-window quality
score has to be computed separately. It can be computed directly from the
stored `signal` -- there is no need to return to the WFDB sources -- but note
the trace is z-scored, so scale-invariant measures (template matching) work
while amplitude-based ones (perfusion index) do not. Rejection statistics
(how many windows a patient lost, and to which test) are counted at runtime in
`fail_reasons` but are NOT persisted, so recovering them does require re-running
this script.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import paths

# Redirects library caches only when PPG_CACHE_ROOT is set, and never
# reassigns HOME. The research version set HOME unconditionally to a scratch
# directory, which on any other machine repoints where every library looks for
# credentials and configuration.
paths.configure_caches()

# Limit each worker to 1 thread (avoid oversubscription across MP workers)
os.environ['OMP_NUM_THREADS']      = '1'
os.environ['MKL_NUM_THREADS']      = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS']  = '1'

import sys
import json
import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


import biobss

# ── PaPaGei imports ────────────────────────────────────────────────────────────
paths.add_papagei_to_path()

from preprocessing.ppg import preprocess_one_ppg_signal
from linearprobing.utils import resample_batch_signal

# ── Config ─────────────────────────────────────────────────────────────────────
WIN_S         = 10
TARGET_FS     = 125.0
TARGET_LEN    = int(WIN_S * TARGET_FS)
Q15_THRESHOLD = 0.9
NAN_FRAC_MAX  = 0.05
FLATLINE_ATOL = 1e-6
MIN_UNIQUE    = 10

# ── Plot config ────────────────────────────────────────────────────────────────
PLOT_PROB_PASSED      = 0.00001
PLOT_PROB_FAILED      = 0.00001
# NOTE: this is per-worker. Total plots ≈ MAX_PLOTS_PER_BUCKET * n_workers
MAX_PLOTS_PER_BUCKET  = 2    # Lower because we have n_workers each saving up to this many
PLOT_DIR = paths.data_root() / "ds2" / "ppg_filter_vis"

INPUT_DIRS = {
    'III': paths.data_root() / "MIMIC_III_ppg_by_patient",
    'IV':  paths.data_root() / "MIMIC_IV_ppg_by_patient",
}
OUTPUT_DIRS = {
    'III': paths.data_root() / "MIMIC_III_ppg_filtered_by_patient",
    'IV':  paths.data_root() / "MIMIC_IV_ppg_filtered_by_patient",
}


# ── Quality helpers ────────────────────────────────────────────────────────────
import biobss

def _maybe_plot_flat_window(window, fs, pid, win_idx, dataset, flat_frac, plot_count):
    """Save a few examples of windows rejected by BioBSS flatline detection."""
    bucket = 'rejected_flatline'
    if plot_count[bucket] >= MAX_PLOTS_PER_BUCKET:
        return
    if np.random.random() >= PLOT_PROB_FAILED * 10:
        return
    
    out_dir = PLOT_DIR / dataset / bucket
    out_dir.mkdir(parents=True, exist_ok=True)
    
    t = np.arange(len(window)) / fs
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.plot(t, window, color='firebrick', linewidth=0.8)
    ax.set_title(f"[{bucket}] #{plot_count[bucket]} | Patient {pid} | "
                 f"win {win_idx} | flat_frac={flat_frac:.3f}")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('PPG amplitude')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(out_dir / f"{plot_count[bucket]}_pid{pid}_w{win_idx}.png", dpi=80)
    plt.close()
    plot_count[bucket] += 1

def _flat_fraction_biobss(sig, change_threshold=0.001, min_duration = 10):
    """
    Use BioBSS detect_flatline_segments with paper-cited default params.
    Returns fraction of samples in detected flat segments.
    """
    if sig is None or len(sig) < min_duration + 1:
        return 1.0
    
    valid = sig[np.isfinite(sig)]
    if len(valid) < min_duration + 1:
        return 1.0
    
    try:
        flat_segs = biobss.sqatools.detect_flatline_segments(
            sig,
            change_threshold=change_threshold,
            min_duration=min_duration,
        )
    except Exception:
        return 0.0
    
    if not flat_segs:
        return 0.0
    
    # BioBSS returns list of (start, end) tuples
    total_flat = sum(int(e) - int(s) for s, e in flat_segs)
    return total_flat / len(sig)

def _is_flatline(sig, atol=FLATLINE_ATOL, min_unique=MIN_UNIQUE):
    if sig is None or len(sig) == 0:
        return True
    valid = sig[np.isfinite(sig)]
    if len(valid) < 100:
        return True
    if valid.max() - valid.min() <= atol:
        return True
    if len(np.unique(valid)) < min_unique:
        return True
    return False


def _interpolate_nans(sig):
    nan_mask = np.isnan(sig)
    if not nan_mask.any():
        return sig
    x = np.arange(len(sig))
    return np.interp(x, x[~nan_mask], sig[~nan_mask])


# ── Plotting helpers (now take per-worker plot counter) ────────────────────────
def _save_window_plot(window, fs, bucket, pid, win_idx, dataset, plot_index):
    out_dir = PLOT_DIR / dataset / bucket
    out_dir.mkdir(parents=True, exist_ok=True)

    t = np.arange(len(window)) / fs
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.plot(t, window, color='steelblue', linewidth=0.8)
    ax.set_title(f"[{bucket}] #{plot_index} | Patient {pid} | win {win_idx}")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('PPG amplitude')
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / f"{plot_index}_pid{pid}_w{win_idx}.png"
    plt.savefig(out_path, dpi=80)
    plt.close()


def _save_segment_plot(raw_signal, fs, reason, pid, dataset, plot_index):
    out_dir = PLOT_DIR / dataset / reason
    out_dir.mkdir(parents=True, exist_ok=True)

    n_to_plot = min(int(60 * fs), len(raw_signal))
    t = np.arange(n_to_plot) / fs

    fig, ax = plt.subplots(figsize=(15, 3))
    ax.plot(t, raw_signal[:n_to_plot], color='firebrick', linewidth=0.5)
    ax.set_title(f"[{reason}] #{plot_index} | Patient {pid} | "
                 f"first {n_to_plot/fs:.1f}s of {len(raw_signal)/fs:.1f}s")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('PPG amplitude (raw)')
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / f"{plot_index}_pid{pid}.png"
    plt.savefig(out_path, dpi=80)
    plt.close()


def _maybe_plot_window(window, fs, pid, win_idx, dataset, plot_count):
    bucket = 'passed'

    if plot_count[bucket] >= MAX_PLOTS_PER_BUCKET:
        return
    if np.random.random() >= PLOT_PROB_PASSED:
        return

    _save_window_plot(window, fs, bucket, pid, win_idx, dataset,
                      plot_index=plot_count[bucket])
    plot_count[bucket] += 1


def _maybe_plot_segment_fail(raw, fs, reason, pid, dataset, plot_count):
    if reason not in plot_count:
        return
    if plot_count[reason] >= MAX_PLOTS_PER_BUCKET:
        return
    if np.random.random() >= PLOT_PROB_FAILED * 10:
        return

    _save_segment_plot(raw, fs, reason, pid, dataset, plot_index=plot_count[reason])
    plot_count[reason] += 1


# ── Per-segment filter ─────────────────────────────────────────────────────────
def filter_one_segment(raw_signal_json, fs, pid, dataset, plot_count):
    try:
        raw = np.array(json.loads(raw_signal_json), dtype=np.float32)
    except Exception:
        return [], "json_parse_failed"

    if len(raw) == 0:
        return [], "empty_signal"

    nan_frac = np.isnan(raw).mean()
    if nan_frac > NAN_FRAC_MAX:
        return [], "too_many_nans"
    if nan_frac > 0:
        raw = _interpolate_nans(raw)

    if _is_flatline(raw):
        _maybe_plot_segment_fail(raw, fs, 'raw_flatline', pid, dataset, plot_count)
        return [], "raw_flatline"

    try:
        processed, _, _, _ = preprocess_one_ppg_signal(waveform=raw, frequency=float(fs))
    except Exception as e:
        return [], f"preprocess_failed:{type(e).__name__}"


    if _is_flatline(processed):
        _maybe_plot_segment_fail(processed, fs, 'post_preprocess_flatline',
                                  pid, dataset, plot_count)
        return [], "post_preprocess_flatline"

    seg_len_raw_fs = int(float(fs) * WIN_S)
    if seg_len_raw_fs <= 0 or len(processed) < seg_len_raw_fs:
        return [], "too_short"

    n_windows = len(processed) // seg_len_raw_fs
    valid_windows = []

    for w in range(n_windows):
        window = processed[w * seg_len_raw_fs:(w + 1) * seg_len_raw_fs]
        if _is_flatline(window):
            continue

        flat_frac = _flat_fraction_biobss(window, min_duration=int(0.5 * fs))  # 0.5s min flat duration
        if flat_frac > 0.25:
            # Plot a few for sanity check
            _maybe_plot_flat_window(window, fs, pid, w, dataset, flat_frac, plot_count)
            continue

        # Plot a few sample windows for sanity check
        _maybe_plot_window(window, fs, pid, w, dataset, plot_count)
        
        # ── Z-score normalize (PaPaGei paper step 4) ─────────────────────
        # Done on the original-fs window; resample below preserves z-scored
        # distribution since resampling is linear (preserves zero mean).
        window = window.astype(np.float32)
        std = window.std()
        if std < 1e-8:
            continue
        window = (window - window.mean()) / std
        # ─────────────────────────────────────────────────────────────────


        # Resample to 125 Hz (PaPaGei paper step 5)
        if not np.isclose(fs, TARGET_FS):
            window_125 = resample_batch_signal(
                window[np.newaxis, :],
                fs_original=float(fs), fs_target=TARGET_FS, axis=-1,
            )[0]
        else:
            window_125 = window

        
        if len(window_125) > TARGET_LEN:
            window_125 = window_125[:TARGET_LEN]
        elif len(window_125) < TARGET_LEN:
            pad_needed = TARGET_LEN - len(window_125)
            if pad_needed <= 2:
                window_125 = np.pad(window_125, (0, pad_needed))
            else:
                continue

        valid_windows.append({
            'window_idx':   w,
            'signal_125Hz': window_125.astype(np.float32)
        })

    if not valid_windows:
        return [], "no_window_passed_flatline"
    return valid_windows, "ok"


# ── Per-patient worker (top-level for pickling) ────────────────────────────────
def process_one_patient_worker(pid_dir_str, output_dir_str, dataset, resume):
    """
    Worker function. Returns stats dict.
    Must take only picklable args (no Path objects... actually Path is picklable,
    but using str makes the API more portable).
    """
    pid_dir = Path(pid_dir_str)
    output_dir = Path(output_dir_str)
    pid = pid_dir.name

    # Each worker has its own local plot counter
    plot_count = {
    'passed':                   0,
    'rejected_flatline':        0,
    'raw_flatline':             0,
    'post_preprocess_flatline': 0,
}

    # Set per-worker random seed (otherwise all workers might pick same plot samples)
    np.random.seed(os.getpid() % 100000)

    out_pid_dir = output_dir / pid
    out_path = out_pid_dir / "ppg.parquet"

    if resume and out_path.exists():
        return {'pid': pid, 'status': 'skipped_exists', 'n_input': 0, 'n_output': 0,
                'fail_reasons': {}, 'plot_count': plot_count}

    in_path = pid_dir / "ppg.parquet"
    if not in_path.exists():
        return {'pid': pid, 'status': 'no_input', 'n_input': 0, 'n_output': 0,
                'fail_reasons': {}, 'plot_count': plot_count}

    try:
        df = pd.read_parquet(in_path)
    except Exception:
        return {'pid': pid, 'status': 'read_failed', 'n_input': 0, 'n_output': 0,
                'fail_reasons': {}, 'plot_count': plot_count}

    if df.empty:
        return {'pid': pid, 'status': 'empty_input', 'n_input': 0, 'n_output': 0,
                'fail_reasons': {}, 'plot_count': plot_count}

    all_windows = []
    fail_reasons = Counter()
    n_input_segments = len(df)

    for _, row in df.iterrows():
        valid, reason = filter_one_segment(
            row['signal'], row['fs'], pid, dataset, plot_count,
        )
        if not valid:
            fail_reasons[reason] += 1
            continue

        for w in valid:
            all_windows.append({
                'patient_id':         pid,
                'record_id':          row.get('record_id'),
                'segment':            row.get('segment'),
                'segment_start_time': row['start_time'],
                'window_idx':         w['window_idx'],
                'window_start_time':  pd.to_datetime(row['start_time']) +
                                       pd.Timedelta(seconds=w['window_idx'] * WIN_S),
                # 'q15':                w['q15'],
                'fs':                 TARGET_FS,
                'signal':             json.dumps(
                    w['signal_125Hz'].tolist(), separators=(",", ":")
                ),
            })

    if not all_windows:
        return {'pid': pid, 'status': 'no_valid_window',
                'n_input': n_input_segments, 'n_output': 0,
                'fail_reasons': dict(fail_reasons), 'plot_count': plot_count}

    out_pid_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_windows).to_parquet(out_path, index=False)

    return {'pid': pid, 'status': 'ok',
            'n_input': n_input_segments, 'n_output': len(all_windows),
            'fail_reasons': dict(fail_reasons), 'plot_count': plot_count}

# ── Pid filter helper ─────────────────────────────────────────────────────────
def _pid_matches_prefixes(pid, prefixes):
    """Return True if pid starts with any of the given prefixes."""
    if not prefixes:
        return True   # no filter → keep all
    return any(pid.startswith(p) for p in prefixes)

# ── q15 distribution plot ─────────────────────────────────────────────────────
def plot_q15_distribution(out_dir, dataset, all_q15):
    if not all_q15:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_q15, bins=50, edgecolor='black', color='steelblue', alpha=0.8)
    ax.axvline(x=Q15_THRESHOLD, color='red', linestyle='--',
                label=f'Threshold (q15={Q15_THRESHOLD})')
    ax.axvline(x=0.95, color='orange', linestyle='--', label='High-quality (0.95)')
    ax.set_xlabel('q15 (15th percentile of templatematch quality)')
    ax.set_ylabel('Count')
    ax.set_title(f'{dataset}: q15 distribution of {len(all_q15):,} passed windows')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = PLOT_DIR / dataset / 'q15_distribution.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Saved q15 distribution to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(dataset, limit=None, resume=True, n_workers=8, pid_prefixes = None):
    in_dir = INPUT_DIRS[dataset]
    out_dir = OUTPUT_DIRS[dataset]
    out_dir.mkdir(parents=True, exist_ok=True)
    (PLOT_DIR / dataset).mkdir(parents=True, exist_ok=True)


    # Collect all patient dirs, then filter by prefix
    all_pid_dirs = sorted(p for p in in_dir.iterdir() if p.is_dir())
    pid_dirs = [p for p in all_pid_dirs if _pid_matches_prefixes(p.name, pid_prefixes)]

    if limit:
        pid_dirs = pid_dirs[:limit]

    print(f"\n{'='*70}")
    print(f"Filtering {dataset}: {len(pid_dirs)} patients with {n_workers} workers")
    print(f"  Input:    {in_dir}")
    print(f"  Output:   {out_dir}")
    print(f"  Plots:    {PLOT_DIR / dataset}")
    print(f"  Resume:   {resume}")
    print(f"  q15 thr:  {Q15_THRESHOLD}")
    print(f"  Max plots/bucket/worker: {MAX_PLOTS_PER_BUCKET} (total ≈ {MAX_PLOTS_PER_BUCKET * n_workers})")
    print(f"{'='*70}\n", flush=True)

    # Worker function with fixed args, only takes pid_dir_str
    worker_fn = partial(
        process_one_patient_worker,
        output_dir_str=str(out_dir),
        dataset=dataset,
        resume=resume,
    )

    pid_dir_strs = [str(p) for p in pid_dirs]

    all_stats = []
    # Use spawn context to avoid fork issues (esp. on Linux with multi-threaded libs)
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        for stat in tqdm(
            pool.imap_unordered(worker_fn, pid_dir_strs, chunksize=1),
            total=len(pid_dir_strs),
            desc=f"Filtering {dataset}",
        ):
            all_stats.append(stat)

    # ── Aggregate ──
    total_in_seg = sum(s['n_input'] for s in all_stats)
    total_out_win = sum(s['n_output'] for s in all_stats)
    overall_fail = Counter()
    overall_plot_count = Counter()
    for s in all_stats:
        for k, v in s.get('fail_reasons', {}).items():
            overall_fail[k] += v
        for k, v in s.get('plot_count', {}).items():
            overall_plot_count[k] += v


    # ── Summary ──
    print(f"\n=== {dataset} Summary ===")
    print(f"Total patients processed: {len(all_stats)}")
    print(f"Total input segments:     {total_in_seg:,}")
    print(f"Total output windows:     {total_out_win:,}")
    if total_in_seg > 0:
        print(f"Windows per segment:      {total_out_win / total_in_seg:.1f}")

    statuses = Counter(s['status'] for s in all_stats)
    print(f"\nPatient statuses:")
    for s, c in statuses.most_common():
        print(f"  {s}: {c}")

    print(f"\nFail reasons (per segment):")
    for r, c in overall_fail.most_common():
        print(f"  {r}: {c:,}")

    print(f"\nPlot counts saved (summed across workers):")
    for k, v in overall_plot_count.most_common():
        print(f"  {k}: {v}")

    retention_rates = [
        s['n_output'] / s['n_input']
        for s in all_stats
        if s['status'] == 'ok' and s['n_input'] > 0
    ]
    if retention_rates:
        ret = np.array(retention_rates)
        print(f"\nWindows-per-segment ratio across {len(ret)} ok patients:")
        print(f"  Median: {np.median(ret):.2f}")
        print(f"  Mean:   {ret.mean():.2f}")

    prefix_tag = "_".join(pid_prefixes) if pid_prefixes else "all"
    stats_path = out_dir / f"_filter_stats_{prefix_tag}.json"
    
    with open(stats_path, 'w') as f:
        json.dump({
            'dataset':         dataset,
            'n_workers':       n_workers,
            'total_patients':  len(all_stats),
            'total_in_seg':    total_in_seg,
            'total_out_win':   total_out_win,
            'statuses':        dict(statuses),
            'fail_reasons':    dict(overall_fail),
            'plot_counts':     dict(overall_plot_count),
            'per_patient':     all_stats,
        }, f, indent=2, default=str)
    print(f"\nStats saved to: {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['III', 'IV', 'both'], default='both')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--n-workers', type=int, default=8,
                        help='Number of worker processes (default: 8)')
    parser.add_argument('--pid-prefixes', type=str, default=None,
                        help='Comma-separated pid prefixes to process '
                             '(e.g. "07,08,09" → only pids starting with 07/08/09). '
                             'Default: process all.')
    args = parser.parse_args()

    resume = not args.no_resume

    pid_prefixes = (
        [p.strip() for p in args.pid_prefixes.split(',') if p.strip()]
        if args.pid_prefixes else None
    )

    if args.dataset == 'both':
        main('III', limit=args.limit, resume=resume,
             n_workers=args.n_workers, pid_prefixes=pid_prefixes)
        main('IV',  limit=args.limit, resume=resume,
             n_workers=args.n_workers, pid_prefixes=pid_prefixes)
    else:
        main(args.dataset, limit=args.limit, resume=resume,
             n_workers=args.n_workers, pid_prefixes=pid_prefixes)