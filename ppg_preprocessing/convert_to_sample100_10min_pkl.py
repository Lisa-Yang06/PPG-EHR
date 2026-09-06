"""
Convert filtered PPG parquets -> per-patient pickle of CONTIGUOUS 10-MINUTE
segments (60 consecutive 10-second windows).

The unit of a "sample" is not a single 10s window but a 10-minute contiguous
chunk of PPG: 60 windows in a row from the same (record_id, segment), each
adjacent pair exactly 10 seconds apart.

NAMING, BECAUSE IT HAS ALREADY MISLED ONE READER
------------------------------------------------
The research tree held two near-identical copies of this script differing in
two lines -- n_windows_per_seg and out_filename -- and the file named
"..._5min_pkl.py" was the one producing the 10-MINUTE pkl. That is the
version here, renamed to say what it produces. The genuinely 5-minute variant
(30 windows -> ppg_sample100_5min.pkl) is not part of this pipeline.

The single source of truth is CONFIG below: n_windows_per_seg = 60.

Why a longer window: single 10s windows aligned poorly with sEHR (val loss
stuck near the random baseline log(256) = 5.545), so the PPG side needs
enough context to carry HRV structure that can correlate with a patient's
codes within +-12 months. HRV work conventionally treats 5 minutes as the
floor for stable LF/HF; 10 minutes sits above it.

Strictness: a segment requires 60 consecutive windows whose neighbouring
window_start_time values are exactly 10 seconds apart (small float
tolerance), AND all 60 share one record_id + segment. Relaxing this (say,
tolerating one or two dropped windows) belongs in find_contiguous_segments.

Output: ppg_sample100_10min.pkl, one per patient, read by the alignment
training scripts via CONFIG['ppg_pkl_filename'].
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import paths

# Opt-in, and never reassigns HOME -- see paths.configure_caches. The research
# version overrode HOME itself, pointing it at a scratch directory, which on
# anyone else's machine repoints where every library looks for credentials.
paths.configure_caches()
import json
import time
import pickle
import argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import joblib

# ── Config (kept consistent with alignment training + earlier convert) ───────
CONFIG = {
    'ppg_IV_filtered_dir':   str(paths.data_root() / "MIMIC_IV_ppg_filtered_by_patient"),
    'ppg_III_filtered_dir':  str(paths.data_root() / "MIMIC_III_ppg_filtered_by_patient"),
    'history_db_IV_path':    str(paths.data_root() / "patient_history_db_IV.joblib"),
    'history_db_III_path':   str(paths.data_root() / "patient_history_db_III.joblib"),

    'window_months':   12,         # +-12 months, matches alignment training
    'n_samples':       100,        # cap on segments per patient
    'expected_sig_len': 1250,      # 10s at 125Hz
    'win_s':            10,        # one window = 10s
    'n_windows_per_seg': 60,       # 60 * 10s = 10 minutes
    'time_tolerance_s': 0.5,       # treat 10s+-0.5s as still "exactly 10s"
                                   # (defends against float ts precision wobble;
                                   # if filter pipeline did its job right this
                                   # tolerance is never actually consumed)
    'out_filename':    "ppg_sample100_10min.pkl",
}


def get_window_delta():
    days_half_window = int(CONFIG['window_months'] * 30.44)
    return np.timedelta64(days_half_window, 'D')


def load_history(dataset):
    """Load + key-normalize EHR history db, matching alignment training script."""
    if dataset == 'IV':
        raw = joblib.load(CONFIG['history_db_IV_path'])
        return {str(k): v for k, v in raw.items()}
    else:
        raw = joblib.load(CONFIG['history_db_III_path'])
        return {str(k).zfill(6): v for k, v in raw.items()}


def list_ready_pids(dataset):
    ppg_dir = Path(CONFIG[f'ppg_{dataset}_filtered_dir'])
    return sorted(p.name for p in ppg_dir.iterdir() if p.is_dir())


# ── Core: find all valid 10-minute segments in a single patient's parquet ───
def find_contiguous_segments(df):
    """
    Given a single patient's full parquet DataFrame (one row per 10s window),
    return a list of dicts, one per valid contiguous 10-minute segment.

    A segment is valid iff:
      - n_windows_per_seg (60) consecutive rows from the SAME
        (record_id, segment) group
      - rows are sorted by window_start_time
      - every adjacent time diff is exactly 10s (within time_tolerance_s)

    Each returned dict has:
      'row_indices':       np.ndarray (n_windows_per_seg,) of original df row
                           positions, in time order (used to pull signals)
      'center_time':       np.datetime64 of the middle window's start
      'start_time':        np.datetime64 of window 0's start
      'record_id':         the segment's record_id
      'segment':           the segment's segment id
      'window_idx_start':  window_idx of window 0
    """
    n_required = CONFIG['n_windows_per_seg']
    expected_gap = np.timedelta64(CONFIG['win_s'] * 1000, 'ms')  # 10000 ms
    tol = np.timedelta64(int(CONFIG['time_tolerance_s'] * 1000), 'ms')

    found = []

    # Group by (record_id, segment). For each group, sort by time, then slide
    # a window of size n_required and check that all n_required-1 gaps == 10s.
    grouped = df.groupby(['record_id', 'segment'], sort=False)
    for (rec_id, seg_id), group in grouped:
        if len(group) < n_required:
            continue

        # Sort by window_start_time; remember the original df index of each row
        # so we can pull the signal later without re-parsing JSON for rows we
        # don't end up using.
        g_sorted = group.sort_values('window_start_time')
        orig_idx = g_sorted.index.to_numpy()
        times = pd.to_datetime(g_sorted['window_start_time'].values).values
        widx  = g_sorted['window_idx'].to_numpy()

        # Slide a window of length n_required over this group's sorted rows.
        # For each candidate, check all n_required-1 gaps are ~10s.
        n = len(g_sorted)
        gaps = np.diff(times)  # length n-1
        # A gap is "good" if |gap - 10s| <= tol
        good_gap = (gaps >= expected_gap - tol) & (gaps <= expected_gap + tol)

        # Find runs of good_gap. A segment starting at i exists iff
        # good_gap[i:i+n_required-1] are all True.
        # We can scan with a sliding sum for efficiency, but n is small here
        # (typically thousands), so a straightforward loop is fine.
        i = 0
        while i + n_required <= n:
            # Check n_required-1 gaps from position i:
            # good_gap[i], good_gap[i+1], ..., good_gap[i+n_required-2]
            window_ok = good_gap[i:i + n_required - 1].all()
            if window_ok:
                rows = orig_idx[i:i + n_required]  # n_required df indices, in time order
                found.append({
                    'row_indices':      rows,
                    'center_time':      times[i + n_required // 2 - 1],
                    'start_time':       times[i],
                    'record_id':        rec_id,
                    'segment':          seg_id,
                    'window_idx_start': int(widx[i]),
                })
                # Move past this segment to avoid massive overlap. Step by
                # n_required so segments within the same group don't overlap.
                # (Overlap would inflate the cap-100 quota with near-duplicate
                # chunks that are 99% identical, which doesn't add CLIP
                # diversity. Non-overlapping is the safer default.)
                i += n_required
            else:
                # Skip past the first bad gap in this window. Any segment
                # starting at i'<=i+gap_position can't be valid because it
                # would have to include that same bad gap.
                bad_in_window = np.where(~good_gap[i:i + n_required - 1])[0]
                first_bad_at = bad_in_window[0]  # at least one, by definition
                i += first_bad_at + 1

    return found


def _segment_has_sehr(center_time_dt64, hist_ts, window_delta):
    """True if at least one EHR timestamp falls within the +-window of center."""
    lo = np.searchsorted(hist_ts, center_time_dt64 - window_delta, side='left')
    hi = np.searchsorted(hist_ts, center_time_dt64 + window_delta, side='right')
    return hi > lo


# ── Per-patient conversion ─────────────────────────────────────────────────────
def convert_one_patient(args):
    """Convert a single patient's ppg.parquet -> ppg_sample100_10min.pkl.

    Returns a small result dict for aggregation/logging in the parent process.
    """
    pid_str, dataset, ppg_dir, history_pat = args
    t0 = time.time()
    ppg_path = Path(ppg_dir) / pid_str / "ppg.parquet"
    out_path = Path(ppg_dir) / pid_str / CONFIG['out_filename']

    result = {
        'pid': pid_str, 'dataset': dataset, 'status': None,
        'n_total_windows': 0, 'n_candidate_segments': 0,
        'n_group_a': 0, 'n_group_b': 0,
        'n_written': 0, 'elapsed_s': 0.0, 'error': None,
    }

    # ── EHR history presence check ──
    if history_pat is None:
        result['status'] = 'WARN_NO_EHR_HISTORY'
        print(f"{pid_str}: NO sEHR HISTORY DB DETECTED", flush=True)
        result['elapsed_s'] = time.time() - t0
        return result

    for key in ('timestamps', 'codes', 'code_type'):
        if key not in history_pat:
            result['status'] = 'WARN_EHR_MISSING_KEY'
            result['error'] = f"history missing key: '{key}'"
            print(f"{pid_str}: WARN EHR missing key {key}", flush=True)
            result['elapsed_s'] = time.time() - t0
            return result

    hist_ts = history_pat['timestamps']

    # ── Read parquet ──
    try:
        df = pd.read_parquet(str(ppg_path))
    except Exception as e:
        result['status'] = 'ERROR_READ_PARQUET'
        result['error'] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"{pid_str}: PPG PARQUET READING ERROR: {e}", flush=True)
        result['elapsed_s'] = time.time() - t0
        return result

    if df.empty:
        result['status'] = 'SKIP_EMPTY_PARQUET'
        print(f"{pid_str}: EMPTY PPG PARQUET", flush=True)
        result['elapsed_s'] = time.time() - t0
        return result

    result['n_total_windows'] = len(df)

    # ── Find all candidate segments ──
    candidates = find_contiguous_segments(df)
    result['n_candidate_segments'] = len(candidates)

    if not candidates:
        # Patient has windows but no run of 60 consecutive ones with 10s gaps.
        # This is expected for short / fragmented records.
        result['status'] = 'SKIP_NO_5MIN_SEGMENTS'
        result['elapsed_s'] = time.time() - t0
        return result

    # ── Split into group A (has sEHR within +-window_months of CENTER time) / B ──
    window_delta = get_window_delta()
    has_sehr = np.array([
        _segment_has_sehr(c['center_time'], hist_ts, window_delta)
        for c in candidates
    ])
    idx_a = np.where(has_sehr)[0]
    idx_b = np.where(~has_sehr)[0]
    result['n_group_a'] = len(idx_a)
    result['n_group_b'] = len(idx_b)

    # ── Sample up to n_samples, prioritizing group A ──
    n_target = CONFIG['n_samples']
    rng = np.random.default_rng(seed=abs(hash(pid_str)) % (2**32))

    if len(idx_a) >= n_target:
        chosen_idx = rng.choice(idx_a, size=n_target, replace=False)
    else:
        n_needed_from_b = n_target - len(idx_a)
        if len(idx_b) >= n_needed_from_b:
            chosen_b = rng.choice(idx_b, size=n_needed_from_b, replace=False)
        else:
            chosen_b = idx_b  # take everything available
        chosen_idx = np.concatenate([idx_a, chosen_b])

    chosen_idx = np.sort(chosen_idx)
    chosen_segments = [candidates[i] for i in chosen_idx]

    # ── Parse signals for the chosen segments only ──
    # For each chosen segment, decode 30 JSON signal strings -> (30, 1250) float32.
    signals_per_seg  = []
    centers          = []
    starts           = []
    record_ids       = []
    segments_ids     = []
    widx_starts      = []
    n_parse_errors_total = 0

    expected_sig_len = CONFIG['expected_sig_len']
    n_per_seg        = CONFIG['n_windows_per_seg']

    for seg in chosen_segments:
        seg_sigs = []
        seg_ok = True
        for row_pos in seg['row_indices']:
            row = df.iloc[row_pos]
            try:
                sig = np.array(json.loads(row['signal']), dtype=np.float32)
            except Exception:
                n_parse_errors_total += 1
                seg_ok = False
                break
            if len(sig) != expected_sig_len:
                n_parse_errors_total += 1
                seg_ok = False
                break
            seg_sigs.append(sig)

        if not seg_ok or len(seg_sigs) != n_per_seg:
            continue  # drop this segment, try the next chosen one

        signals_per_seg.append(np.stack(seg_sigs, axis=0))  # (30, 1250)
        centers.append(seg['center_time'])
        starts.append(seg['start_time'])
        record_ids.append(seg['record_id'])
        segments_ids.append(seg['segment'])
        widx_starts.append(seg['window_idx_start'])

    if not signals_per_seg:
        result['status'] = 'SKIP_NO_VALID_SIGNALS'
        result['elapsed_s'] = time.time() - t0
        return result

    out_data = {
        'patient_id':            pid_str,
        'dataset':               dataset,
        'signals':               np.stack(signals_per_seg, axis=0),       # (N, 30, 1250)
        'center_times':          np.array(centers, dtype='datetime64[ns]'),
        'segment_start_times':   np.array(starts,  dtype='datetime64[ns]'),
        'record_id':             record_ids,
        'segment':               segments_ids,
        'window_idx_start':      widx_starts,
    }

    # ── Write pkl ──
    try:
        with open(out_path, 'wb') as f:
            pickle.dump(out_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        result['status'] = 'ERROR_WRITE_PKL'
        result['error'] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"{pid_str}: PKL WRITE ERROR: {e}", flush=True)
        result['elapsed_s'] = time.time() - t0
        return result

    result['status'] = 'OK'
    result['n_written'] = len(signals_per_seg)
    if n_parse_errors_total:
        result['error'] = f"{n_parse_errors_total} window parse errors skipped"
    result['elapsed_s'] = time.time() - t0
    return result


# ── Driver ──────────────────────────────────────────────────────────────────
def run_dataset(dataset, n_workers, limit=None):
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset}  (target: contiguous segments of "
          f"{CONFIG['n_windows_per_seg']} windows = "
          f"{CONFIG['n_windows_per_seg'] * CONFIG['win_s'] // 60} min, "
          f"max {CONFIG['n_samples']}/patient)")
    print(f"{'='*70}")

    print(f"Loading EHR history ({dataset})...")
    history = load_history(dataset)
    print(f"  {len(history)} patients in history db")

    ppg_dir = CONFIG[f'ppg_{dataset}_filtered_dir']
    pids = list_ready_pids(dataset)
    if limit:
        pids = pids[:limit]
    print(f"  {len(pids)} patients with filtered PPG ready")

    tasks = [(pid, dataset, ppg_dir, history.get(pid)) for pid in pids]

    t_start = time.time()
    results = []
    n_done = 0
    print(f"\nConverting with {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        for res in pool.imap_unordered(convert_one_patient, tasks, chunksize=4):
            results.append(res)
            n_done += 1
            if n_done % 100 == 0 or n_done == len(tasks):
                elapsed = time.time() - t_start
                rate = n_done / elapsed
                eta = (len(tasks) - n_done) / rate if rate > 0 else float('inf')
                print(
                    f"  [{n_done}/{len(tasks)}] {elapsed:.0f}s elapsed, "
                    f"{rate:.1f} patients/s, ETA {eta:.0f}s",
                    flush=True,
                )

    elapsed_total = time.time() - t_start

    # ── Summary ──
    status_counts = {}
    for r in results:
        status_counts[r['status']] = status_counts.get(r['status'], 0) + 1

    print(f"\n{dataset} conversion done in {elapsed_total:.0f}s "
          f"({elapsed_total/60:.1f} min)")
    print(f"  Status breakdown:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"    {status}: {count}")

    ok_results = [r for r in results if r['status'] == 'OK']
    if ok_results:
        avg_cand    = np.mean([r['n_candidate_segments'] for r in ok_results])
        avg_a       = np.mean([r['n_group_a'] for r in ok_results])
        avg_b       = np.mean([r['n_group_b'] for r in ok_results])
        avg_written = np.mean([r['n_written'] for r in ok_results])
        n_capped_by_a = sum(1 for r in ok_results if r['n_group_a'] >= CONFIG['n_samples'])
        print(f"  Avg candidate segments/patient: {avg_cand:.0f}")
        print(f"  Avg group A (sEHR in +-{CONFIG['window_months']}mo of center): {avg_a:.0f}")
        print(f"  Avg group B (no sEHR): {avg_b:.0f}")
        print(f"  Avg segments written/patient: {avg_written:.0f}")
        print(f"  Patients with >={CONFIG['n_samples']} group-A segments "
              f"(100% sEHR-covered samples): {n_capped_by_a}/{len(ok_results)}")

        # Distribution for "patients with written < n_target"
        n_target = CONFIG['n_samples']
        short_written = [r for r in ok_results if r['n_written'] < n_target]
        if short_written:
            print(f"\n  Patients with written < {n_target} ({len(short_written)} of {len(ok_results)}):")
            scarce = [r for r in short_written if r['n_candidate_segments'] < n_target]
            parse_loss = [r for r in short_written if r['n_candidate_segments'] >= n_target]
            if scarce:
                ns = sorted(r['n_candidate_segments'] for r in scarce)
                print(f"    (a) {len(scarce)} patients have <{n_target} candidate "
                      f"segments (genuine scarcity of contiguous data)")
                print(f"        n_candidate_segments distribution: "
                      f"min={ns[0]}, median={ns[len(ns)//2]}, max={ns[-1]}")
                print(f"        example pids (first 10): {[r['pid'] for r in scarce[:10]]}")
            if parse_loss:
                gaps = [n_target - r['n_written'] for r in parse_loss]
                print(f"    (b) {len(parse_loss)} patients had >={n_target} candidates "
                      f"but wrote less (parse errors)")
                print(f"        gap distribution: min={min(gaps)}, "
                      f"median={sorted(gaps)[len(gaps)//2]}, max={max(gaps)}")

    warn_no_ehr = [r for r in results if r['status'] == 'WARN_NO_EHR_HISTORY']
    if warn_no_ehr:
        print(f"\n  !!! WARNING: {len(warn_no_ehr)} patients had NO EHR history. "
              f"Affected pids (first 20): {[r['pid'] for r in warn_no_ehr[:20]]}")

    errors = [r for r in results if r['status'] and r['status'].startswith('ERROR')]
    if errors:
        print(f"\n  !!! {len(errors)} hard errors. First 5:")
        for r in errors[:5]:
            print(f"    {r['pid']}: {r['status']} - {r['error']}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['III', 'IV', 'both'], default='both')
    parser.add_argument('--n-workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None,
                         help='Only process first N patients (for testing)')
    args = parser.parse_args()

    datasets = ['III', 'IV'] if args.dataset == 'both' else [args.dataset]
    t_start = time.time()
    for ds in datasets:
        run_dataset(ds, args.n_workers, limit=args.limit)

    print(f"\n{'='*70}")
    print(f"ALL DONE in {time.time() - t_start:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()