"""
Per-admission metadata for downstream_features_admission_postdisch.joblib.

Two things this file is for:

  1. Candidate gate inputs -- signal quality, EHR volume, PPG timing, length of
     stay -- for any fusion model that has to decide how much to trust each
     modality.
  2. Confounder audit. Every field here is a candidate free predictor: if it
     forecasts the outcome on its own, a gate that reads it smuggles that
     prediction into the logit, and "fusion helped" stops being interpretable.
     Nothing here is safe until it has been scored on its own.

DESCRIBES THE BLOCKS THE EMBEDDING ACTUALLY USED
------------------------------------------------
Stage 4 reconstructs every non-overlapping 10-min block inside
[admittime, dischtime], then caps at max_blocks_per_sample=100 by taking a
linspace over the globally time-sorted eligible blocks. Across the cohort that
is 192,884 used out of 769,144 eligible -- a quarter.

Metadata computed over all 769k would describe signal the model never saw. This
file reproduces the cap exactly (same global sort, same linspace) and computes
everything over the SELECTED blocks. That is also four times cheaper, since SQI
is the only part that has to decode waveform data.

The reproduction is checked, not trusted: the number of blocks selected here
must equal n_ppg_blocks_used stored by stage 4. A mismatch means the selection
logic has drifted and every SQI figure below describes different signal from
the embedding.

WHAT IS NEW RELATIVE TO THE PRE-DISCHARGE VERSION
--------------------------------------------------
  anchor        the prediction origin moved from admittime to dischtime, so
                every timing field is reported against BOTH. Distance from
                admit says where in the illness course monitoring happened;
                distance from discharge says how far the signal sits from the
                point the label starts counting. They are not independent --
                gap_from_admit + gap_to_disch = los_h -- and the correlation
                block at the end makes that visible rather than implied.

  los_h         length of stay, the most suspicious confounder in this design.
                It bounds how much PPG can exist, it shifts where the label
                window opens, and it tracks severity. It was not available
                before because the old anchor was the admission day alone.

  *__prior      carried through from the label file: was this disease already
                coded on or before discharge. 81-90% of positives have it, and
                the sEHR window [disch-12mo, disch] contains the codes that set
                it, so the sEHR branch can nearly read the label off its own
                input. Any downstream claim about fusion needs a
                "sEHR + prior" baseline, and this file exists partly so that
                prior can be audited like any other variable.

  EHR counts    counted over the window actually fed to the model,
                [dischtime - 12mo, dischtime], not the old matched/strict
                split. Diagnosis codes carry the admittime of their stay, so
                the index admission's own codes fall inside this window --
                that is where *__prior comes from, and n_codes_index_adm
                measures it directly.

Usage:
    python extract_postdisch_metadata.py                 # pass 1, minutes
    python extract_postdisch_metadata.py --with-sqi      # pass 2, hours
    python extract_postdisch_metadata.py --with-sqi --limit 300   # sample first
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import paths

import os

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

FEATURES = str(paths.data_root() / "ds1/downstream_features_admission_postdisch.joblib")
DB_PATH = {'III': str(paths.data_root() / "patient_history_db_III.joblib"),
           'IV':  str(paths.data_root() / "patient_history_db_IV.joblib")}
PPG_DIR = {'III': str(paths.data_root() / "MIMIC_III_ppg_filtered_by_patient"),
           'IV':  str(paths.data_root() / "MIMIC_IV_ppg_filtered_by_patient")}
OUT_DIR = str(paths.data_root() / "ds1")

# these must match stage 4 exactly or the reproduced selection is meaningless
PPG_PARQUET = "ppg.parquet"
SIGNAL_COL = "signal"
N_WINDOWS_PER_SEG = 60
WINDOW_SIGNAL_LEN = 1250
WIN_S = 10
GAP_TOL = (9.9, 10.1)
MAX_BLOCKS_PER_SAMPLE = 100
FS = 125
WINDOW_MONTHS = 12

DISEASES = ('AF', 'HF', 'eMACE')
HORIZONS = (30, 90, 180, 365)


def canon_pid(raw, dataset):
    s = str(raw).strip()
    return s.zfill(6) if dataset == 'III' else s


# ══════════════════════════════════════════════════════════════════════════
# PPG blocks -- reconstruction identical to stage 4
# ══════════════════════════════════════════════════════════════════════════
def read_ppg_frame(pid, dataset, need_signal):
    """Parquet for one patient. Signal column only when SQI is requested."""
    p = Path(PPG_DIR[dataset]) / pid / PPG_PARQUET
    if not p.exists():
        return None
    cols = ['record_id', 'segment', 'window_start_time']
    if need_signal:
        cols.append(SIGNAL_COL)
    try:
        df = pd.read_parquet(p, columns=cols)
    except Exception:
        print(f"  !! unreadable parquet: {p}")
        return None
    if len(df) == 0:
        return None
    df['window_start_time'] = pd.to_datetime(df['window_start_time'])
    return df


def assemble_blocks(df):
    """Every non-overlapping run of 60 windows ~10 s apart within a segment.

    Byte-for-byte the same rule as stage 4's load_patient_ppg: consecutive runs
    inside one (record_id, segment), reset after each complete block so blocks
    never overlap, and a global sort by start time at the end -- groupby only
    orders within a segment, and the linspace cap depends on the global order.
    """
    lo, hi = GAP_TOL
    mid = N_WINDOWS_PER_SEG // 2
    pos_l, start_l, center_l, end_l = [], [], [], []

    for (_rec, _seg), g in df.groupby(['record_id', 'segment'], sort=False):
        g = g.sort_values('window_start_time')
        times = g['window_start_time'].values.astype('datetime64[ns]')
        pos = g.index.values
        run = []
        for i in range(len(times)):
            if i == 0:
                run = [0]
            else:
                dt = (times[i] - times[i - 1]) / np.timedelta64(1, 's')
                run = run + [i] if lo <= dt <= hi else [i]
            if len(run) == N_WINDOWS_PER_SEG:
                pos_l.append([pos[j] for j in run])
                start_l.append(times[run[0]])
                center_l.append(times[run[mid]])
                end_l.append(times[run[-1]] + np.timedelta64(WIN_S, 's'))
                run = []

    if not pos_l:
        return None
    starts = np.array(start_l, dtype='datetime64[ns]')
    order = np.argsort(starts)
    return {
        'pos':     [pos_l[i] for i in order],
        'starts':  starts[order],
        'centers': np.array(center_l, dtype='datetime64[ns]')[order],
        'ends':    np.array(end_l, dtype='datetime64[ns]')[order],
    }


def select_blocks(blocks, admit, disch):
    """Stage 4's eligibility and cap: whole block inside the stay, then cap.

    Containment is on the FULL block, not its centre: the label window opens at
    dischtime, so a block straddling discharge would carry signal from after
    the prediction origin.
    """
    ok = (blocks['starts'] >= admit) & (blocks['ends'] <= disch)
    idx = np.where(ok)[0]
    n_eligible = len(idx)
    if n_eligible > MAX_BLOCKS_PER_SAMPLE:
        sel = np.linspace(0, n_eligible - 1, MAX_BLOCKS_PER_SAMPLE).astype(int)
        idx = idx[sel]
    return idx, n_eligible


# ══════════════════════════════════════════════════════════════════════════
# SQI
# ══════════════════════════════════════════════════════════════════════════
def sqi_for_windows(sigs):
    """(acf, skew, tmpl) per 10-s window; NaN where a metric cannot be formed.

    acf   height of the highest autocorrelation peak in the plausible
          heart-rate lag range. A pulsatile trace repeats; noise does not.
    skew  waveform asymmetry. A clean PPG rises fast and falls slowly, so it is
          positively skewed; artefact tends toward symmetry.
    tmpl  correlation of each detected beat against the subject's own mean
          beat. Needs peak detection, which is exactly what fails on the
          artefact windows it is meant to flag -- those become NaN and are
          skipped, so this metric is blind in the situation it exists for.
          Kept for now so the distributions can decide.
    """
    from scipy.stats import skew as _skew
    from scipy.signal import find_peaks

    lag_lo, lag_hi = int(FS * 0.4), int(FS * 1.5)      # 40-150 bpm
    acf_v, skew_v, tmpl_v = [], [], []

    for x in sigs:
        x = np.asarray(x, dtype=np.float64)
        if x.size != WINDOW_SIGNAL_LEN or not np.isfinite(x).all():
            acf_v.append(np.nan); skew_v.append(np.nan); tmpl_v.append(np.nan)
            continue
        sd = x.std()
        if sd < 1e-12:                                  # flat line
            acf_v.append(np.nan); skew_v.append(np.nan); tmpl_v.append(np.nan)
            continue
        z = (x - x.mean()) / sd

        ac = np.correlate(z, z, mode='full')[len(z) - 1:]
        ac = ac / ac[0]
        acf_v.append(float(np.max(ac[lag_lo:lag_hi]))
                     if lag_hi < len(ac) else np.nan)

        skew_v.append(float(_skew(x)))

        try:
            pk, _ = find_peaks(z, distance=int(FS * 0.4), height=0.0)
            if len(pk) < 3:
                tmpl_v.append(np.nan)
            else:
                half = int(FS * 0.35)
                beats = [z[p - half:p + half] for p in pk
                         if p - half >= 0 and p + half <= len(z)]
                beats = [b for b in beats if len(b) == 2 * half]
                if len(beats) < 3:
                    tmpl_v.append(np.nan)
                else:
                    B = np.stack(beats)
                    tpl = B.mean(axis=0)
                    if tpl.std() < 1e-12:
                        tmpl_v.append(np.nan)
                    else:
                        cs = [np.corrcoef(b, tpl)[0, 1] for b in B
                              if b.std() > 1e-12]
                        tmpl_v.append(float(np.mean(cs)) if cs else np.nan)
        except Exception:
            tmpl_v.append(np.nan)

    return np.array(acf_v), np.array(skew_v), np.array(tmpl_v)


def summarise(v):
    """median and p10 of the finite entries, plus how many were finite.

    p10 alongside the median because the embedding is a flat mean over every
    used window -- bad windows are already inside it, and a median hides
    precisely the tail that put them there.
    """
    f = v[np.isfinite(v)]
    if f.size == 0:
        return np.nan, np.nan, 0
    return float(np.median(f)), float(np.percentile(f, 10)), int(f.size)


# ══════════════════════════════════════════════════════════════════════════
# EHR
# ══════════════════════════════════════════════════════════════════════════
CODE_TYPES = ('diagnosis', 'procedure', 'medication')
MAX_SEQ_LEN = 512


def ehr_counts(entry, admit, disch, tokenizer=None):
    """Reproduce stage 4's forward_ehr selection, step for step.

    The counts here are only meaningful if they describe the SAME codes the
    encoder consumed, so this mirrors forward_ehr rather than re-deriving a
    reasonable-looking window. Stage 4 does, in order:

        days = int(window_months * 30.44)              # 12 -> 365, not "a year"
        anchor_day = Timestamp(anchor_time).normalize()
        hist_day   = timestamps floored to the day
        mask = (hist_day >= anchor_day - 365d) & (hist_day <= anchor_day)
        if len(sub) > 512: keep the 512 with the latest timestamps
        indices = tokenizer.encode(sub_codes)
        valid   = indices != 0
        pooled  = sum(out * valid) / sum(valid)

    Three details matter and are easy to get wrong:

      day granularity   history_db timestamps are mixed -- 19% carry a time,
                        81% sit at midnight -- and forward_ehr floors both
                        sides anyway. Comparing raw timestamps would put a
                        same-day medication (stored 00:00) BEFORE an admission
                        at 14:30 and classify this stay's own codes as prior
                        history.
      365 days          int(12 * 30.44) = 365, not twelve calendar months.
      valid != 0        codes outside the tokenizer's vocabulary encode to 0
                        and are excluded from the pooling, so the number the
                        encoder actually averaged over is smaller than the
                        number of codes in the window. Passing a tokenizer here
                        measures that; without one, n_codes_fed is an upper
                        bound and is flagged as such.

    Stage 4 stores no code count, so there is nothing to reconcile against
    directly. The one equivalence that IS checkable is
    (n_codes_window > 0) == (ehr_mask == 1), because forward_ehr returns None
    when the window is empty; main() asserts it.

    Counts returned
    ---------------
    n_codes_window        codes inside the window
    n_codes_after_trunc   after the 512 cap -- what was tokenized
    n_codes_fed           after dropping out-of-vocabulary codes; what the
                          pooling averaged over. Exact with a tokenizer,
                          otherwise = n_codes_after_trunc and marked in
                          n_codes_fed_is_upper_bound.
    n_codes_before_admit  inside the window, on days before the admission day
    n_codes_index_adm     inside the window, on days from admission to
                          discharge inclusive -- this stay's own codes.
                          Procedures and medications carry their real
                          timestamps during the stay, so `== admittime` would
                          have counted diagnoses only.
    """
    out = {
        'n_codes_window': 0,
        'n_codes_after_trunc': 0,
        'n_codes_fed': 0,
        'n_codes_unk': np.nan,
        'n_unique_codes_window': 0,
        'n_codes_before_admit': 0,
        'n_codes_index_adm': 0,
        'n_codes_after_disch_day': 0,
    }
    for t in CODE_TYPES:
        out[f'n_codes_window_{t}'] = 0
    if entry is None:
        return out

    ts_raw = pd.to_datetime(entry.get('timestamps', []))
    if len(ts_raw) == 0:
        return out
    ts = np.asarray(ts_raw).astype('datetime64[ns]')
    codes = np.asarray(entry.get('codes', []), dtype=object)
    types = np.asarray(entry.get('code_type', []), dtype=object)

    # 三个数组是平行的。截断到最短会无声地丢掉尾部的 code,
    # 下面所有计数(包括 confounder audit 依赖的那些)都会对
    # 恰好 history_db 有问题的那批患者偏低。这是 history 文件
    # 的错误,不该在这里掩盖。
    if not (len(ts) == len(codes) == len(types)):
        raise ValueError(
            f"history entry has mismatched lengths -- timestamps {len(ts)}, "
            f"codes {len(codes)}, code_type {len(types)}. These are parallel "
            f"arrays; fix the history file rather than truncating here.")

    unknown = set(types.tolist()) - set(CODE_TYPES)
    if unknown:
        raise ValueError(
            f"unexpected code_type values {sorted(unknown)}; the per-type "
            f"counts would silently read zero for them")

    # forward_ehr's window, reproduced exactly
    day = ts.astype('datetime64[D]').astype('datetime64[ns]')
    anchor_day = np.datetime64(pd.Timestamp(disch).normalize())
    admit_day = np.datetime64(pd.Timestamp(admit).normalize())
    lo = anchor_day - np.timedelta64(int(WINDOW_MONTHS * 30.44), 'D')
    win = (day >= lo) & (day <= anchor_day)

    out['n_codes_window'] = int(win.sum())
    out['n_unique_codes_window'] = int(len(set(codes[win].tolist())))
    out['n_codes_before_admit'] = int((win & (day < admit_day)).sum())
    out['n_codes_index_adm'] = int(
        (win & (day >= admit_day) & (day <= anchor_day)).sum())
    # inside the day-level window but after dischtime itself: these sit in the
    # sEHR input AND in the label window (disch, disch+N]. Only reachable when
    # a readmission starts on the discharge date; counted so it stays visible.
    out['n_codes_after_disch_day'] = int(
        (win & (ts > np.datetime64(pd.Timestamp(disch)))).sum())
    for t in CODE_TYPES:
        out[f'n_codes_window_{t}'] = int((win & (types == t)).sum())

    if not win.any():
        return out

    # the 512 cap: keep the latest by timestamp
    sub_codes, sub_ts = codes[win], ts[win]
    if len(sub_codes) > MAX_SEQ_LEN:
        order = np.argsort(sub_ts)
        keep = np.sort(order[-MAX_SEQ_LEN:])
        sub_codes = sub_codes[keep]
    out['n_codes_after_trunc'] = int(len(sub_codes))

    if tokenizer is None:
        # Without the tokenizer we know how many codes were passed to the encoder,
        # but cannot determine how many were mapped to [UNK].
        out['n_codes_fed'] = int(len(sub_codes))
        out['n_codes_unk'] = np.nan
    else:
        idx = np.asarray(tokenizer.encode(sub_codes.tolist()))

        pad_id = tokenizer.special_tokens['[PAD]']
        unk_id = tokenizer.special_tokens['[UNK]']

        # Stage 4 masks only PAD tokens. OOV codes become [UNK] and remain valid.
        out['n_codes_fed'] = int((idx != pad_id).sum())
        out['n_codes_unk'] = int((idx == unk_id).sum())
    return out


# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default=FEATURES)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--with-sqi', action='store_true')
    ap.add_argument('--tokenizer',
                    default=str(paths.data_root() / "medical_tokenizer_combined.json"),
                    help="stage 4's tokenizer; without it n_codes_fed is only "
                         "an upper bound")
    ap.add_argument('--no-tokenizer', action='store_true')
    ap.add_argument('--limit', type=int, default=0,
                    help="process only the first N admissions (sampling run)")
    args = ap.parse_args()

    if args.with_sqi:
        try:
            import scipy  # noqa: F401
        except ImportError:
            sys.exit("--with-sqi needs scipy")

    feat = joblib.load(args.features)
    if args.limit:
        feat = feat[:args.limit]
    print(f"{len(feat)} admissions from {os.path.basename(args.features)}")

    # Loaded so n_codes_fed counts what the pooling actually averaged over:
    # out-of-vocabulary codes encode to 0 and forward_ehr's `valid` mask drops
    # them. Without it the count is an upper bound, which is flagged per row.
    tokenizer = None
    if not args.no_tokenizer:
        try:
            sys.path.insert(0, os.path.join(_ROOT, "sehr_bert"))
            from sEHR_BERT import MedicalCodeTokenizer
            tokenizer = MedicalCodeTokenizer()
            tokenizer.load(args.tokenizer)
            print(f"  tokenizer loaded, vocab {tokenizer.vocab_size}")
        except Exception as e:
            raise RuntimeError(f"  !! tokenizer unavailable ({e}); n_codes_fed will be an "
                  f"upper bound")

    dbs, lookups = {}, {}
    for ds, p in DB_PATH.items():
        dbs[ds] = joblib.load(p)
        lookups[ds] = {canon_pid(k, ds): k for k in dbs[ds].keys()}

    # group by patient so each parquet is read once
    by_pat = {}
    for i, d in enumerate(feat):
        by_pat.setdefault((d['dataset'], str(d['patient_id'])), []).append(i)

    rows = [None] * len(feat)
    n_sel_mismatch = 0
    n_elig_mismatch = 0
    n_len_bad = 0

    for (ds, pid), idxs in tqdm(by_pat.items(), desc="admissions"):
        need_ppg = any(feat[i]['ppg_mask'] == 1 for i in idxs)
        df = read_ppg_frame(pid, ds, args.with_sqi) if need_ppg else None
        blocks = assemble_blocks(df) if df is not None else None

        entry = dbs[ds].get(lookups[ds].get(pid))

        for i in idxs:
            d = feat[i]
            admit = np.datetime64(pd.Timestamp(d['admit_time']))
            disch = np.datetime64(pd.Timestamp(d['dischtime']))
            los_h = (disch - admit) / np.timedelta64(1, 'h')

            r = {'patient_id': pid, 'dataset': ds, 'hadm_id': d.get('hadm_id'),
                 'admit_time': d['admit_time'], 'dischtime': d['dischtime'],
                 'los_h': float(los_h),
                 'ppg_mask': d['ppg_mask'], 'ehr_mask': d['ehr_mask'],
                 'n_ppg_blocks_available_file': d.get('n_ppg_blocks_available'),
                 'n_ppg_blocks_used_file': d.get('n_ppg_blocks_used')}
            for dis in DISEASES:
                r[f'{dis}__prior'] = d.get(f'{dis}__prior')
                for N in HORIZONS:
                    r[f'{dis}__{N}'] = d.get(f'{dis}__{N}')

            # EHR metadata must be filled for EHR-only admissions too, so
            # this has to precede any PPG-related early exit
            r.update(ehr_counts(entry, admit, disch, tokenizer))

            if blocks is None:
                r.update(n_blocks_eligible=0, n_blocks_selected=0,
                         n_records=0, span_h=np.nan)
                rows[i] = r
                continue

            sel, n_elig = select_blocks(blocks, admit, disch)
            r['n_blocks_eligible'] = int(n_elig)
            r['n_blocks_selected'] = int(len(sel))
            # Two independent checks. `eligible` depends only on block
            # assembly and the containment rule, so it validates the
            # reconstruction; `used` additionally depends on the linspace cap.
            # Matching on `used` alone would pass even if a different set of
            # blocks had been assembled, as long as the count came out the same.
            if (d.get('n_ppg_blocks_available') is not None
                    and int(d['n_ppg_blocks_available']) != n_elig):
                n_elig_mismatch += 1
            if (d.get('n_ppg_blocks_used') is not None
                    and int(d['n_ppg_blocks_used']) != len(sel)):
                n_sel_mismatch += 1

            if len(sel) == 0:
                r.update(n_records=0, span_h=np.nan)
                rows[i] = r
                continue

            st, ct, en = (blocks['starts'][sel], blocks['centers'][sel],
                          blocks['ends'][sel])
            h = lambda x: float(x / np.timedelta64(1, 'h'))
            r['gap_first_from_admit_h'] = h(st[0] - admit)
            r['gap_median_from_admit_h'] = h(np.median(ct - admit))
            r['gap_last_from_admit_h'] = h(en[-1] - admit)
            r['gap_first_to_disch_h'] = h(disch - st[0])
            r['gap_median_to_disch_h'] = h(np.median(disch - ct))
            r['gap_last_to_disch_h'] = h(disch - en[-1])
            r['span_h'] = h(en[-1] - st[0])
            r['covered_minutes'] = float(len(sel) * 10)
            r['n_records'] = int(pd.Series(
                [df.at[p[0], 'record_id'] for p in
                 [blocks['pos'][k] for k in sel]]).nunique())

            if args.with_sqi:
                sigs = []
                for k in sel:
                    for ix in blocks['pos'][k]:
                        a = np.asarray(json.loads(df.at[ix, SIGNAL_COL]),
                                       dtype=np.float32)
                        if a.size != WINDOW_SIGNAL_LEN:
                            n_len_bad += 1
                            continue
                        sigs.append(a)
                if sigs:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        acf, skw, tpl = sqi_for_windows(sigs)
                    for nm, v in (('acf', acf), ('skew', skw), ('tmpl', tpl)):
                        med, p10, nfin = summarise(v)
                        r[f'{nm}_median'] = med
                        r[f'{nm}_p10'] = p10
                        r[f'frac_windows_{nm}_failed'] = (
                            1.0 - nfin / len(v) if len(v) else np.nan)
                    r['n_windows_sqi'] = len(sigs)
            rows[i] = r

    md = pd.DataFrame(rows)
    tag = '_sqi' if args.with_sqi else ''
    out = os.path.join(args.out_dir, f'admission_metadata_postdisch{tag}.csv')
    md.to_csv(out, index=False)

    # ── self-checks ──
    print("\n" + "=" * 96)
    print("SELF-CHECK  (all must be 0)")
    print("=" * 96)
    if 'n_codes_window' not in md.columns:
        raise RuntimeError(
            "n_codes_window is missing, so ehr_counts never ran. The EHR "
            "columns do not exist and no check below means anything.")
    c1 = int(((md.n_blocks_selected > 0) != (md.ppg_mask == 1)).sum())
    c2 = int(((md.n_codes_window > 0) != (md.ehr_mask == 1)).sum())
    print(f"  (n_blocks_selected > 0) != ppg_mask   : {c1}")
    print(f"  (n_codes_window > 0)  != ehr_mask     : {c2}")
    if c2:
        print("     -> forward_ehr returns None exactly when its window is")
        print("        empty, so this can only differ if the window here is")
        print("        not the window stage 4 used: check the 365-day width,")
        print("        the day flooring, and that the anchor is dischtime.")
    print(f"  eligibility differs from n_ppg_blocks_available : "
          f"{n_elig_mismatch}")
    if n_elig_mismatch:
        print("     -> block assembly or the containment rule differs from")
        print("        stage 4. Nothing computed over blocks is comparable.")
    print(f"  selection differs from n_ppg_blocks_used       : "
          f"{n_sel_mismatch}")
    if n_sel_mismatch:
        print("     -> the cap was not reproduced. Every field computed over")
        print("        the selected blocks describes different signal from the")
        print("        embedding; do not use them until this is 0.")
    n_leak = int(md.get('n_codes_after_disch_day', pd.Series([0])).sum())
    print(f"  codes inside the sEHR window but after dischtime: {n_leak}")
    if n_leak:
        print("     -> these sit in the input AND in the label window. Only")
        print("        possible when a readmission begins on the discharge")
        print("        date; small counts are a stated limitation, not a bug.")
    if args.with_sqi:
        print(f"  windows dropped for length != {WINDOW_SIGNAL_LEN} : {n_len_bad}")

    both = md[(md.ppg_mask == 1) & (md.ehr_mask == 1)]
    print(f"\n  cohort: {len(md)} admissions | both-modality {len(both)}")

    # ── distributions ──
    print("\n" + "=" * 96)
    print("DISTRIBUTIONS  (both-modality cohort)")
    print("=" * 96)
    fields = ['los_h', 'n_blocks_eligible', 'n_blocks_selected', 'n_records',
              'span_h', 'covered_minutes',
              'gap_first_from_admit_h', 'gap_median_from_admit_h',
              'gap_last_from_admit_h', 'gap_first_to_disch_h',
              'gap_median_to_disch_h', 'gap_last_to_disch_h']
    if args.with_sqi:
        fields += ['acf_median', 'acf_p10', 'skew_median', 'skew_p10',
                   'tmpl_median', 'tmpl_p10', 'frac_windows_tmpl_failed',
                   'n_windows_sqi']
    print(f"  {'field':28s} {'n':>5s} {'p10':>10s} {'p50':>10s} "
          f"{'p90':>10s} {'IQR':>10s}")
    for f in fields:
        if f not in both:
            continue
        v = pd.to_numeric(both[f], errors='coerce').dropna()
        if v.empty:
            print(f"  {f:28s} {'--':>5s}")
            continue
        q = v.quantile([.1, .25, .5, .75, .9])
        print(f"  {f:28s} {len(v):>5d} {q[.1]:>10.4g} {q[.5]:>10.4g} "
              f"{q[.9]:>10.4g} {q[.75]-q[.25]:>10.4g}")
    print()
    print("  A near-zero IQR means the field is constant on this cohort and")
    print("  cannot inform a gate. n_blocks_selected saturating at "
          f"{MAX_BLOCKS_PER_SAMPLE} means")
    print("  the cap binds for most admissions, so it carries far less")
    print("  information than n_blocks_eligible does.")

    # ── EHR counts ──
    print("\n" + "=" * 96)
    print("EHR COUNTS  (window actually fed to the model: [disch-12mo, disch])")
    print("=" * 96)
    for f in ['n_codes_window', 'n_codes_after_trunc', 'n_codes_fed',
            'n_codes_unk', 'n_unique_codes_window',
            'n_codes_before_admit', 'n_codes_index_adm',
            'n_codes_window_diagnosis',
            'n_codes_window_procedure',
            'n_codes_window_medication']:
        if f not in both:
            continue
        v = pd.to_numeric(both[f], errors='coerce').dropna()
        q = v.quantile([.1, .5, .9])
        print(f"  {f:28s} p10={q[.1]:>8.4g} p50={q[.5]:>8.4g} "
              f"p90={q[.9]:>8.4g}  zero={int((v == 0).sum())}")
    print()
    print("  n_codes_window -> n_codes_after_trunc reflects the 512-token cap.")
    print("  n_codes_unk counts tokens mapped to [UNK]; these remain valid")
    print("  encoder inputs and are included in the pooled representation.")

    print()
    print("  n_codes_index_adm is this stay's own codes. Diagnosis codes carry")
    print("  their stay's admittime, so they sit inside the sEHR window and are")
    print("  what sets *__prior. With 81-90% of positives already prior=1, the")
    print("  sEHR branch can nearly read the label off its own input -- which")
    print("  is why a 'sEHR + prior' baseline is mandatory downstream, and why")
    print("  this column needs auditing like any other candidate.")

    # ── correlations ──
    print("\n" + "=" * 96)
    print("CORRELATION  (|r| >= 0.6 among numeric candidates)")
    print("=" * 96)
    cand = [f for f in fields + ['n_codes_window', 'n_codes_index_adm',
                                 'n_codes_before_admit']
            if f in both and pd.to_numeric(both[f], errors='coerce').notna().sum() > 10]
    C = both[cand].apply(pd.to_numeric, errors='coerce').corr()
    seen = set()
    for a in cand:
        for b in cand:
            if a == b or (b, a) in seen:
                continue
            seen.add((a, b))
            r = C.loc[a, b]
            if np.isfinite(r) and abs(r) >= 0.6:
                print(f"  {a:28s} {b:28s} r={r:+.3f}")
    print()
    print("  gap_*_from_admit and gap_*_to_disch are two views of one quantity:")
    print("  their sum is los_h by construction, so they will correlate")
    print("  strongly and only one anchor should enter a gate. Which one is a")
    print("  question about meaning, not fit -- distance from admit says where")
    print("  in the illness course the monitoring sat, distance from discharge")
    print("  says how far the signal is from where the label starts counting.")

    print(f"\nSaved {out}")
    if not args.with_sqi:
        print("Re-run with --with-sqi for the signal-quality fields.")


if __name__ == '__main__':
    main()