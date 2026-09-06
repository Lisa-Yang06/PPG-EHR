"""
Rebuild the sample set from the admission table, label it, and count how much
of it carries PPG. Stages 1-3.5 of the plan, in one run.

WHAT CHANGED AND WHY
--------------------
The previous sample set (1638 rows) was built by treating every day-floored
DIAGNOSIS-CODE timestamp as an "admission", then dropping every admission on or
after the patient's first MACE onset. Both parts are now wrong for this task:

  the anchor      a day with a diagnosis code is not necessarily a
                  hospitalisation, and the new label window opens at
                  dischtime, which such a day does not have. The stay table
                  gives real admissions with real admittime/dischtime, so
                  hadm_id becomes the sample id and nothing has to be inferred.

  the drop rule   "admission_day >= onset_day -> DROP" existed because the old
                  target was time-to-FIRST-onset, and post-onset rows are not
                  prediction. The new label is "is this disease coded after
                  this stay", which is perfectly well defined afterwards -- and
                  the dropped rows are exactly the follow-up admissions of
                  patients who already have cardiovascular disease, i.e. where
                  the positives are densest. Removing the rule is the point of
                  this rebuild.

LABEL
-----
    label(admission, disease, N)
        = 1   if any code of that disease falls in (dischtime, dischtime + N]
        = NA  if no such code AND the patient died on or before dischtime + N
        = 0   otherwise

A death censors only when the outcome is otherwise unknown. If the code already
appeared inside the window the event is observed and the label stays 1 whatever
happened next; censoring an observed positive would discard the scarcest thing
in this dataset.

Death comes from dod, not from the stay table's deathtime: deathtime is
populated only for deaths DURING that stay, so it is empty for exactly the
deaths that matter here -- discharged alive, dies three weeks later. deathtime
only fills gaps dod leaves.

STAGE 3.5 IS THE DECISION POINT
-------------------------------
PPG availability is computed here directly from the parquet window timestamps
rather than waiting for feature extraction, because it is cheap and it decides
whether extraction is worth running at all. The previous cohort had 953
PPG-carrying admissions and 3/7/9/17 positives among them, which is why nothing
downstream could be resolved. If removing the drop rule does not move that
number materially, the expensive stage should not be started.

WHAT IS NOT HANDLED
-------------------
Loss to follow-up: an admission whose patient has no coded activity after
dischtime + N is scored 0, though the honest answer is "not observed". MIMIC
has no follow-up-end field. The effect is one-directional -- 0s at long
horizons are contaminated, so positive rates there are underestimates, more so
for patients who rarely present.

Out-of-hospital death recording is itself incomplete (III via a social-security
index, IV via state records), so "no dod" is not proof of survival. Coverage is
reported; the limitation cannot be closed in code.

Usage:
    python build_admission_labels.py
    python build_admission_labels.py --no-test-only     # ignore the split
    python build_admission_labels.py --skip-ppg         # stages 1-3 only
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
os.environ.setdefault('MPLCONFIGDIR', str(paths.data_root() / ".cache/matplotlib"))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

SPLITS_PATH = str(paths.data_root() / "splits/alignment_splits_combined/splits.json")
STAY_CSV = {'III': str(paths.data_root() / "ds1/data/iii.csv"),
            'IV':  str(paths.data_root() / "ds1/data/iv.csv")}
DOD_CSV = {'III': str(paths.data_root() / "ds3/data/iii_dod.csv"),
           'IV':  str(paths.data_root() / "ds3/data/iv_dod.csv")}
DB_PATH = {'III': str(paths.data_root() / "patient_history_db_III.joblib"),
           'IV':  str(paths.data_root() / "patient_history_db_IV.joblib")}
PPG_DIR = {'III': str(paths.data_root() / "MIMIC_III_ppg_filtered_by_patient"),
           'IV':  str(paths.data_root() / "MIMIC_IV_ppg_filtered_by_patient")}
PPG_PARQUET = "ppg.parquet"
OUT_DIR = str(paths.data_root() / "ds1")

# ICD-10-CM, dots stripped; ICD-9 is already mapped upstream.
#
# Ordered by how directly a peripheral photoplethysmogram could carry
# information about the condition. This is mechanistic reasoning, not a
# clinical guideline, and wants review by someone with clinical training:
#
#   AF      a PPG measures pulse timing and atrial fibrillation IS a timing
#           abnormality -- the most direct target available.
#   HF      changes stroke volume and peripheral perfusion, hence pulse
#           amplitude, upstroke and waveform shape. Indirect but grounded.
#   AMI     perturbs haemodynamics acutely, but as a code says little about the
#           waveform at the time of measurement.
#   CAD_UA  coronary narrowing sits in vessels a peripheral PPG never sees.
#   STROKE  cerebrovascular, no expected peripheral pulse signature.
#
# Composites are included so that the cost of mixing an informative component
# with an uninformative one is measured rather than assumed.
DISEASES = {
    'AF':     (("I48",), frozenset()),
    'HF':     (("I50",), frozenset()),
    'AMI':    (("I21", "I22"), frozenset()),
    'STROKE': (("I60", "I61", "I62", "I63"), frozenset({"I64"})),
    'eMACE': (
        ("I21", "I22",      # acute MI
        "I60", "I61", "I62", "I63", "I64",   # stroke
        "I50"),            # heart failure
        frozenset()
    ),
    'AF_HF':  (("I48", "I50"), frozenset()),
}


def norm_code(c):
    return str(c).replace('.', '').upper().strip()


def hits(codes, prefixes, exact):
    for c in codes:
        if (prefixes and c.startswith(prefixes)) or c in exact:
            return True
    return False


def canon_pid(raw, dataset):
    """III is stored zero-padded to 6, IV as-is."""
    s = str(raw).strip()
    return s.zfill(6) if dataset == 'III' else s


def load_test_pids():
    with open(SPLITS_PATH) as f:
        sp = json.load(f)
    return {'III': {canon_pid(p, 'III') for p in sp['test'].get('III', [])},
            'IV':  {canon_pid(p, 'IV') for p in sp['test'].get('IV', [])}}


def load_admissions(test_pids):
    """One row per real hospitalisation, from the stay table."""
    frames = []
    for ds, path in STAY_CSV.items():
        df = pd.read_csv(path, dtype={'subject_id': str})
        for c in ('admittime', 'dischtime', 'deathtime'):
            df[c] = pd.to_datetime(df[c], errors='coerce')
        df['dataset'] = ds
        df['patient_id'] = df['subject_id'].map(lambda s: canon_pid(s, ds))
        n_all = len(df)
        if test_pids is not None:
            df = df[df['patient_id'].isin(test_pids[ds])]
        n_bad = int(df['dischtime'].isna().sum())
        df = df.dropna(subset=['admittime', 'dischtime'])
        df = df[df['dischtime'] > df['admittime']]
        print(f"  {ds}: {n_all} stays -> {len(df)} usable"
              f"  ({'test-only, ' if test_pids is not None else ''}"
              f"{n_bad} lacked dischtime)")
        frames.append(df[['patient_id', 'dataset', 'hadm_id', 'admittime',
                          'dischtime', 'deathtime']])
    return pd.concat(frames, ignore_index=True)


def load_dod():
    frames = []
    for ds, path in DOD_CSV.items():
        if not os.path.exists(path):
            print(f"  !! {path} missing; {ds} deaths rely on deathtime alone")
            continue
        df = pd.read_csv(path, dtype={'subject_id': str})
        df['dod'] = pd.to_datetime(df['dod'], errors='coerce')
        df['dataset'] = ds
        df['patient_id'] = df['subject_id'].map(lambda s: canon_pid(s, ds))
        df = (df.dropna(subset=['dod']).sort_values('dod')
                .drop_duplicates(['patient_id', 'dataset'], keep='first'))
        print(f"  {ds}: {len(df)} patients with a dod")
        frames.append(df[['patient_id', 'dataset', 'dod']])
    return (pd.concat(frames, ignore_index=True) if frames else
            pd.DataFrame(columns=['patient_id', 'dataset', 'dod']))


def patient_codes(pid, db, lookup):
    raw = lookup.get(pid)
    if raw is None:
        return np.array([], dtype=object), np.array([], dtype='datetime64[ns]')
    d = db[raw]
    codes, times = d.get('codes', []), d.get('timestamps', [])
    kc, kt = [], []
    for i in range(min(len(codes), len(times))):
        if times[i] is None or pd.isna(times[i]):
            continue
        kc.append(norm_code(codes[i]))
        kt.append(times[i])
    if not kc:
        return np.array([], dtype=object), np.array([], dtype='datetime64[ns]')
    ts = pd.to_datetime(kt).values.astype('datetime64[ns]')
    o = np.argsort(ts)
    return np.array(kc, dtype=object)[o], ts[o]


def ppg_span(pid, dataset):
    """(first, last) PPG window timestamp for a patient, or None.

    Read straight from the parquet's window_start_time column -- no block
    assembly, no signal decoding. Availability per admission only needs to know
    whether any window falls inside the stay, and answering that here avoids
    running feature extraction before knowing whether it is worth running.
    """
    p = Path(PPG_DIR[dataset]) / pid / PPG_PARQUET
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, columns=['window_start_time'])
    except Exception:
        return None
    if len(df) == 0:
        return None
    t = pd.to_datetime(df['window_start_time']).values.astype('datetime64[ns]')
    return np.sort(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--horizons', type=int, nargs='+',
                    default=[30, 90, 180, 365])
    ap.add_argument('--no-test-only', action='store_true',
                    help="drop the test-split restriction (see the warning it "
                         "prints before doing so)")
    ap.add_argument('--skip-ppg', action='store_true')
    args = ap.parse_args()

    print("=" * 100)
    print("STAGE 1  admissions from the stay table")
    print("=" * 100)
    if args.no_test_only:
        print("  !! test-split restriction disabled. MedicalBERT was pretrained")
        print("     on the non-test patients' codes, and this label is built")
        print("     from codes, so the sEHR branch has seen the answer for")
        print("     those patients. Any number produced this way is inflated.")
        test_pids = None
    else:
        test_pids = load_test_pids()
        print(f"  test patients: III={len(test_pids['III'])}, "
              f"IV={len(test_pids['IV'])}")

    adm = load_admissions(test_pids)
    print(f"  admissions: {len(adm)}  patients: {adm.patient_id.nunique()}")
    per_pat = adm.groupby(['patient_id', 'dataset']).size()
    print(f"  per patient: median {per_pat.median():.0f}, "
          f"p90 {per_pat.quantile(.9):.0f}, max {per_pat.max()}")
    los = (adm['dischtime'] - adm['admittime']).dt.total_seconds() / 86400
    print(f"  length of stay (days): p25 {los.quantile(.25):.1f}, "
          f"p50 {los.median():.1f}, p90 {los.quantile(.9):.1f}")
    print("    The label window opens at discharge, so a long stay pushes its")
    print("    window later in the illness course, and stay length tracks")
    print("    severity. A property of the definition, not an error.")

    print("\n" + "=" * 100)
    print("STAGE 2  death, for censoring")
    print("=" * 100)
    dod = load_dod()
    adm = adm.merge(dod, on=['patient_id', 'dataset'], how='left')
    adm['death'] = adm['dod'].fillna(adm['deathtime'])
    n_dod = int(adm['dod'].notna().sum())
    n_dt = int((adm['dod'].isna() & adm['deathtime'].notna()).sum())
    print(f"  death known for {int(adm['death'].notna().sum())}/{len(adm)} "
          f"admissions (dod {n_dod}, deathtime-only {n_dt})")
    print(f"  patients with any death record: "
          f"{adm.loc[adm.death.notna()].patient_id.nunique()}"
          f"/{adm.patient_id.nunique()}")

    print("\n" + "=" * 100)
    print("STAGE 3  disease labels")
    print("=" * 100)
    dbs, lookups = {}, {}
    for ds, p in DB_PATH.items():
        dbs[ds] = joblib.load(p)
        lookups[ds] = {canon_pid(k, ds): k for k in dbs[ds].keys()}

    code_cache = {}
    rows = []
    for r in tqdm(adm.itertuples(), total=len(adm), desc="  labelling"):
        key = (r.patient_id, r.dataset)
        if key not in code_cache:
            code_cache[key] = patient_codes(r.patient_id, dbs[r.dataset],
                                            lookups[r.dataset])
        codes, times = code_cache[key]

        rec = {'patient_id': r.patient_id, 'dataset': r.dataset,
               'hadm_id': r.hadm_id, 'admittime': r.admittime,
               'dischtime': r.dischtime, 'death': r.death,
               'n_codes_total': int(codes.size)}
        d0 = np.datetime64(r.dischtime)
        dth = np.datetime64(r.death) if pd.notna(r.death) else None
        before = times <= d0
        for name, (pref, exact) in DISEASES.items():
            rec[f'{name}__prior'] = int(hits(codes[before], pref, exact))
            for N in args.horizons:
                hi = d0 + np.timedelta64(N, 'D')
                win = (times > d0) & (times <= hi)
                if hits(codes[win], pref, exact):
                    rec[f'{name}__{N}'] = 1
                elif dth is not None and dth <= hi:
                    rec[f'{name}__{N}'] = np.nan
                else:
                    rec[f'{name}__{N}'] = 0
        rows.append(rec)
    lab = pd.DataFrame(rows)

    # ── stage 3.5 ──
    if not args.skip_ppg:
        print("\n" + "=" * 100)
        print("STAGE 3.5  PPG availability per admission")
        print("=" * 100)
        spans = {}
        pats = lab[['patient_id', 'dataset']].drop_duplicates()
        for r in tqdm(pats.itertuples(), total=len(pats), desc="  reading ppg"):
            spans[(r.patient_id, r.dataset)] = ppg_span(r.patient_id, r.dataset)
        n_win, has = [], []
        for r in lab.itertuples():
            t = spans.get((r.patient_id, r.dataset))
            if t is None:
                n_win.append(0); has.append(0); continue
            a = np.datetime64(r.admittime); d = np.datetime64(r.dischtime)
            k = int(np.searchsorted(t, d, 'right') - np.searchsorted(t, a, 'left'))
            n_win.append(k); has.append(int(k > 0))
        lab['n_ppg_windows'] = n_win
        lab['ppg_mask'] = has
        print(f"  admissions with >=1 PPG window in [admit, disch]: "
              f"{int(lab.ppg_mask.sum())}/{len(lab)}")
        print(f"  patients contributing them: "
              f"{lab.loc[lab.ppg_mask == 1].patient_id.nunique()}")
        q = lab.loc[lab.ppg_mask == 1, 'n_ppg_windows']
        if len(q):
            print(f"  windows per such admission: p25 {q.quantile(.25):.0f}, "
                  f"p50 {q.median():.0f}, p90 {q.quantile(.9):.0f}")
    else:
        lab['ppg_mask'] = np.nan
        lab['n_ppg_windows'] = np.nan

    out = os.path.join(args.out_dir, 'admission_disease_labels.joblib')
    joblib.dump(lab, out)

    # ── distribution ──
    ppg = lab[lab.ppg_mask == 1] if not args.skip_ppg else lab.iloc[0:0]
    summary = []
    for name in DISEASES:
        print("\n" + "=" * 100)
        print(f"{name}   prefixes={DISEASES[name][0] or '(none)'}   "
              f"exact={sorted(DISEASES[name][1]) or '(none)'}")
        print("=" * 100)
        pr = int(lab[f'{name}__prior'].sum())
        print(f"  coded on/before discharge (prevalent): "
              f"{pr}/{len(lab)} ({pr/len(lab):.1%})")
        print(f"  {'N':>5s} | {'usable':>7s} {'n_pos':>6s} {'rate':>6s} "
              f"{'pats':>6s} | {'PPG usable':>10s} {'PPG pos':>8s} "
              f"{'PPG pats':>8s} | {'prevalent':>10s}")
        for N in args.horizons:
            c = f'{name}__{N}'
            s = lab[c]
            us, np_ = int(s.notna().sum()), int((s == 1).sum())
            pat = lab.loc[s == 1, 'patient_id'].nunique()
            prev = int(((s == 1) & (lab[f'{name}__prior'] == 1)).sum())
            if len(ppg):
                sp = ppg[c]
                pu, pp = int(sp.notna().sum()), int((sp == 1).sum())
                ppat = ppg.loc[sp == 1, 'patient_id'].nunique()
            else:
                pu = pp = ppat = 0
            print(f"  {N:>5d} | {us:>7d} {np_:>6d} "
                  f"{(np_/us if us else np.nan):>5.1%} {pat:>6d} | "
                  f"{pu:>10d} {pp:>8d} {ppat:>8d} | "
                  f"{prev:>4d} ({prev/np_ if np_ else np.nan:>5.1%})")
            summary.append({'disease': name, 'horizon': N,
                            'n_usable': us, 'n_pos': np_, 'n_patients': pat,
                            'n_usable_ppg': pu, 'n_pos_ppg': pp,
                            'n_patients_ppg': ppat,
                            'n_prevalent_pos': prev,
                            'frac_prevalent': prev/np_ if np_ else np.nan})

    sdf = pd.DataFrame(summary)
    csv = os.path.join(args.out_dir, 'admission_label_distribution.csv')
    sdf.to_csv(csv, index=False)
    print(f"\nSaved {out}\nSaved {csv}")

    print("\n" + "=" * 100)
    print("DECISION POINT")
    print("=" * 100)
    print("  'PPG pos' and 'PPG pats' are the numbers that decide whether")
    print("  feature extraction is worth running. The previous sample set had")
    print("  953 PPG-carrying rows and 3/7/9/17 positives among them, from")
    print("  3/7/9/14 patients -- too few to resolve anything, which is why")
    print("  the drop rule was removed. Compare against those.")
    print()
    print("  Count PATIENTS, not admissions. Cross-validation groups by")
    print("  patient, so ten admissions from one patient carry roughly one")
    print("  patient's worth of independent information.")
    print()
    print("  'prevalent' is the share of positives already coded on or before")
    print("  discharge. High means the label is mostly established disease")
    print("  being re-coded: the sEHR input can nearly read the answer off")
    print("  itself, leaving little for a waveform to add. It also means the")
    print("  task is identification rather than prediction, and must be")
    print("  described that way. AF and HF should differ here -- paroxysmal AF")
    print("  can be absent from one stay and coded at the next, while heart")
    print("  failure tends to be carried forward.")
    print()
    print("  Compare each single disease against eMACE. If the composite gains")
    print("  positives mainly from stroke and coronary anatomy -- components a")
    print("  peripheral waveform cannot carry -- it is diluting whatever signal")
    print("  AF and HF might offer.")


if __name__ == '__main__':
    main()