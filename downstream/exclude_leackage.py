"""
Drop the admissions whose sEHR input window and label window overlap.

THE OVERLAP
-----------
Stage 4 selects sEHR codes at DAY granularity

    hist_day <= discharge_day                     (forward_ehr)

while the label window uses the exact timestamp

    dischtime < code_time <= dischtime + N days   (build_admission_labels)

so the hours between dischtime and midnight of the discharge day belong to
both. A code landing there is simultaneously an encoder input and a label
event. 42 of 3956 admissions contain 127 such codes.

Only one of those codes is a target diagnosis: a single admission carries an
I48 timestamped two minutes AFTER its own discharge, and it is the only I48
in that admission's label window, so its AF label at every horizon comes from
its own input. (Record identifiers are omitted here -- MIMIC is credentialed
under a PhysioNet DUA and this repository redistributes none of it.) The remaining 126 are medications and procedures
that no target definition matches today. They would matter the moment the
disease definition changed, which is why the rule dropped here is structural
("the two windows overlap") rather than "this code happened to be an I48".

WHY DROPPING IS NOT FREE
------------------------
These 42 are not a random 1%. Their event rate runs 2-3.5x the cohort's, in the
same direction across all twelve disease x horizon combinations:

    AF__365      29.0%  vs 11.7%
    HF__365      35.5%  vs 17.5%
    eMACE__365   39.4%  vs 19.6%

Coding that continues past the discharge timestamp plausibly marks a
complicated discharge, and a complicated discharge tracks severity. Removing
them therefore shifts the cohort slightly toward lower risk. The size is small
-- 13 of 636 eMACE__365 positives, 8 of 2186 both-modality admissions -- but
the direction is known and belongs in the limitations rather than in a
footnote.

The alternative was to make forward_ehr's window exact and re-extract the sEHR
embeddings, which removes the overlap at its source and introduces no
selection. That was not taken here on cost grounds; this file records what was
traded for it.

WHAT IT WRITES
--------------
Filtered copies of all three artefacts, suffixed _clean, keyed identically on
(dataset, patient_id, hadm_id). Leaving one unfiltered would silently
misalign every downstream join, so the script verifies the three key sets are
equal before writing anything.

Usage:
    python drop_overlapping_admissions.py
    python drop_overlapping_admissions.py --dry-run
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

import numpy as np
import pandas as pd
import joblib

DS = str(paths.data_root() / "ds1")
LABELS = f"{DS}/admission_disease_labels.joblib"
FEATURES = f"{DS}/downstream_features_admission_postdisch.joblib"
METADATA = f"{DS}/admission_metadata_postdisch_sqi.csv"

KEY = ['dataset', 'patient_id', 'hadm_id']
DISEASES = ('AF', 'HF', 'eMACE')
HORIZONS = (30, 90, 180, 365)


def keyset(df_or_list):
    """(dataset, patient_id, hadm_id) as a set of tuples, ids as strings.

    patient_id is forced to str because read_csv infers MIMIC-III ids as
    integers and drops their leading zeros -- '041034' becomes 41034, which
    matches nothing in the joblib files and produces an empty intersection
    rather than an error.
    """
    if isinstance(df_or_list, pd.DataFrame):
        it = df_or_list[KEY].itertuples(index=False)
        return {(str(a), str(b), str(c)) for a, b, c in it}
    return {(str(d['dataset']), str(d['patient_id']), str(d['hadm_id']))
            for d in df_or_list}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', default=METADATA)
    ap.add_argument('--labels', default=LABELS)
    ap.add_argument('--features', default=FEATURES)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    md = pd.read_csv(args.metadata, dtype={'patient_id': str})
    lab = joblib.load(args.labels)
    feat = joblib.load(args.features)
    lab['patient_id'] = lab['patient_id'].astype(str)
    print(f"metadata {len(md)} | labels {len(lab)} | features {len(feat)}")

    if 'n_codes_after_disch_day' not in md.columns:
        raise SystemExit(
            "metadata has no n_codes_after_disch_day column, so the overlapping "
            "admissions cannot be identified. Re-run the metadata extraction.")

    k_md, k_lab, k_feat = keyset(md), keyset(lab), keyset(feat)
    if not (k_md == k_lab == k_feat):
        raise SystemExit(
            f"the three files do not describe the same admissions "
            f"(metadata {len(k_md)}, labels {len(k_lab)}, features {len(k_feat)}; "
            f"pairwise intersections {len(k_md & k_lab)}, {len(k_md & k_feat)}). "
            f"Filtering them separately would misalign every downstream join.")

    bad = md['n_codes_after_disch_day'].fillna(0) > 0
    drop = keyset(md[bad])
    print(f"\ndropping {len(drop)} admissions "
          f"({md.loc[bad, 'patient_id'].nunique()} patients, "
          f"{int((bad & (md.ppg_mask == 1)).sum())} of them both-modality)")

    print(f"\n  {'label':12s} {'positives':>10s} {'dropped':>8s} {'remaining':>10s}"
          f" {'rate before':>12s} {'rate after':>11s}")
    for dis in DISEASES:
        for N in HORIZONS:
            c = f'{dis}__{N}'
            if c not in md.columns:
                continue
            s = pd.to_numeric(md[c], errors='coerce')
            n_pos = int((s == 1).sum())
            n_drop = int(((s == 1) & bad).sum())
            before = s.mean()
            after = s[~bad].mean()
            print(f"  {c:12s} {n_pos:>10d} {n_drop:>8d} {n_pos-n_drop:>10d}"
                  f" {before:>11.2%} {after:>10.2%}")
    print("\n  The rate falls at every horizon: the dropped admissions carry")
    print("  2-3.5x the cohort's event rate, so this exclusion is not neutral.")

    if args.dry_run:
        print("\n--dry-run, nothing written")
        return

    md_c = md[~bad].reset_index(drop=True)
    lab_c = lab[~lab.apply(
        lambda r: (str(r['dataset']), str(r['patient_id']),
                   str(r['hadm_id'])) in drop, axis=1)].reset_index(drop=True)
    feat_c = [d for d in feat
              if (str(d['dataset']), str(d['patient_id']),
                  str(d['hadm_id'])) not in drop]

    kc = (keyset(md_c), keyset(lab_c), keyset(feat_c))
    if not (kc[0] == kc[1] == kc[2]):
        raise SystemExit(
            f"filtered files disagree: metadata {len(kc[0])}, labels "
            f"{len(kc[1])}, features {len(kc[2])}. Not written.")
    expected = len(k_md) - len(drop)
    if len(kc[0]) != expected:
        raise SystemExit(
            f"expected {expected} rows after dropping {len(drop)} from "
            f"{len(k_md)}, got {len(kc[0])}. Not written.")

    out_md = args.metadata.replace('.csv', '_clean.csv')
    out_lab = args.labels.replace('.joblib', '_clean.joblib')
    out_feat = args.features.replace('.joblib', '_clean.joblib')
    md_c.to_csv(out_md, index=False)
    joblib.dump(lab_c, out_lab)
    joblib.dump(feat_c, out_feat)

    print(f"\n{len(kc[0])} admissions retained, all three files aligned")
    for p in (out_md, out_lab, out_feat):
        print(f"  {p}")
    print("\n  Read the _clean files downstream. The originals are untouched,")
    print("  so the exclusion stays auditable rather than being folded in.")


if __name__ == '__main__':
    main()