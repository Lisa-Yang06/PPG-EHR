"""Keep only codes with at least 50 records; drop rows with a missing code.

Stage 3 of the sEHR side, run once per cohort:

    icd9_to_10.py / preprocess_medication.py  ->  *_icd10_mapped.csv, prescriptions_cleaned.csv
    keep_50.py                                ->  *_icd10_mapped_50.csv, prescriptions_cleaned_50.csv
    sEHR_BERT.py                              reads the _50 files, both cohorts

WHY --cohort EXISTS
-------------------
The research code hardcoded the MIMIC-III directory and was edited by hand to
produce the MIMIC-IV files; that edit was never committed. The filter itself
never depended on the cohort -- same threshold, same columns -- so the cohort is
a directory argument here and nothing else changes between the two runs.

Usage:
    python keep_50.py --cohort III
    python keep_50.py --cohort IV
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import paths


from pathlib import Path
import pandas as pd

def keep_50(input_path: str, output_path: str, code_col: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
        df[code_col] = df[code_col].astype("string")
    else:
        df = pd.read_csv(input_path)
        df = pd.read_csv(input_path, dtype={code_col: "string"})

    df[code_col] = df[code_col].astype("string").str.strip()

    df.loc[df[code_col] == "", code_col] = pd.NA

    df = df.dropna(subset=[code_col])

    code_counts = df[code_col].value_counts()
    codes_to_keep = code_counts[code_counts >= 50].index.tolist()
    df_filtered = df[df[code_col].isin(codes_to_keep)]

    df_filtered.to_csv(output_path, index=False)

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--cohort', choices=['III', 'IV'], required=True,
                help="which cleaned directory to filter in place")
ap.add_argument('--src-dir', type=Path, default=None,
                help="default: $PPG_DATA_ROOT/MIMIC_<cohort>_sEHR_cleaned")
args = ap.parse_args()

# The two cohorts are not named in parallel on disk; MIMIC_IV_sEHR_cleaned is
# the directory sEHR_BERT.py reads.
src_dir = args.src_dir or (paths.data_root() / f"MIMIC_{args.cohort}_sEHR_cleaned")

# (input stem, code column) -- the same three tables in both cohorts.
TABLES = [
    ("diagnose_icd10_mapped",  "diagnosis_code_icd10"),
    ("procedure_icd10_mapped", "procedure_code_icd10"),
    ("prescriptions_cleaned",  "ndc"),
]

for stem, code_col in TABLES:
    src = src_dir / f"{stem}.csv"
    dst = src_dir / f"{stem}_50.csv"
    print(f"\n{stem}: {src} -> {dst}", flush=True)
    keep_50(str(src), str(dst), code_col)
    out = pd.read_csv(str(dst), dtype={code_col: "string"})
    print(f"  shape after keeping 50+: {out.shape}")
    print(f"  unique codes after keeping 50+: {out[code_col].nunique()}")
