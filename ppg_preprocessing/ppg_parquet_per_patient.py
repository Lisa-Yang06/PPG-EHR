"""Split one combined MIMIC-III shard into per-patient parquet files.

Stage 3 of the MIMIC-III PPG chain:

    extract_ppg_III.py   raw waveform  -> ppg_segments_raw_III_<group>.parquet
    concat_ppg.py        combine       -> MIMIC_III_ppg_combined/
    THIS FILE            split         -> MIMIC_III_ppg_by_patient/<pid>/ppg.parquet
    ppg_full_preprocessing.py          -> MIMIC_III_ppg_filtered_by_patient/

Splitting only. No column is added, dropped or rewritten, so `segment` still
carries the "a+b+c" chain names concat_ppg.py produced.

MIMIC-IV does not use this script: concat_ppg.py writes IV per patient in one
step. The asymmetry is real -- III is sharded by patient group and IV is not.
"""
print("SCRIPT STARTED", flush=True)

import os
import sys
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import gc
from tqdm import tqdm
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import paths

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--p_dir', type=str, required=True,
                    help="patient group to split, e.g. p00")
parser.add_argument('--src-dir', type=Path, default=None,
                    help="default: $PPG_DATA_ROOT/MIMIC_III_ppg_combined")
parser.add_argument('--dst-dir', type=Path, default=None,
                    help="default: $PPG_DATA_ROOT/MIMIC_III_ppg_by_patient")
args = parser.parse_args()

src_dir = args.src_dir or (paths.data_root() / "MIMIC_III_ppg_combined")
dst_dir = args.dst_dir or (paths.data_root() / "MIMIC_III_ppg_by_patient")
dst_dir.mkdir(parents=True, exist_ok=True)

parquet_path = src_dir / f"ppg_segments_combined_III_{args.p_dir}.parquet"
print(f"Processing {parquet_path}...", flush=True)

# only read patient_id
pf = pq.ParquetFile(str(parquet_path))
pids = pf.read(columns=['patient_id']).to_pandas()['patient_id'].astype(str).unique()
print(f"  {len(pids)} patients", flush=True)

for pid in tqdm(pids, desc=f"Writing {args.p_dir}"):
    pid_dir = dst_dir / pid
    pid_dir.mkdir(exist_ok=True)
    out_path = pid_dir / "ppg.parquet"

    if out_path.exists():
        continue

    df = pd.read_parquet(str(parquet_path), filters=[('patient_id', '==', pid)])
    print(f"Length of matched rows: {len(df)}")
    df.to_parquet(str(out_path), index=False)

    del df
    gc.collect()

print("Done.", flush=True)