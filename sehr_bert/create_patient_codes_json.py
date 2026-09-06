import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import paths


import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def build_patient_history(diag_file, proc_file, med_file,
                          proc_time_col='chartdate', med_time_col='starttime'):
    def load_and_clean(path, code_col_orig, time_col_orig, ctype):
        print(f"Reading {ctype} from {path}...")
        # Specify low_memory=False to avoid DtypeWarning
        df = pd.read_csv(path, low_memory=False)
        print(f"{ctype} data rows: {df.shape[0]}")
        # 1. Validate expected columns exist before renaming (rename silently skips missing cols)
        required = {'subject_id', code_col_orig, time_col_orig}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns in {ctype} file ({path}):\n"
                f"  Missing : {sorted(missing)}\n"
                f"  Available: {sorted(df.columns.tolist())}"
            )
        df = df.rename(columns={
            'subject_id': 'pid',
            code_col_orig: 'code',
            time_col_orig: 'ts'
        })
        
        # 2. Keep ONLY the 4 essential columns and clean code strings
        df = df[['pid', 'code', 'ts']]
        df['type'] = ctype
        # Fix floating point trap: cast to string and remove '.0' suffix ONLY for medication
        if ctype == 'medication':
            df['code'] = df['code'].astype(str).str.replace(r'\.0$', '', regex=True)
            print(f"medication rows after code fix: {df.shape[0]}")
        else:
            df['code'] = df['code'].astype(str)
        
        # 3. Robust datetime conversion
        # errors='coerce' will turn bad dates into NaT
        df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
        
        # 4. Drop rows with any missing crucial info
        df = df.dropna(subset=['pid', 'code', 'ts'])
        return df

    # Load each file individually with only necessary columns
    diag = load_and_clean(diag_file, 'diagnosis_code_icd10', 'admittime', 'diagnosis')
    print(f"Diagnosis data loaded. Shape: {diag.shape}")
    proc = load_and_clean(proc_file, 'procedure_code_icd10', proc_time_col, 'procedure')
    print(f"Procedure data loaded. Shape: {proc.shape}")
    med = load_and_clean(med_file, 'ndc', med_time_col, 'medication')
    print(f"Medication data loaded. Shape: {med.shape}")

    # 4. Verify column alignment before concat to prevent silent NaN errors
    dfs = [diag, proc, med]
    names = ['Diagnosis', 'Procedure', 'Medication']
    base_cols = sorted(diag.columns.tolist())
    
    for df, name in zip(dfs[1:], names[1:]):
        current_cols = sorted(df.columns.tolist())
        if current_cols != base_cols:
            raise ValueError(
                f"❌ Column mismatch error!\n"
                f"Reference (Diagnosis) columns: {base_cols}\n"
                f"Current ({name}) columns: {current_cols}\n"
                f"Check load_and_clean() logic."
            )

    # Now concat is safe and clean
    print("Combining all modalities into a clean timeline...")
    all_data = pd.concat(dfs, ignore_index=True)
    print(f"Combined data shape: {all_data.shape}")

    patient_history = {}
    print("Grouping by patient and sorting...")
    
    # Sort entire dataset by time once (faster than sorting each group)
    all_data = all_data.sort_values(['pid', 'ts'])
    
    for pid, group in all_data.groupby('pid'):
        patient_history[str(pid)] = {
            'codes': group['code'].tolist(),
            # Store as datetime64[ns] array for efficient window matching
            'timestamps': group['ts'].values, 
            'code_type': group['type'].tolist()
        }
        
    print(f"✓ Success! History built for {len(patient_history)} patients.")
    print("Sample jsons:")
    for i, (pid, data) in enumerate(list(patient_history.items())[:3]):
        print(f"  Patient {pid}: {len(data['codes'])} codes")
    return patient_history

def build_and_save(data_path, proc_time_col, med_time_col, save_path):
    history = build_patient_history(
        Path(data_path) / "diagnose_icd10_mapped_50.csv",
        Path(data_path) / "procedure_icd10_mapped_50.csv",
        Path(data_path) / "prescriptions_cleaned_50.csv",
        proc_time_col=proc_time_col,
        med_time_col=med_time_col,
    )
    joblib.dump(history, save_path)
    print(f"Database saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["IV", "III"], required=True)
    args = parser.parse_args()

    if args.dataset == "IV":
        build_and_save(
            str(paths.data_root() / "MIMICIV_sEHR_cleaned"),
            proc_time_col="chartdate",
            med_time_col="starttime",
            save_path=str(paths.data_root() / "patient_history_db_IV.joblib"),
        )
    else:
        build_and_save(
            str(paths.data_root() / "MIMIC_III_sEHR_cleaned"),
            proc_time_col="admittime",
            med_time_col="startdate",
            save_path=str(paths.data_root() / "patient_history_db_III.joblib"),
        )

    import numpy as np
    import pandas as pd

    path = str(paths.data_root() / "patient_history_db_III.joblib")
    print(f"Loading {path}...")
    history = joblib.load(path)

    all_pids = list(history.keys())
    print(f"\n--- Basic Stats ---")
    print(f"Total patients: {len(all_pids)}")

    print(f"\n--- Consistency Check (Random 5) ---")
    rng = np.random.default_rng()
    sample_pids = rng.choice(all_pids, size=5)

    for pid in sample_pids:
        data = history[pid]
        codes = data['codes']
        ts = data['timestamps']
        types = data['code_type']
        
        is_consistent = (len(codes) == len(ts) == len(types))
        
        is_sorted = np.all(np.diff(ts).astype(float) >= 0)
        
        unique_types = set(types)
        
        print(f"Patient {pid}:")
        print(f"  - Count: {len(codes)} events")
        print(f"  - Length Consistent: {'✅' if is_consistent else '❌'}")
        print(f"  - Chronologically Sorted: {'✅' if is_sorted else '❌'}")
        print(f"  - Modalities present: {unique_types}")
        print(f"  - First Event: {ts[0]} ({types[0]})")
        print(f"  - Last Event: {ts[-1]} ({types[-1]})")

    print(f"\n--- Coverage Summary ---")
    type_counts = {'diagnosis': 0, 'procedure': 0, 'medication': 0}
    for pid in all_pids:
        for t in history[pid]['code_type']:
            if t in type_counts:
                type_counts[t] += 1

    print(f"Total events in DB: {type_counts}")