"""
Input Data Format Specification
This file defines how input data should be formatted for preprocess_medbert.py
and implements global patient-level splitting (Train/Val/Test).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import paths


import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm

def calculate_vocab_size(
    diagnosis_file: str,
    procedure_file: str,
    medication_file: str,
    patient_id_col: str = 'subject_id',
    diag_time_col: str = 'admittime',
    proc_time_col: str = 'admittime',
    med_time_col: str = 'startdate',
    diag_code_col: str = 'diagnosis_code_icd10',
    proc_code_col: str = 'procedure_code_icd10',
    med_code_col: str = 'ndc',
) -> int:
    all_codes = set()
    
    def get_valid_codes(path, code_col, time_col, name, is_med=False):
        df = pd.read_csv(path, low_memory=False)
        if code_col not in df.columns:
            raise KeyError(f"❌ {name} file missing required column: '{code_col}'")
        if patient_id_col not in df.columns:
            raise KeyError(f"❌ {name} file missing required column: '{patient_id_col}'")
        temp_time = pd.to_datetime(df[time_col], errors='coerce')
        mask = df[patient_id_col].notna() & df[code_col].notna() & temp_time.notna()
        valid_codes = df.loc[mask, code_col]
        if is_med:
            return valid_codes.astype(str).str.replace(r'\.0$', '', regex=True).unique()
        else:
            return valid_codes.astype(str).unique()

    all_codes.update(get_valid_codes(diagnosis_file, diag_code_col, diag_time_col, 'Diagnosis'))
    all_codes.update(get_valid_codes(procedure_file, proc_code_col, proc_time_col, 'Procedure'))
    all_codes.update(get_valid_codes(medication_file, med_code_col, med_time_col, 'Medication', is_med=True))

    all_codes.discard('nan')
    all_codes.discard('None')

    unique_codes_count = len(all_codes)
    vocab_size = unique_codes_count + 5
    print(f"\n📊 Vocab Audit: {unique_codes_count} unique codes + 5 special = {vocab_size} Total")
    return vocab_size


def load_data_from_csv(
    diagnosis_file: str,
    procedure_file: str,
    medication_file: str,
    patient_id_col: str = 'subject_id',
    diag_time_col: str = 'admittime',
    proc_time_col: str = 'admittime',
    med_time_col: str = 'startdate',
    diag_code_col: str = 'diagnosis_code_icd10',
    proc_code_col: str = 'procedure_code_icd10',
    med_code_col: str = 'ndc',
) -> List[Dict]:
    diagnosis = pd.read_csv(diagnosis_file)
    procedure = pd.read_csv(procedure_file)
    medication = pd.read_csv(medication_file)
    
    diagnosis['code_type'] = 'diagnosis'
    procedure['code_type'] = 'procedure'
    medication['code_type'] = 'medication'

    procedure = procedure.rename(columns={proc_code_col: 'code', proc_time_col: 'timestamp'})
    diagnosis = diagnosis.rename(columns={diag_code_col: 'code', diag_time_col: 'timestamp'})
    medication = medication.rename(columns={med_code_col: 'code', med_time_col: 'timestamp'})
    
    diagnosis['timestamp'] = pd.to_datetime(diagnosis['timestamp'], errors='coerce')
    procedure['timestamp'] = pd.to_datetime(procedure['timestamp'], errors='coerce')
    medication['timestamp'] = pd.to_datetime(medication['timestamp'], errors='coerce')
    
    medication['code'] = medication['code'].astype(str).str.replace(r'\.0$', '', regex=True)
    diagnosis['code'] = diagnosis['code'].astype(str)
    procedure['code'] = procedure['code'].astype(str)
    
    needed_cols = [patient_id_col, 'code', 'timestamp', 'code_type']
    cleaned_dfs = []
    
    for df, name in zip([diagnosis, procedure, medication], ['Diagnosis', 'Procedure', 'Medication']):
        df[patient_id_col] = df[patient_id_col].astype(str).str.zfill(6)
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            raise KeyError(f"❌ {name} table is missing columns after rename: {missing}")
        cleaned_dfs.append(df[needed_cols])
    
    all_data = pd.concat(cleaned_dfs, ignore_index=True)
    bad_procs = all_data[(all_data['code_type'] == 'procedure') & (all_data['timestamp'].isna())]
    print(f"DEBUG: Procedure rows with invalid timestamps: {len(bad_procs)}")
    n_before = len(all_data)
    all_data = all_data.dropna(subset=[patient_id_col, 'code', 'timestamp'])
    n_after = len(all_data)
    print(f"Dropped {n_before - n_after} rows with missing values")

    formatted_data = []
    for pid, group in tqdm(all_data.groupby(patient_id_col), desc="Formatting sequences"):
        group = group.sort_values('timestamp')
        formatted_data.append({
            'patient_id': str(pid),
            'codes': group['code'].tolist(),
            'timestamps': group['timestamp'].dt.to_pydatetime().tolist(),
            'code_types': group['code_type'].tolist()
        })
            
    print(f"✓ Processed {len(formatted_data)} unique patients.")
    return formatted_data

if __name__ == "__main__":
    # Example usage
    vocab_size = calculate_vocab_size(str(paths.data_root() / "MIMIC_III_cleaned/diagnose_icd10_mapped_50.csv"), 
                                      str(paths.data_root() / "MIMIC_III_cleaned/procedure_icd10_mapped_50.csv"), 
                                      str(paths.data_root() / "MIMIC_III_cleaned/prescriptions_cleaned_50.csv"))