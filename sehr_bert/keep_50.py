"""
for doagnose, procedure, and medication codes, only keep those codes that have at least 50 records
also filter out ones with missing codes
"""
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

keep_50(str(paths.data_root() / "MIMIC_III_sEHR_cleaned/diagnose_icd10_mapped.csv"),
          str(paths.data_root() / "MIMIC_III_sEHR_cleaned/diagnose_icd10_mapped_50.csv"),
          "diagnosis_code_icd10")
print("diagnose done.")
print("total shape after keeping 50+:", pd.read_csv(str(paths.data_root() / "MIMIC_III_sEHR_cleaned/diagnose_icd10_mapped_50.csv")).shape)
print("total unique codes after keeping 50+:",
      pd.read_csv(str(paths.data_root() / "MIMIC_III_sEHR_cleaned/diagnose_icd10_mapped_50.csv"))["diagnosis_code_icd10"].nunique())

keep_50(str(paths.data_root() / "MIMIC_III_sEHR_cleaned/procedure_icd10_mapped.csv"),
          str(paths.data_root() / "MIMIC_III_sEHR_cleaned/procedure_icd10_mapped_50.csv"),
          "procedure_code_icd10")
print("procedure done.")
print("total shape after keeping 50+:", pd.read_csv(str(paths.data_root() / "MIMIC_III_sEHR_cleaned/procedure_icd10_mapped_50.csv")).shape)
print("total unique codes after keeping 50+:",
      pd.read_csv(str(paths.data_root() / "MIMIC_III_sEHR_cleaned/procedure_icd10_mapped_50.csv"))["procedure_code_icd10"].nunique())    

keep_50(str(paths.data_root() / "MIMIC_III_sEHR_cleaned/prescriptions_cleaned.csv"),
          str(paths.data_root() / "MIMIC_III_sEHR_cleaned/prescriptions_cleaned_50.csv"),
          "ndc")
print("medication done.")
print("total shape after keeping 50+:", pd.read_csv(str(paths.data_root() / "MIMIC_III_sEHR_cleaned/prescriptions_cleaned_50.csv")).shape)
print("total unique codes after keeping 50+:",
      pd.read_csv(str(paths.data_root() / "MIMIC_III_sEHR_cleaned/prescriptions_cleaned_50.csv"))["ndc"].nunique())