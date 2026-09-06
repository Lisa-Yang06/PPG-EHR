"""
filter out rows with missing ndc/ndc=0
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import paths


import pandas as pd
df = pd.read_csv(str(paths.data_root() / "MIMICIV_sEHR_raw/mimic_iv_pres_raw.csv"))

df["ndc"] = df["ndc"].astype("string").str.strip()
df = df[df["ndc"].notna() & (df["ndc"] != "0")]

df.to_csv(str(paths.data_root() / "MIMICIV_sEHR_cleaned/prescriptions_cleaned.csv"), index=False)
print("after cleaning, shape:", df.shape)