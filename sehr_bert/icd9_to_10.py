"""Map MIMIC-IV ICD-9 diagnosis and procedure codes to ICD-10.

Diagnoses go through the `icdmappings` package; procedures go through the CMS
General Equivalence Mappings, which are not redistributed here.

    Download: CMS "2018 ICD-10 PCS General Equivalence Mappings", the
    gem_i9pcs.txt file inside it, and pass it with --gem-i9pcs.
    https://www.cms.gov/medicare/coding/icd10/2018-icd-10-pcs-gem

Reads the raw MIMIC-IV sEHR extracts and writes the mapped CSVs that
keep_50.py then thins.

Usage:
    python icd9_to_10.py --gem-i9pcs /path/to/gem_i9pcs.txt
"""
import argparse
import os
import sys

from icdmappings import Mapper
from collections import defaultdict
import pandas as pd
import time
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import paths

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument('--gem-i9pcs', type=str, required=True,
                 help="CMS gem_i9pcs.txt (2018 ICD-10 PCS General "
                      "Equivalence Mappings). Not bundled: it is a CMS "
                      "release with its own distribution terms.")
_args = _ap.parse_args()

print(">>> Loading CSVs ...")
t_all = time.time()

mapper = Mapper()

GEM_I9PCS_PATH = _args.gem_i9pcs

def norm_code(code):
    """Keep as string, remove decimals/spaces, preserve existing leading zeros."""
    if code is None or pd.isna(code):
        return None
    s = str(code).strip().replace(".", "")
    return s if s else None

def load_gem_i9pcs(path):
    """
    Parse gem_i9pcs.txt into:
      source_code -> list of rows
    Each row keeps target + decoded flags.
    """
    gem = defaultdict(list)

    with open(path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue

            # GEM text rows are source, target, flags
            parts = line.split()
            if len(parts) < 3:
                continue

            source, target, flags = parts[0], parts[1], parts[2]

            # Expect 5-digit flag string
            if len(flags) != 5 or not flags.isdigit():
                continue

            gem[source].append({
                "target": target,
                "flags": flags,
                "approximate": int(flags[0]),
                "no_map": int(flags[1]),
                "combination": int(flags[2]),
                "scenario": int(flags[3]),
                "choice_list": int(flags[4]),
            })

    return gem

GEM_I9PCS = load_gem_i9pcs(GEM_I9PCS_PATH)

def choose_single_proc_target(candidates):
    uniq = sorted(set(candidates))
    return uniq[0] if uniq else None

'''"""
tests = ["250.00","25000","414.01","491.21","530.81","V58.67","401.9","4019"]
for c in tests:
    r = mapper.map(c, source="icd9", target="icd10")
    print(c, "->", r, type(r))
'''
diagnose_path = str(paths.data_root() / "MIMICIV_sEHR_raw/mimic_iv_diag_raw.csv")
procedure_path = str(paths.data_root() / "MIMICIV_sEHR_raw/mimic_iv_proc_raw.csv")

diagnose = pd.read_csv(diagnose_path, dtype={"icd_code": str})
procedure = pd.read_csv(procedure_path, dtype={"icd_code": str})

print(f"    diagnose loaded: shape={diagnose.shape}, cols={list(diagnose.columns)[:6]}...")
print(f"    procedure loaded: shape={procedure.shape}, cols={list(procedure.columns)[:6]}...")

print(f"\n>>> Debug: diagnose columns = {list(diagnose.columns)}")
print(f">>> Debug: procedure columns = {list(procedure.columns)}")
if len(diagnose) > 0:
    print(f">>> Debug: diagnose first row:\n{diagnose.head(1).to_string()}")
    if "icd_version" in diagnose.columns:
        print(f">>> Debug: diagnose icd_version unique values: {diagnose['icd_version'].unique()[:10]}")
        print(f">>> Debug: diagnose icd_version dtype: {diagnose['icd_version'].dtype}")
if len(procedure) > 0:
    print(f">>> Debug: procedure first row:\n{procedure.head(1).to_string()}")
    if "icd_version" in procedure.columns:
        print(f">>> Debug: procedure icd_version unique values: {procedure['icd_version'].unique()[:10]}")
        print(f">>> Debug: procedure icd_version dtype: {procedure['icd_version'].dtype}")


def map_icd(row):
    '''only keep the first three characters of the mapped ICD-10 diagnose codes'''
    def shorten3(code: str):
        if code is None:
            return None
        s = str(code).strip().replace('.', '')
        return s[:3] if s else None

    # Handle both int and string types for icd_version
    icd_version = row.get("icd_version")
    if pd.isna(icd_version):
        return None
    try:
        icd_version = int(icd_version) if isinstance(icd_version, (int, float)) else int(str(icd_version).strip())
    except (ValueError, TypeError):
        return None
    
    icd_code = row.get("icd_code")
    if pd.isna(icd_code) or not icd_code:
        return None
    
    if icd_version == 9:
        lst = mapper.map(norm_code(icd_code), source="icd9", target="icd10")

        if lst is None:
            return None
        if isinstance(lst, str):
            lst = [x for x in re.split(r'[,\s;/]+', lst.strip()) if x]
        elif isinstance(lst, (list, tuple, set)):
            lst = list(lst)
            if lst and all(isinstance(x, str) and len(x) == 1 for x in lst):
                lst = [''.join(lst)]
            else:
                lst = [str(x).strip() for x in lst if str(x).strip()]
        else:
            s = str(lst).strip()
            lst = [s] if s else []

        # keep only the first 3 chars of each mapped code, and sort them to ensure deterministic order
        short = sorted({shorten3(x) for x in lst if shorten3(x)})
        return ",".join(short) if short else None

    else:
        # return truncated original if already version 10
        icd_code = row.get("icd_code")
        if pd.isna(icd_code) or not icd_code:
            return None
        return shorten3(norm_code(icd_code))


def map_icd_proc(row): # shouldn't use the mapper; mapper is only for diagnose codes!
    '''only convert ICD-9 to ICD-10 procedure codes, keep original length (no shortening)'''

    # Handle both int and string types for icd_version
    icd_version = row.get("icd_version")
    if pd.isna(icd_version):
        return None
    try:
        icd_version = int(icd_version) if isinstance(icd_version, (int, float)) else int(str(icd_version).strip())
    except (ValueError, TypeError):
        print(f"Warning: invalid icd_version '{icd_version}' in procedure row, treating as unmapped")
        return None
    
    icd_code = row.get("icd_code")
    if pd.isna(icd_code) or not icd_code:
        print(f"Warning: missing or invalid icd_code in procedure {row.name}, treating as unmapped")
        return None
    
    if icd_version == 9:
        src = norm_code(icd_code)
        if not src:
            return None

        rows = GEM_I9PCS.get(src, [])
        if not rows:
            print(f"Warning: no GEM mapping found for ICD-9 procedure code '{src}' in row {row.name}")
            return None

        # 00000 or 10000
        single_rows = [
            r for r in rows
            if r["no_map"] == 0 and r["combination"] == 0
        ]

        if not single_rows:
            print(f"Warning: no SINGLE mapping found for ICD-9 procedure code '{src}' in row {row.name}, candidates: {[r['target'] for r in rows]}")
            return None

        candidates = [r["target"] for r in single_rows if r["target"] != "NoPCS"]
        if not candidates:
            return None

        return sorted(set(candidates))[0]

    elif icd_version == 10:
        # return original if already version 10
        icd_code = row.get("icd_code")
        if pd.isna(icd_code) or not icd_code:
            return None
        return norm_code(icd_code)

    else:
        print(f"Found ICD version {icd_version}, returning None")
        return None


print(">>> Mapping diagnose -> diagnosis_code_icd10 ...")
t0 = time.time()
try:
    diagnose["diagnosis_code_icd10"] = diagnose.apply(map_icd, axis=1)
    print(f"    diagnose mapping done in {time.time()-t0:.1f}s; "
          f"non-null={diagnose['diagnosis_code_icd10'].notna().sum()} / total={len(diagnose)}")
    print(f"    Debug: null count = {diagnose['diagnosis_code_icd10'].isna().sum()}")
    print(f"    Debug: sample results:\n{diagnose[['icd_version', 'icd_code', 'diagnosis_code_icd10']].head(5).to_string()}")
except Exception as e:
    print(f"    ERROR in diagnose mapping: {e}")
    import traceback
    traceback.print_exc()
mask_d = diagnose["diagnosis_code_icd10"].fillna("").str.contains(",")
multi_d = diagnose[mask_d]

print("number of rows in diagnose that have multiple mapping：", multi_d.shape[0], "/", diagnose.shape[0])
print(multi_d[["icd_version","icd_code","diagnosis_code_icd10"]].head(10).to_string(index=False))

# print number of NAs (Nones)
print(f"diagnose mapping result: {diagnose['diagnosis_code_icd10'].isna().sum()} NAs out of {len(diagnose)} rows")
diagnose = diagnose[diagnose["diagnosis_code_icd10"].notna()].copy()

diagnose.to_csv(str(paths.data_root() / "MIMICIV_sEHR_cleaned/diagnose_icd10_mapped.csv"), index=False)

print(">>> Mapping procedure -> procedure_code_icd10 ...")
t1 = time.time()
try:
    procedure["procedure_code_icd10"] = procedure.apply(map_icd_proc, axis=1)
    print(f"    procedure mapping done in {time.time()-t1:.1f}s; "
          f"non-null={procedure['procedure_code_icd10'].notna().sum()} / total={len(procedure)}")
    print(f"    Debug: null count = {procedure['procedure_code_icd10'].isna().sum()}")
    print(f"    Debug: sample results:\n{procedure[['icd_version', 'icd_code', 'procedure_code_icd10']].head(5).to_string()}")
except Exception as e:
    print(f"    ERROR in procedure mapping: {e}")
    import traceback
    traceback.print_exc()

mask_p = procedure["procedure_code_icd10"].fillna("").str.contains(",")
multi_p = procedure[mask_p]

print("number of rows in procedure that have multiple mapping：", multi_p.shape[0], "/", procedure.shape[0])
print(multi_p[["icd_version","icd_code","procedure_code_icd10"]].head(10).to_string(index=False))

print(f">>> All done in {time.time()-t_all:.1f}s")

# print number of NAs (Nones)
print("checking how many procedure rows are deleted due to multiple matching")
total = len(procedure)
mapped = procedure['procedure_code_icd10'].notna().sum()
print(f"Procedure mapping rate: {mapped}/{total} = {mapped/total:.1%}")

procedure = procedure[procedure["procedure_code_icd10"].notna()].copy()
procedure.to_csv(str(paths.data_root() / "MIMICIV_sEHR_cleaned/procedure_icd10_mapped.csv"), index=False)
