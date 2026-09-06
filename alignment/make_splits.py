import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import paths


import json
import joblib
import numpy as np
from pathlib import Path

SEED = 42
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
III_ppg_by_patient_dir = Path(str(paths.data_root() / "MIMIC_III_ppg_by_patient"))
IV_ppg_by_patient_dir  = Path(str(paths.data_root() / "MIMIC_IV_ppg_by_patient"))
III_history_path       = str(paths.data_root() / "patient_history_db_III.joblib")
IV_history_path        = str(paths.data_root() / "patient_history_db_IV.joblib")

alignment_split_path = str(paths.data_root() / "splits/alignment_splits_combined/splits.json")
sehr_split_path      = str(paths.data_root() / "splits/sEHR_splits_combined/splits.json")

Path(alignment_split_path).parent.mkdir(parents=True, exist_ok=True)
Path(sehr_split_path).parent.mkdir(parents=True, exist_ok=True)


"""
Sanity check on the patient history databases used for splitting and training.
Checks:
  1. '0' code still exists (should be 0 after cleanup)
  2. Other suspicious codes (empty, NaN, very short, etc.)
  3. Length consistency (codes, timestamps, code_type aligned)
  4. Basic stats per patient
"""
import joblib
import numpy as np
from collections import Counter

III_history_path = str(paths.data_root() / "patient_history_db_III.joblib")
IV_history_path  = str(paths.data_root() / "patient_history_db_IV.joblib")


def check_history(path, name):
    print(f"\n{'='*60}")
    print(f"Checking: {name} ({path})")
    print(f"{'='*60}")
    
    h = joblib.load(path)
    print(f"Total patients: {len(h)}")
    
    # ── 1. Count '0' codes ───────────────────────────────────────────────
    n_zero = 0
    patients_with_zero = 0
    for v in h.values():
        n_in_pat = sum(1 for c in v['codes'] if str(c) == '0')
        if n_in_pat > 0:
            patients_with_zero += 1
        n_zero += n_in_pat
    print(f"\n[1] '0' codes:")
    print(f"  Total instances: {n_zero}")
    print(f"  Patients affected: {patients_with_zero} / {len(h)}")
    if n_zero == 0:
        print(f"  ✓ Clean")
    else:
        print(f"  ✗ Should be 0!")
    
    # ── 2. Other suspicious codes ────────────────────────────────────────
    suspicious_counter = Counter()
    n_total = 0
    for v in h.values():
        for c in v['codes']:
            n_total += 1
            s = str(c)
            # Empty
            if s == '' or s == 'nan' or s.lower() == 'none':
                suspicious_counter[f"empty/nan/none ({s!r})"] += 1
            # Very short non-special codes
            elif len(s) == 1 and s != '0':
                suspicious_counter[f"len-1 code ({s!r})"] += 1
            # Looks like just digits (might be unmapped or weird)
            elif s.isdigit() and len(s) < 3:
                suspicious_counter[f"short numeric ({s!r})"] += 1
    
    print(f"\n[2] Suspicious codes ({n_total:,} total codes scanned):")
    if suspicious_counter:
        for k, v in suspicious_counter.most_common(20):
            print(f"  {k}: {v:,} ({v/n_total:.2%})")
    else:
        print(f"  ✓ None found")
    
    # ── 3. Length consistency ────────────────────────────────────────────
    mismatches = 0
    sample_mismatch = None
    for pid, v in h.items():
        n_c = len(v['codes'])
        n_t = len(v['timestamps'])
        n_ct = len(v['code_type'])
        if not (n_c == n_t == n_ct):
            mismatches += 1
            if sample_mismatch is None:
                sample_mismatch = (pid, n_c, n_t, n_ct)
    
    print(f"\n[3] Length consistency (codes vs timestamps vs code_type):")
    if mismatches == 0:
        print(f"  ✓ All {len(h)} patients have aligned arrays")
    else:
        print(f"  ✗ {mismatches} patients have mismatched lengths!")
        print(f"  Example: pid={sample_mismatch[0]}, "
              f"codes={sample_mismatch[1]}, ts={sample_mismatch[2]}, "
              f"code_type={sample_mismatch[3]}")
    
    # ── 4. Per-patient stats ─────────────────────────────────────────────
    codes_per_pat = [len(v['codes']) for v in h.values()]
    print(f"\n[4] Per-patient code count:")
    print(f"  Min:    {min(codes_per_pat)}")
    print(f"  Median: {int(np.median(codes_per_pat))}")
    print(f"  Mean:   {np.mean(codes_per_pat):.1f}")
    print(f"  Max:    {max(codes_per_pat)}")
    
    # Empty / very small patients
    n_empty = sum(1 for n in codes_per_pat if n == 0)
    n_tiny = sum(1 for n in codes_per_pat if 0 < n < 5)
    print(f"  Patients with 0 codes: {n_empty}")
    print(f"  Patients with 1-4 codes: {n_tiny}")
    
    # ── 5. Sample first patient ──────────────────────────────────────────
    print(f"\n[5] Sample patient (first one):")
    first_pid = list(h.keys())[0]
    v = h[first_pid]
    print(f"  pid: {first_pid!r}")
    print(f"  n_codes: {len(v['codes'])}")
    print(f"  First 10 codes:      {v['codes'][:10]}")
    print(f"  First 10 timestamps: {[str(t)[:10] for t in v['timestamps'][:10]]}")
    print(f"  First 10 code_types: {v['code_type'][:10]}")


check_history(III_history_path, "III")
check_history(IV_history_path,  "IV")

# Backup before overwriting (safety) 
import shutil
for p in [alignment_split_path, sehr_split_path]:
    if Path(p).exists():
        shutil.copy(p, p + ".bak")
        print(f"Backed up: {p}.bak")

# ── Load EHR histories ────────────────────────────────────────────────────────
print("Loading EHR histories...")
history_III_raw = joblib.load(III_history_path)
history_IV_raw  = joblib.load(IV_history_path)

ehr_pids_III = set(str(k).zfill(6) for k in history_III_raw.keys())
ehr_pids_IV  = set(str(k).zfill(6) for k in history_IV_raw.keys())
print(f"EHR patients - III: {len(ehr_pids_III)}, IV: {len(ehr_pids_IV)}")

# ── Load PPG patient IDs ───────────────────────────────────────────────────────
print("Loading PPG patient IDs...")
ppg_pids_III = set(p.name for p in III_ppg_by_patient_dir.iterdir() if p.is_dir())
ppg_pids_IV  = set(p.name for p in IV_ppg_by_patient_dir.iterdir()  if p.is_dir())
print(f"PPG patients - III: {len(ppg_pids_III)}, IV: {len(ppg_pids_IV)}")

# ── Matched patients (have BOTH PPG and EHR) ──────────────────────────────────
matched_III = ppg_pids_III & ehr_pids_III
matched_IV  = ppg_pids_IV  & ehr_pids_IV
overlap     = matched_III & matched_IV
print(f"Matched - III: {len(matched_III)}, IV: {len(matched_IV)}, overlap: {len(overlap)}")

pid_to_datasets = {}
for pid in matched_III:
    pid_to_datasets.setdefault(pid, []).append('III')
for pid in matched_IV:
    pid_to_datasets.setdefault(pid, []).append('IV')

# ── Helper: shuffle + split ratio ─────────────────────────────────────────────
def shuffle_split(items, frac1):
    """Returns (first frac1, rest)."""
    items_arr = np.array(sorted(items))
    np.random.shuffle(items_arr)
    n1 = int(len(items_arr) * frac1)
    return set(items_arr[:n1].tolist()), set(items_arr[n1:].tolist())

# ── Step 1: Pretrain (70%) / Test (30%) for matched patients ─────────────────
print("\nStep 1: Split matched patients into pretrain (70%) / test (30%)...")

overlap_pretrain, overlap_test = shuffle_split(overlap, 0.70)
III_pretrain, III_test         = shuffle_split(matched_III - overlap, 0.70)
IV_pretrain,  IV_test          = shuffle_split(matched_IV  - overlap, 0.70)

print(f"  Pretrain candidates: overlap={len(overlap_pretrain)}, "
      f"III-only={len(III_pretrain)}, IV-only={len(IV_pretrain)}")
print(f"  Test:                overlap={len(overlap_test)}, "
      f"III-only={len(III_test)}, IV-only={len(IV_test)}")

# ── Step 2: Further split pretrain into train (90%) / val (10%) ───────────────
print("\nStep 2: Split pretrain into alignment_train (90%) / alignment_val (10%)...")

# Important: split overlap, III-only, IV-only INDEPENDENTLY,
# so train and val both have a representative mix.
overlap_val,  overlap_train  = shuffle_split(overlap_pretrain, 0.10)
III_val,      III_train      = shuffle_split(III_pretrain,     0.10)
IV_val,       IV_train       = shuffle_split(IV_pretrain,      0.10)

print(f"  Train: overlap={len(overlap_train)}, III-only={len(III_train)}, IV-only={len(IV_train)}")
print(f"  Val:   overlap={len(overlap_val)},   III-only={len(III_val)},   IV-only={len(IV_val)}")

# ── Step 3: Compose alignment splits by dataset ──────────────────────────────
align_train_III = sorted((overlap_train | III_train) & matched_III)
align_train_IV  = sorted((overlap_train | IV_train)  & matched_IV)
align_val_III   = sorted((overlap_val   | III_val)   & matched_III)
align_val_IV    = sorted((overlap_val   | IV_val)    & matched_IV)
align_test_III  = sorted((overlap_test  | III_test)  & matched_III)
align_test_IV   = sorted((overlap_test  | IV_test)   & matched_IV)

# Sanity: no leakage between splits
for ds, train_, val_, test_ in [
    ('III', set(align_train_III), set(align_val_III), set(align_test_III)),
    ('IV',  set(align_train_IV),  set(align_val_IV),  set(align_test_IV)),
]:
    assert train_ & val_ == set(), f"{ds}: train-val overlap!"
    assert train_ & test_ == set(), f"{ds}: train-test overlap!"
    assert val_ & test_ == set(), f"{ds}: val-test overlap!"
print("  ✓ Alignment train/val/test are mutually disjoint")

# ── Step 4: Save alignment splits ────────────────────────────────────────────
alignment_splits = {
    'train': {'III': align_train_III, 'IV': align_train_IV},
    'val':   {'III': align_val_III,   'IV': align_val_IV},
    'test':  {'III': align_test_III,  'IV': align_test_IV},
    'pid_to_datasets': pid_to_datasets,
}

with open(alignment_split_path, 'w') as f:
    json.dump(alignment_splits, f, indent=2)

print(f"\nAlignment splits saved to {alignment_split_path}")
print(f"  train: III={len(align_train_III)}, IV={len(align_train_IV)}, "
      f"total={len(align_train_III) + len(align_train_IV)}")
print(f"  val:   III={len(align_val_III)},   IV={len(align_val_IV)},   "
      f"total={len(align_val_III) + len(align_val_IV)}")
print(f"  test:  III={len(align_test_III)},  IV={len(align_test_IV)},  "
      f"total={len(align_test_III) + len(align_test_IV)}")

# ── Step 5: sEHR-BERT split: exclude alignment_val AND alignment_test ─────────
print("\nStep 5: Building sEHR-BERT split...")
print("  BERT can see all EHR patients EXCEPT alignment_val and alignment_test")

align_val_all  = set(align_val_III)  | set(align_val_IV)
align_test_all = set(align_test_III) | set(align_test_IV)
holdout = align_val_all | align_test_all

# BERT pool = all EHR patients - holdout
bert_pool = (ehr_pids_III | ehr_pids_IV) - holdout
print(f"  BERT pool size: {len(bert_pool)} (= all EHR - alignment_val - alignment_test)")
print(f"    Of which {len(bert_pool & (matched_III | matched_IV))} have PPG too (= alignment_train)")
print(f"    Remaining {len(bert_pool - (matched_III | matched_IV))} are EHR-only (no PPG)")

# Split BERT pool into BERT train (90%) / BERT val (10%)
bert_pool_arr = np.array(sorted(bert_pool))
np.random.shuffle(bert_pool_arr)
n_bert_train = int(len(bert_pool_arr) * 0.90)
bert_train_set = set(bert_pool_arr[:n_bert_train].tolist())
bert_val_set   = set(bert_pool_arr[n_bert_train:].tolist())

# Split into III / IV
bert_train_III = sorted(bert_train_set & ehr_pids_III)
bert_train_IV  = sorted(bert_train_set & ehr_pids_IV)
bert_val_III   = sorted(bert_val_set   & ehr_pids_III)
bert_val_IV    = sorted(bert_val_set   & ehr_pids_IV)

# Sanity: bert train/val have NO overlap with alignment_val or alignment_test
bert_seen = bert_train_set | bert_val_set
leak_val  = bert_seen & align_val_all
leak_test = bert_seen & align_test_all
assert len(leak_val) == 0,  f"BERT seen ∩ alignment_val:  {len(leak_val)} patients!"
assert len(leak_test) == 0, f"BERT seen ∩ alignment_test: {len(leak_test)} patients!"
print("  ✓ BERT will not see alignment_val or alignment_test")

bert_splits = {
    'train': {'III': bert_train_III, 'IV': bert_train_IV},
    'val':   {'III': bert_val_III,   'IV': bert_val_IV},
}

with open(sehr_split_path, 'w') as f:
    json.dump(bert_splits, f, indent=2)

print(f"\nsEHR-BERT splits saved to {sehr_split_path}")
print(f"  train: III={len(bert_train_III)}, IV={len(bert_train_IV)}, "
      f"total={len(bert_train_III) + len(bert_train_IV)}")
print(f"  val:   III={len(bert_val_III)},   IV={len(bert_val_IV)},   "
      f"total={len(bert_val_III) + len(bert_val_IV)}")