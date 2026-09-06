"""
Combined MIMIC-III + IV PPG-sEHR contrastive alignment, 10-minute segments,
PaPaGei frozen.

This is the configuration the reported results come from.

SAMPLE
------
One sample is a CONTIGUOUS 10-MINUTE PPG chunk -- n_windows_per_seg = 60
consecutive 10s windows from one record_id + segment, adjacent pairs exactly
10s apart -- paired with that patient's sEHR codes within +-12 months of the
chunk's centre time. The 60 windows pass through PaPaGei independently and
their 512-d embeddings are mean-pooled into one embedding per sample before
the projection head.

Why not a single 10s window: validation loss stalled near the random-guess
log(B) = 5.545 baseline while training loss fell to ~5.0 -- overfitting on
snippets too short to carry stable HRV structure. HRV analysis treats five
minutes as the floor for stable LF/HF; ten sits above it.

WHAT TRAINS AND WHAT DOES NOT
-----------------------------
PaPaGei is FROZEN end to end (requires_grad = False on every parameter, and
.eval() during training), so it is a fixed feature extractor. What alignment
changes on the PPG side is the projection head, not the encoder -- a
distinction worth keeping when describing the result.

sEHR-BERT is fine-tuned except its first three transformer blocks, which are
frozen. Both towers project to projection_dim = 128 and the temperature is
fixed at 0.1 (logit_scale_const = 10), not learned.

COHORTS
-------
Loads both cohorts' filtered PPG directories and both history DBs. Every
sample carries (pid, dataset) so the right PPG directory and history DB are
used. Train/val patient ids come from splits.json, produced by
make_splits.py, which also emits the sEHR-BERT pretraining split.

INPUTS
------
Requires ppg_sample100_10min.pkl in each patient's filtered PPG directory,
written by ppg_preprocessing/convert_to_sample100_10min_pkl.py, which
prioritises segments whose centre time has sEHR codes within +-12 months and
caps each patient at 100 segments.

Also requires the MLM-pretrained sEHR-BERT checkpoint and the PaPaGei-s
weights, both under PPG_CKPT_ROOT.

Writes to alignment_results_combined_10min_freezeppg/ under PPG_DATA_ROOT.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import paths

# Opt-in, and never reassigns HOME -- see paths.configure_caches.
paths.configure_caches()

import sys
import json
import time
import pickle
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

# Make sure we import correct sEHR_BERT
SEHR_DIR = os.path.join(_ROOT, "sehr_bert")
if SEHR_DIR not in sys.path:
    sys.path.insert(0, SEHR_DIR)
from sEHR_BERT import MedicalBERT, MedicalCodeTokenizer

# ── PaPaGei setup ─────────────────────────────────────────────────────────────
paths.add_papagei_to_path()

from linearprobing.utils import load_model_without_module_prefix
from models.resnet import ResNet1DMoE

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    # PPG dirs (BOTH datasets)
    'ppg_IV_filtered_dir':    str(paths.data_root() / "MIMIC_IV_ppg_filtered_by_patient"),
    'ppg_III_filtered_dir':   str(paths.data_root() / "MIMIC_III_ppg_filtered_by_patient"),
    
    # EHR history (BOTH)
    'history_db_IV_path':     str(paths.data_root() / "patient_history_db_IV.joblib"),
    'history_db_III_path':    str(paths.data_root() / "patient_history_db_III.joblib"),
    
    'tokenizer_path':         str(paths.data_root() / "medical_tokenizer_combined.json"),
    'split_dir':              str(paths.data_root() / "splits/alignment_splits_combined"),
    'bert_ckpt_path':         str(paths.ckpt_root() / "best_medbert_model_20260617-190122.pt"),
    'papagei_ckpt_path':      str(paths.ckpt_root() / "papagei_s (1).pt"),
    'output_dir':             str(paths.data_root() / "alignment_results_combined_10min_freezeppg"),

    # 5-minute PPG segment configuration
    # Each "sample" in ppg_sample100_5min.pkl is 30 consecutive 10s windows
    # (5 minutes total). Each 10s window has 1250 samples @ 125Hz. We average
    # PaPaGei embeddings across the 30 windows to get one 5-min embedding per
    # sample. Anchor for the sEHR +-window is the segment's CENTER time, which
    # convert_to_sample100_10min_pkl.py wrote to data['center_times'].
    'ppg_pkl_filename':       "ppg_sample100_10min.pkl",
    'n_windows_per_seg':      60,           # 60 * 10s = 10 minutes
    'window_signal_len':      1250,         # 10s @ 125Hz

    'batch_size':          256,
    'projection_dim':      128,
    'epochs':              500,
    'early_stop_patience': 10,
    'seed':                42,

    'max_seq_len':         512,
    'window_months':       12,         # ±12 months around PPG window_start_time

    'target_fs':           125.0,
    'win_s':               10,

    'papagei_lr':          1e-4,
    'bert_lr':             1e-4,
    'mlp_lr':              1e-4,
    'weight_decay':        1e-4,

    'num_workers':         4, # only for data loading, not for forwarding, updating weights
}

PAPAGEI_MODEL_CONFIG = {
    'base_filters': 32,
    'kernel_size':  3,
    'stride':       2,
    'groups':       1,
    'n_block':      18,
    'n_classes':    512,
    'n_experts':    3,
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── PPG sample loader (reads pre-converted pkl, not the raw parquet) ──────────
def _load_ppg_sample100(pid_str, dataset):
    """Load a patient's ppg_sample100.pkl for given dataset ('III' or 'IV').

    This replaces the old _get_ppg_rows(), which read the full ppg.parquet
    (~180MB/patient, JSON-encoded signal column) directly. That was the
    measured IO bottleneck (per-batch timing showed wait_for_data dominating
    compute by 2-3 orders of magnitude). ppg_sample100.pkl is produced by
    convert_to_sample100_pkl.py: up to 100 pre-sampled windows per patient
    (prioritizing windows with sEHR codes within the alignment window),
    stored as plain numpy arrays (~0.5MB/patient, no per-read JSON parsing).

    Returns None if the pkl is missing or fails to load — caller skips this
    sample, same fallback behavior as the old parquet path.
    """
    t0 = time.time()
    ppg_dir = CONFIG[f'ppg_{dataset}_filtered_dir']
    pkl_path = Path(ppg_dir) / pid_str / CONFIG['ppg_pkl_filename']
    if not pkl_path.exists():
        return None
    try:
        file_size_mb = pkl_path.stat().st_size / (1024 * 1024)
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        t_read = time.time() - t0
        #print(
        #    f"    [READ] {pid_str} ({dataset}): {t_read:.3f}s "
        #    f"{file_size_mb:.2f}MB ({file_size_mb / max(t_read, 1e-6):.1f} MB/s) "
        #    f"{len(data['signals'])} windows",
        #    flush=True,
        #)
        return data
    except Exception as e:
        # Possible reasons:
        #   - convert script is still writing this file (race condition)
        #   - convert script crashed mid-write, file is corrupt
        # Either way, skip this sample. collate_fn will drop None.
        print(f"  WARN: failed to read {pid_str} ({dataset}): "
              f"{type(e).__name__}: {str(e)[:80]}",
              flush=True)
        return None


# ── Dataset ───────────────────────────────────────────────────────────────────
class AlignedDataset(Dataset):
    """
    Multi-dataset (III + IV) alignment dataset.
    self.entries: list of (pid_str, dataset) tuples.
    """
    def __init__(self, history_IV, history_III, tokenizer,
                 pids_IV, pids_III, training=True):
        self.history_IV  = history_IV
        self.history_III = history_III
        self.tokenizer   = tokenizer
        self.training    = training

        # Build entries list: only keep pids where BOTH EHR history AND filtered PPG dir exist
        self.entries = []  # list of (pid, dataset)
        
        # IV
        missing_iv_ehr = 0
        missing_iv_ppg = 0
        for pid in pids_IV:
            pid = str(pid)
            if pid not in self.history_IV:
                missing_iv_ehr += 1
                continue
            ppg_path = Path(CONFIG['ppg_IV_filtered_dir']) / pid / CONFIG['ppg_pkl_filename']
            if not ppg_path.exists():
                missing_iv_ppg += 1
                continue
            self.entries.append((pid, 'IV'))
        
        # III
        missing_iii_ehr = 0
        missing_iii_ppg = 0
        for pid in pids_III:
            pid = str(pid)
            if pid not in self.history_III:
                missing_iii_ehr += 1
                continue
            ppg_path = Path(CONFIG['ppg_III_filtered_dir']) / pid / CONFIG['ppg_pkl_filename']
            if not ppg_path.exists():
                missing_iii_ppg += 1
                continue
            self.entries.append((pid, 'III'))
        
        n_iv  = sum(1 for _, d in self.entries if d == 'IV')
        n_iii = sum(1 for _, d in self.entries if d == 'III')

        print(f"Dataset built:")
        print(f"  IV  kept: {n_iv}/{len(pids_IV)} "
              f"(missing EHR: {missing_iv_ehr}, missing PPG: {missing_iv_ppg})")
        print(f"  III kept: {n_iii}/{len(pids_III)} "
              f"(missing EHR: {missing_iii_ehr}, missing PPG: {missing_iii_ppg})")
        print(f"  Total: {len(self.entries)} patient-dataset entries")
        
        # ±12 months as np.timedelta64
        days_half_window = int(CONFIG['window_months'] * 30.44)
        self.window_delta = np.timedelta64(days_half_window, 'D')
        print(f"  sEHR window: ±{days_half_window} days = ±{CONFIG['window_months']} months")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        t_item_start = time.time()
        pid_str, dataset = self.entries[idx]

        data = _load_ppg_sample100(pid_str, dataset)
        if data is None or len(data['signals']) == 0:
            print(f"  No valid PPG samples for {pid_str} ({dataset}), skipping sample.")
            return None

        # Random segment for training, fixed segment for val/test (deterministic),
        # mirroring the old single-window sampling but at the 5-min segment level.
        # data['signals'] has shape (N, 30, 1250) — N segments, each is 30 consecutive
        # 10s windows. We pick one of the N segments and pass all 30 windows to the
        # encoder (averaged downstream in forward()).
        t0 = time.time()
        n_segments = len(data['signals'])
        if self.training:
            sel = np.random.randint(0, n_segments)
        else:
            sel = np.random.RandomState(42).randint(0, n_segments)

        ppg_seg = data['signals'][sel]  # (30, 1250) float32, the 5-min segment
        t_sample_json = time.time() - t0

        n_per_seg = CONFIG['n_windows_per_seg']
        sig_len   = CONFIG['window_signal_len']
        if ppg_seg.shape != (n_per_seg, sig_len):
            print(f"  Invalid PPG segment shape for {pid_str} ({dataset}): {ppg_seg.shape} "
                  f"(expected ({n_per_seg}, {sig_len}))")
            return None

        # (30, 1, 1250): unsqueeze the channel dim so each window matches what
        # PaPaGei expects as a single-sample input (1 channel, 1250 samples).
        # forward() will reshape to (B*30, 1, 1250) for the encoder pass.
        ppg_tensor = torch.tensor(ppg_seg, dtype=torch.float32).unsqueeze(1)

        # Anchor sEHR window at the SEGMENT'S CENTER time (window 15's start
        # time, written by convert_to_sample100_5min_pkl.py). This is the
        # time-point most representative of the 5-min chunk.
        ppg_time = np.datetime64(pd.to_datetime(data['center_times'][sel]))

        # Select correct history db
        t0 = time.time()
        history = self.history_IV if dataset == 'IV' else self.history_III
        pat_history = history.get(pid_str)
        if not pat_history:
            print(f"  No valid patient history for {pid_str} ({dataset}), skipping sample.")
            return None

        for key in ('timestamps', 'codes', 'code_type'):
            if key not in pat_history:
                raise KeyError(f"Patient {pid_str} ({dataset}) history missing key: '{key}'")

        hist_ts   = pat_history['timestamps']
        start_idx = np.searchsorted(hist_ts, ppg_time - self.window_delta, side='left')
        end_idx   = np.searchsorted(hist_ts, ppg_time + self.window_delta, side='right')

        codes  = pat_history['codes'][start_idx:end_idx]
        types  = pat_history['code_type'][start_idx:end_idx]
        sub_ts = hist_ts[start_idx:end_idx]

        if len(codes) == 0:
            print(f"  No valid sEHR codes for {pid_str} ({dataset}), skipping sample.")
            return None

        # Truncate around ppg_time if too long
        if len(codes) > CONFIG['max_seq_len']:
            mid  = np.searchsorted(sub_ts, ppg_time, side='left')
            half = CONFIG['max_seq_len'] // 2
            lo   = max(0, mid - half)
            hi   = lo + CONFIG['max_seq_len']
            if hi > len(codes):
                hi = len(codes)
                lo = hi - CONFIG['max_seq_len']
            codes  = codes[lo:hi]
            types  = types[lo:hi]
            sub_ts = sub_ts[lo:hi]
        t_history = time.time() - t0

        t0 = time.time()
        indices = self.tokenizer.encode(codes)
        t_tokenize = time.time() - t0

        t_item_total = time.time() - t_item_start
        #print(
        #    f"    [ITEM] {pid_str} ({dataset}): "
        #    f"sample={t_sample_json:.3f}s history={t_history:.3f}s "
        #    f"tokenize={t_tokenize:.3f}s | item_total={t_item_total:.3f}s",
        #    flush=True,
        #)

        return {
            'pid':             pid_str,
            'dataset':         dataset,
            'ppg_wave':        ppg_tensor,
            'code_indices':    torch.tensor(indices, dtype=torch.long),
            'code_types':      types,
            'timestamps':      sub_ts,
            'first_timestamp': pd.to_datetime(hist_ts[0]),
        }


def collate_fn(batch):
    original_size = len(batch)
    batch = [b for b in batch if b is not None]
    valid_size = len(batch)

    if valid_size < original_size:
        print(f"WARNING: Dropped {original_size - valid_size}/{original_size} samples.")

    if not batch:
        raise RuntimeError(f"All {original_size} samples in this batch were invalid.")

    padded_codes = torch.nn.utils.rnn.pad_sequence(
        [b['code_indices'] for b in batch], batch_first=True, padding_value=0
    )

    return {
        'pids':             [b['pid'] for b in batch],
        'datasets':         [b['dataset'] for b in batch],
        'ppg_waves':        torch.stack([b['ppg_wave'] for b in batch]),
        'code_indices':     padded_codes,
        'attention_mask':   (padded_codes != 0),
        'first_timestamps': [b['first_timestamp'] for b in batch],
        'raw_batch':        batch,
    }


# ── Model (unchanged from IV-only) ────────────────────────────────────────────
class AlignmentModel(nn.Module):
    def __init__(self, bert_ckpt, papagei_ckpt, vocab_size, projection_dim):
        super().__init__()

        self.bert = MedicalBERT(vocab_size=vocab_size)
        bert_ckpt_obj = torch.load(bert_ckpt, map_location="cpu")
        if isinstance(bert_ckpt_obj, dict) and "model_state_dict" in bert_ckpt_obj:
            bert_state = bert_ckpt_obj["model_state_dict"]
        else:
            bert_state = bert_ckpt_obj

        ckpt_vocab_size = bert_state['code_embedding.weight'].shape[0]
        print(f"BERT ckpt code_embedding vocab_size: {ckpt_vocab_size}")
        print(f"Current tokenizer vocab_size:        {vocab_size}")
        if ckpt_vocab_size != vocab_size:
            print(f"!!! VOCAB MISMATCH !!!")

        missing, unexpected = self.bert.load_state_dict(bert_state, strict=False)
        print(f"BERT missing keys: {len(missing)}, unexpected: {len(unexpected)}")

        papagei = ResNet1DMoE(in_channels=1, **PAPAGEI_MODEL_CONFIG)
        self.papagei = load_model_without_module_prefix(papagei, papagei_ckpt)

        self.mlp_p = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, projection_dim),
        )
        self.mlp_s = nn.Sequential(
            nn.Linear(320, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, projection_dim),
        )
        self.register_buffer('logit_scale_const', torch.tensor(1.0 / 0.1))  # = 10

        # ── Freeze PaPaGei entirely (use as fixed feature extractor) ─────
        for param in self.papagei.parameters():
            param.requires_grad = False
        # ─────────────────────────────────────────────────────────────────

        for i in range(3):
            for param in self.bert.transformer_blocks[i].parameters():
                param.requires_grad = False

    def forward(self, ppg_wave, code_indices, timestamps, first_timestamps, code_types, mask):
        # ppg_wave is (B, n_windows_per_seg, 1, signal_len)
        # = (B, 30, 1, 1250) for 5-min segments. Each row is a patient's 
        # 5-minute PPG chunk: 30 consecutive 10s windows.
        #
        # PaPaGei expects (N, 1, 1250) — single-window input. We flatten the
        # batch and per-segment-window dims into one big "batch", encode each
        # window independently, then reshape back and mean-pool over the 30
        # windows to get one 5-min embedding per patient.
        if ppg_wave.dim() != 4:
            raise ValueError(
                f"Expected ppg_wave shape (B, n_windows, 1, signal_len), "
                f"got {tuple(ppg_wave.shape)}"
            )
        B, n_win, n_ch, sig_len = ppg_wave.shape
        # (B*30, 1, 1250)
        flat = ppg_wave.reshape(B * n_win, n_ch, sig_len)

        ppg_out = self.papagei(flat)
        ppg_rep = ppg_out[0] if isinstance(ppg_out, (tuple, list)) else ppg_out
        if ppg_rep.dim() == 3:
            ppg_rep = ppg_rep.mean(dim=1)
        if ppg_rep.dim() != 2 or ppg_rep.shape[-1] != 512:
            raise ValueError(f"Unexpected PaPaGei output shape: {ppg_rep.shape}")

        # (B*30, 512) -> (B, 30, 512) -> (B, 512) by mean over the 30 windows.
        # Each patient's 5-min embedding is the average of its 30 per-10s
        # embeddings (decided in the design: simple mean pool over windows).
        ppg_rep = ppg_rep.view(B, n_win, -1).mean(dim=1)

        s_out = self.bert(
            code_indices, timestamps,
            first_timestamps=first_timestamps,
            code_types=code_types,
            attention_mask=~mask,
            return_embeddings=True,
        )
        mask_exp = mask.unsqueeze(-1).expand(s_out.size()).float()
        s_rep = torch.sum(s_out * mask_exp, dim=1) / torch.clamp(mask_exp.sum(dim=1), min=1e-9)

        z_p = F.normalize(self.mlp_p(ppg_rep), dim=-1)
        z_s = F.normalize(self.mlp_s(s_rep),   dim=-1)
        return z_p, z_s, self.logit_scale_const


def clip_loss(z_p, z_s, temperature):
    labels = torch.arange(z_p.shape[0], device=z_p.device)
    logits = torch.matmul(z_p, z_s.T) * temperature
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    set_seed(CONFIG['seed'])

    # ── File existence check ──
    required_files = [
        'history_db_IV_path', 'history_db_III_path',
        'bert_ckpt_path', 'tokenizer_path', 'papagei_ckpt_path',
    ]
    for key in required_files:
        if not os.path.exists(CONFIG[key]):
            raise FileNotFoundError(f"Not found [{key}]: {CONFIG[key]}")

    for key in ('ppg_IV_filtered_dir', 'ppg_III_filtered_dir'):
        if not os.path.isdir(CONFIG[key]):
            raise FileNotFoundError(f"PPG filtered dir not found [{key}]: {CONFIG[key]}")

    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load EHR (IV + III) ──
    print("\nLoading EHR history (IV)...")
    history_IV_raw = joblib.load(CONFIG['history_db_IV_path'])
    history_IV = {str(k): v for k, v in history_IV_raw.items()}
    print(f"  IV EHR patients: {len(history_IV)}")

    print("Loading EHR history (III)...")
    history_III_raw = joblib.load(CONFIG['history_db_III_path'])
    history_III = {str(k).zfill(6): v for k, v in history_III_raw.items()}
    # print sample pids in sEHR of MIMIC-III
    print(f"  Sample III EHR pids: {list(history_III.keys())[:5]}")
    print(f"  III EHR patients: {len(history_III)}")

    # ── Tokenizer ──
    tokenizer = MedicalCodeTokenizer()
    tokenizer.load(CONFIG['tokenizer_path'])
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # ── Load splits (BOTH IV and III) ──
    with open(os.path.join(CONFIG['split_dir'], "splits.json")) as f:
        splits = json.load(f)

    train_pids_IV  = [str(p) for p in splits['train'].get('IV',  [])]
    train_pids_III = [str(p) for p in splits['train'].get('III', [])]
    val_pids_IV    = [str(p) for p in splits['val'].get('IV',  [])]
    val_pids_III   = [str(p) for p in splits['val'].get('III', [])]

    print(f"\nsplits.json sizes:")
    print(f"  Train: IV={len(train_pids_IV)}, III={len(train_pids_III)}")
    print(f"  Val:   IV={len(val_pids_IV)},   III={len(val_pids_III)}")

    # ── Show how many III pids are actually ready ──
    iii_ready = {p.name for p in Path(CONFIG['ppg_III_filtered_dir']).iterdir() if p.is_dir()}
    print(f"\nIII filtered PPG dir: {len(iii_ready)} patients ready")
    train_iii_ready = set(train_pids_III) & iii_ready
    val_iii_ready   = set(val_pids_III)   & iii_ready
    print(f"  Train III in splits ∩ ready: {len(train_iii_ready)}/{len(train_pids_III)}")
    print(f"  Val   III in splits ∩ ready: {len(val_iii_ready)}/{len(val_pids_III)}")

    iv_ready = {p.name for p in Path(CONFIG['ppg_IV_filtered_dir']).iterdir() if p.is_dir()}
    print(f"IV filtered PPG dir: {len(iv_ready)} patients ready")

    # ── Build datasets (AlignedDataset will filter to ready pids automatically) ──
    print("\nBuilding train dataset...")
    train_ds = AlignedDataset(
        history_IV=history_IV, history_III=history_III, tokenizer=tokenizer,
        pids_IV=train_pids_IV, pids_III=train_pids_III, training=True,
    )

    print("\nBuilding val dataset...")
    val_ds = AlignedDataset(
        history_IV=history_IV, history_III=history_III, tokenizer=tokenizer,
        pids_IV=val_pids_IV, pids_III=val_pids_III, training=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
        collate_fn=collate_fn, num_workers=CONFIG['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=CONFIG['num_workers'], pin_memory=True
    )

    # ── Model + optimizer ──
    model = AlignmentModel(
        bert_ckpt=CONFIG['bert_ckpt_path'],
        papagei_ckpt=CONFIG['papagei_ckpt_path'],
        vocab_size=tokenizer.vocab_size,
        projection_dim=CONFIG['projection_dim'],
    )
    model.to(device)

    print("\nTrainable params:")
    print(f"  PaPaGei:   {sum(p.numel() for p in model.papagei.parameters() if p.requires_grad):,}")
    print(f"  sEHR-BERT: {sum(p.numel() for p in model.bert.parameters() if p.requires_grad):,}")
    print(f"  MLPs:      {sum(p.numel() for p in list(model.mlp_p.parameters()) + list(model.mlp_s.parameters())):,}")

    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.bert.parameters() if p.requires_grad],         'lr': CONFIG['bert_lr']},
        {'params': list(model.mlp_p.parameters()) + list(model.mlp_s.parameters()), 'lr': CONFIG['mlp_lr']},
    ], weight_decay=CONFIG['weight_decay'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # ── Loop ──
    start_ts          = datetime.now().strftime("%Y%m%d-%H%M%S")
    best_path         = os.path.join(CONFIG['output_dir'], f"best_aligned_model_{start_ts}.pt")
    loss_log_path     = os.path.join(CONFIG['output_dir'], f"loss_log_{start_ts}.json")
    best_val_loss     = float('inf')
    early_stop_counter = 0
    loss_log          = []

    print(f"\nStarting training (epochs={CONFIG['epochs']}, batch_size={CONFIG['batch_size']})...")

    for epoch in range(CONFIG['epochs']):
        model.train()
        model.papagei.eval()
        total_train_loss = 0.0
        n_train_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        t_prev_batch_end = time.time()

        for batch_idx, batch in enumerate(pbar):
            t_batch_start = time.time()

            batch_ts_dt = [
                [pd.to_datetime(t) for t in item['timestamps']]
                for item in batch['raw_batch']
            ]
            z_p, z_s, temp = model(
                batch['ppg_waves'].to(device),
                batch['code_indices'].to(device),
                batch_ts_dt,
                batch['first_timestamps'],
                [item['code_types'] for item in batch['raw_batch']],
                batch['attention_mask'].to(device),
            )
            loss = clip_loss(z_p, z_s, temp)

            if batch_idx == 0:
                # Show batch composition: how many IV vs III in this batch
                ds_counts = {'IV': 0, 'III': 0}
                for d in batch['datasets']:
                    ds_counts[d] = ds_counts.get(d, 0) + 1
                print(
                    f"[ep {epoch+1} batch 0] effective B: {z_p.shape[0]}, "
                    f"composition: {ds_counts}, "
                    f"loss: {loss.item():.4f}, "
                    f"log(B): {np.log(z_p.shape[0]):.4f}"
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_compute = time.time() - t_batch_start
            t_wait = t_batch_start - t_prev_batch_end
            t_prev_batch_end = time.time()
            print(
                f"  [TIME ep{epoch+1} batch{batch_idx+1}] "
                f"wait_for_data={t_wait:.2f}s compute={t_compute:.2f}s "
                f"total={t_wait + t_compute:.2f}s",
                flush=True,
            )

            total_train_loss += loss.item()
            n_train_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'temp': f"{temp.item():.4f}"})

        avg_train_loss = total_train_loss / n_train_batches

        # ── Val ──
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        val_batch_idx = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                batch_ts_dt = [
                    [pd.to_datetime(t) for t in item['timestamps']]
                    for item in batch['raw_batch']
                ]
                z_p, z_s, temp = model(
                    batch['ppg_waves'].to(device),
                    batch['code_indices'].to(device),
                    batch_ts_dt,
                    batch['first_timestamps'],
                    [item['code_types'] for item in batch['raw_batch']],
                    batch['attention_mask'].to(device),
                )
                val_loss = clip_loss(z_p, z_s, temp)

                if val_batch_idx == 0:
                    ds_counts = {'IV': 0, 'III': 0}
                    for d in batch['datasets']:
                        ds_counts[d] = ds_counts.get(d, 0) + 1
                    print(f"VAL batch 0: effective B: {z_p.shape[0]}, "
                          f"composition: {ds_counts}, "
                          f"loss: {val_loss.item():.4f}, "
                          f"log(B): {np.log(z_p.shape[0]):.4f}")
                val_batch_idx += 1

                total_val_loss += val_loss.item()
                n_val_batches += 1

        avg_val_loss = total_val_loss / n_val_batches
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # Log loss
        loss_log.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss':   avg_val_loss,
        })
        with open(loss_log_path, 'w') as f:
            json.dump(loss_log, f, indent=2)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save({
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss':           avg_train_loss,
                'val_loss':             best_val_loss,
                'config':               CONFIG,
            }, best_path)
            print(f"  New best: {best_val_loss:.4f}, saved to {best_path}")
        else:
            early_stop_counter += 1
            print(f"  No improvement: {early_stop_counter}/{CONFIG['early_stop_patience']}")

        if early_stop_counter >= CONFIG['early_stop_patience']:
            print(f"Early stopping at epoch {epoch+1}.")
            break

        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(CONFIG['output_dir'], f"checkpoint_ep{epoch+1}_{start_ts}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    train()