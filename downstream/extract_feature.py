"""
STAGE 4: Post-discharge admission-level dual-modality feature extraction.

For each REAL hospital admission from admission_disease_labels.joblib:
  - sample id: (dataset, patient_id, hadm_id)
  - prediction origin: dischtime

sEHR:
  - default window: [dischtime - 12mo, dischtime]
    (set ehr_anchor='admit' below if pre-admission EHR is desired)
  - all code types, latest 512
  - original MedicalBERT -> v_ehr_pre (320)
  - aligned MedicalBERT + mlp_s + L2 norm -> v_ehr_post (128)

PPG:
  - reconstruct the same non-overlapping consecutive 10-min blocks used
    previously: 60 consecutive ~10-s windows
  - require the ENTIRE 10-min block to lie inside [admittime, dischtime]
  - if >100 qualifying blocks, take 100 approximately uniformly over time
  - each 10-min block:
        60 x 10-s PaPaGei embeddings -> mean -> h_k (512)
  - admission pre:
        mean_k(h_k)
  - admission post:
        each h_k -> mlp_p -> L2 normalize -> z_k
        mean_k(z_k) -> L2 normalize

No first-onset exclusion is used. Recurrent/prevalent disease is allowed by
the new endpoint definition.

Missing modality -> feature = NaN, mask = 0.

Primary labels carried forward:
  AF, HF, eMACE at 30/90/180/365 days, plus their prior indicators.
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

# Opt-in, and never reassigns HOME -- see paths.configure_caches.
paths.configure_caches()

import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm


SEHR_DIR = os.path.join(_ROOT, "sehr_bert")
if SEHR_DIR not in sys.path:
    sys.path.insert(0, SEHR_DIR)
from sEHR_BERT import MedicalBERT, MedicalCodeTokenizer

paths.add_papagei_to_path()
from models.resnet import ResNet1DMoE


CONFIG = {
    # New Stage-3 output
    'label_path':
        str(paths.data_root() / "ds1/admission_disease_labels.joblib"),

    'ppg_IV_filtered_dir':
        str(paths.data_root() / "MIMIC_IV_ppg_filtered_by_patient"),
    'ppg_III_filtered_dir':
        str(paths.data_root() / "MIMIC_III_ppg_filtered_by_patient"),

    'history_db_IV_path':
        str(paths.data_root() / "patient_history_db_IV.joblib"),
    'history_db_III_path':
        str(paths.data_root() / "patient_history_db_III.joblib"),

    'tokenizer_path':
        str(paths.data_root() / "medical_tokenizer_combined.json"),

    'alignment_ckpt':
        str(paths.data_root() / "alignment_results_combined_10min_freezeppg/best_aligned_model_20260709-111121.pt"),

    'orig_bert_ckpt':
        str(paths.ckpt_root() / "best_medbert_model_20260617-190122.pt"),

    # Proposed new output name
    'output_path':
        str(paths.data_root() / "ds1/downstream_features_admission_postdisch.joblib"),

    'ppg_parquet_filename': "ppg.parquet",
    'signal_col': "signal",

    # admission aggregation
    'max_blocks_per_sample': 100,

    # EHR
    'max_seq_len': 512,
    'projection_dim': 128,
    'window_months': 12,

    # Current prediction origin is discharge.
    # Change to 'admit' if you explicitly want pre-admission EHR only.
    'ehr_anchor': 'discharge',

    # 10-min PPG definition -- kept from previous Stage 4
    'n_windows_per_seg': 60,
    'window_signal_len': 1250,
    'win_s': 10,
    'gap_tol': (9.9, 10.1),
}

PRIMARY_TARGETS = ('AF', 'HF', 'eMACE')
HORIZONS = (30, 90, 180, 365)


PAPAGEI_MODEL_CONFIG = {
    'base_filters': 32,
    'kernel_size': 3,
    'stride': 2,
    'groups': 1,
    'n_block': 18,
    'n_classes': 512,
    'n_experts': 3,
}


# ── Model loading ─────────────────────────────────────────────────────────
def load_aligned_components(ckpt_path, vocab_size, device):
    print(f"Loading alignment ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model_state_dict']

    epoch = ckpt.get('epoch')
    val_loss = ckpt.get('val_loss')
    print(f"  epoch={epoch}, val_loss={val_loss}")

    papagei = ResNet1DMoE(
        in_channels=1,
        **PAPAGEI_MODEL_CONFIG
    )
    papagei_state = {
        k.replace('papagei.', '', 1): v
        for k, v in state.items()
        if k.startswith('papagei.')
    }
    miss_p, unexp_p = papagei.load_state_dict(
        papagei_state,
        strict=False
    )
    print(
        f"  PaPaGei missing={len(miss_p)}, "
        f"unexpected={len(unexp_p)}"
    )
    papagei.eval().to(device)

    bert = MedicalBERT(vocab_size=vocab_size)
    bert_state = {
        k.replace('bert.', '', 1): v
        for k, v in state.items()
        if k.startswith('bert.')
    }
    miss_b, unexp_b = bert.load_state_dict(
        bert_state,
        strict=False
    )
    print(
        f"  BERT missing={len(miss_b)}, "
        f"unexpected={len(unexp_b)}"
    )
    bert.eval().to(device)

    mlp_p = nn.Sequential(
        nn.Linear(512, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, CONFIG['projection_dim']),
    )

    mlp_s = nn.Sequential(
        nn.Linear(320, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, CONFIG['projection_dim']),
    )

    mlp_p.load_state_dict({
        k.replace('mlp_p.', '', 1): v
        for k, v in state.items()
        if k.startswith('mlp_p.')
    })

    mlp_s.load_state_dict({
        k.replace('mlp_s.', '', 1): v
        for k, v in state.items()
        if k.startswith('mlp_s.')
    })

    mlp_p.eval().to(device)
    mlp_s.eval().to(device)

    for m in (papagei, bert, mlp_p, mlp_s):
        for p in m.parameters():
            p.requires_grad = False

    return papagei, bert, mlp_p, mlp_s


# ── PPG forward ───────────────────────────────────────────────────────────
def forward_ppg_segments(
    signals_list,
    papagei,
    mlp_p,
    device,
    gpu_chunk=50,
):
    """
    signals_list:
        list/array of K 10-min blocks,
        each block has shape (60, 1250).

    Returns
    -------
    v_ppg_pre:
        mean of K 10-min PaPaGei representations, shape (512,)

    v_ppg_post:
        each 10-min representation is projected and L2-normalized exactly
        as in alignment; those aligned vectors are averaged across the
        admission and normalized again, shape (128,)
    """
    n_win = CONFIG['n_windows_per_seg']
    sig_len = CONFIG['window_signal_len']
    n_seg = len(signals_list)

    if n_seg == 0:
        return None, None

    pre_sum = torch.zeros(512, device=device)
    post_sum = torch.zeros(
        CONFIG['projection_dim'],
        device=device
    )

    with torch.no_grad():
        for s in range(0, n_seg, gpu_chunk):
            chunk = np.stack(
                signals_list[s:s + gpu_chunk],
                axis=0
            )
            c = chunk.shape[0]

            t = torch.tensor(
                chunk,
                dtype=torch.float32
            ).view(
                -1,
                1,
                sig_len
            ).to(device)

            out = papagei(t)
            emb = out[0] if isinstance(
                out,
                (tuple, list)
            ) else out

            if emb.dim() == 3:
                emb = emb.mean(dim=1)

            # Same 10-min representation as before:
            # 60 x 10-s embeddings -> one 512-d vector
            block_pre = emb.view(
                c,
                n_win,
                -1
            ).mean(dim=1)

            # Alignment-space representation PER 10-min block
            block_post = F.normalize(
                mlp_p(block_pre),
                dim=-1
            )

            pre_sum += block_pre.sum(dim=0)
            post_sum += block_post.sum(dim=0)

            del t, out, emb, block_pre, block_post

        v_ppg_pre = pre_sum / n_seg

        # Mean aligned directions across the admission, then renormalize.
        v_ppg_post = F.normalize(
            (post_sum / n_seg).unsqueeze(0),
            dim=-1
        ).squeeze(0)

    return (
        v_ppg_pre.cpu().numpy(),
        v_ppg_post.cpu().numpy(),
    )


# ── EHR forward ───────────────────────────────────────────────────────────
def forward_ehr(
    history_entry,
    anchor_time,
    bert,
    bert_orig,
    mlp_s,
    tokenizer,
    device,
):
    """
    sEHR window:
        [anchor_time - 12mo, anchor_time]

    anchor_time is discharge by default for the current post-discharge
    prediction task. Set CONFIG['ehr_anchor']='admit' to preserve the old
    pre-admission-only definition.

    ALL code types, latest 512.

    Returns
    -------
    v_ehr_pre:
        original MedicalBERT pooled embedding, shape (320,)

    v_ehr_post:
        aligned MedicalBERT -> mlp_s -> L2 normalized, shape (128,)
    """
    hist_ts = np.array(
        pd.to_datetime(history_entry['timestamps'])
    )
    hist_codes = np.asarray(
        history_entry['codes']
    )
    hist_types = np.asarray(
        history_entry['code_type']
    )

    days = int(CONFIG['window_months'] * 30.44)
    window_delta = np.timedelta64(days, 'D')

    anchor_day = pd.Timestamp(
        pd.to_datetime(anchor_time)
    ).normalize()
    anchor_day_np = np.datetime64(anchor_day)

    hist_ts_np = hist_ts.astype('datetime64[ns]')
    
    if len(hist_ts_np) > 1 and np.any(hist_ts_np[1:] < hist_ts_np[:-1]):
        raise ValueError(
            "history_entry timestamps are not sorted chronologically; "
            "cannot safely use hist_ts_np[0] as first_timestamp."
        )

    hist_day_np = (
        hist_ts_np
        .astype('datetime64[D]')
        .astype('datetime64[ns]')
    )

    start_day_np = anchor_day_np - window_delta

    mask = (
        (hist_day_np >= start_day_np)
        &
        (hist_day_np <= anchor_day_np)
    )

    if not mask.any():
        return None, None

    sub_codes = hist_codes[mask]
    sub_ts = hist_ts_np[mask]
    sub_types = hist_types[mask]

    # Truncate to latest 512
    if len(sub_codes) > CONFIG['max_seq_len']:
        order = np.argsort(sub_ts)
        keep = order[-CONFIG['max_seq_len']:]
        keep = np.sort(keep)

        sub_codes = sub_codes[keep]
        sub_ts = sub_ts[keep]
        sub_types = sub_types[keep]

    first_ts = pd.to_datetime(hist_ts_np[0])

    indices = tokenizer.encode(
        sub_codes.tolist()
    )
    indices_t = torch.tensor(
        [indices],
        dtype=torch.long
    ).to(device)

    valid = (indices_t != 0)
    sub_ts_dt = [
        pd.to_datetime(t)
        for t in sub_ts
    ]

    with torch.no_grad():
        # aligned BERT
        out = bert(
            indices_t,
            [sub_ts_dt],
            first_timestamps=[first_ts],
            code_types=[sub_types.tolist()],
            attention_mask=~valid,
            return_embeddings=True,
        )

        m = valid.unsqueeze(-1).expand(
            out.size()
        ).float()

        pooled = (
            torch.sum(out * m, dim=1)
            /
            torch.clamp(
                m.sum(dim=1),
                min=1e-9
            )
        )

        ehr_post = F.normalize(
            mlp_s(pooled),
            dim=-1
        )

        # original pretrained BERT
        out_o = bert_orig(
            indices_t,
            [sub_ts_dt],
            first_timestamps=[first_ts],
            code_types=[sub_types.tolist()],
            attention_mask=~valid,
            return_embeddings=True,
        )

        m_o = valid.unsqueeze(-1).expand(
            out_o.size()
        ).float()

        pooled_orig = (
            torch.sum(out_o * m_o, dim=1)
            /
            torch.clamp(
                m_o.sum(dim=1),
                min=1e-9
            )
        )

    return (
        pooled_orig.cpu().squeeze(0).numpy(),
        ehr_post.cpu().squeeze(0).numpy(),
    )


# ── PPG: read full parquet, find ALL consecutive 10-min blocks ────────────
def load_patient_ppg(pid, dataset):
    """
    Read the patient's full ppg.parquet and reconstruct every consecutive
    NON-OVERLAPPING 10-minute block:

        60 windows, each ~10 s apart,
        within one (record_id, segment).

    This preserves the previous Stage-4 reconstruction logic.

    Returns
    -------
    {
        'signals':      (N, 60, 1250),
        'start_times':  (N,),
        'center_times': (N,),
        'end_times':    (N,)
    }

    end_time is defined as:
        start time of the 60th 10-s window + 10 s

    so the entire signal block spans approximately 10 minutes.
    """
    ppg_dir = (
        CONFIG['ppg_IV_filtered_dir']
        if dataset == 'IV'
        else CONFIG['ppg_III_filtered_dir']
    )

    pq_path = (
        Path(ppg_dir)
        / pid
        / CONFIG['ppg_parquet_filename']
    )

    if not pq_path.exists():
        return None

    try:
        df = pd.read_parquet(
            pq_path,
            columns=[
                'record_id',
                'segment',
                'window_start_time',
                CONFIG['signal_col'],
            ]
        )
    except Exception:
        print(
            f"  !! Failed to read PPG parquet "
            f"for {pid} ({dataset}): {pq_path}"
        )
        return None

    if len(df) == 0:
        return None

    df['window_start_time'] = pd.to_datetime(
        df['window_start_time']
    )

    n_win = CONFIG['n_windows_per_seg']
    lo, hi = CONFIG['gap_tol']
    mid = n_win // 2

    blocks_pos = []
    blocks_start = []
    blocks_center = []
    blocks_end = []

    for (_rec, _seg), g in df.groupby(
        ['record_id', 'segment'],
        sort=False
    ):
        g = g.sort_values('window_start_time')

        times = g[
            'window_start_time'
        ].values.astype('datetime64[ns]')

        pos = g.index.values

        run = []

        for i in range(len(times)):
            if i == 0:
                run = [0]
            else:
                dt = (
                    (times[i] - times[i - 1])
                    / np.timedelta64(1, 's')
                )

                if lo <= dt <= hi:
                    run.append(i)
                else:
                    run = [i]

            if len(run) == n_win:
                blocks_pos.append([
                    pos[j]
                    for j in run
                ])

                blocks_start.append(
                    times[run[0]]
                )

                blocks_center.append(
                    times[run[mid]]
                )

                # Last window START + 10 sec = end of full block
                blocks_end.append(
                    times[run[-1]]
                    + np.timedelta64(
                        CONFIG['win_s'],
                        's'
                    )
                )

                # preserve previous non-overlapping behavior
                run = []

    if not blocks_pos:
        return None

    sig_len = CONFIG['window_signal_len']

    needed = sorted({
        ix
        for blk in blocks_pos
        for ix in blk
    })

    sigcache = {}

    for ix in needed:
        sigcache[ix] = np.asarray(
            json.loads(
                df.at[
                    ix,
                    CONFIG['signal_col']
                ]
            ),
            dtype=np.float32
        )

    good_sig = []
    good_start = []
    good_center = []
    good_end = []

    for blk, sta, ctr, end in zip(
        blocks_pos,
        blocks_start,
        blocks_center,
        blocks_end,
    ):
        if all(
            len(sigcache[ix]) == sig_len
            for ix in blk
        ):
            good_sig.append(
                np.stack([
                    sigcache[ix]
                    for ix in blk
                ])
            )
            good_start.append(sta)
            good_center.append(ctr)
            good_end.append(end)
        else:
            print(
                "Found PPG 10s window "
                "!= 1250 samples"
            )

    if not good_sig:
        return None

    signals = np.stack(
        good_sig
    ).astype(np.float32)

    starts = np.array(
        good_start,
        dtype='datetime64[ns]'
    )
    centers = np.array(
        good_center,
        dtype='datetime64[ns]'
    )
    ends = np.array(
        good_end,
        dtype='datetime64[ns]'
    )

    # IMPORTANT:
    # groupby guarantees ordering only within each record/segment.
    # Sort ALL reconstructed blocks globally by time before temporal sampling.
    order = np.argsort(starts)

    signals = signals[order]
    starts = starts[order]
    centers = centers[order]
    ends = ends[order]

    # Timing sanity check.
    durations_s = (
        (ends - starts)
        / np.timedelta64(1, 's')
    )

    bad_dur = (
        (durations_s < 590)
        |
        (durations_s > 610)
    )

    if bad_dur.any():
        print(
            f"  !! {pid} ({dataset}): "
            f"{int(bad_dur.sum())}/{len(durations_s)} "
            f"reconstructed blocks outside "
            f"590-610 s duration"
        )

    return {
        'signals': signals,
        'start_times': starts,
        'center_times': centers,
        'end_times': ends,
    }


def canon_pid(raw, dataset):
    s = str(raw).strip()
    return s.zfill(6) if dataset == 'III' else s


def label_value(row, col):
    """
    Preserve censoring:
      1 / 0 stay numeric
      NaN stays NaN
    """
    x = getattr(row, col)
    if pd.isna(x):
        return np.nan
    return int(x)


def run():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # ── Labels / real admissions ──────────────────────────────────────────
    print(
        f"\nLoading admission labels: "
        f"{CONFIG['label_path']}"
    )

    labels = joblib.load(
        CONFIG['label_path']
    )

    if not isinstance(labels, pd.DataFrame):
        labels = pd.DataFrame(labels)

    required = {
        'patient_id',
        'dataset',
        'hadm_id',
        'admittime',
        'dischtime',
    }

    missing = required - set(labels.columns)

    if missing:
        raise ValueError(
            f"Label file is missing required columns: "
            f"{sorted(missing)}"
        )

    labels['admittime'] = pd.to_datetime(
        labels['admittime']
    )
    labels['dischtime'] = pd.to_datetime(
        labels['dischtime']
    )

    # canonical patient ids, just in case
    labels['patient_id'] = [
        canon_pid(pid, ds)
        for pid, ds in zip(
            labels['patient_id'],
            labels['dataset']
        )
    ]

    # New sample id must be unique.
    keys = [
        'dataset',
        'patient_id',
        'hadm_id'
    ]

    if labels.duplicated(keys).any():
        dup = int(
            labels.duplicated(
                keys,
                keep=False
            ).sum()
        )
        raise ValueError(
            f"{dup} rows have duplicate "
            f"(dataset, patient_id, hadm_id)"
        )

    print(
        f"  {len(labels)} real admission samples, "
        f"{labels[['dataset', 'patient_id']].drop_duplicates().shape[0]} "
        f"patients"
    )

    # Check current target columns.
    for disease in PRIMARY_TARGETS:
        prior_col = f'{disease}__prior'

        if prior_col not in labels.columns:
            raise ValueError(
                f"Missing {prior_col}"
            )

        for N in HORIZONS:
            c = f'{disease}__{N}'
            if c not in labels.columns:
                raise ValueError(
                    f"Missing {c}"
                )

    # ── Histories ─────────────────────────────────────────────────────────
    print("\nLoading histories...")

    hist_IV = joblib.load(
        CONFIG['history_db_IV_path']
    )
    hist_IV = {
        str(k): v
        for k, v in hist_IV.items()
    }

    hist_III = joblib.load(
        CONFIG['history_db_III_path']
    )
    hist_III = {
        str(k).zfill(6): v
        for k, v in hist_III.items()
    }

    # ── Tokenizer + models ────────────────────────────────────────────────
    tokenizer = MedicalCodeTokenizer()
    tokenizer.load(
        CONFIG['tokenizer_path']
    )

    papagei, bert, mlp_p, mlp_s = (
        load_aligned_components(
            CONFIG['alignment_ckpt'],
            tokenizer.vocab_size,
            device
        )
    )

    bert_orig = MedicalBERT(
        vocab_size=tokenizer.vocab_size
    )

    orig_sd = torch.load(
        CONFIG['orig_bert_ckpt'],
        map_location='cpu'
    )

    orig_sd = orig_sd.get(
        'model_state_dict',
        orig_sd
    )

    miss_o, unexp_o = (
        bert_orig.load_state_dict(
            orig_sd,
            strict=False
        )
    )

    print(
        f"  orig BERT missing={len(miss_o)}, "
        f"unexpected={len(unexp_o)}"
    )

    bert_orig.eval().to(device)

    for p in bert_orig.parameters():
        p.requires_grad = False

    # ── Extraction ────────────────────────────────────────────────────────
    final = []

    n_ppg_ok = 0
    n_ehr_ok = 0
    n_both = 0
    n_neither = 0

    n_blocks_reconstructed = 0
    n_blocks_eligible = 0
    n_blocks_used = 0

    n_patient_groups = (
        labels[
            ['dataset', 'patient_id']
        ]
        .drop_duplicates()
        .shape[0]
    )

    print("\nExtracting...")

    # Group by dataset + patient to avoid any cross-dataset PID collision.
    grouped = labels.groupby(
        ['dataset', 'patient_id'],
        sort=False
    )

    for (dataset, pid), rows in tqdm(
        grouped,
        total=n_patient_groups
    ):
        history = (
            hist_IV
            if dataset == 'IV'
            else hist_III
        )

        history_entry = history.get(pid)

        # Read / reconstruct PPG once per patient.
        ppg_data = load_patient_ppg(
            pid,
            dataset
        )

        if ppg_data is not None:
            ppg_signals = ppg_data['signals']
            ppg_starts = ppg_data['start_times']
            ppg_centers = ppg_data['center_times']
            ppg_ends = ppg_data['end_times']

            n_blocks_reconstructed += len(
                ppg_signals
            )
        else:
            ppg_signals = None

            ppg_starts = np.array(
                [],
                dtype='datetime64[ns]'
            )
            ppg_centers = np.array(
                [],
                dtype='datetime64[ns]'
            )
            ppg_ends = np.array(
                [],
                dtype='datetime64[ns]'
            )

        for row in rows.itertuples():
            admit_time = pd.Timestamp(
                row.admittime
            )
            disch_time = pd.Timestamp(
                row.dischtime
            )

            admit_np = np.datetime64(
                admit_time
            )
            disch_np = np.datetime64(
                disch_time
            )

            # ── EHR ───────────────────────────────────────────────────────
            v_ehr_pre = None
            v_ehr_post = None

            if CONFIG['ehr_anchor'] == 'discharge':
                ehr_anchor_time = disch_time
            elif CONFIG['ehr_anchor'] == 'admit':
                ehr_anchor_time = admit_time
            else:
                raise ValueError(
                    "CONFIG['ehr_anchor'] must be "
                    "'admit' or 'discharge'"
                )

            if history_entry is not None:
                v_ehr_pre, v_ehr_post = (
                    forward_ehr(
                        history_entry,
                        ehr_anchor_time,
                        bert,
                        bert_orig,
                        mlp_s,
                        tokenizer,
                        device
                    )
                )

            ehr_mask = int(
                v_ehr_pre is not None
            )

            # ── PPG ───────────────────────────────────────────────────────
            v_ppg_pre = None
            v_ppg_post = None

            n_ppg_available = 0
            n_ppg_used_this = 0

            if (
                ppg_signals is not None
                and len(ppg_starts) > 0
            ):
                # STRICT full-block containment.
                #
                # Old code used block CENTER inside [admit, disch].
                # For post-discharge prediction that is not enough:
                # no part of the 10-min signal may cross dischtime.
                in_window = (
                    (ppg_starts >= admit_np)
                    &
                    (ppg_ends <= disch_np)
                )

                if in_window.any():
                    seg_idxs = np.where(
                        in_window
                    )[0]

                    n_ppg_available = len(
                        seg_idxs
                    )

                    n_blocks_eligible += (
                        n_ppg_available
                    )

                    # Arrays were globally sorted by block start.
                    # Therefore linspace gives deterministic temporal coverage.
                    if (
                        len(seg_idxs)
                        >
                        CONFIG[
                            'max_blocks_per_sample'
                        ]
                    ):
                        sel = np.linspace(
                            0,
                            len(seg_idxs) - 1,
                            CONFIG[
                                'max_blocks_per_sample'
                            ]
                        ).astype(int)

                        seg_idxs = seg_idxs[sel]

                    n_ppg_used_this = len(
                        seg_idxs
                    )
                    n_blocks_used += (
                        n_ppg_used_this
                    )

                    # Hard timing sanity check.
                    if not np.all(
                        ppg_starts[seg_idxs]
                        >= admit_np
                    ):
                        raise RuntimeError(
                            "Selected PPG block starts "
                            "before admission"
                        )

                    if not np.all(
                        ppg_ends[seg_idxs]
                        <= disch_np
                    ):
                        raise RuntimeError(
                            "Selected PPG block crosses "
                            "discharge"
                        )

                    sigs = [
                        ppg_signals[i]
                        for i in seg_idxs
                    ]

                    v_ppg_pre, v_ppg_post = (
                        forward_ppg_segments(
                            sigs,
                            papagei,
                            mlp_p,
                            device
                        )
                    )

            ppg_mask = int(
                v_ppg_pre is not None
            )

            # ── Fill missing with NaN ─────────────────────────────────────
            if v_ppg_pre is None:
                v_ppg_pre = np.full(
                    512,
                    np.nan,
                    dtype=np.float32
                )
                v_ppg_post = np.full(
                    128,
                    np.nan,
                    dtype=np.float32
                )

            if v_ehr_pre is None:
                v_ehr_pre = np.full(
                    320,
                    np.nan,
                    dtype=np.float32
                )
                v_ehr_post = np.full(
                    128,
                    np.nan,
                    dtype=np.float32
                )

            # ── Counters ─────────────────────────────────────────────────
            if ppg_mask and ehr_mask:
                n_both += 1
            elif ppg_mask:
                n_ppg_ok += 1
            elif ehr_mask:
                n_ehr_ok += 1
            else:
                n_neither += 1

            rec = {
                'patient_id':
                    pid,

                'dataset':
                    dataset,

                'hadm_id':
                    row.hadm_id,

                # Keep old key name for downstream compatibility.
                'admit_time':
                    admit_time,

                'dischtime':
                    disch_time,

                'ehr_anchor_time':
                    pd.Timestamp(
                        ehr_anchor_time
                    ),

                'v_ppg_pre_mlp':
                    v_ppg_pre.astype(
                        np.float32
                    ),

                'v_ppg_post_mlp':
                    v_ppg_post.astype(
                        np.float32
                    ),

                'ppg_mask':
                    ppg_mask,

                # Useful audit fields:
                'n_ppg_blocks_available':
                    n_ppg_available,

                'n_ppg_blocks_used':
                    n_ppg_used_this,

                'v_ehr_pre_mlp':
                    v_ehr_pre.astype(
                        np.float32
                    ),

                'v_ehr_post_mlp':
                    v_ehr_post.astype(
                        np.float32
                    ),

                'ehr_mask':
                    ehr_mask,
            }

            # Carry forward primary disease labels.
            # IMPORTANT: preserve NaN censoring.
            for disease in PRIMARY_TARGETS:
                prior_col = (
                    f'{disease}__prior'
                )

                rec[prior_col] = (
                    label_value(
                        row,
                        prior_col
                    )
                )

                for N in HORIZONS:
                    c = f'{disease}__{N}'
                    rec[c] = label_value(
                        row,
                        c
                    )

            final.append(rec)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 70)

    print(
        f"Total samples: {len(final)}"
    )

    print(
        f"  PPG 10-min blocks reconstructed "
        f"(patient-level total): "
        f"{n_blocks_reconstructed}"
    )

    print(
        f"  PPG blocks eligible across admissions: "
        f"{n_blocks_eligible}"
    )

    print(
        f"  PPG blocks actually used after "
        f"max-{CONFIG['max_blocks_per_sample']} cap: "
        f"{n_blocks_used}"
    )

    print(
        f"  both modalities: {n_both}"
    )
    print(
        f"  EHR only:        {n_ehr_ok}"
    )
    print(
        f"  PPG only:        {n_ppg_ok}"
    )
    print(
        f"  neither:         {n_neither}"
    )

    outdf = pd.DataFrame(final)

    for disease in PRIMARY_TARGETS:
        print(
            "\n" + "-" * 70
        )
        print(disease)
        print("-" * 70)

        for N in HORIZONS:
            c = f'{disease}__{N}'

            usable = outdf[c].notna()
            pos = outdf[c] == 1
            ppg = outdf['ppg_mask'] == 1

            n_usable = int(
                usable.sum()
            )
            n_pos = int(
                pos.sum()
            )

            n_pos_pat = (
                outdf.loc[
                    pos,
                    ['dataset', 'patient_id']
                ]
                .drop_duplicates()
                .shape[0]
            )

            n_ppg_usable = int(
                (usable & ppg).sum()
            )

            n_ppg_pos = int(
                (pos & ppg).sum()
            )

            n_ppg_pos_pat = (
                outdf.loc[
                    pos & ppg,
                    ['dataset', 'patient_id']
                ]
                .drop_duplicates()
                .shape[0]
            )

            print(
                f"  {N:>3d}d: "
                f"usable={n_usable:>4d}, "
                f"pos={n_pos:>4d} "
                f"({n_pos_pat} pats), "
                f"PPG usable={n_ppg_usable:>4d}, "
                f"PPG pos={n_ppg_pos:>4d} "
                f"({n_ppg_pos_pat} pats)"
            )

    joblib.dump(
        final,
        CONFIG['output_path']
    )

    print(
        f"\nSaved to "
        f"{CONFIG['output_path']}"
    )


if __name__ == "__main__":
    run()