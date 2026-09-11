# PPG–EHR: EHR-Guided PPG Representation Learning

Code for *EHR-Guided PPG Representation Learning with Admission-Adaptive
Multimodal Fusion*.

Photoplethysmography carries fine-grained physiological detail but none of the
clinical context in an electronic health record. This repository aligns
PaPaGei-s PPG embeddings with sEHR-BERT clinical-code embeddings by
bidirectional contrastive learning on MIMIC-III/IV, then evaluates the result
on post-discharge prediction of atrial fibrillation, heart failure and
expanded MACE at 30/90/180/365 days, using an admission-adaptive
mixture-of-experts over an sEHR expert, a PPG expert and a joint expert.

> **Status.** All four pipeline stages are here and every path is read from
> the environment, but the repository has not been run end to end from a clean
> checkout. See [Known gaps](#known-gaps).
>
> **No licence has been chosen. Until one is added, no use is granted.**

## Data

**This repository contains no data and no model weights.** MIMIC-III and
MIMIC-IV are credentialed datasets distributed by PhysioNet under a Data Use
Agreement; the required training and access are yours to obtain:

- [MIMIC-IV](https://physionet.org/content/mimiciv/) ·
  [MIMIC-IV Waveform, matched](https://physionet.org/content/mimic4wdb/)
- [MIMIC-III](https://physionet.org/content/mimiciii/) ·
  [MIMIC-III Waveform, matched](https://physionet.org/content/mimic3wdb-matched/)

Nothing derived from them is committed here either. `.gitignore` blocks the
common formats as a backstop, but the rule is the policy, not the pattern list.

The PPG encoder is [PaPaGei](https://github.com/Nokia-Bell-Labs/papagei-foundation-model),
imported rather than vendored — it has its own licence and release history.
Clone it separately, obtain its weights under its terms, and point
`PAPAGEI_REPO` at the clone.

## Setup

```bash
cp .env.example .env    # then edit the four paths
source .env
python paths.py         # prints what resolved -- run this first
```

| variable | holds |
|---|---|
| `PPG_DATA_ROOT` | derived data — PPG parquet trees, history DBs, splits, task dirs |
| `PPG_CKPT_ROOT` | weights — sEHR-BERT, the aligned model, PaPaGei-s |
| `PAPAGEI_REPO` | the PaPaGei clone |
| `PPG_CACHE_ROOT` | *optional* — a writable cache dir, for clusters with a small home |

Python **3.10**, the version this was developed and run against. Later versions
accept syntax 3.10 rejects, so 3.10 is the floor that matters.

Every stage below is a batch job, not an interactive command. Extraction and
filtering want a high-memory node; alignment wants a GPU.

## Pipeline

**1. PPG preprocessing** — raw waveforms to per-patient 10-minute blocks.

```bash
python ppg_preprocessing/extract_ppg_IV.py  --waveform-dir <mimic4wdb>/0.1.0/waves
python ppg_preprocessing/extract_ppg_III.py --waveform-dir <mimic3wdb-matched>/1.0

python ppg_preprocessing/concat_ppg.py --cohort IV
python ppg_preprocessing/concat_ppg.py --cohort III --group p00      # once per group
python ppg_preprocessing/ppg_parquet_per_patient.py --p_dir p00      # III only

python ppg_preprocessing/ppg_full_preprocessing.py        --dataset IV
python ppg_preprocessing/convert_to_sample100_10min_pkl.py --dataset IV
```

MIMIC-III stays sharded by patient group (`p00`..`p09`) from extraction until
`ppg_parquet_per_patient.py`; MIMIC-IV is a single file that goes straight to
per-patient. Both cohorts then run the same filter and block builder.

**2. sEHR-BERT** — code cleaning to a pretrained clinical-code encoder.

```bash
python sehr_bert/icd9_to_10.py --gem-i9pcs <gem_i9pcs.txt>
python sehr_bert/preprocess_medication.py
python sehr_bert/keep_50.py --cohort III
python sehr_bert/keep_50.py --cohort IV
python sehr_bert/create_patient_codes_json.py
python sehr_bert/sEHR_BERT.py                 # masked-language-model pretraining
```

**3. Contrastive alignment** — the PPG and sEHR projections into one space.

```bash
python alignment/make_splits.py       # patient-level; every later stage reads these
python alignment/train_alignment.py
```

**4. Downstream evaluation** — admission-level labels, features, and the models.

```bash
python downstream/build_admission_labels.py
python downstream/extract_feature.py
python downstream/metadata.py --with-sqi     # the gate's four scalars
python downstream/exclude_leackage.py        # drops label/input window overlap
python downstream/score_fusion.py --disease AF     # then HF, then eMACE
```

## Where the reported numbers come from

`downstream/score_fusion.py` produces every row of the results and ablation
tables, in one run per disease, and writes three CSVs into `--out-dir`:

| file | contents |
|---|---|
| `score_fusion_postdisch*.csv` | per-fold metrics and selected hyperparameters, all rows |
| `score_fusion_postdisch_ci*.csv` | point estimates with bootstrap 95% CIs |
| `score_fusion_postdisch_hpsurface*.csv` | inner-validation AUROC for every (lr, wd) candidate |

Row names map onto the paper as:

| paper | CSV `setting` |
|---|---|
| sEHR (pre-alignment) | `A_EHR_pre` |
| sEHR (aligned) | `sEHR_post` |
| PPG (pre-alignment) | `PPG_pre` |
| PPG (aligned) | `B_PPG_post` |
| PPG–EHR MoE (E+P+J) | `H_gate_with_logits` |
| ablation E+P / E+J / P+J | `H_gate_EP` / `H_gate_EJ` / `H_gate_PJ` |

The point estimate is the mean of the five outer-fold metrics. The interval is
built the same way: resample patients with replacement *within* each outer test
fold, recompute that fold's metric, average the five, and take the 2.5th and
97.5th percentiles of the bootstrap means. Resampling is by patient because one
patient can contribute several admissions.

`plain_mlp_torch.py`, `gated_fusion.py`, `alpha_fusion.py` and
`train_probe_regularized.py` are imported by `score_fusion.py` for the shared
training, hyperparameter-selection and bootstrap code. They also run
standalone, as the single-modality and embedding-fusion variants the reported
model was chosen against.

## Known gaps

- **`concat_ppg.py --cohort III` is not validated.** It calls the same chaining
  function the MIMIC-IV branch calls, but it has not been checked against the
  MIMIC-III shards the reported results were built from, which no longer exist.
  Verify this stage first if you are reproducing MIMIC-III from raw waveforms.
- **The hyperparameter-sensitivity figures** are drawn from
  `*_hpsurface*.csv` by a plotting script that is not yet in this repository.
- **No `requirements.txt` yet.** Beyond the usual numpy/pandas/torch/sklearn
  stack, the less obvious ones are `neurokit2` and `biobss` (PPG quality) and
  `icdmappings` (ICD-9 to ICD-10).
- **Most scripts resolve `PPG_DATA_ROOT` at import**, building their file paths
  as module-level constants, so the environment has to be set even to print
  `--help`. The error names the missing variable and what it is for.

## Conventions

No absolute paths in source: anything outside the repository comes from
[`paths.py`](paths.py), anything inside is resolved relative to the importing
file. PaPaGei imports go through `paths.add_papagei_to_path()` so the path
insertion happens in one place. Cache redirection is opt-in —
`paths.configure_caches()` does nothing unless `PPG_CACHE_ROOT` is set, and it
never reassigns `HOME`.
