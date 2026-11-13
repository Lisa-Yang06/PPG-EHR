# 🫀 PPG–EHR: EHR-Guided PPG Representation Learning and Admission-Adaptive Fusion

## 🔬 Overview
Photoplethysmography carries fine-grained physiological detail but none of the
clinical context in an electronic health record. This repository aligns
PaPaGei-s PPG embeddings with sEHR-BERT clinical-code embeddings by
bidirectional contrastive learning on MIMIC-III/IV, and evaluates the result on
post-discharge prediction of atrial fibrillation, heart failure and expanded
MACE at 30/90/180/365 days, using an admission-adaptive mixture-of-experts over
an sEHR expert, a PPG expert and a joint expert.

## 📂 Data

No data is included here. MIMIC-III and MIMIC-IV are
credentialed datasets distributed by [PhysioNet](https://physionet.org/) under
a Data Use Agreement; both the clinical modules and the matched waveform
databases are needed.

The PPG encoder is [PaPaGei](https://github.com/Nokia-Bell-Labs/papagei-foundation-model). 
Clone it separately, obtain its weights under
its own terms, and point `PAPAGEI_REPO` at the clone.

## ⚙️ Setup

```bash
cp .env.example .env    # then edit the paths
source .env
python paths.py         # prints what resolved
```

| variable | holds |
|---|---|
| `PPG_DATA_ROOT` | derived data — PPG parquet trees, history DBs, splits, task dirs |
| `PPG_CKPT_ROOT` | weights — sEHR-BERT, the aligned model, PaPaGei-s |
| `PAPAGEI_REPO` | the PaPaGei clone |
| `PPG_CACHE_ROOT` | *optional* — a writable cache dir, for clusters with a small home |

These stages are intended for batch execution. Waveform extraction and preprocessing 
can require substantial memory; contrastive alignment requires a GPU.

## 🔄 Pipeline

**1. PPG preprocessing**
Extract waveforms, filter PPG segments, and construct per-patient 10-minute blocks. 
MIMIC-III is processed by patient group (p00–`p09`); MIMIC-IV follows a single-file path. 
Repeat the group-specific MIMIC-III commands for each group.

```bash
python ppg_preprocessing/extract_ppg_IV.py  --waveform-dir <mimic4wdb>/0.1.0/waves
python ppg_preprocessing/extract_ppg_III.py --waveform-dir <mimic3wdb-matched>/1.0
python ppg_preprocessing/concat_ppg.py --cohort IV
python ppg_preprocessing/concat_ppg.py --cohort III --group p00      # once per group
python ppg_preprocessing/ppg_parquet_per_patient.py --p_dir p00      # III only
python ppg_preprocessing/ppg_full_preprocessing.py        --dataset both
python ppg_preprocessing/convert_to_sample100_10min_pkl.py --dataset both
```

**2. sEHR-BERT** 
Standardize clinical codes, construct longitudinal patient histories, and pretrain 
sEHR-BERT using masked language modeling.
```bash
python sehr_bert/icd9_to_10.py --gem-i9pcs <gem_i9pcs.txt>
python sehr_bert/preprocess_medication.py
python sehr_bert/keep_50.py --cohort III
python sehr_bert/keep_50.py --cohort IV
python sehr_bert/create_patient_codes_json.py
python sehr_bert/sEHR_BERT.py                 # masked-language-model pretraining
```

**3. Contrastive alignment.** 
Create patient-level splits and train the bidirectional contrastive alignment model.

```bash
python alignment/make_splits.py
python alignment/train_alignment.py
```

**4. Downstream evaluation.**
Construct post-discharge labels, extract admission-level representations and metadata, 
exclude overlapping input and outcome windows, and evaluate the downstream models. 
Run scoring separately for AF, HF, and eMACE.

```bash
python downstream/build_admission_labels.py
python downstream/extract_feature.py
python downstream/metadata.py --with-sqi     # the gate's four scalars
python downstream/exclude_leackage.py        # drops label/input window overlap
python downstream/score_fusion.py --disease AF     # then HF, then eMACE
```

## 📊 Evaluation & Outputs

Downstream evaluation uses five outer and three inner patient-disjoint cross-validation folds. 
downstream/score_fusion.py evaluates the primary predictors and expert-combination ablations. 
For each disease, it writes per-fold metrics, bootstrap confidence intervals, 
and inner-validation hyperparameter results as CSV files under the specified --out-dir.

The principal CSV setting values correspond to the following models:

| paper | CSV `setting` |
|---|---|
| sEHR (pre-alignment) | `A_EHR_pre` |
| sEHR (aligned) | `sEHR_post` |
| PPG (pre-alignment) | `PPG_pre` |
| PPG (aligned) | `B_PPG_post` |
| PPG–EHR MoE (E+P+J) | `H_gate_with_logits` |
| ablation E+P / E+J / P+J | `H_gate_EP` / `H_gate_EJ` / `H_gate_PJ` |

`plain_mlp_torch.py`, `gated_fusion.py`, `alpha_fusion.py` and
`train_probe_regularized.py` supply the shared training,
hyperparameter-selection and bootstrap code, and also run standalone as the
single-modality and embedding-fusion variants.
