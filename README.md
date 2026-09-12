# PPG–EHR

Code accompanying the submission. Anonymised for review.

Photoplethysmography carries fine-grained physiological detail but none of the
clinical context in an electronic health record. This repository aligns
PaPaGei-s PPG embeddings with sEHR-BERT clinical-code embeddings by
bidirectional contrastive learning on MIMIC-III/IV, and evaluates the result on
post-discharge prediction of atrial fibrillation, heart failure and expanded
MACE at 30/90/180/365 days, using an admission-adaptive mixture-of-experts over
an sEHR expert, a PPG expert and a joint expert.

## Data

No data and no model weights are included here. MIMIC-III and MIMIC-IV are
credentialed datasets distributed by [PhysioNet](https://physionet.org/) under
a Data Use Agreement; both the clinical modules and the matched waveform
databases are needed.

The PPG encoder is [PaPaGei](https://github.com/Nokia-Bell-Labs/papagei-foundation-model),
imported rather than vendored. Clone it separately, obtain its weights under
its own terms, and point `PAPAGEI_REPO` at the clone.

## Setup

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

Python 3.10. Every stage below is a batch job: extraction and filtering want a
high-memory node, alignment wants a GPU.

## Pipeline

**1. PPG preprocessing** — raw waveforms to per-patient 10-minute blocks.
MIMIC-III stays sharded by patient group (`p00`..`p09`) until
`ppg_parquet_per_patient.py`; MIMIC-IV is a single file throughout. Both then
run the same filter and block builder.

```bash
python ppg_preprocessing/extract_ppg_IV.py  --waveform-dir <mimic4wdb>/0.1.0/waves
python ppg_preprocessing/extract_ppg_III.py --waveform-dir <mimic3wdb-matched>/1.0
python ppg_preprocessing/concat_ppg.py --cohort IV
python ppg_preprocessing/concat_ppg.py --cohort III --group p00      # once per group
python ppg_preprocessing/ppg_parquet_per_patient.py --p_dir p00      # III only
python ppg_preprocessing/ppg_full_preprocessing.py        --dataset both
python ppg_preprocessing/convert_to_sample100_10min_pkl.py --dataset both
```

**2. sEHR-BERT** — code cleaning to a pretrained clinical-code encoder.

```bash
python sehr_bert/icd9_to_10.py --gem-i9pcs <gem_i9pcs.txt>
python sehr_bert/preprocess_medication.py
python sehr_bert/keep_50.py --cohort III
python sehr_bert/keep_50.py --cohort IV
python sehr_bert/create_patient_codes_json.py
python sehr_bert/sEHR_BERT.py                 # masked-language-model pretraining
```

**3. Contrastive alignment.** `make_splits.py` writes the patient-level splits
every later stage reads.

```bash
python alignment/make_splits.py
python alignment/train_alignment.py
```

**4. Downstream evaluation.**

```bash
python downstream/build_admission_labels.py
python downstream/extract_feature.py
python downstream/metadata.py --with-sqi     # the gate's four scalars
python downstream/exclude_leackage.py        # drops label/input window overlap
python downstream/score_fusion.py --disease AF     # then HF, then eMACE
```

## Reported results

`downstream/score_fusion.py` produces every row of the results and ablation
tables, one run per disease, and writes per-fold metrics, bootstrap confidence
intervals and the inner-validation hyperparameter surface as three CSVs in
`--out-dir`. Row names map onto the paper as:

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
