# PPG–EHR Multimodal Alignment

CLIP-style contrastive alignment of PaPaGei-s PPG embeddings with sEHR-BERT
clinical-code embeddings, trained on MIMIC-III/IV and evaluated on
post-discharge disease identification.

> **Status: staging.** Code is being migrated here module by module from the
> working repository. Not yet complete or runnable end to end.
> **No licence has been chosen yet — until one is added, no use is granted.**

---

## Data availability

**This repository contains no data and no model weights.**

MIMIC-III and MIMIC-IV are credentialed datasets distributed by PhysioNet
under a Data Use Agreement. To run any of this you must complete the required
training and obtain access yourself:

- MIMIC-IV — https://physionet.org/content/mimiciv/
- MIMIC-IV Waveform (matched subset) — https://physionet.org/content/mimic4wdb/
- MIMIC-III Waveform (matched subset) — https://physionet.org/content/mimic3wdb-matched/

Nothing derived from these datasets is committed here either. `.gitignore`
blocks the common formats as a backstop, but the rule is the policy, not the
pattern list.

## External dependency

The PPG encoder is [PaPaGei](https://github.com/Nokia-Bell-Labs/papagei-foundation-model),
imported rather than vendored — it carries its own licence and release history.
Clone it separately and point `PAPAGEI_REPO` at it.

## Setup

```bash
cp .env.example .env    # then edit the paths
source .env
python paths.py         # prints what resolved; run this first
```

Four roots, all resolved in [`paths.py`](paths.py):

| variable | holds |
|---|---|
| `PPG_DATA_ROOT` | derived data — history DBs, PPG parquet trees, splits, task dirs |
| `PPG_CKPT_ROOT` | model weights — sEHR-BERT, alignment, PaPaGei-s |
| `PAPAGEI_REPO` | the PaPaGei clone |
| `PPG_CACHE_ROOT` | *optional* — writable cache dir, for clusters with a small home |

Python **3.10** (the version this was developed and run against; 3.12 accepts
some syntax 3.10 rejects, so 3.10 is the floor that matters).

## Conventions

**No absolute paths in source.** Anything outside the repository comes from
`paths.py`. Anything inside it is resolved relative to the importing file:

```python
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
```

**PaPaGei imports go through the helper**, so the path insertion happens in one
place rather than being repeated per file:

```python
import paths
paths.add_papagei_to_path()
from preprocessing.ppg import preprocess_one_ppg_signal
```

**Cache redirection is opt-in.** `paths.configure_caches()` is a no-op unless
`PPG_CACHE_ROOT` is set, and it never reassigns `HOME`.

## Repository layout

Filled in as modules land.

| stage | status |
|---|---|
| PPG preprocessing | not yet migrated |
| sEHR-BERT | not yet migrated |
| Contrastive alignment | not yet migrated |
| Downstream evaluation | not yet migrated |
