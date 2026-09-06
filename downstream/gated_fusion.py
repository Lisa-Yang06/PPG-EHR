"""
GATED FUSION -- post-discharge downstream cohort.

This is a minimal adaptation of the previous gated-fusion script. The fusion
model, optimizer, four-setting comparison, shared-across-horizons design,
patient-grouped outer CV, gate-input ablations, and case-level analysis are
kept. The changes are limited to what the rebuilt admission cohort requires:

  * clean post-discharge feature/metadata files;
  * merge by (dataset, patient_id, hadm_id);
  * AF, HF, and eMACE are run as three separate disease families;
  * each disease has one shared gate across 30/90/180/365 days;
  * censored (NaN) labels are removed only for that disease x horizon row;
  * outer folds are disease-specific and stratified on the 365-day STATUS:
        0 = usable negative, 1 = positive, 2 = censored;
    NaN is never converted to 0;
  * gate scalars are updated to the new metadata definitions:
        acf_p10, skew_p10, n_codes_fed, gap_median_from_admit_h.

Fusion is unchanged:

    z_fused,i = alpha_i * z_PPG,i + (1 - alpha_i) * z_sEHR,i
    alpha_i   = sigmoid(g(x_i))

Main gate inputs are the same advisor-specified categories as before:
PPG embedding, sEHR embedding, PPG signal quality, EHR-code amount,
PPG-to-admission timing, and prediction horizon.

SETTINGS
--------
    1   EHR_post
    2   PPG_post
    1b  EHR_post + gate scalars
    3   fixed alpha, selected inside the inner CV
    4   learned alpha_i

GATE INPUT VARIANTS
-------------------
    default            embeddings + scalars + horizon
    --no-emb-in-gate   scalars + horizon
    --emb-only-gate    embeddings + horizon

sEHR SOURCE FOR THE MIXER AND THE GATE  (--ehr-source)
------------------------------------------------------
    aligned   v_ehr_post_mlp. The default; unchanged behaviour.
    randmlp   L2(mlp_s_random(v_ehr_pre_mlp)) -- same architecture and same L2
              as the aligned projection, but never trained.
    pca       PCA(128) of v_ehr_pre_mlp, block-scaled. SETTING 4 ONLY.

WHY pca TOUCHES SETTING 4 ALONE
-------------------------------
0e_EHR_pre_pca128 showed a label-blind PCA of the same 320-d input reaching
0a_EHR_pre to within ~0.00x, so 128 dimensions are not what costs
v_ehr_post its 0.024-0.083 on HF and eMACE -- the aligned projection lost
the useful directions rather than the budget. That makes PCA-128 the strong
128-d sEHR side the learned gate has never had: setting 4 has only ever
mixed v_ehr_post (degraded) or a random projection.

Settings 3 and the alpha sweep keep v_ehr_post on purpose. The comparison
this mode is for is setting 4 against setting 4 in the aligned and randmlp
runs -- same folds, same code, same gate, three different sEHR sides.
Swapping the mixer settings too would have changed two things at once.

The PCA is fitted INSIDE each outer fold rather than once over the cohort:
it reads the feature matrix, so a cohort-wide fit would let the test fold
shape the principal axes. Unsupervised leakage is still leakage. randmlp
needs no such care -- a random projection never looks at the data.

EXPECT THE GATE-INPUT PROBLEM TO RECUR. The randmlp run showed setting 4
degrading in 11/12 cells while setting 3, which mixes but does not READ the
embedding, improved in 8/12 -- the gate struggles to learn from 128
dimensions with no semantic structure. PCA axes are not aligned-space
coordinates either, so pair this with --no-emb-in-gate to separate "a
stronger sEHR side helps the fusion" from "the gate cannot use the
embedding as an input".

Why the option exists: v_ehr_post scores BELOW v_ehr_pre on HF and eMACE, so
the fusion is built on an sEHR vector weaker than the one available. randmlp
holds the bottleneck and the normalisation fixed while removing only the
alignment TRAINING, which both measures what that training contributed and
supplies a mixer input that does not depend on it.

    adds  0c_EHR_randmlp                     (= the mixer's alpha=0 endpoint)
          1br_EHR_randmlp_plus_gate_scalars  (what setting 4 must beat)

The PPG side is identical in both modes, so the sweep's alpha=1 endpoint must
equal 2_PPG_post in BOTH runs -- checked automatically at the end.

LIMITATION, and it is not a footnote: with a random sEHR projection the two
128-d blocks no longer share a learned coordinate system, so the mixer's
shared-w constraint (train_probe_regularized.py:150-156) loses its
justification. The mixture remains a well-defined restriction of the
concatenated probe, but not a principled one. alpha values from a randmlp run
are NOT comparable to an aligned run's -- the gate's input distribution
changed too.

SETTING 4's REGULARISATION  (--gate-wd-fix)
------------------------------------------
Setting 4 as originally written has three coupled faults, all found while
fixing the same ones in score_fusion.py:

    weight decay hit the gate BIAS, which encodes the baseline alpha, so alpha
    was pulled toward sigmoid(0) = 0.5 -- not a neutral value, just an artifact
    of the parametrisation (the optimum is ~0.25);

    one wd covered the gate AND the 128-d head, so a single knob controlled
    both "how much does alpha vary" and "how much is the classifier shrunk";

    WEIGHT_DECAY_GRID stops at 1e-1, which never flattens alpha (measured: sd
    still 0.054 there), so setting 4 could never degenerate to setting 3 and
    "the learned gate loses to a fixed alpha" was not separable from "the gate
    was never allowed to stop varying". Every AF fold selected the grid edge.

--gate-wd-fix repairs all three: bias excluded, head held at DEFAULT_WD so the
selected wd has ONE meaning, grid widened to 10. OFF by default, so an
unflagged run reproduces the published setting-4 numbers exactly; with it on
the output filename gains _gatefix and nothing is overwritten.

Expect it to close a fairness gap, not to win: setting 4's alpha=0 endpoint is
1_EHR_post, which sits 0.024-0.083 below 0a_EHR_pre on HF and eMACE, and no
amount of regularisation fixes a representation deficit.

Usage:
    python gated_fusion.py
    python gated_fusion.py --gate-wd-fix --disease AF
    python gated_fusion.py --no-emb-in-gate
    python gated_fusion.py --emb-only-gate
    python gated_fusion.py --gate-hidden 16
    python gated_fusion.py --ehr-source randmlp --rand-seed 0
"""
import os
os.environ.setdefault('MPLCONFIGDIR', str(paths.data_root() / ".cache/matplotlib"))

import argparse
import inspect
import sys

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import paths

# train_probe_regularized.py sits beside this file in the published
# layout, so the _HERE insert above already resolves it.
import train_probe_regularized as reg

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

FEATURES_PATH = (
    str(paths.data_root() / "ds1/downstream_features_admission_postdisch_clean.joblib")
)
METADATA_CSV = (
    str(paths.data_root() / "ds1/admission_metadata_postdisch_sqi_clean.csv")
)
MERGE_KEY = ['dataset', 'patient_id', 'hadm_id']

DISEASES = ('AF', 'HF', 'eMACE')
HORIZONS = (30, 90, 180, 365)
HORIZON_DAYS = {30: 30, 90: 90, 180: 180, 365: 365}

# Same four advisor scalar categories as the previous script, with the new
# admission-level definitions.
#
# SQI     p10 captures the low-quality tail among the 10-s windows that
#         actually contributed to the admission-level PPG embedding.
# CODE    n_codes_fed is the number of non-PAD code tokens actually fed to the
#         sEHR encoder after the 512-token cap.
# TIMING  gap_median_from_admit_h is the median temporal location, relative to
#         admission, of the selected PPG blocks that were pooled into the
#         admission representation. This keeps the advisor's original
#         "PPG-to-admission timing" category while better matching a pooled PPG
#         representation than the first block alone.
SQI_FIELDS = ['acf_p10', 'skew_p10']
CODE_FIELDS = ['n_codes_fed']
TIMING_FIELDS = ['gap_median_from_admit_h']
SCALAR_FIELDS = SQI_FIELDS + CODE_FIELDS + TIMING_FIELDS
LOG_FIELDS = {'n_codes_fed', 'gap_median_from_admit_h'}

N_EPOCHS = 300
PATIENCE = 30
MIN_POS_FOR_EARLYSTOP = 4

# Regularisation and step size are SELECTED on the same inner split that
# early-stops the run, not fixed. Setting 3's C and alpha go through the inner
# CV, so leaving setting 4's weight decay hard-coded would confound "the
# learned gate is worse" with "the learned gate was never tuned".
#
# 4 x 2 = 8 candidates, each a full training run. The grids stay coarse
# because the validation AUROC that ranks them comes from a split holding a
# handful of positive patients; a finer grid would not locate the optimum, it
# would only give argmax more equally-noisy candidates to pick between.
WEIGHT_DECAY_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
LR_GRID = [3e-3, 1e-2]

# --gate-wd-fix only. The default grid stops at 1e-1 and every AF fold picked
# that edge, so setting 4 could never degenerate to setting 3 and "the learned
# gate loses to a fixed alpha" was not separable from "the gate was never
# allowed to stop varying".
#
# HOW FAR THE GRID HAS TO REACH -- measured on THIS gate, not borrowed.
# An earlier version of this comment justified a top of 10 with numbers taken
# from score_fusion.py's gate (4 inputs, lr=0.05, zero-init). That transfer was
# wrong: this gate has 261 inputs, lr from [3e-3, 1e-2] and default random
# init, and all three differences push the same way. Re-measured here at
# lr=3e-3, 300 epochs:
#
#     wd     alpha sd   ||w_gate||
#     10       0.265       1.45      <- the old top; alpha still varies fully
#     30       0.150       0.63
#     100      0.049       0.19
#     300      0.016       0.063     <- effectively constant
#     1000      NaN         NaN      <- diverges
#
# Adam's update size is gradient-magnitude independent, so decay and gradient
# balance at ||w|| ~ sqrt(n_gate_inputs) / wd. With 261 inputs that is ~16/wd,
# which the table matches -- and it is why a value tuned on a 4-input gate is
# meaningless here.
#
# STABILITY CEILING: decoupled decay is p <- p * (1 - lr*wd), so |1-lr*wd| > 1
# once lr*wd > 2 and the weights blow up. That is exactly the wd=1000 NaN
# (lr*wd = 3). The grid must therefore satisfy lr*wd <= 2 for EVERY lr in
# LR_GRID; with max lr 1e-2 that caps wd at 200, and 100 is the largest round
# value inside it. At lr=1e-2, wd=100 gives lr*wd = 1, i.e. the weights are
# zeroed each step and alpha is genuinely constant -- so the pair that reaches
# the degenerate endpoint exists in the grid even though wd=100 alone does not
# flatten it at lr=3e-3.
#
# Zero-initialising the gate does NOT substitute for this: measured at wd=10
# with zero init, alpha sd is still 0.272, because the gradient pushes the
# weights up regardless of where they start.
# 30 is not decorative. The measured transition from "alpha varies fully" to
# "alpha is constant" happens ENTIRELY between 10 and 100, and a decade grid
# puts nothing inside it -- forcing the selection into a binary choice when the
# useful answer may be an intermediate amount of adaptivity. 300 is left out:
# at lr=1e-2 it exceeds the stability ceiling and is skipped, and at lr=3e-3 it
# reproduces what (100, 1e-2) already gives.
WEIGHT_DECAY_GRID_WIDE = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 30.0, 100.0]

# lr*wd beyond this makes AdamW's decoupled decay flip sign and amplify.
MAX_LR_TIMES_WD = 2.0

# Used when a fold cannot support a validation split: without one there is
# nothing to early-stop on AND nothing to select on, so the fixed epoch budget
# runs at these values. Counted and reported separately.
DEFAULT_WD = 1e-4
DEFAULT_LR = 3e-3


def label_col(disease, horizon):
    return f'{disease}__{horizon}'


def GATE_INPUT_NAME(args):
    if args.no_emb_in_gate:
        return 'scalars+horizon only'
    if args.emb_only_gate:
        return 'embeddings+horizon (no scalars)'
    return 'embeddings+scalars+horizon'


def GATE_INPUT_TAG(args):
    if args.no_emb_in_gate:
        return '_noemb'
    if args.emb_only_gate:
        return '_embonly'
    return ''

class ConvexMixer(BaseEstimator, TransformerMixin):
    """fused = (1 - alpha) * v_ehr + alpha * v_ppg.

    Input X is [v_ehr | v_ppg]; alpha IS the PPG weight, so alpha=0 reproduces
    the EHR_post probe exactly and alpha=1 the PPG_post probe exactly. Placed
    AFTER the scaler so both blocks are standardised first -- the mixture is
    then a restriction of the concatenated probe with the two coefficient
    vectors constrained parallel.

    reg.FusionMixer computes (ve + a*vp)/(1+a) instead, where a=2 means a PPG
    weight of 2/3, so its grid never reaches pure PPG and its alpha is not
    comparable to the one swept here.
    """
    def __init__(self, alpha=0.0, dim=128):
        self.alpha = alpha
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ve, vp = X[:, :self.dim], X[:, self.dim:]
        return (1.0 - self.alpha) * ve + self.alpha * vp


def build_pca_estimator(n_components=128, whiten=False):
    """StandardScaler -> PCA(128) -> the SAME L2 logistic every other row uses.

    WHAT THIS ISOLATES
    ------------------
    v_ehr_post (128, aligned) sits 0.024-0.083 below v_ehr_pre (320) on HF and
    eMACE. Two candidate causes were confounded: 128 dimensions may not hold
    the information, or the alignment may have kept the wrong 128. This splits
    them, because PCA is the best LINEAR 128-d compression of the same input:

        0e ~ 0a   128 dimensions suffice, so the aligned projection lost the
                  useful directions rather than the budget -- and a better
                  objective could recover them.
        0e << 0a  the budget itself is binding; no 128-d projection will do,
                  and fusion should use the 320-d vector (which is what
                  score_fusion.py already does).

    WHY PCA RATHER THAN A TASK-TRAINED PROJECTION
    ---------------------------------------------
    A supervised 320->256->128 mirror of mlp_s is ~115k parameters against
    1480 rows and 52-155 positives, so losing to 0a would be uninterpretable --
    overfitting and an insufficient budget look identical. PCA is unsupervised,
    has no such capacity, and never touches the label, so a deficit here can
    only be the dimension. It answers the question that was actually asked.

    NESTING, AND WHICH DIRECTION IT RUNS
    ------------------------------------
    Any linear function of the top-128 PCs is a linear function of the 320
    inputs, so 0e's hypothesis class is CONTAINED in 0a's -- the opposite
    direction from the concat rows, where the new model is the larger one. 0e
    therefore cannot beat 0a on the training objective, and 0a - 0e on the
    outer folds is what the 192 discarded components were worth.

    THE READING IS ASYMMETRIC -- this matters
    -----------------------------------------
    PCA maximises VARIANCE, not label-relevance. It is the best linear 128-d
    compression in the reconstruction sense, not the discriminative one. So:

        0e ~ 0a   CONCLUSIVE. Even a label-blind compression keeps what the
                  probe needs, so 128 dimensions are not the constraint and
                  the aligned projection lost the useful directions.
        0e << 0a  AMBIGUOUS. Either 128 dimensions really are too few, OR PCA
                  kept high-variance directions that happen not to carry the
                  label. A supervised 320->128 projection could still beat it.

    Verified on synthetic data where the label depended on 5 of 320 iid
    Gaussian dimensions: PCA-128 scored 0.664 against 0.727 for the full 320,
    purely because those 5 dimensions were not the high-variance ones. Real
    BERT embeddings are unlikely to be that adversarial -- the signal is
    probably spread over high-variance directions -- but the possibility is
    why a deficit here does NOT close the question on its own.

    No leakage: PCA sits inside the Pipeline, so it is fitted on the training
    part of each fold only. svd_solver='full' for determinism -- the default
    would pick the randomized solver at this shape.
    """
    return Pipeline([
        ('sc', StandardScaler()),
        ('pca', PCA(n_components=n_components, whiten=whiten,
                    svd_solver='full')),
        ('lr', LogisticRegression(penalty='l2', solver='lbfgs',
                                  class_weight='balanced', max_iter=5000)),
    ])


def build_convex_estimator(dim):
    return Pipeline([
        ('sc', StandardScaler()),
        ('mix', ConvexMixer(dim=dim)),
        ('lr', LogisticRegression(penalty='l2', solver='lbfgs',
                                  class_weight='balanced', max_iter=5000)),
    ])


CONVEX_ALPHA_GRID = [float(round(a, 2)) for a in np.linspace(0.0, 1.0, 11)]
SWEEP_ALPHA_GRID = [float(round(a, 2)) for a in np.linspace(0.0, 1.0, 21)]


# ══════════════════════════════════════════════════════════════════════════
# Randomly-initialised sEHR projection  (--ehr-source randmlp)
# ══════════════════════════════════════════════════════════════════════════
def random_project_ehr(VE_pre, seed):
    """v_ehr_pre (320) -> L2(mlp_s_random(v_ehr_pre)) (128).

    WHAT THIS IS FOR
    ----------------
    v_ehr_post scores BELOW v_ehr_pre on HF and eMACE, so the fusion's sEHR
    side is weaker than the sEHR representation available. Three things change
    between pre and post -- the 320->128 bottleneck, the row-wise L2, and the
    alignment training -- and the probe cannot separate them. Replacing only
    the TRAINING with a random draw holds the other two fixed, so this variant
    both (a) measures what alignment training contributed to the sEHR side and
    (b) supplies a 128-d sEHR vector for the mixer that does not depend on it.

    Architecture is copied bit-for-bit from the mlp_s that produced
    v_ehr_post_mlp (identify/extract_feature.py:186-193), then L2-normalised
    exactly as extract_feature.py:437 does. Any deviation would reintroduce the
    dimensional or normalisation confound this variant exists to remove.

    IMPLEMENTATION NOTES, each of which is load-bearing:

    eval()          Dropout(0.1) would otherwise zero 10% of the hidden units
                    at random on every call. That is noise, not a fixed random
                    projection; in eval mode Dropout is the identity.

    fork_rng        seeding the GLOBAL torch RNG here would shift the stream
                    that run_setting4 later draws its gate initialisation from,
                    so every setting-4 number would move for a reason having
                    nothing to do with the sEHR side. Forking keeps the global
                    stream untouched, and the baseline run reproduces bit-for-
                    bit.

    default init    nn.Linear's Kaiming-uniform default is exactly the
                    distribution the trained mlp_s started from, so this is
                    that same network before it saw data. Hand-rolled init
                    would answer a different question.

    CAVEAT for the writeup: with a random sEHR projection the two 128-d blocks
    no longer share a learned coordinate system, so the mixer's shared-w
    constraint (train_probe_regularized.py:150-156) loses its justification --
    coordinate j of the two vectors are unrelated. The mixture is still a
    well-defined restriction of the concatenated probe, but it is no longer a
    principled one. This belongs in the limitations, not in a footnote.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        mlp = nn.Sequential(
            nn.Linear(320, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
        )
    mlp.eval()
    with torch.no_grad():
        z = mlp(torch.from_numpy(VE_pre).float())
        z = torch.nn.functional.normalize(z, dim=-1)
    return z.numpy().astype(np.float64)


def EHR_SOURCE_TAG(args):
    if args.ehr_source == 'randmlp':
        return f'_randehr_s{args.rand_seed}'
    if args.ehr_source == 'pca':
        return '_pcaehr'
    return ''


def GATE_WD_FIX_TAG(args):
    return '_gatefix' if args.gate_wd_fix else ''


def DISEASE_TAG(args):
    """Keeps parallel per-disease jobs from overwriting each other's CSV."""
    return '' if args.disease == 'all' else f'_{args.disease}'

# ══════════════════════════════════════════════════════════════════════════
def load_joined(features_path, metadata_csv):
    """Load the clean Stage-4 features and metadata, joined one-to-one by stay."""
    feat = joblib.load(features_path)

    rows = []
    for d in feat:
        r = {
            'dataset': str(d['dataset']),
            'patient_id': str(d['patient_id']),
            'hadm_id': str(d['hadm_id']),
            'admit_time': pd.Timestamp(d['admit_time']),
            'dischtime': pd.Timestamp(d['dischtime']),
            'ppg_mask': d['ppg_mask'],
            'ehr_mask': d['ehr_mask'],
        }
        for disease in DISEASES:
            for horizon in HORIZONS:
                c = label_col(disease, horizon)
                r[c] = d.get(c, np.nan)
        rows.append(r)

    fdf = pd.DataFrame(rows)
    fdf['_idx'] = np.arange(len(fdf))

    mdf = pd.read_csv(
        metadata_csv,
        dtype={'dataset': str, 'patient_id': str, 'hadm_id': str},
    )

    dup_l = int(fdf.duplicated(MERGE_KEY).sum())
    dup_r = int(mdf.duplicated(MERGE_KEY).sum())
    if dup_l or dup_r:
        raise ValueError(
            f"duplicate merge keys: features {dup_l}, metadata {dup_r}. "
            f"The join would not be one-to-one."
        )

    merged = fdf.merge(
        mdf,
        on=MERGE_KEY,
        how='left',
        suffixes=('', '_meta'),
        validate='one_to_one',
        indicator=True,
    )
    n_missing = int((merged['_merge'] != 'both').sum())
    if n_missing:
        raise ValueError(
            f"{n_missing} feature admissions have no matching metadata row; "
            f"the clean files are not aligned."
        )
    merged = merged.drop(columns=['_merge'])

    print(
        f"  joined {len(merged)} rows "
        f"({len(fdf)} features, {len(mdf)} metadata)"
    )
    return feat, merged


def build_scalar_matrix(sub):
    """Gate scalar block: log1p for right-skewed count/timing fields; no fill."""
    cols = []
    for f in SCALAR_FIELDS:
        x = pd.to_numeric(sub[f], errors='coerce').to_numpy(dtype=np.float64)
        n_bad = int((~np.isfinite(x)).sum())
        if n_bad:
            raise ValueError(
                f"{f}: {n_bad} missing/non-finite values on the both-modality "
                f"cohort. ppg_mask==1 should imply SQI/timing exist and "
                f"ehr_mask==1 should imply n_codes_fed exists. Investigate; "
                f"do not impute."
            )
        if f in LOG_FIELDS:
            shift = min(0.0, float(x.min()))
            x = np.log1p(x - shift)
        cols.append(x)
    return np.stack(cols, axis=1)


def assign_folds(both, disease):
    """One fold per admission, patient-grouped, 365-day-status stratified.

    Stratification uses the disease-specific 365-day STATUS, not just positives:
        0 = usable negative
        1 = positive
        2 = censored (NaN)

    This keeps NaN distinct from a negative while retaining the old design of
    assigning folds once at admission level before long-format expansion.
    """
    groups = both['patient_id'].astype(str).to_numpy()
    s365 = pd.to_numeric(both[label_col(disease, 365)], errors='coerce')

    y_strat = np.full(len(both), 2, dtype=np.int64)  # censored
    y_strat[s365.eq(0).to_numpy()] = 0
    y_strat[s365.eq(1).to_numpy()] = 1

    n0 = int((y_strat == 0).sum())
    n1 = int((y_strat == 1).sum())
    n2 = int((y_strat == 2).sum())
    print(
        f"  {disease}: 365d stratification status "
        f"negative={n0}, positive={n1}, censored={n2}"
    )

    fold_of = np.full(len(both), -1, dtype=int)
    skf = StratifiedGroupKFold(
        n_splits=reg.N_FOLDS_OUTER,
        shuffle=True,
        random_state=reg.RANDOM_SEED,
    )
    for k, (_tr, te) in enumerate(
        skf.split(np.zeros((len(both), 1)), y_strat, groups=groups)
    ):
        fold_of[te] = k

    assert (fold_of >= 0).all(), "some admission got no fold"

    chk = (
        pd.DataFrame({'g': groups, 'f': fold_of})
        .groupby('g')['f']
        .nunique()
    )
    assert int((chk > 1).sum()) == 0, "patient spans multiple folds"
    return fold_of


# ══════════════════════════════════════════════════════════════════════════
class GatedFusion(nn.Module):
    """alpha_i = sigmoid(g(x_i)); shared head on the convex representation."""

    def __init__(self, dim, gate_in_dim, hidden=0):
        super().__init__()
        if hidden and hidden > 0:
            self.gate = nn.Sequential(
                nn.Linear(gate_in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.gate = nn.Linear(gate_in_dim, 1)
        self.head = nn.Linear(dim, 1)

    def forward(self, z_ehr, z_ppg, x_gate):
        alpha = torch.sigmoid(self.gate(x_gate))
        fused = alpha * z_ppg + (1.0 - alpha) * z_ehr
        return self.head(fused).squeeze(-1), alpha.squeeze(-1)


def _split_param_groups(model, wd, gate_wd_fix, wd_head=None):
    """Optimizer param groups for one (wd) candidate.

    Default (gate_wd_fix=False) reproduces the original exactly: a single
    weight decay over every parameter.

    With the fix, three groups:

      gate weights  wd          the quantity actually being selected --
                                how much alpha varies with its inputs
      gate biases   0.0         the bias sets the BASELINE alpha; decaying it
                                pulls alpha toward sigmoid(0) = 0.5, which is
                                not a neutral value but an artifact of the
                                parametrisation. The optimum here is ~0.25.
      head          DEFAULT_WD  held FIXED rather than selected, so the
                                selected wd has one meaning instead of two.

    The head must keep a real penalty for the degenerate endpoint to be
    meaningful: at the top of the wide grid the gate weights vanish and alpha
    becomes constant, which should reproduce setting 3 -- a constant-alpha
    mixture with a working head. Letting the selected wd hit the head too
    would crush the head at the same time and leave no model at all.
    """
    if not gate_wd_fix:
        return [{'params': list(model.parameters()), 'weight_decay': wd}]
    gw = [p for n, p in model.gate.named_parameters() if not n.endswith('bias')]
    gb = [p for n, p in model.gate.named_parameters() if n.endswith('bias')]
    if wd_head is None:
        # Unchanged path: setting 4 pins the head at DEFAULT_WD. Note that
        # this decays the head BIAS too, which at 1e-4 is immaterial but is
        # wrong in principle -- sklearn's LogisticRegression never penalises
        # the intercept. It is left alone here so setting 4 reproduces
        # bit-for-bit; alpha_fusion.py passes wd_head and takes the split
        # below.
        return [
            {'params': gw, 'weight_decay': wd},
            {'params': gb, 'weight_decay': 0.0},
            {'params': list(model.head.parameters()),
             'weight_decay': DEFAULT_WD},
        ]
    # wd_head given: the head's decay is a SELECTED hyperparameter, so its
    # bias has to come out of the decay for the same reason the gate's did.
    # DEFAULT_WD = 1e-4 on a 128-input head is not a light penalty, it is no
    # penalty at all: decoupled decay shrinks by (1 - lr*wd) per step, so at
    # lr=3e-3 the entire 300-step run removes 1e-4 of the weight norm. The
    # sklearn head that beats it by 0.031 selects C near 1e-5..1e-4, i.e. a
    # penalty coefficient 1/(2C) of 5000-50000. Pinning it was the bug.
    hw = [p for n, p in model.head.named_parameters() if not n.endswith('bias')]
    hb = [p for n, p in model.head.named_parameters() if n.endswith('bias')]
    return [
        {'params': gw, 'weight_decay': wd},
        {'params': gb, 'weight_decay': 0.0},
        {'params': hw, 'weight_decay': wd_head},
        {'params': hb, 'weight_decay': 0.0},
    ]


def _train_once(ze_t, zp_t, xg_t, y_t, y, h_idx, tr_i, va_i, scorable,
                dim, gate_dim, hidden, seed, wd, lr, gate_wd_fix=False,
                wd_head=None, n_epochs=None, return_epoch=False,
                eval_every=1, make_model=None, make_groups=None):
    """One training run at fixed (wd, lr). Returns (model, best_val_auroc).

    best_val_auroc is None when there is no validation split, in which case the
    fixed epoch budget runs and the final model is returned. That value is what
    ranks candidates, so a fold returning None cannot select and the caller
    falls back to the defaults.

    The seed is applied here rather than by the caller so every candidate starts
    from the same initialisation and differs only by the hyperparameter.
    """
    # make_model / make_groups let a caller substitute a different module and
    # its own parameter-group split while reusing this loop -- the early
    # stopping, the eval_every schedule, the epoch accounting. Duplicating the
    # loop for every model variant is how two implementations drift apart.
    # Both default to None, so every existing call is unchanged.
    torch.manual_seed(seed)
    model = GatedFusion(dim, gate_dim, hidden) if make_model is None \
        else make_model()
    groups = (_split_param_groups(model, wd, gate_wd_fix, wd_head)
              if make_groups is None else make_groups(model, wd, wd_head))
    opt = torch.optim.AdamW(groups, lr=lr)

    n_p = max(float(y[tr_i].sum()), 1.0)
    n_n = float(len(tr_i)) - n_p
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(n_n / n_p, dtype=torch.float32) # positive sample BCE loss * (n_n/n_p)
    )

    # n_epochs lets a caller refit for an EXACT step count -- the one chosen
    # on the inner folds -- instead of the module budget. Default None keeps
    # the module budget, so every existing call is unchanged.
    best_va, best_state, bad = -np.inf, None, 0
    best_ep = 0
    # eval_every>1 scores the validation split only every k-th step. With
    # FULL-BATCH updates consecutive steps barely differ, and measured on this
    # model the per-step validation is 93% of the wall clock (the forward pass
    # plus the state_dict clone, not roc_auc_score, which is 5%). `bad` is
    # incremented BY eval_every so PATIENCE keeps meaning "epochs without
    # improvement" rather than silently becoming k times longer. Default 1
    # reproduces the original loop exactly.
    budget = N_EPOCHS if n_epochs is None else int(n_epochs)
    for _ep in range(budget):
        model.train()
        opt.zero_grad()
        logit, _ = model(ze_t[tr_i], zp_t[tr_i], xg_t[tr_i])
        lossf(logit, y_t[tr_i]).backward()
        opt.step()

        if va_i is None:
            continue
        if eval_every > 1 and (_ep + 1) % eval_every and _ep + 1 < budget:
            continue
        model.eval()
        with torch.no_grad():
            lv, _ = model(ze_t[va_i], zp_t[va_i], xg_t[va_i])
        lv = lv.numpy()
        yv, hv = y[va_i], h_idx[va_i]

        aucs = []
        for hi in scorable:
            m = hv == hi
            if len(np.unique(yv[m])) >= 2:
                aucs.append(roc_auc_score(yv[m], lv[m]))
        if not aucs:
            continue
        va = float(np.mean(aucs))

        if va > best_va + 1e-5:
            best_va, bad, best_ep = va, 0, _ep + 1
            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }
        else:
            bad += eval_every
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    va_out = best_va if np.isfinite(best_va) else None
    if return_epoch:
        return model, va_out, (best_ep if best_ep else budget)
    return model, va_out


def fit_gate(ze, zp, xg, y, groups, h_idx, dim, gate_dim, hidden, seed,
             gate_wd_fix=False):
    """Joint fit of g and head, with (weight_decay, lr) chosen on the inner split.

    Returns (model, used_early_stopping, chosen_wd, chosen_lr).

    ONE INNER SPLIT, TWO JOBS
    -------------------------
    The split is made once and every candidate trains against it, so the same
    held-out patients both stop each run and rank the runs against each other --
    the structure GridSearchCV uses for (C, alpha) in the sklearn settings.
    Selecting on the outer test fold would be the same error as reading a swept
    alpha off the test set.

    Input rows are already restricted to NON-CENSORED disease x horizon rows.
    Positives are counted as distinct patients, not long-format rows, because
    the four horizon copies of one admission are not four independent cases.
    Validation AUROC is computed per horizon and averaged: a pooled AUROC would
    rank a 365-day positive against a 30-day negative and be dominated by
    whichever horizon has the highest event rate.

    When no usable split exists there is no validation signal, so nothing can be
    selected AND nothing can be stopped: the fixed epoch budget runs at
    DEFAULT_WD / DEFAULT_LR and the fold is counted. Stopping or selecting on
    the training rows would pick the most overfitted candidate.
    """
    pos_patients = np.unique(groups[y == 1])
    n_pos_patients = len(pos_patients)
    use_es = n_pos_patients >= MIN_POS_FOR_EARLYSTOP

    if use_es:
        ever_pos = np.isin(groups, pos_patients).astype(int)
        n_inner = min(4, n_pos_patients)
        splitter = StratifiedGroupKFold(
            n_splits=n_inner,
            shuffle=True,
            random_state=seed,
        )
        tr_i, va_i = next(
            iter(
                splitter.split(
                    np.zeros((len(y), 1)),
                    ever_pos,
                    groups=groups,
                )
            )
        )
        scorable = [
            hi for hi in range(len(HORIZONS))
            if len(np.unique(y[va_i][h_idx[va_i] == hi])) >= 2
        ]
        if not scorable:
            use_es = False
    if not use_es:
        tr_i, va_i, scorable = np.arange(len(y)), None, []

    ze_t = torch.tensor(ze, dtype=torch.float32)
    zp_t = torch.tensor(zp, dtype=torch.float32)
    xg_t = torch.tensor(xg, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    if not use_es:
        model, _ = _train_once(
            ze_t, zp_t, xg_t, y_t, y, h_idx, tr_i, None, [],
            dim, gate_dim, hidden, seed, DEFAULT_WD, DEFAULT_LR,
            gate_wd_fix,
        )
        return model, False, DEFAULT_WD, DEFAULT_LR

    grid = WEIGHT_DECAY_GRID_WIDE if gate_wd_fix else WEIGHT_DECAY_GRID
    best = (None, -np.inf, DEFAULT_WD, DEFAULT_LR)
    for wd in grid:
        for lr in LR_GRID:
            # Skip combinations where the decoupled decay would amplify rather
            # than shrink (see MAX_LR_TIMES_WD). Silently training a diverging
            # candidate would hand argmax a NaN validation score at best and a
            # garbage model at worst.
            if lr * wd > MAX_LR_TIMES_WD:
                continue
            m, va = _train_once(
                ze_t, zp_t, xg_t, y_t, y, h_idx, tr_i, va_i, scorable,
                dim, gate_dim, hidden, seed, wd, lr, gate_wd_fix,
            )
            if va is not None and va > best[1]:
                best = (m, va, wd, lr)

    if best[0] is None:
        model, _ = _train_once(
            ze_t, zp_t, xg_t, y_t, y, h_idx, tr_i, None, [],
            dim, gate_dim, hidden, seed, DEFAULT_WD, DEFAULT_LR,
            gate_wd_fix,
        )
        return model, False, DEFAULT_WD, DEFAULT_LR
    return best[0], True, best[2], best[3]


def _pf(vals):
    """Per-fold values as a CSV string, matching run_cv's auroc_per_fold.

    Setting 4 and its diagnostic rows are the ONLY ones this script builds by
    hand, so without this they were the only rows in the table missing the
    per-fold columns -- and they are precisely the rows the --ehr-source modes
    exist to compare. assign_folds depends on the disease and RANDOM_SEED
    alone, never on --ehr-source, so setting 4 in an aligned run and setting 4
    in a pca run are scored on the SAME outer folds and can be differenced
    fold by fold. A mean-only comparison throws that pairing away: the
    fold-to-fold spread is largely shared between the two runs, so the sd of
    the per-fold difference is far smaller than the difference of the sds.
    """
    return ','.join(f'{v:.6f}' for v in vals)


def ranks(s):
    """1 = highest risk."""
    order = np.argsort(-s, kind='mergesort')
    rk = np.empty(len(s), dtype=int)
    rk[order] = np.arange(1, len(s) + 1)
    return rk


# ══════════════════════════════════════════════════════════════════════════
def horizon_valid_folds(both, fold_of, disease):
    """Global fold ids that are trainable and scorable for each horizon.

    NaN labels are excluded only for the corresponding horizon. A fold is valid
    when the NON-CENSORED training and test slices both contain classes 0 and 1.
    """
    valid = {}
    for horizon in HORIZONS:
        s = pd.to_numeric(
            both[label_col(disease, horizon)],
            errors='coerce',
        ).to_numpy(dtype=np.float64)
        usable = np.isfinite(s)
        ok = set()
        for k in range(reg.N_FOLDS_OUTER):
            te = (fold_of == k) & usable
            tr = (fold_of != k) & usable
            if (
                te.any()
                and tr.any()
                and len(np.unique(s[tr])) >= 2
                and len(np.unique(s[te])) >= 2
            ):
                ok.add(k)
        valid[horizon] = ok
    return valid


def subset_folds(fold_of, usable):
    """Convert admission-level global fold ids to indices of a horizon subset."""
    f = fold_of[usable]
    return [
        (np.where(f != k)[0], np.where(f == k)[0])
        for k in range(reg.N_FOLDS_OUTER)
    ]


def run_setting4(
    both,
    VE,
    VP,
    M_scalar,
    fold_of,
    valid_folds,
    disease,
    args,
    VE_pre=None,
):
    """One shared learned-alpha model across four horizons for one disease.

    Censoring is row-level: a NaN at one horizon removes only that long-format
    row from fitting/scoring. The admission remains available at other horizons.
    """
    dim = VE.shape[1]
    hz_vals = np.array([HORIZON_DAYS[h] / 365.0 for h in HORIZONS])
    Y = np.stack(
        [
            pd.to_numeric(
                both[label_col(disease, h)],
                errors='coerce',
            ).to_numpy(dtype=np.float64)
            for h in HORIZONS
        ],
        axis=1,
    )
    groups_adm = both['patient_id'].astype(str).to_numpy()

    def expand(adm_idx):
        a = np.repeat(adm_idx, len(HORIZONS))
        h = np.tile(np.arange(len(HORIZONS)), len(adm_idx))
        return a, h

    per_h = {
        h: {'tr': [], 'te': [], 'ap': [], 'a_mu': [], 'a_sd': []}
        for h in HORIZONS
    }
    # 4b uses the SAME gate and the SAME per-admission alpha as setting 4 and
    # differs in one thing only: the classifier on top of the fused vector is
    # the L2 logistic with C chosen by inner CV that settings 1/2/3 all use,
    # instead of the torch head trained at a fixed wd with early stopping.
    # Setting 4 has come in below setting 3 in every run so far, and run 6383
    # sharpened it -- the alpha sweep at 0.50 scored 0.825 while setting 4, at
    # a learned alpha averaging 0.51, scored 0.743. Same mixing weight, 0.08
    # apart. This row says whether that gap is the head or the per-admission
    # variation in alpha, which no existing row can separate.
    per_h4b = {h: {'te': [], 'ap': [], 'C': []} for h in HORIZONS}
    # Setting 4's OWN endpoints, scored as AUROC. s0/s1 are already computed
    # for the case-level ranks; only the scoring is new, so this costs nothing.
    #
    # These are EXACT decompositions, not approximations. head is w.x + b, so
    #
    #     s_i = w.((1-a_i) ze_i + a_i zp_i) + b = (1-a_i) s0_i + a_i s1_i
    #
    # i.e. setting 4's score is the per-admission convex combination of these
    # two columns. 4e1 > 4 therefore means the sEHR side is actively dragging
    # the mixture below its own PPG half.
    #
    # READ THEM AGAINST EACH OTHER, NOT AGAINST 0e. Both use the SAME w, and
    # that w is fitted to the MIXTURE, not to either block alone -- so 4e0 is
    # a weight vector applied to a block it was not optimised for, while 0e
    # gets a probe fitted to that block with C selected by inner CV. 4e0 sits
    # BELOW 0e by construction, and the gap widens as alpha moves away from 0
    # (at alpha ~ 0.5, w is a compromise between two different blocks). An
    # earlier note here read 4e0 against 0e's 0.81 as if they were
    # commensurable; they are not. What 4e0 supports is the comparison
    # 4e0-in-this-run vs 4e0-in-another-run, and 4e0 vs 4e1 within a run.
    per_hE = {h: {'a0': [], 'a1': []} for h in HORIZONS}
    es_flags, cases = [], []
    wd_sel_folds, lr_sel_folds = [], []
    n_params = None

    for fold_id in range(reg.N_FOLDS_OUTER):
        te_adm = np.where(fold_of == fold_id)[0]
        tr_adm = np.where(fold_of != fold_id)[0]
        if len(te_adm) == 0 or len(tr_adm) == 0:
            continue

        # Expand ALL four horizon copies first. This preserves the old scaler
        # behaviour on training covariates. NaN labels are removed only from
        # the rows that enter the loss / metrics below.
        a_tr_all, h_tr_all = expand(tr_adm)
        a_te_all, h_te_all = expand(te_adm)

        y_tr_all = Y[a_tr_all, h_tr_all]
        y_te_all = Y[a_te_all, h_te_all]
        usable_tr = np.isfinite(y_tr_all)
        usable_te = np.isfinite(y_te_all)

        if not usable_tr.any():
            continue

        a_tr = a_tr_all[usable_tr]
        h_tr = h_tr_all[usable_tr]
        y_tr = y_tr_all[usable_tr].astype(np.int64)
        if len(np.unique(y_tr)) < 2:
            continue

        sc_e = StandardScaler()
        sc_p = StandardScaler()
        sc_g = StandardScaler()

        if args.ehr_source == 'pca':
            # Fitted HERE, inside the outer fold, not once over the cohort.
            # PCA reads the feature matrix, so fitting it on all rows would let
            # the test fold shape the principal axes -- unsupervised leakage,
            # but leakage. randmlp needs no such care because a random
            # projection never looks at the data at all.
            #
            # NOT whitened. An earlier version was, to put both blocks at
            # unit variance per column so alpha would be a clean
            # per-coordinate weight. Run 6419-6420 measured what that cost:
            # 0g (whitened PCA through the standard probe) came in BELOW 0e
            # (the same 128 components unwhitened) in 12/12 cells, mean
            # -0.046, worst on AF at -0.085 -- which is exactly where setting
            # 4 collapsed hardest. Whitening rescales every component to unit
            # variance, so the low-variance components (mostly noise) arrive
            # at the same scale as the high-variance ones carrying the signal,
            # and the variance ordering that makes PCA work is destroyed.
            #
            # Two synthetic checks failed to reproduce this, once at the
            # default C and once with C selected by CV, so the effect was
            # dismissed twice before the real data settled it. Synthetic
            # spectra were evidently far flatter than these embeddings'.
            #
            # Instead the block is divided by sqrt(mean eigenvalue), a SINGLE
            # scalar. Relative variance WITHIN the block is untouched, and the
            # block's mean column variance becomes 1, matching what
            # StandardScaler leaves on the PPG side -- so alpha is still a
            # meaningful block-level weight without the per-column damage.
            ehr_pipe = Pipeline([
                ('sc', StandardScaler()),
                ('pca', PCA(n_components=VE.shape[1], svd_solver='full')),
            ])
            ze_tr_all = ehr_pipe.fit_transform(VE_pre[a_tr_all])
            ze_te_all = ehr_pipe.transform(VE_pre[a_te_all])
            # sqrt of the MEAN retained eigenvalue, taken from the TRAIN fit
            # and applied to both sides. One scalar, so the relative variance
            # between components is untouched; the block's mean column
            # variance becomes 1, matching the PPG side's StandardScaler.
            blk = float(np.sqrt(np.mean(ehr_pipe['pca'].explained_variance_)))
            if blk > 0:
                ze_tr_all = ze_tr_all / blk
                ze_te_all = ze_te_all / blk
        else:
            ze_tr_all = sc_e.fit_transform(VE[a_tr_all])
            ze_te_all = sc_e.transform(VE[a_te_all])
        zp_tr_all = sc_p.fit_transform(VP[a_tr_all])
        zp_te_all = sc_p.transform(VP[a_te_all])

        # What the GATE reads as its sEHR block must be what the MIXER fuses.
        # In aligned/randmlp modes VE already IS the fused vector, so the gate
        # keeps reading it raw and those runs stay bit-identical. In pca mode
        # the fused vector is built per fold from VE_pre, so the gate has to be
        # pointed at that instead -- otherwise it would be deciding how much to
        # trust a vector it never sees.
        if args.ehr_source == 'pca':
            GE_tr, GE_te = ze_tr_all, ze_te_all
        else:
            GE_tr, GE_te = VE[a_tr_all], VE[a_te_all]

        if args.no_emb_in_gate:
            G_tr_all = np.hstack([
                M_scalar[a_tr_all],
                hz_vals[h_tr_all][:, None],
            ])
            G_te_all = np.hstack([
                M_scalar[a_te_all],
                hz_vals[h_te_all][:, None],
            ])
        elif args.emb_only_gate:
            G_tr_all = np.hstack([
                GE_tr,
                VP[a_tr_all],
                hz_vals[h_tr_all][:, None],
            ])
            G_te_all = np.hstack([
                GE_te,
                VP[a_te_all],
                hz_vals[h_te_all][:, None],
            ])
        else:
            G_tr_all = np.hstack([
                GE_tr,
                VP[a_tr_all],
                M_scalar[a_tr_all],
                hz_vals[h_tr_all][:, None],
            ])
            G_te_all = np.hstack([
                GE_te,
                VP[a_te_all],
                M_scalar[a_te_all],
                hz_vals[h_te_all][:, None],
            ])

        xg_tr_all = sc_g.fit_transform(G_tr_all)
        xg_te_all = sc_g.transform(G_te_all)

        ze_tr = ze_tr_all[usable_tr]
        zp_tr = zp_tr_all[usable_tr]
        xg_tr = xg_tr_all[usable_tr]

        model, used_es, wd_sel, lr_sel = fit_gate(
            ze_tr,
            zp_tr,
            xg_tr,
            y_tr,
            groups_adm[a_tr],
            h_tr,
            dim,
            xg_tr.shape[1],
            args.gate_hidden,
            reg.RANDOM_SEED + fold_id,
            args.gate_wd_fix,
        )
        es_flags.append(used_es)
        wd_sel_folds.append(wd_sel)
        lr_sel_folds.append(lr_sel)
        if n_params is None:
            n_params = sum(p.numel() for p in model.parameters())

        with torch.no_grad():
            s_tr, alpha_tr = model(
                torch.tensor(ze_tr, dtype=torch.float32),
                torch.tensor(zp_tr, dtype=torch.float32),
                torch.tensor(xg_tr, dtype=torch.float32),
            )
            s_te_all, alpha_te_all = model(
                torch.tensor(ze_te_all, dtype=torch.float32),
                torch.tensor(zp_te_all, dtype=torch.float32),
                torch.tensor(xg_te_all, dtype=torch.float32),
            )
            s0_te_all = model.head(
                torch.tensor(ze_te_all, dtype=torch.float32)
            ).squeeze(-1)
            s1_te_all = model.head(
                torch.tensor(zp_te_all, dtype=torch.float32)
            ).squeeze(-1)

        s_tr = s_tr.numpy()
        alpha_tr = alpha_tr.numpy()
        s_te_all = s_te_all.numpy()
        alpha_te_all = alpha_te_all.numpy()
        s0_te_all = s0_te_all.numpy()
        s1_te_all = s1_te_all.numpy()

        # ── 4b: identical mixture, different classifier ──
        f_tr = (alpha_tr[:, None] * zp_tr
                + (1.0 - alpha_tr[:, None]) * ze_tr)
        f_te = (alpha_te_all[:, None] * zp_te_all
                + (1.0 - alpha_te_all[:, None]) * ze_te_all)
        s4b_te_all = None
        C4b = np.nan
        g_tr_adm = groups_adm[a_tr]
        if int(y_tr.sum()) >= reg.MIN_POS_FOR_INNER_CV:
            try:
                inner = StratifiedGroupKFold(
                    n_splits=reg.N_FOLDS_INNER, shuffle=True,
                    random_state=reg.RANDOM_SEED,
                )
                gs = GridSearchCV(
                    LogisticRegression(penalty='l2', solver='lbfgs',
                                       class_weight='balanced', max_iter=5000),
                    {'C': list(reg.C_GRID)}, scoring='roc_auc', cv=inner,
                    n_jobs=-1, error_score=np.nan, refit=True,
                )
                gs.fit(f_tr, y_tr, groups=g_tr_adm)
                if np.isfinite(gs.best_score_):
                    s4b_te_all = gs.decision_function(f_te)
                    C4b = float(gs.best_params_['C'])
            except (ValueError, AttributeError):
                s4b_te_all = None
        if s4b_te_all is None:
            m = LogisticRegression(penalty='l2', solver='lbfgs',
                                   class_weight='balanced', max_iter=5000,
                                   C=reg.DEFAULT_C).fit(f_tr, y_tr)
            s4b_te_all = m.decision_function(f_te)
            C4b = reg.DEFAULT_C

        for hi, horizon in enumerate(HORIZONS):
            # Censored rows are excluded from fitting and from every metric, so
            # including them here would describe the gate's behaviour on
            # admissions that are not in the analysis at this horizon -- the
            # alpha column would not correspond to the AUROC beside it.
            m_alpha = (h_te_all == hi) & usable_te
            if m_alpha.any():
                per_h[horizon]['a_mu'].append(
                    float(alpha_te_all[m_alpha].mean())
                )
                per_h[horizon]['a_sd'].append(
                    float(alpha_te_all[m_alpha].std())
                )

            # Metrics use only non-censored rows and only folds that the
            # per-horizon baselines can also train and score on.
            if fold_id not in valid_folds[horizon]:
                continue

            m_te = (h_te_all == hi) & usable_te
            m_tr = h_tr == hi
            if not m_te.any():
                continue

            y_te = y_te_all[m_te].astype(np.int64)
            if len(np.unique(y_te)) < 2:
                continue

            sl_s = s_te_all[m_te]
            sl_0 = s0_te_all[m_te]
            sl_1 = s1_te_all[m_te]
            aa = alpha_te_all[m_te]
            adm_here = a_te_all[m_te]

            per_h[horizon]['te'].append(roc_auc_score(y_te, sl_s))
            per_h[horizon]['ap'].append(
                average_precision_score(y_te, sl_s)
            )

            per_hE[horizon]['a0'].append(roc_auc_score(y_te, sl_0))
            per_hE[horizon]['a1'].append(roc_auc_score(y_te, sl_1))

            sl_4b = s4b_te_all[m_te]
            per_h4b[horizon]['te'].append(roc_auc_score(y_te, sl_4b))
            per_h4b[horizon]['ap'].append(
                average_precision_score(y_te, sl_4b)
            )
            per_h4b[horizon]['C'].append(C4b)

            if m_tr.any() and len(np.unique(y_tr[m_tr])) >= 2:
                per_h[horizon]['tr'].append(
                    roc_auc_score(y_tr[m_tr], s_tr[m_tr])
                )

            r_l = ranks(sl_s)
            r_0 = ranks(sl_0)
            r_1 = ranks(sl_1)
            n_slice = int(m_te.sum())
            for j in np.where(y_te == 1)[0]:
                adm_j = adm_here[j]
                cases.append({
                    'disease': disease,
                    'horizon': horizon,
                    'fold': fold_id,
                    'patient_id': groups_adm[adm_j],
                    'dataset': both['dataset'].to_numpy()[adm_j],
                    'hadm_id': both['hadm_id'].astype(str).to_numpy()[adm_j],
                    'admit_time': both['admit_time'].to_numpy()[adm_j],
                    'n_test': n_slice,
                    'alpha_learned': float(aa[j]),
                    'score_a0': float(sl_0[j]),
                    'score_learned': float(sl_s[j]),
                    'score_a1': float(sl_1[j]),
                    'rank_a0': int(r_0[j]),
                    'rank_learned': int(r_l[j]),
                    'rank_a1': int(r_1[j]),
                    'rank_gain_vs_a0': int(r_0[j] - r_l[j]),
                })

    rows = []
    for horizon in HORIZONS:
        d = per_h[horizon]
        s = pd.to_numeric(
            both[label_col(disease, horizon)],
            errors='coerce',
        )
        rows.append({
            'probe': 'Learned_alpha',
            'setting': '4_learned_alpha',
            'disease': disease,
            'horizon': horizon,
            'train_auroc': float(np.mean(d['tr'])) if d['tr'] else np.nan,
            'auroc_mean': float(np.mean(d['te'])) if d['te'] else np.nan,
            'auroc_std_fold': float(np.std(d['te'])) if d['te'] else np.nan,
            'auprc_mean': float(np.mean(d['ap'])) if d['ap'] else np.nan,
            'auprc_std_fold': float(np.std(d['ap'])) if d['ap'] else np.nan,
            'auroc_per_fold': _pf(d['te']),
            'auprc_per_fold': _pf(d['ap']),
            'C_median': np.nan,
            'n_valid_folds': len(d['te']),
            'alpha_mean': float(np.mean(d['a_mu'])) if d['a_mu'] else np.nan,
            'alpha_std_within': (
                float(np.mean(d['a_sd'])) if d['a_sd'] else np.nan
            ),
            'n_alpha_folds': len(d['a_mu']),
            'gate_params': n_params,
            'folds_without_earlystop': int(
                sum(1 for f in es_flags if not f)
            ),
            'wd_per_fold': ','.join(f'{w:g}' for w in wd_sel_folds),
            'lr_per_fold': ','.join(f'{l:g}' for l in lr_sel_folds),
            'n_samples': int(s.notna().sum()),
            'n_pos': int((s == 1).sum()),
        })
        dE = per_hE[horizon]
        for key, nm in [('a0', '4e0_setting4_head_on_sEHR_only'),
                        ('a1', '4e1_setting4_head_on_PPG_only')]:
            rows.append({
                'probe': nm, 'setting': nm, 'disease': disease,
                'horizon': horizon, 'train_auroc': np.nan,
                'auroc_mean': (float(np.mean(dE[key])) if dE[key] else np.nan),
                'auroc_std_fold': (float(np.std(dE[key])) if dE[key]
                                   else np.nan),
                'auroc_per_fold': _pf(dE[key]),
                'auprc_mean': np.nan, 'auprc_std_fold': np.nan,
                'auprc_per_fold': '',
                'C_median': np.nan, 'n_valid_folds': len(dE[key]),
                'n_samples': int(s.notna().sum()),
                'n_pos': int((s == 1).sum()),
            })

        d4 = per_h4b[horizon]
        rows.append({
            'probe': 'Learned_alpha_sklearn_head',
            'setting': '4b_learned_alpha_sklearn_head',
            'disease': disease,
            'horizon': horizon,
            'train_auroc': np.nan,
            'auroc_mean': float(np.mean(d4['te'])) if d4['te'] else np.nan,
            'auroc_std_fold': float(np.std(d4['te'])) if d4['te'] else np.nan,
            'auprc_mean': float(np.mean(d4['ap'])) if d4['ap'] else np.nan,
            'auprc_std_fold': float(np.std(d4['ap'])) if d4['ap'] else np.nan,
            'auroc_per_fold': _pf(d4['te']),
            'auprc_per_fold': _pf(d4['ap']),
            'C_median': (float(np.median(d4['C'])) if d4['C'] else np.nan),
            'n_valid_folds': len(d4['te']),
            'n_samples': int(s.notna().sum()),
            'n_pos': int((s == 1).sum()),
        })
    return rows, cases


# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default=FEATURES_PATH)
    ap.add_argument('--metadata', default=METADATA_CSV)
    ap.add_argument('--out-dir', default=reg.OUT_DIR)
    ap.add_argument('--gate-hidden', type=int, default=0)
    ap.add_argument(
        '--no-emb-in-gate',
        action='store_true',
        help='gate sees only the four scalars + horizon',
    )
    ap.add_argument(
        '--emb-only-gate',
        action='store_true',
        help='gate sees the two embeddings + horizon, no scalars',
    )
    ap.add_argument(
        '--ehr-source',
        choices=['aligned', 'randmlp', 'pca'],
        default='aligned',
        help=(
            'sEHR vector fed to the MIXER and the GATE (settings 3, sweep, 4). '
            '"aligned" = v_ehr_post_mlp, the default and the published '
            'behaviour. "randmlp" = a randomly-initialised mlp_s applied to '
            'v_ehr_pre_mlp. "pca" = PCA(128) of v_ehr_pre_mlp, block-scaled by '
            'sqrt(mean eigenvalue), fitted INSIDE each outer fold; it changes '
            'SETTING 4 ONLY -- see the '
            'module docstring. The PPG side is untouched in every mode, and '
            'the standalone rows 0a/0b/1/2/1b are computed identically, so any '
            'mode reproduces the baseline on them.'
        ),
    )
    ap.add_argument(
        '--rand-seed',
        type=int,
        default=0,
        help='seed for --ehr-source randmlp; appears in the output filename',
    )
    ap.add_argument(
        '--gate-wd-fix',
        action='store_true',
        help=(
            'setting 4 only. Exclude the gate BIAS from weight decay, hold the '
            "head's decay fixed at DEFAULT_WD so the selected wd means only "
            '"how much does alpha vary", and widen the wd grid to reach the '
            'constant-alpha point where setting 4 degenerates to setting 3. '
            'Without all three, "the learned gate loses to a fixed alpha" '
            'cannot be separated from "the gate was never allowed to stop '
            'varying". OFF by default so an unflagged run reproduces the '
            'published setting-4 numbers exactly; the output filename is '
            'suffixed when it is on.'
        ),
    )
    ap.add_argument(
        '--disease',
        choices=list(DISEASES) + ['all'],
        default='all',
        help=(
            'run one disease family instead of all three, so the three can go '
            'out as parallel jobs. Folds are already assigned per disease '
            '(assign_folds is called with it), and nothing is shared across '
            'diseases except the final CSV, so splitting changes NOTHING in '
            'any number -- only which rows land in which file. The output '
            'filename carries the disease so parallel jobs cannot overwrite '
            'each other; concatenate the CSVs afterwards.'
        ),
    )
    args = ap.parse_args()

    if args.no_emb_in_gate and args.emb_only_gate:
        sys.exit(
            '--no-emb-in-gate and --emb-only-gate are mutually exclusive.'
        )

    if 'folds' not in inspect.signature(reg.run_cv).parameters:
        sys.exit(
            "train_probe_regularized.run_cv does not accept a `folds` "
            "argument. The baseline settings must use the same admission-level "
            "outer folds as setting 4."
        )

    print('=' * 112)
    print('GATED FUSION -- clean post-discharge cohort')
    print('=' * 112)

    feat, df = load_joined(args.features, args.metadata)
    both = df[
        (df['ppg_mask'] == 1) & (df['ehr_mask'] == 1)
    ].reset_index(drop=True)
    print(
        f"  both-modality cohort: {len(both)} admissions, "
        f"{both['patient_id'].nunique()} patients"
    )

    idx = both['_idx'].to_numpy()
    VE = np.stack([
        feat[i]['v_ehr_post_mlp'] for i in idx
    ]).astype(np.float64)
    VP = np.stack([
        feat[i]['v_ppg_post_mlp'] for i in idx
    ]).astype(np.float64)

    # Pre-alignment references. 0a vs 1 and 0b vs 2 isolate what the alignment
    # projection did to each modality on its own, which is a separate question
    # from whether fusing them helps. Their dimensionality differs from the
    # 128-d aligned space (BERT hidden 320, PaPaGei 512), so they can only be
    # standalone probes -- they cannot enter the mixer or the gate.
    VE_pre = np.stack([
        feat[i]['v_ehr_pre_mlp'] for i in idx
    ]).astype(np.float64)
    VP_pre = np.stack([
        feat[i]['v_ppg_pre_mlp'] for i in idx
    ]).astype(np.float64)

    M_scalar = build_scalar_matrix(both)
    groups = both['patient_id'].astype(str).to_numpy()
    dim = VE.shape[1]
    print(f"  pre-alignment dims: sEHR {VE_pre.shape[1]}, "
          f"PPG {VP_pre.shape[1]}  (aligned space is {dim})")

    # VE_fus is what the MIXER and the GATE consume. VE stays the aligned
    # vector so setting 1 remains the aligned-sEHR baseline in both modes.
    # The PPG side is never touched: VP feeds the mixer in both modes, which
    # is what makes the sweep's alpha=1 endpoint comparable ACROSS runs.
    if args.ehr_source == 'randmlp':
        VE_fus = random_project_ehr(VE_pre, args.rand_seed)
        print(
            f"  sEHR into mixer/gate: RANDOM mlp_s(seed={args.rand_seed}) "
            f"on v_ehr_pre -> {VE_fus.shape[1]}d, L2-normalised"
        )
        print(
            "  PPG into mixer/gate: v_ppg_post_mlp (unchanged)\n"
            "  NOTE alpha is NOT comparable to an --ehr-source aligned run: "
            "the gate's input distribution changed too."
        )
    elif args.ehr_source == 'pca':
        VE_fus = VE          # settings 3 / sweep deliberately unchanged
        print(
            "  sEHR into SETTING 4 only: PCA(128) of v_ehr_pre, block-scaled, "
            "fitted per outer fold\n"
            "  settings 3 and the alpha sweep keep v_ehr_post_mlp, so the "
            "comparison for setting 4\n"
            "  is against setting 4 in the aligned and randmlp runs -- same "
            "folds, same code, different sEHR side."
        )
    else:
        VE_fus = VE
        print("  sEHR into mixer/gate: v_ehr_post_mlp (aligned)")

    print(f"  gate scalars: {SCALAR_FIELDS}")
    print(f"  log1p fields: {sorted(LOG_FIELDS)}")
    print(
        f"  gate: "
        f"{'linear' if not args.gate_hidden else f'MLP({args.gate_hidden})'}"
        f" | AdamW wd from "
        f"{WEIGHT_DECAY_GRID_WIDE if args.gate_wd_fix else WEIGHT_DECAY_GRID}"
        f", lr from {LR_GRID}"
        f" | inputs: {GATE_INPUT_NAME(args)}\n"
    )

    # Splitting by disease is safe because assign_folds() is called per disease
    # and no state crosses between them; the only thing the three share is the
    # output CSV. Each job therefore produces exactly the rows it would have
    # produced inside a combined run.
    diseases = DISEASES if args.disease == 'all' else (args.disease,)
    if args.disease != 'all':
        print(f"  running ONE disease: {args.disease} "
              f"(of {', '.join(DISEASES)}) -- concatenate the CSVs afterwards")

    all_rows = []
    all_cases = []

    for disease in diseases:
        print('\n' + '#' * 112)
        print(f'DISEASE: {disease}')
        print('#' * 112)

        # One disease-specific fold assignment shared by its four horizons.
        # 365d positives are the positive stratum; 365d usable negatives and
        # censored admissions are separate strata rather than collapsing NaN to 0.
        fold_of = assign_folds(both, disease)
        valid_folds = horizon_valid_folds(both, fold_of, disease)

        for horizon in HORIZONS:
            skipped = sorted(
                set(range(reg.N_FOLDS_OUTER)) - valid_folds[horizon]
            )
            if skipped:
                print(
                    f"  {horizon}d: fold(s) {skipped} unusable after "
                    f"horizon-specific censoring; excluded from ALL settings"
                )

        print(
            '  folds assigned on admissions, grouped by patient, stratified '
            'on 365d status {negative, positive, censored}'
        )

        # ── settings 1, 2, 1b, 3: per horizon on NON-CENSORED rows ──
        for horizon in HORIZONS:
            c = label_col(disease, horizon)
            s = pd.to_numeric(both[c], errors='coerce')
            usable = s.notna().to_numpy()
            y = s[usable].to_numpy(dtype=np.int64)
            groups_h = groups[usable]
            VE_h = VE[usable]
            VP_h = VP[usable]
            VE_fus_h = VE_fus[usable]
            VE_pre_h = VE_pre[usable]
            VP_pre_h = VP_pre[usable]
            M_h = M_scalar[usable]
            folds_h = subset_folds(fold_of, usable)

            n = len(y)
            n_pos = int(y.sum())
            print(
                f"  {horizon}d usable={n}, pos={n_pos}, "
                f"censored={int((~usable).sum())}"
            )

            standalone = [
                (VE_pre_h, 'EHR_pre', '0a_EHR_pre'),
                (VP_pre_h, 'PPG_pre', '0b_PPG_pre'),
                (VE_h, 'EHR_post', '1_EHR_post'),
                (VP_h, 'PPG_post', '2_PPG_post'),
                (
                    np.hstack([VE_h, M_h]),
                    'EHR_post+gate_scalars',
                    '1b_EHR_post_plus_gate_scalars',
                ),
            ]
            if args.ehr_source == 'randmlp':
                # 0c is the alpha=0 endpoint of the new mixer, so it is not
                # optional: without it a fusion number cannot be attributed
                # between the sEHR side and the PPG side. 1br gives the same
                # scalar side information to a plain probe on the random sEHR
                # vector, which is what setting 4 must beat to have shown
                # anything beyond reading its own gate inputs.
                standalone += [
                    (VE_fus_h, 'EHR_randmlp', '0c_EHR_randmlp'),
                    (
                        np.hstack([VE_fus_h, M_h]),
                        'EHR_randmlp+gate_scalars',
                        '1br_EHR_randmlp_plus_gate_scalars',
                    ),
                ]

            # ── 0e: best LINEAR 128-d compression of v_ehr_pre ──
            # Independent of --ehr-source (it is a function of v_ehr_pre only),
            # so it is identical in both modes; computed here as well as in
            # concat_fusion so the randmlp run carries all FOUR 128-d sEHR
            # representations -- aligned, random, PCA, plus the 320-d original
            # -- in one table instead of across two CSVs.
            standalone_pca = (VE_pre_h, 'EHR_pre_pca128', '0e_EHR_pre_pca128')

            for X, tag, setting in standalone:
                r = reg.run_cv(
                    X,
                    y,
                    groups_h,
                    tag,
                    folds=folds_h,
                )
                r.update(
                    setting=setting,
                    disease=disease,
                    horizon=horizon,
                    n_samples=n,
                    n_pos=n_pos,
                )
                all_rows.append(r)

            X0e, tag0e, set0e = standalone_pca
            r0e = reg.run_cv(
                X0e, y, groups_h, tag0e, folds=folds_h,
                make_est=lambda: build_pca_estimator(128),
            )
            r0e.update(setting=set0e, disease=disease, horizon=horizon,
                       n_samples=n, n_pos=n_pos)
            all_rows.append(r0e)

            # ── 0g: 0e's twin, WHITENED. The only difference. ──
            # SETTLED, and kept as the evidence. When --ehr-source pca still
            # whitened, setting 4 scored 0.74 against 0e's 0.808; this row
            # isolated why. 0g came in below 0e in 12/12 cells (mean -0.046,
            # AF 30d -0.085), so whitening alone accounted for the deficit and
            # the pca path dropped it (9e57035).
            #
            # Mechanism: whitening rescales every component to unit variance,
            # so a single L2 penalty no longer protects the high-variance
            # directions the signal lives in. Two synthetic checks missed this
            # -- one at the default C=1 where regularisation barely binds --
            # while the real runs select C near 1e-4, where it binds hard.
            #
            # Retained rather than deleted: it is the control that makes the
            # unwhitened choice defensible, and it costs one probe per cell.
            r0g = reg.run_cv(
                VE_pre_h, y, groups_h, 'EHR_pre_pca128_whitened',
                folds=folds_h,
                make_est=lambda: build_pca_estimator(128, whiten=True),
            )
            r0g.update(setting='0g_EHR_pre_pca128_whitened', disease=disease,
                       horizon=horizon, n_samples=n, n_pos=n_pos)
            all_rows.append(r0g)

            r = reg.run_cv(
                np.hstack([VE_fus_h, VP_h]),
                y, groups_h, 'Fixed_alpha_innerCV',
                fusion=True, dim=dim, folds=folds_h,
                make_est=lambda: build_convex_estimator(dim),
                alpha_grid=CONVEX_ALPHA_GRID,
            )
            r.update(
                setting='3_fixed_alpha_inner_cv',
                disease=disease,
                horizon=horizon,
                n_samples=n,
                n_pos=n_pos,
            )
            all_rows.append(r)

            # ── manual alpha sweep: one full CV per grid point ──
            # Every point here is scored on the OUTER test fold, so the best
            # of them is the maximum of 21 noisy numbers and nobody could have
            # selected it without looking at the test set. It answers "could
            # any fixed mixing weight beat the pre-alignment baselines", which
            # is an upper bound, not a deployable result. Setting 3 above is
            # the deployable version of the same question.
            for a in SWEEP_ALPHA_GRID:
                rs = reg.run_cv(
                    np.hstack([VE_fus_h, VP_h]),
                    y,
                    groups_h,
                    f'Sweep_alpha={a:.2f}',
                    fusion=True,
                    dim=dim,
                    folds=folds_h,
                    make_est=lambda: build_convex_estimator(dim),
                    alpha_grid=[a],          # frozen; C is still selected
                )
                rs.update(
                    setting=f'S_alpha_{a:.2f}',
                    disease=disease,
                    horizon=horizon,
                    n_samples=n,
                    n_pos=n_pos,
                    sweep_alpha=a,
                )
                all_rows.append(rs)

        # ── setting 4: one shared gate across four horizons ──
        rows4, cases4 = run_setting4(
            both,
            VE_fus,
            VP,
            M_scalar,
            fold_of,
            valid_folds,
            disease,
            args,
            VE_pre,
        )
        all_rows.extend(rows4)
        all_cases.extend(cases4)

    res = pd.DataFrame(all_rows)
    res['memorised'] = res['train_auroc'] > 0.995

    order = [
        '0a_EHR_pre',
        '0e_EHR_pre_pca128',
        '0g_EHR_pre_pca128_whitened',
        '0b_PPG_pre',
        '1_EHR_post',
        '2_PPG_post',
        '1b_EHR_post_plus_gate_scalars',
        '3_fixed_alpha_inner_cv',
        '4_learned_alpha',
        '4b_learned_alpha_sklearn_head',
        '4e0_setting4_head_on_sEHR_only',
        '4e1_setting4_head_on_PPG_only',
    ]
    if args.ehr_source == 'randmlp':
        # Printed next to their aligned counterparts so the substitution is
        # visible in the table rather than only in the filename.
        order = order[:5] + [
            '0c_EHR_randmlp',
            '1br_EHR_randmlp_plus_gate_scalars',
        ] + order[5:]

    # ── main result table ──
    for disease in diseases:
        for horizon in HORIZONS:
            sub = res[
                (res.disease == disease) &
                (res.horizon == horizon)
            ]
            if sub.empty:
                continue

            n = int(sub.iloc[0]['n_samples'])
            n_pos = int(sub.iloc[0]['n_pos'])
            print('=' * 112)
            print(f'{disease} {horizon}d   n={n}  n_pos={n_pos}')
            print('=' * 112)
            if args.ehr_source == 'pca':
                # These two rows sit adjacent and invite subtraction, but in
                # this mode they do not share an sEHR side, so 4 - 3 mixes the
                # question being asked with a change of input.
                print('  !! pca mode: rows 3/S_alpha use v_ehr_post; rows 4/4b/'
                      '4e* use PCA(128) of v_ehr_pre.')
                print('     4 vs 3 is NOT a like-for-like comparison here. '
                      'Compare 4 against setting 4 in the aligned run.')
            print(
                f"  {'setting':34s} {'TRAIN':>7s} {'AUROC':>14s} "
                f"{'AUPRC':>14s} {'alpha':>14s} {'folds':>6s}"
            )
            for setting in order:
                rr = sub[sub.setting == setting]
                if rr.empty:
                    continue
                r = rr.iloc[0]
                # Setting 4 reports a learned per-admission alpha (mean+-sd);
                # setting 3 reports the one the inner CV SELECTED, which
                # run_cv returns as alpha_median. Without the second branch
                # the fixed-alpha comparator -- the row this whole analysis
                # turns on -- showed '--' in the alpha column while the value
                # sat in the CSV.
                if np.isfinite(r.get('alpha_mean', np.nan)):
                    a = f"{r['alpha_mean']:.3f}±{r['alpha_std_within']:.3f}"
                elif np.isfinite(r.get('alpha_median', np.nan)):
                    a = f"{r['alpha_median']:.2f} sel"
                else:
                    a = '--'
                flag = '  <-- MEMORISED' if r['memorised'] else ''
                print(
                    f"  {setting:34s} {r['train_auroc']:>7.3f} "
                    f"{r['auroc_mean']:>7.3f}±{r['auroc_std_fold']:.3f} "
                    f"{r['auprc_mean']:>7.3f}±{r['auprc_std_fold']:.3f} "
                    f"{a:>14s} "
                    f"{int(r['n_valid_folds']):>4d}/{reg.N_FOLDS_OUTER}"
                    f"{flag}"
                )
            print()

    # ── comparability check ──
    print('=' * 112)
    print('FOLD-COUNT CHECK  (every setting must agree per disease x horizon)')
    print('=' * 112)
    bad = 0
    for disease in diseases:
        for horizon in HORIZONS:
            sub = res[
                (res.disease == disease) &
                (res.horizon == horizon)
            ]
            # The 21 sweep rows all abbreviate to 'S' and would print 21
            # identical entries, burying the settings anyone reads. They are
            # still CHECKED -- uniq is taken over every row -- just collapsed
            # to one column in the display.
            counts = {
                r['setting']: int(r['n_valid_folds'])
                for _, r in sub.iterrows()
            }
            uniq = set(counts.values())
            shown = {k: v for k, v in counts.items()
                     if not k.startswith('S_alpha_')}
            sweep = {v for k, v in counts.items() if k.startswith('S_alpha_')}
            if sweep:
                shown['Ssweep'] = sorted(sweep)[0] if len(sweep) == 1 else -1
            ok = len(uniq) == 1
            bad += int(not ok)
            print(
                f"  {disease:6s} {horizon:>3d}d  "
                + '  '.join(
                    f"{k.split('_')[0]}={c}" for k, c in shown.items()
                )
                + ('   OK' if ok else '   *** MISMATCH ***')
            )
    if bad:
        print(
            '\n  !! Setting 4 and the per-horizon baselines are not being '
            'scored on identical fold sets somewhere. Fix before interpreting '
            'those comparisons.'
        )
    else:
        print('\n  All settings scored on identical fold counts.')
    print()

    # ── gate parameter / early-stop summary per disease ──
    for disease in diseases:
        r4 = res[
            (res.setting == '4_learned_alpha') &
            (res.disease == disease) &
            (res.horizon == 365)
        ]
        if r4.empty:
            continue
        r4 = r4.iloc[0]
        print(
            f"  {disease}: setting 4 gate+head parameters "
            f"{int(r4['gate_params'])} vs {int(r4['n_pos'])} positives at 365d"
        )
        print(
            f"    alpha statistics pooled over {int(r4['n_alpha_folds'])}/"
            f"{reg.N_FOLDS_OUTER} outer folds"
        )
        print(f"  gate weight_decay per outer fold: {r4.get('wd_per_fold', '')}")
        print(f"  gate lr per outer fold:            {r4.get('lr_per_fold', '')}")
        print("    Selected on the inner split, never on the outer test fold. A")
        print("    grid-edge winner in every fold means the grid should be widened")
        print("    in that direction; a value that jumps between folds means the")
        print("    inner split cannot identify it and the gate's regularisation is")
        print("    effectively arbitrary.")
        if r4['folds_without_earlystop']:
            print(
                f" !! {int(r4['folds_without_earlystop'])} fold(s) had too "
                f"few positive patients for a usable validation split; those "
                f"ran the fixed {N_EPOCHS}-epoch budget with NO early stopping "
                f"and NO hyperparameter selection (wd={DEFAULT_WD:g}, "
                f"lr={DEFAULT_LR:g})."
            )

    # ── write outputs ──
    tag = (GATE_INPUT_TAG(args) + EHR_SOURCE_TAG(args)
           + GATE_WD_FIX_TAG(args) + DISEASE_TAG(args))
    os.makedirs(args.out_dir, exist_ok=True)
    f_res = f'gated_fusion_postdisch_results{tag}.csv'
    f_cas = f'gated_fusion_postdisch_cases{tag}.csv'
    res.to_csv(os.path.join(args.out_dir, f_res), index=False)
    cdf = pd.DataFrame(all_cases)
    cdf.to_csv(os.path.join(args.out_dir, f_cas), index=False)
    print(f'\nSaved {f_res} and {f_cas}\n')

    # ── case-level report ──
    print('=' * 112)
    print('CASE-LEVEL: every positive, alpha=0 -> learned -> alpha=1')
    print('=' * 112)
    print(
        '  Rank is within that disease/horizon slice of its own usable test '
        'fold; 1 = highest risk.\n'
    )

    for disease in diseases:
        for horizon in HORIZONS:
            if len(cdf) == 0:
                continue
            sub = cdf[
                (cdf.disease == disease) &
                (cdf.horizon == horizon)
            ]
            if len(sub) == 0:
                continue
            print(
                f"  {disease} {horizon}d "
                f"({len(sub)} positives in valid folds)"
            )
            print(
                f"    {'patient':12s} {'hadm':12s} {'fold':>4s} {'n_te':>5s} "
                f"{'alpha':>6s} {'rank a0':>8s} {'learned':>8s} "
                f"{'rank a1':>8s} {'gain':>6s}"
            )
            for _, c in sub.iterrows():
                print(
                    f"    {str(c['patient_id']):12s} {str(c['hadm_id']):12s} "
                    f"{int(c['fold']):>4d} {int(c['n_test']):>5d} "
                    f"{c['alpha_learned']:>6.3f} "
                    f"{int(c['rank_a0']):>8d} "
                    f"{int(c['rank_learned']):>8d} "
                    f"{int(c['rank_a1']):>8d} "
                    f"{int(c['rank_gain_vs_a0']):>+6d}"
                )
            up = int((sub['rank_gain_vs_a0'] > 0).sum())
            dn = int((sub['rank_gain_vs_a0'] < 0).sum())
            print(
                f"    -> up {up}, down {dn}, same {len(sub)-up-dn}; "
                f"mean rank change "
                f"{sub['rank_gain_vs_a0'].mean():+.1f}\n"
            )
    

    print('=' * 112)
    print('HOW TO READ')
    print('=' * 112)
    print(
        '  Main comparison for the default/scalar gate: 4 vs 1b. 1b gives a '
        'plain EHR probe the same scalar side information without adaptive '
        'fusion.'
    )
    print()
    print(
        '  alpha_std_within near 0 means the learned gate is effectively '
        'constant rather than admission-adaptive.'
    )
    print()
    print(
        '  Settings 1/2/3/1b are fitted separately per horizon; setting 4 is '
        'one shared model across the four horizons WITHIN each disease. '
        'Censored labels are omitted only at their own horizon.'
    )

    # ── alpha sweep ──
    sw = res[res.setting.str.startswith('S_alpha_')].copy()
    if len(sw):
        print('=' * 104)
        print('ALPHA SWEEP -- fused = (1-alpha)*sEHR + alpha*PPG')
        print('=' * 104)
        print('  alpha is frozen at each point and C is still chosen by the')
        print('  inner CV, so each row is "this mixing weight, best-regularised".')
        print('  The peak is read off the OUTER test folds: it is the maximum')
        print('  of 21 noisy numbers and is a DIAGNOSTIC UPPER BOUND, not a')
        print('  result. Compare it against the pre-alignment baselines to see')
        print('  whether any fixed weight could beat them at all; compare')
        print('  setting 3 against them to see whether one can be selected.\n')
        for disease in diseases:
            for horizon in HORIZONS:
                s = sw[(sw.disease == disease) & (sw.horizon == horizon)]
                if s.empty:
                    continue
                ref = res[(res.disease == disease) & (res.horizon == horizon)]

                def _get(setting):
                    rr = ref[ref.setting == setting]
                    return float(rr.iloc[0]['auroc_mean']) if len(rr) else np.nan

                a0 = _get('0a_EHR_pre')
                b0 = _get('0b_PPG_pre')
                e1 = _get('1_EHR_post')
                s3 = _get('3_fixed_alpha_inner_cv')
                best = s.loc[s['auroc_mean'].idxmax()]

                print(f"  {disease} {horizon}d   "
                      f"EHR_pre {a0:.3f}  PPG_pre {b0:.3f}  "
                      f"EHR_post {e1:.3f}  setting3 {s3:.3f}")
                print(f"    sweep peak alpha={best['sweep_alpha']:.2f} "
                      f"AUROC {best['auroc_mean']:.3f}"
                      f"±{best['auroc_std_fold']:.3f}"
                      f"   vs EHR_pre {best['auroc_mean'] - a0:+.3f}"
                      f"   vs PPG_pre {best['auroc_mean'] - b0:+.3f}")
                curve = '  '.join(
                    f"{r['sweep_alpha']:.2f}:{r['auroc_mean']:.3f}"
                    for _, r in s.sort_values('sweep_alpha').iterrows()
                )
                print(f"    {curve}\n")

    # ── randmlp run: endpoint self-checks and the two headline deltas ──
    if args.ehr_source == 'randmlp':
        print('=' * 112)
        print(f'RANDOM sEHR PROJECTION (seed {args.rand_seed}) -- CHECKS')
        print('=' * 112)
        print(
            "  The mixer is exact at both ends, so these are equalities, not\n"
            "  approximations. A mismatch is a bug, not noise.\n"
            "    alpha=0.00  ==  0c_EHR_randmlp    (same run)\n"
            "    alpha=1.00  ==  2_PPG_post        (and equal to the ALIGNED\n"
            "                                       run's 2_PPG_post, since\n"
            "                                       the PPG side never moved)\n"
        )
        tol = 5e-4
        n_bad = 0
        for disease in diseases:
            for horizon in HORIZONS:
                ref = res[(res.disease == disease) & (res.horizon == horizon)]

                def _g(setting):
                    rr = ref[ref.setting == setting]
                    return (float(rr.iloc[0]['auroc_mean'])
                            if len(rr) else np.nan)

                a0_sweep = _g('S_alpha_0.00')
                a1_sweep = _g('S_alpha_1.00')
                c0 = _g('0c_EHR_randmlp')
                p2 = _g('2_PPG_post')
                for lhs, rhs, name in [
                    (a0_sweep, c0, 'alpha=0 vs 0c_EHR_randmlp'),
                    (a1_sweep, p2, 'alpha=1 vs 2_PPG_post'),
                ]:
                    if np.isfinite(lhs) and np.isfinite(rhs) \
                            and abs(lhs - rhs) > tol:
                        n_bad += 1
                        print(f"  !! {disease} {horizon}d  {name}: "
                              f"{lhs:.6f} vs {rhs:.6f}  "
                              f"(diff {lhs - rhs:+.6f})")
        print(f"  endpoint mismatches beyond {tol:g}: {n_bad}"
              f"{'  -- OK' if n_bad == 0 else '  -- INVESTIGATE'}\n")

        print('=' * 112)
        print('HEADLINE DELTAS  (AUROC; paired per-fold values in the CSV)')
        print('=' * 112)
        print(
            "  vs EHR_pre     did fusion beat the strongest single modality?\n"
            "  vs EHR_randmlp did the PPG side contribute anything, or is the\n"
            "                 number carried by the random sEHR vector alone?\n"
            "  The second is the one that attributes the result. A fusion that\n"
            "  beats EHR_pre but not EHR_randmlp has not used the waveform.\n"
        )
        print(f"  {'disease':7s} {'hz':>5s} "
              f"{'EHR_pre':>9s} {'EHR_rand':>9s} "
              f"{'set3':>9s} {'set4':>9s} "
              f"{'3-pre':>8s} {'3-rand':>8s} {'4-pre':>8s} {'4-rand':>8s}")
        for disease in diseases:
            for horizon in HORIZONS:
                ref = res[(res.disease == disease) & (res.horizon == horizon)]

                def _g(setting):
                    rr = ref[ref.setting == setting]
                    return (float(rr.iloc[0]['auroc_mean'])
                            if len(rr) else np.nan)

                a0 = _g('0a_EHR_pre')
                c0 = _g('0c_EHR_randmlp')
                s3 = _g('3_fixed_alpha_inner_cv')
                s4 = _g('4_learned_alpha')
                print(f"  {disease:7s} {horizon:>5d} "
                      f"{a0:>9.3f} {c0:>9.3f} {s3:>9.3f} {s4:>9.3f} "
                      f"{s3 - a0:>+8.3f} {s3 - c0:>+8.3f} "
                      f"{s4 - a0:>+8.3f} {s4 - c0:>+8.3f}")
        print()


if __name__ == '__main__':
    main()