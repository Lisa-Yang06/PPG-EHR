"""
PLAIN 320 -> 128 -> 1, NO NORMALISATION -- post-discharge cohort. ALL TORCH.

The torch twin of plain_mlp.py. Same rows, same names, same folds, same
grids; the difference is that every MODEL here is a torch module trained by
AdamW, where plain_mlp.py fits five of its nine rows with sklearn's
LogisticRegression under GridSearchCV.

WHY
---
plain_mlp.py mixes two frameworks, so every "fusion vs baseline" difference
in it carries a framework difference as well. Two numbers from its last run
show the size of that:

    N_probe(sklearn convex solve) - N_mlp(AdamW)   -0.0031   3/12
    N_probe_on_128 - A320_selC (both sklearn)      -0.0111   0/12

The second is optimiser-matched, so "a learned 128-d loses to the raw 320-d"
survives. The rest of that table does not have the same protection. Here
every row is on one framework and one grid, so a difference is a difference
of representation or of fusion form and nothing else.

Row for row against plain_mlp.py's CSV, the gap IS the framework effect.

WHAT REPLACES C
---------------
sklearn minimises C*sum(loss) + 0.5*||w||^2, i.e. mean(loss) +
||w||^2/(2Cn), so its penalty weight is lambda = 1/(2Cn). That role is taken
here by weight decay on the head, searched over the same 6 x 4 grid, rho-
filtered, that every other torch row in this file searches -- 21 candidates
against the 21-point C grid it replaces.

For the PINNED reference row, C = 1e-5 at n ~ 1800 gives lambda ~ 28, so
--wd-a320 defaults to 30, which is also the modal decay the searched rows
picked on the last real run. It remains a hand-picked value.

StandardScaler and the metrics stay sklearn: preprocessing and measurement,
not models.

    N_mlp_320_128_1   x -> Linear(320,128) -> Linear(128,1), trained
                      end to end. No L2 normalisation between the layers,
                      no PPG, no alpha, no gate.

    N_probe_on_128    that model's 128-d hidden layer, FROZEN, with a fresh
                      Linear(128,1) trained on it, decay by the same inner
                      3-fold every other row here gets.

WHY NO NORMALISATION HERE
-------------------------
linear_fusion.py's M_ rows put F.normalize between the two layers. That is
not decoration: with a trainable P and no normalisation the convex mixture
(1-a)Px + a*zp is degenerate, because for any a' the substitution

    w' = (a/a')w      P' = P (1-a)a' / ((1-a')a)

reproduces the logit exactly, so alpha carries no information about the
function and the selected alpha reports the ratio of the two decays instead
of anything about the data.

Nothing here mixes, so nothing here needs the unit sphere, and the model
becomes what a strong-EHR-baseline proposal actually asks for. The cost of
dropping it is worth stating: with normalisation the score is
(u.x + c) / ||Px + b||, a genuinely different model class; without it,

    score = w.(Px + b) = (P'w).x + const = u.x + const

which is A320_sEHR_pre's hypothesis class restricted to the <=128-dim row
space of P. A_sEHR_pca128 - A320 = -0.001 in 12/12 cells says a label-blind
128-d subspace already loses nothing, so this row cannot EXCEED A320 through
capacity. What it can differ by is (a) AdamW with early stopping instead of a
convex solve, and (b) an L2 penalty on the two FACTORS rather than on the
composite u, which is a nuclear-norm-like penalty and not ridge.

HOW TO READ N_probe_on_128 -- IMPORTANT
---------------------------------------
There is no test leakage. P is fitted on outer-train only, the probe is
fitted on outer-train only, the outer test fold is untouched by both.

But this row is NOT the same kind of number as A_sEHR_pca128, and putting
them side by side without saying so is misleading:

    A_sEHR_pca128     128 dims chosen WITHOUT looking at any label. The
                      probe AUROC is a statement about the REPRESENTATION.

    N_probe_on_128    128 dims chosen BY fitting the downstream label. The
                      probe AUROC is a statement about a two-stage FITTING
                      PROCEDURE, and the representation cannot be called
                      general-purpose.

The C selection inside this row is also optimistic in a way the others are
not: its inner-validation rows were part of the data P was fitted on, so the
inner AUROC that picks C is inflated. That affects only WHICH C is picked;
the reported test AUROC is clean either way. Fixing it properly means
refitting the encoder inside every inner split, which is what --oof-alpha
does for alpha in alpha_fusion and costs one extra encoder fit per inner
fold. Not done here.

PROTOCOL IS COPIED, NOT REIMPLEMENTED
-------------------------------------
Outer folds, inner 3-fold selection, the candidate grid, the parameter-group
split and the per-fold packing all come from alpha_fusion.py and
gated_fusion.py. af._adapter_pack is called with the full [pre|post|ppg]
matrix even though only v_ehr_pre is read, so the per-fold scaler discipline
is byte-identical to the M_ and H_ rows. The ONLY difference from
linear_fusion's M_only is the missing F.normalize, so a difference in the
numbers cannot come from a difference in protocol, and every row here is
comparable fold by fold to everything already computed.

POOLED OOF AND THE BOOTSTRAP INTERVAL
-------------------------------------
Every row now reports TWO uncertainty surfaces, and they answer different
questions. Reporting either as if it were the other would be wrong.

  auroc_mean +- auroc_std_fold   the five per-fold AUROCs, averaged, and their
                                 spread. "How much does the answer move when
                                 the partition moves."

  auroc_oof (ci_low-ci_high)     ONE AUROC on all five folds' outer-test
                                 predictions concatenated, and a 95% interval
                                 from resampling PATIENTS. "How well are these
                                 admissions ranked, and how much would that
                                 move on another sample of patients."

The pooled estimate is NOT the mean of the fold estimates and need not equal
it. It also carries an assumption the fold mean does not: each outer fold
selects its own wd/lr/epoch and returns RAW LOGITS on its own scale, so
pooling ranks fold-A admissions against fold-B admissions on scales that were
never made commensurable. AUROC and AUPRC are rank-based, which makes this
well defined, not free -- a fold whose logits sit higher drags its rows up the
joint ranking.

The interval covers PATIENT SAMPLING ONLY. It contains nothing about
fold-assignment or hyperparameter-selection variability, which is the part
auroc_std_fold gestures at. Neither number is a superset of the other.

Resampling is by PATIENT, not admission: a patient contributing several
admissions makes those rows correlated, and resampling admissions would count
them as independent and return an interval that is too narrow. The outer CV
groups by patient for the same reason. The metric stays admission-level; only
the resampling unit is the patient. A replicate with one class has no AUROC
and is redrawn, so at low-positive cells the accepted replicates are
conditioned on containing a positive -- n_boot_draws exposes how often that
happened.

The per-admission predictions are written to a companion _oof.csv, so a paired
delta-AUROC between two rows, a calibration curve or an error-case listing can
be built later without retraining.

Usage:
    python plain_mlp.py --disease AF
"""
import os
os.environ.setdefault('MPLCONFIGDIR', str(paths.data_root() / ".cache/matplotlib"))

import argparse
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# StandardScaler and the two metrics stay: the first is deterministic
# preprocessing that a hand-written mean/std would reproduce to the last bit,
# the second are metrics. Neither is a MODEL, and models are what this file
# exists to put on one framework.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

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
import gated_fusion as gf
import alpha_fusion as af

from gated_fusion import (
    DISEASES, HORIZONS, FEATURES_PATH, METADATA_CSV,
    label_col, load_joined, assign_folds, subset_folds, build_scalar_matrix,
)

N_DIM = af.N_COMPONENTS          # 128

# alpha IS the PPG weight, so 0.0 reproduces N_probe_on_128 exactly and 1.0 is
# PPG alone. Uniform on the full convex range, matching alpha_fusion's
# ADAPTER_ALPHA_GRID at the default 21 points so a selected value here can be
# read beside one from there. Rebound from --alpha-points, because the torch
# fixed-alpha rows cost len(ALPHA_GRID) x 21 x 3 trainings per cell and 21 is
# not always affordable.
ALPHA_GRID = [round(0.05 * i, 2) for i in range(21)]

# ── pooled OOF + patient-cluster bootstrap ────────────────────────────────
# Deliberately NOT reg.RANDOM_SEED, though both are 42 today. Fold assignment
# and CI resampling are two different sources of randomness; tying them to one
# constant means a robustness check that re-seeds the folds silently moves
# every interval too, and the two can then no longer be audited apart.
BOOTSTRAP_SEED = 42
N_BOOT = 2000
# A replicate with one class in it has no AUROC, so it is discarded and
# redrawn until N_BOOT valid ones exist. On a cell with very few positives
# that loop could run a long time; this caps the draws and reports how many
# were needed instead of hanging.
BOOT_MAX_DRAW_FACTOR = 20


def _alpha_grid(n):
    return [round(i / (n - 1), 4) for i in range(n)] if n > 1 else [0.0]


def _boot_patient_indices(y, groups, n_boot=N_BOOT, seed=BOOTSTRAP_SEED,
                          max_draw_factor=BOOT_MAX_DRAW_FACTOR):
    """Row indices for n_boot patient-clustered bootstrap replicates.

    The independence unit is the PATIENT, not the admission: one patient can
    contribute several admissions and those share age, comorbidity and
    everything else stable about them, so resampling admissions would treat
    correlated rows as independent evidence and return an interval that is
    too narrow. The outer CV groups by patient for the same reason.

    Each replicate draws len(patients) patients WITH REPLACEMENT and takes
    every admission of each drawn patient; a patient drawn twice contributes
    its admissions twice. The number of ADMISSIONS per replicate therefore
    varies -- that is the cluster bootstrap working, not a defect.

    Returns (list of index arrays, n_drawn). n_drawn > len(list) means
    replicates were rejected for being single-class.
    """
    y = np.asarray(y)
    pats, inv = np.unique(np.asarray(groups), return_inverse=True)
    rows_by_pat = [np.flatnonzero(inv == i) for i in range(len(pats))]
    rng = np.random.default_rng(seed)

    out, drawn = [], 0
    max_draws = max_draw_factor * n_boot
    while len(out) < n_boot and drawn < max_draws:
        drawn += 1
        pick = rng.integers(0, len(pats), size=len(pats))
        idx = np.concatenate([rows_by_pat[i] for i in pick])
        yb = y[idx]
        # Both classes must be present or neither metric is defined. Using ONE
        # condition for both keeps the AUROC and AUPRC intervals on the same
        # replicates, so they are paired rather than independently drawn.
        if yb.min() == yb.max():
            continue
        out.append(idx)
    return out, drawn


_BOOT_CACHE = {}


def pooled_oof_ci(y, oof, groups, n_boot=N_BOOT, seed=BOOTSTRAP_SEED):
    """Pooled out-of-fold AUROC/AUPRC and their patient-bootstrap 95% CIs.

    POOLED IS NOT THE MEAN OF THE FOLD METRICS. mean(AUROC_1..5) asks "how
    well does this method do on a fold", pooled asks "how well are these 1647
    admissions ranked". They are different quantities and need not agree.

    The pooled number carries an assumption the fold mean does not: each outer
    fold selects its OWN wd/lr/epoch and returns RAW LOGITS on its own scale,
    so pooling ranks fold-A patients against fold-B patients on scales that
    were never made commensurable. AUROC and AUPRC are rank-based, which makes
    this well defined but not free -- a fold whose logits happen to sit higher
    drags its rows up the joint ranking. Standard practice; still an
    assumption, and it belongs beside the number.

    The interval covers PATIENT SAMPLING ONLY. It says nothing about
    fold-assignment or hyperparameter-selection variability, which is what
    auroc_std_fold gestures at. The two are not substitutes and must not be
    presented as if either covered the other.
    """
    y = np.asarray(y)
    oof = np.asarray(oof, dtype=np.float64)
    groups = np.asarray(groups)

    # Folds skipped by the single-class guard in the outer loop leave NaN
    # here. They are absent, not zero.
    ok = np.isfinite(oof)
    r = {'n_oof': int(ok.sum()), 'auroc_oof': np.nan, 'auprc_oof': np.nan,
         'auroc_ci_low': np.nan, 'auroc_ci_high': np.nan,
         'auprc_ci_low': np.nan, 'auprc_ci_high': np.nan,
         'n_boot_valid': 0, 'n_boot_draws': 0}
    if not ok.any() or len(np.unique(y[ok])) < 2:
        return r

    yo, so, go = y[ok], oof[ok], groups[ok]
    r['auroc_oof'] = float(roc_auc_score(yo, so))
    r['auprc_oof'] = float(average_precision_score(yo, so))

    # The replicates depend only on (y, groups, seed), which are identical for
    # every model row at one disease x horizon. Caching therefore does two
    # things: it saves rebuilding the same draws eleven times, and it
    # GUARANTEES every row is bootstrapped on the same patients -- the
    # property a later paired delta-AUROC between two rows depends on.
    ck = (ok.tobytes(), yo.tobytes(), go.tobytes(), n_boot, seed)
    if ck not in _BOOT_CACHE:
        _BOOT_CACHE[ck] = _boot_patient_indices(yo, go, n_boot, seed)
    boots, drawn = _BOOT_CACHE[ck]

    aucs = np.empty(len(boots))
    aps = np.empty(len(boots))
    for i, idx in enumerate(boots):
        aucs[i] = roc_auc_score(yo[idx], so[idx])
        aps[i] = average_precision_score(yo[idx], so[idx])

    r['n_boot_valid'] = len(boots)
    r['n_boot_draws'] = drawn
    if len(boots):
        lo, hi = np.percentile(aucs, [2.5, 97.5])
        r['auroc_ci_low'], r['auroc_ci_high'] = float(lo), float(hi)
        lo, hi = np.percentile(aps, [2.5, 97.5])
        r['auprc_ci_low'], r['auprc_ci_high'] = float(lo), float(hi)
    return r


class PlainMLP(nn.Module):
    """320 -> 128 -> 1. Nothing between the layers.

    The projection is named `adapter` and `gate` is set to None so that
    af.adapter_groups -- which reads model.adapter.named_parameters(),
    model.head, and model.gate -- applies with no special case. The forward
    signature takes (ez, zp, xg) because gf._train_once passes three tensors;
    zp and xg are ignored. This model uses NO PPG.
    """

    def __init__(self, pre_dim, dim, freeze_proj=False):
        super().__init__()
        self.pre_dim = pre_dim
        self.adapter = nn.Linear(pre_dim, dim)
        self.gate = None
        self.head = nn.Linear(dim, 1)
        if freeze_proj:
            # N_randproj_frozen. _train_once seeds torch before calling this,
            # so the frozen P is EXACTLY the initialisation the trained row
            # starts from -- the two rows differ by whether it then moves.
            self.adapter.requires_grad_(False)

    def features(self, ez):
        return self.adapter(ez[:, :self.pre_dim])

    def forward(self, ez, zp, xg):
        h = self.features(ez)
        # second return value keeps _train_once's (logit, alpha) contract;
        # there is no alpha in this model, so it is a constant zero and the
        # alpha_* columns are reported as NaN rather than as 0.0, which would
        # read as "the gate chose zero".
        return (self.head(h).squeeze(-1),
                torch.zeros(ez.shape[0], dtype=h.dtype))


class LinearHead(nn.Module):
    """Linear(d, 1). The torch stand-in for sklearn's LogisticRegression.

    forward takes (ez, zp, xg) so gf._train_once drives it unchanged; zp and
    xg are ignored. `gate = None` lets head_only_groups and gate_head_groups
    both apply without a special case.

    What replaces C: sklearn minimises C*sum(loss) + 0.5*||w||^2, i.e.
    mean(loss) + ||w||^2/(2*C*n), so its penalty weight is lambda = 1/(2Cn).
    Here that role is played by weight_decay on the head, searched over the
    SAME grid every other row in this file searches.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.gate = None
        self.head = nn.Linear(in_dim, 1)

    def forward(self, ez, zp, xg):
        return (self.head(ez).squeeze(-1),
                torch.zeros(ez.shape[0], dtype=ez.dtype))


class AlphaOnFrozen(nn.Module):
    """Per-sample convex mixture of two FROZEN 128-d blocks, then a head.

        a_i    = sigmoid(g . scalars_i)
        fused  = (1 - a_i) * h_i + a_i * zp_i
        logit  = w . fused

    Nothing upstream trains: h is N_mlp_320_128_1's hidden layer, already
    fitted and frozen, and zp is v_ppg_post. That is what makes alpha
    identifiable here. With a TRAINABLE projection it is not -- for any a'
    the substitution w' = (a/a')w, P' = P(1-a)a'/((1-a')a) reproduces the
    logit exactly, so the selected alpha would report the ratio of the two
    decays rather than anything about the data. Both blocks being fixed
    numeric matrices removes that freedom.

    Both are put through the same per-coordinate StandardScaler before
    mixing, so each block has expected squared norm equal to its width and
    neither swamps the other. h carries whatever scale training left it at
    and zp has unit norm, so without this alpha would again be meaningless.
    """

    def __init__(self, dim, gate_in_dim, alpha_fixed=None, mode='convex'):
        super().__init__()
        self.alpha_fixed = alpha_fixed
        # THREE fusion forms, one class, so they cannot drift apart:
        #
        #   convex    a = sigmoid(g.x);   (1-a)*h + a*zp
        #             coefficients sum to one and are non-negative
        #   residual  a = sigmoid(g.x);   h + a*zp
        #             sEHR carries no coefficient. NOT the convex form
        #             reparametrised: writing convex as
        #             (1-a(x))[w.h + (a/(1-a))(x) w.zp], the leading (1-a(x))
        #             is a per-sample rescale of the whole vector a linear
        #             head cannot absorb. At a FIXED alpha the two ARE the
        #             same class; per-sample they are not. The residual caps
        #             the PPG:sEHR ratio at 1, the convex form's effective
        #             ratio a/(1-a) is unbounded.
        #   free      (b1,b2) = g.x, no activation;  b1*h + b2*zp
        #             per-patient, unconstrained. Strictly CONTAINS convex,
        #             so a loss here is selection noise and nothing else; a
        #             gain says sum-to-one or non-negativity was binding.
        assert mode in ('convex', 'residual', 'free'), mode
        self.mode = mode
        # A frozen alpha builds no gate at all, so the fixed and gated rows
        # are one class rather than two that can drift apart -- the pattern
        # linear_fusion.LinearFusion already uses.
        self.gate = (None if alpha_fixed is not None
                     else nn.Linear(gate_in_dim,
                                    2 if mode == 'free' else 1))
        self.head = nn.Linear(dim, 1)

    def forward(self, ez, zp, xg):
        # Second return value is the COEFFICIENT PAIR actually applied, (n,2)
        # as [sEHR | PPG], so all three forms report the same thing and the
        # results table can put them in the same columns. gf._train_once
        # discards it; only the refit path reads it.
        if self.mode == 'free':
            b = self.gate(xg)
            return self.head(b[:, :1] * ez + b[:, 1:] * zp).squeeze(-1), b
        if self.gate is None:
            a = torch.full((ez.shape[0], 1), float(self.alpha_fixed),
                           dtype=ez.dtype)
        else:
            a = torch.sigmoid(self.gate(xg))
        if self.mode == 'residual':
            return (self.head(ez + a * zp).squeeze(-1),
                    torch.cat([torch.ones_like(a), a], dim=1))
        return (self.head((1.0 - a) * ez + a * zp).squeeze(-1),
                torch.cat([1.0 - a, a], dim=1))


def gate_head_groups(model, wd, wd_head):
    """Gate and head, biases exempt. Under --wd-mode shared the caller
    passes the same value for both, so this is one decay."""
    g = [{'params': [model.head.weight], 'weight_decay': wd_head},
         {'params': [model.head.bias], 'weight_decay': 0.0}]
    if model.gate is not None:
        g += [{'params': [model.gate.weight], 'weight_decay': wd},
              {'params': [model.gate.bias], 'weight_decay': 0.0}]
    return g


def sel_desc(**cols):
    """'wd 1,30 | lr 0.01,0.003 | epoch 6,3' from name -> per-fold values.

    Built where the values live, by the code that selected them, so a row can
    only ever describe its own search. The previous arrangement had one shared
    pack() read sel/eps from the enclosing scope, and since only the MLP row
    wrote to those, five of nine rows reported the MLP's hyperparameters as
    their own.
    """
    def fmt(v):
        return f'{v:g}' if isinstance(v, float) else str(v)
    return '  '.join(f"{k} {','.join(fmt(v) for v in vals)}"
                     for k, vals in cols.items() if vals)


def torch_linear_cv(X_tr, X_te, y_tr, inner_splits, cands, seed, args,
                    fixed=None):
    """StandardScaler -> Linear(d,1) -> AdamW, weight decay by inner 3-fold.

    The torch counterpart of what a Pipeline([StandardScaler, Logistic]) under
    GridSearchCV(C) did. The scaler is refit INSIDE every inner split, so the
    validation rows never shape the statistics the selection is read off, and
    again on the full outer-train for the refit.

    fixed=(wd, lr) skips selection entirely and trains the full epoch budget
    with no validation -- the analogue of fitting at a pinned C. Returns
    (test scores, wd, lr, epoch).
    """
    d = X_tr.shape[1]
    T = lambda z: torch.tensor(z, dtype=torch.float32)
    Z = lambda n: T(np.zeros((n, 1)))
    mk = lambda: LinearHead(d)

    if fixed is not None: # predefined/fixed hyperparameters, no inner CV
        wd, lr, ep = fixed[0], fixed[1], args.epochs
    else:
        rows = []
        for wd_c, lr_c in cands:
            vas, ee = [], []
            for i_tr, i_va in inner_splits:
                sc_i = StandardScaler().fit(X_tr[i_tr])
                idx = np.concatenate([i_tr, i_va])
                Xi = sc_i.transform(X_tr[idx])
                yi = y_tr[idx]
                vi = np.arange(len(i_tr), len(idx))
                if len(np.unique(yi[vi])) < 2:
                    continue
                _, va, ep_i = gf._train_once(
                    T(Xi), Z(len(idx)), Z(len(idx)),
                    T(yi.astype(np.float64)), yi,
                    np.zeros(len(idx), dtype=np.int64),
                    np.arange(len(i_tr)), vi, [0],
                    d, 1, 0, seed, wd_c, lr_c, True,
                    wd_head=wd_c, n_epochs=args.epochs, return_epoch=True,
                    eval_every=args.eval_every, make_model=mk,
                    make_groups=head_only_groups)
                if va is not None:
                    vas.append(va)
                    ee.append(ep_i)
            if vas:
                rows.append({'mean': float(np.mean(vas)),
                             'epoch': int(np.median(ee)),
                             'wd': wd_c, 'lr': lr_c})
        if rows:
            rows.sort(key=lambda r: -r['mean'])
            b = rows[0]
            wd, lr, ep = b['wd'], b['lr'], b['epoch']
        else:
            # Same off-grid fallback the other rows take when a fold holds too
            # few positives to select anything
            wd, lr, ep = gf.DEFAULT_WD, gf.DEFAULT_LR, args.epochs

    sc = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = sc.transform(X_tr), sc.transform(X_te)
    model, _ = gf._train_once(
        T(Xs_tr), Z(len(X_tr)), Z(len(X_tr)),
        T(y_tr.astype(np.float64)), y_tr,
        np.zeros(len(X_tr), dtype=np.int64), np.arange(len(X_tr)), None, [],
        d, 1, 0, seed, wd, lr, True,
        wd_head=wd, n_epochs=ep, make_model=mk,
        make_groups=head_only_groups)
    with torch.no_grad():
        s_te, _ = model(T(Xs_te), Z(len(X_te)), Z(len(X_te)))
    return s_te.numpy(), wd, lr, ep


def inner_folds(y_tr, g_tr, seed):
    """The inner 3-fold split every row in this file selects on.

    Stratified on "this patient is ever positive" and grouped by patient, the
    same construction run_plain uses, so the baseline rows and the fusion rows
    select on identical splits.
    """
    out = []
    pos = np.unique(g_tr[y_tr == 1])
    if len(pos) < af.N_INNER_E:
        return out
    ever = np.isin(g_tr, pos).astype(int)
    sp = StratifiedGroupKFold(n_splits=af.N_INNER_E, shuffle=True,
                              random_state=seed)
    for i_tr, i_va in sp.split(np.zeros((len(y_tr), 1)), ever, groups=g_tr):
        if len(np.unique(y_tr[i_tr])) >= 2:
            out.append((i_tr, i_va))
    return out


def wd_lr_candidates(args):
    """(wd, lr) under the rho guard -- the grid every torch row here shares."""
    return [(w, lr) for w in args.wd_proj for lr in af.LR_GRID_E
            if lr * w <= af.MAX_RHO]


def run_torch_baseline(X, y, groups, folds, args, tag, fixed=None):
    """A 320-d probe as a torch row. The outer loop reg.run_cv used to run.

    fixed=(wd, lr) reproduces the pinned-C reference: no inner CV, no early
    stopping, the full epoch budget at one configuration.
    """
    cands = wd_lr_candidates(args)
    auc, ap, sw, sl, se = [], [], [], [], []
    # Every admission's outer-test score, in the row order of y. subset_folds
    # makes the five te arrays a partition of arange(len(y)), so no index
    # translation is needed. A fold skipped below leaves its rows NaN.
    oof = np.full(len(y), np.nan, dtype=np.float64)
    n_no_sel = 0
    for fold_id, (tr, te) in enumerate(folds):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        seed = reg.RANDOM_SEED + fold_id
        isp = [] if fixed is not None else inner_folds(y[tr], groups[tr], seed)
        if fixed is None and not isp:
            n_no_sel += 1
        s_te, wd, lr, ep = torch_linear_cv(X[tr], X[te], y[tr], isp, cands,
                                           seed, args, fixed)
        sw.append(wd)
        sl.append(lr)
        se.append(ep)
        oof[te] = s_te
        auc.append(roc_auc_score(y[te], s_te))
        ap.append(average_precision_score(y[te], s_te))
    return {
        # Popped in main before the DataFrame is built -- an array left in a
        # row dict becomes a CSV column of arrays.
        '_oof': oof,
        'probe': tag, 'setting': tag,
        'auroc_mean': float(np.mean(auc)) if auc else np.nan,
        'auroc_std_fold': float(np.std(auc)) if auc else np.nan,
        'auprc_mean': float(np.mean(ap)) if ap else np.nan,
        'auprc_std_fold': float(np.std(ap)) if ap else np.nan,
        'auroc_per_fold': af._pf(auc), 'auprc_per_fold': af._pf(ap),
        'alpha_mean': np.nan, 'alpha_std_within': np.nan,
        'alpha_std_across': np.nan, 'alpha_per_fold': '',
        'n_valid_folds': len(auc), 'folds_without_selection': n_no_sel,
        'folds_alpha_defaulted': 0,
        'sel_desc': (('PINNED, no selection:  ' if fixed is not None else '')
                     + sel_desc(wd=sw, lr=sl, epoch=se)),
        'sel_wd_head': ','.join(f'{v:g}' for v in sw),
        'sel_lr': ','.join(f'{v:g}' for v in sl),
        'sel_wd_proj': '', 'sel_epoch': ','.join(str(v) for v in se),
        'w_proj_norm': np.nan, 'w_head_norm': np.nan, 'h_norm': np.nan,
        # No fusion here, so no coefficients. Empty strings, not NaN: the
        # table formats these columns from text and a NaN prints as 'nan'.
        'coef_ehr_per_fold': '', 'coef_ppg_per_fold': '',
        'coef_ehr_mean': np.nan, 'coef_ppg_mean': np.nan,
    }


def head_only_groups(model, wd, wd_head):
    """Only the head trains. The projection is not in the optimiser at all,
    so no step and no decoupled decay can touch it and it stays at its
    initialisation exactly."""
    return [{'params': [model.head.weight], 'weight_decay': wd_head},
            {'params': [model.head.bias], 'weight_decay': 0.0}]


def candidates(args, frozen=False):
    """(alpha, wd_head, lr, wd_proj) per row. ONE definition, used both by
    run_plain and by the startup print -- they disagreed once, the log
    claiming 113 while the search ran 21."""
    if frozen:
        # No projection decay to search: P is not in the optimiser.
        return [(0.0, wh, lr, 0.0)
                for wh in af.WD_HEAD_GRID for lr in af.LR_GRID_E
                if lr * wh <= af.MAX_RHO]
    if args.wd_mode == 'shared':
        # ONE decay for both layers. The two are not regularising different
        # objects here -- this is a plain MLP, not a residual on a frozen
        # base, so there is no "how far from the pretrained vector" for one
        # of them to bound. Searching them separately cost a factor of six
        # in candidates to select on differences that are not identifiable:
        # in the previous run a single cell picked wd_proj across 0.01 to
        # 100 in its five folds.
        #
        # It also makes this row and N_randproj_frozen search the same
        # number of candidates, so the trained-vs-random comparison carries
        # no selection-noise asymmetry in either direction.
        return [(0.0, w, lr, w)
                for w in args.wd_proj for lr in af.LR_GRID_E
                if lr * w <= af.MAX_RHO]
    return af.adapter_candidates('only', args.wd_proj, args.lr_adapter)


def run_plain(Xa, M, y, groups, folds, gate_input, args, frozen=False):
    """Both rows, from ONE model per outer fold.

    Selection, refit and scoring are the 'only' branch of
    linear_fusion.run_linear with PlainMLP substituted; the probe is then
    fitted on the refit model's hidden layer inside the same fold.

    frozen=True is the random-projection control: P stays at its random
    initialisation and only the head trains, so the row answers whether the
    trained projection is doing anything the head could not do behind a
    random one. It returns ONE row and no probe -- a convex probe on a
    random projection is a different question.

    The control searches wd_head x lr only, since there is no projection
    decay to search. That is 6 candidates against the trained row's 36, so
    the control gets LESS opportunity to overfit the inner CV; if the two
    rows come out equal, that asymmetry is working against the conclusion,
    not for it.
    """
    pre_dim, dim = args.pre_dim, N_DIM
    cands = candidates(args, frozen)

    auc_e, ap_e, auc_p, ap_p = [], [], [], []
    # One outer-test score vector per row tag, in the row order of y. The
    # gated rows keep theirs inside G below, next to their own metrics.
    oof_e = np.full(len(y), np.nan, dtype=np.float64)   # mlp / frozen control
    oof_p = np.full(len(y), np.nan, dtype=np.float64)   # probe on the 128
    oof_x = np.full(len(y), np.nan, dtype=np.float64)   # fixed alpha
    # The fixed-alpha row. ce/cp are the two coefficients it applies,
    # 1-alpha and alpha, reported in the same columns every fusion row uses.
    auc_x, ap_x, a_fix, CXs, ce_x, cp_x = [], [], [], [], [], []
    n_fix_nosel = 0

    # The three gated forms, in ROW_ORDER order: convex, free, residual.
    # One accumulator shape for all three -- auc, ap, the two coefficients
    # (ce = sEHR side, cp = PPG side), the WITHIN-fold spread of the PPG
    # coefficient, and the selected hyperparameters.
    G = [{'auc': [], 'ap': [], 'ce': [], 'cp': [], 'sd': [], 'sel': [],
          'oof': np.full(len(y), np.nan, dtype=np.float64)}
         for _ in range(3)]
    G_TAGS = ('N_gate_alpha', 'N_gate_beta_free', 'N_gate_residual')
    hrn, hrs, prn, prs = [], [], [], []
    # Per-ROW selections. Every row that searches anything writes its own
    # here, so no row can inherit another's hyperparameters.
    sv_mlp, sv_probe = [], []
    sv_fix = []
    sel, eps, was, Cs, hn, wn, hnorm = ([] for _ in range(7))
    n_no_sel = 0

    for fold_id, (tr, te) in enumerate(folds):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        seed = reg.RANDOM_SEED + fold_id
        Xtr, Mtr, ytr, gtr = Xa[tr], M[tr], y[tr], groups[tr]

        packs = []
        # Kept so the learned-alpha row selects on the SAME inner splits the
        # projection was selected on, rather than drawing its own.
        inner_splits = []
        pos = np.unique(gtr[ytr == 1])
        if len(pos) >= af.N_INNER_E:
            ever = np.isin(gtr, pos).astype(int)
            sp = StratifiedGroupKFold(n_splits=af.N_INNER_E, shuffle=True,
                                      random_state=seed)
            for i_tr, i_va in sp.split(np.zeros((len(ytr), 1)), ever,
                                       groups=gtr):
                if len(np.unique(ytr[i_tr])) < 2:
                    continue
                inner_splits.append((i_tr, i_va))
                pk = af._adapter_pack(Xtr, Mtr, ytr, i_tr, i_va, gate_input,
                                      pre_dim, dim)
                if pk is not None:
                    packs.append(pk)

        best = None
        if packs:
            rows = []
            for _a, wh, lr, wp in cands:
                vas, ee = [], []
                for pk in packs:
                    mk = lambda: PlainMLP(pre_dim, dim, frozen)
                    _, va, ep = gf._train_once(
                        pk['ez'], pk['zp'], pk['xg'], pk['yt'], pk['y'],
                        pk['h'], pk['tr_i'], pk['va_i'], [0],
                        dim, pk['gate_dim'], 0, seed, 0.0, lr, True,
                        wd_head=wh, n_epochs=args.epochs, return_epoch=True,
                        eval_every=args.eval_every, make_model=mk,
                        make_groups=(head_only_groups if frozen else
                                     af.adapter_groups(wp, args.lr_adapter)),
                    )
                    if va is not None:
                        vas.append(va)
                        ee.append(ep)
                if vas:
                    rows.append({'mean': float(np.mean(vas)),
                                 'epoch': int(np.median(ee)),
                                 'wh': wh, 'lr': lr, 'wp': wp})
            if rows:
                rows.sort(key=lambda r: -r['mean'])
                best = rows[0]
        if best is None:
            # Same fallback as run_linear / run_adapter, and the same caveat:
            # these values are NOT on any searched grid, so a fold that takes
            # this branch ran a configuration nobody chose. Count it.
            n_no_sel += 1
            wh, lr, ep = gf.DEFAULT_WD, gf.DEFAULT_LR, args.epochs
            wp = args.wd_proj[-1]
        else:
            wh, lr, ep, wp = (best['wh'], best['lr'], best['epoch'],
                              best['wp'])
        sel.append((wh, lr))
        eps.append(ep)
        was.append(wp)
        sv_mlp.append((wp, wh, lr, ep))

        # ── refit on the FULL outer-train, fresh init ──
        E = Xa[:, :pre_dim]
        sc = StandardScaler().fit(E[tr])
        Ze = Xa[:, pre_dim:pre_dim + dim]
        ez_tr = np.hstack([sc.transform(E[tr]), Ze[tr]])
        ez_te = np.hstack([sc.transform(E[te]), Ze[te]])
        T = lambda z: torch.tensor(z, dtype=torch.float32)
        mk = lambda: PlainMLP(pre_dim, dim, frozen)
        model, _ = gf._train_once(
            T(ez_tr), T(np.zeros((len(tr), dim))), T(np.zeros((len(tr), 1))),
            T(y[tr].astype(np.float64)), y[tr],
            np.zeros(len(tr), dtype=np.int64), np.arange(len(tr)), None, [],
            dim, 1, 0, seed, 0.0, lr, True,
            wd_head=wh, n_epochs=ep, make_model=mk,
            make_groups=(head_only_groups if frozen else
                         af.adapter_groups(wp, args.lr_adapter)),
        )

        with torch.no_grad():
            H_tr = model.features(T(ez_tr)).numpy().astype(np.float64)
            H_te = model.features(T(ez_te)).numpy().astype(np.float64)
            s_te = model.head(T(H_te)).squeeze(-1).numpy()
            wn.append(float(model.adapter.weight.norm()))
            hn.append(float(model.head.weight.norm()))
        hnorm.append(float(np.linalg.norm(H_te, axis=1).mean()))

        oof_e[te] = s_te
        auc_e.append(roc_auc_score(y[te], s_te))
        ap_e.append(average_precision_score(y[te], s_te))

        # ── row 2: fresh probe on the FROZEN hidden layer ──
        # Linear(128,1) under AdamW, weight decay by the same inner 3-fold and
        # the same grid the rows above search. Standardised first, so the
        # decay falls evenly across 128 coordinates whose scales are whatever
        # training happened to leave them at.
        if not frozen:
            p_te, C, lr_p, ep_p = torch_linear_cv(
                H_tr, H_te, y[tr], inner_splits, wd_lr_candidates(args),
                seed, args)
            Cs.append(C)
            sv_probe.append((C, lr_p, ep_p))
            oof_p[te] = p_te
            auc_p.append(roc_auc_score(y[te], p_te))
            ap_p.append(average_precision_score(y[te], p_te))

            # The two blocks, stacked only so ONE per-coordinate
            # StandardScaler covers both. h is unnormalised and zp has unit
            # norm, so without this the mixture below would be dominated by
            # whichever block training happened to leave larger and alpha
            # would carry no meaning. No probe is fitted on the stack itself.
            Zp = Xa[:, pre_dim + dim:]
            F_tr = np.hstack([H_tr, Zp[tr]])
            F_te = np.hstack([H_te, Zp[te]])
            scF = StandardScaler().fit(F_tr)

            # ── rows 4-7: the two alpha forms, on two block variants ──
            # The mixture sits after the scaler, so this is the concatenated
            # probe restricted to two parallel coefficient vectors, and
            # alpha=0 is N_probe_on_128 coordinate for coordinate -- a
            # per-coordinate StandardScaler on [h|zp] gives the h columns
            # exactly what a scaler on h alone gives. That end of the grid is
            # this row's own control.
            #
            # alpha and C are searched TOGETHER in one GridSearchCV. Searching
            # C after fixing alpha would pick a penalty for a mixture that the
            # penalty itself changes the best mixture for.
            Fs_tr, Fs_te = scF.transform(F_tr), scF.transform(F_te)

            # Does one alpha mean the same thing for every patient?
            #
            # Per-coordinate standardisation equalises the two blocks ON
            # AVERAGE -- unit-variance columns put both expected row norms at
            # sqrt(dim) = 11.3 -- but it does not equalise their SPREAD. zp is
            # L2-normalised at extraction, so every PPG row starts at length 1
            # and the block arrives almost isotropic; h is not, and its
            # between-patient length differences survive scaling. Where they
            # do, (1-a)h + a*zp is dominated by whichever block is longer for
            # THAT patient, so a fixed alpha buys a different mixture per row
            # and a learned gate can spend itself cancelling ||h|| instead of
            # reading anything clinical.
            #
            # Measured on outer-TRAIN, where the scaler was fitted, so no
            # test-fold statistic enters the table. Diagnostic only: nothing
            # here feeds selection or scoring.
            _hn = np.linalg.norm(Fs_tr[:, :dim], axis=1)
            _pn = np.linalg.norm(Fs_tr[:, dim:], axis=1)
            hrn.append(float(_hn.mean())); hrs.append(float(_hn.std()))
            prn.append(float(_pn.mean())); prs.append(float(_pn.std()))

            # --alpha-gate emb appends the two blocks the gate is weighing.
            # alpha_fusion's emb mode refused the ADAPTED vector because it
            # moves every step and a scaler fitted once on it goes stale; that
            # does not apply here, h having been frozen a stage ago. Cost: the
            # gate goes from 5 parameters to 261.
            def _gm(Fs_block, Mblock):
                return (Mblock if args.alpha_gate == 'scalars'
                        else np.hstack([Fs_block, Mblock]))

            A_tr, A_te = Fs_tr, Fs_te
            gm_tr, gm_te = _gm(A_tr, M[tr]), _gm(A_te, M[te])
            scg = StandardScaler().fit(gm_tr)

            # Inner-split packs built ONCE and shared by every fusion row, so
            # they select on identical splits. Every scaler is refit inside
            # the split, so no selection is read off statistics its own
            # validation rows helped form.
            packs_v = []
            for i_tr, i_va in inner_splits:
                sc_i = StandardScaler().fit(F_tr[i_tr])
                idx = np.concatenate([i_tr, i_va])
                Fi = sc_i.transform(F_tr[idx])
                Gi_raw = _gm(Fi, M[tr][idx])
                scg_i = StandardScaler().fit(Gi_raw[:len(i_tr)])
                yi = y[tr][idx]
                vi = np.arange(len(i_tr), len(idx))
                if len(np.unique(yi[vi])) < 2:
                    continue
                packs_v.append({'F': Fi, 'G': scg_i.transform(Gi_raw),
                                'y': yi, 'vi': vi, 'ntr': len(i_tr)})

            # ── row: alpha as a HYPERPARAMETER ──
            # alpha and the decay are searched TOGETHER. Choosing the decay
            # after fixing alpha would pick a penalty for a mixture the
            # penalty itself changes the best mixture for. alpha=0 reproduces
            # N_probe_on_128 coordinate for coordinate, so that end of the
            # grid is this row's own control.
            rows_x = []
            for a1 in ALPHA_GRID:
                for wx, lrx in wd_lr_candidates(args):
                    vas, ee = [], []
                    for pk in packs_v:
                        mkX = (lambda a=a1: AlphaOnFrozen(dim, 1, a))
                        _, va, ep_i = gf._train_once(
                            T(pk['F'][:, :dim]), T(pk['F'][:, dim:]),
                            T(pk['G']),
                            T(pk['y'].astype(np.float64)), pk['y'],
                            np.zeros(len(pk['y']), dtype=np.int64),
                            np.arange(pk['ntr']), pk['vi'], [0],
                            dim, pk['G'].shape[1], 0, seed, wx, lrx, True,
                            wd_head=wx, n_epochs=args.epochs,
                            return_epoch=True, eval_every=args.eval_every,
                            make_model=mkX, make_groups=gate_head_groups,
                        )
                        if va is not None:
                            vas.append(va)
                            ee.append(ep_i)
                    if vas:
                        rows_x.append({'mean': float(np.mean(vas)),
                                       'epoch': int(np.median(ee)),
                                       'a': a1, 'wd': wx, 'lr': lrx})
            if rows_x:
                rows_x.sort(key=lambda r: -r['mean'])
                bx = rows_x[0]
                a_sel, wdx, lrx, epx = bx['a'], bx['wd'], bx['lr'], bx['epoch']
            else:
                # Too few positives to select. alpha is forced to 0, which
                # makes this fold pure sEHR -- counted, because otherwise it
                # is indistinguishable from the inner CV declining PPG.
                n_fix_nosel += 1
                a_sel, wdx, lrx, epx = (0.0, gf.DEFAULT_WD, gf.DEFAULT_LR,
                                        args.epochs)
            CXs.append(wdx)
            sv_fix.append((a_sel, wdx, lrx, epx))
            mkX = (lambda a=a_sel: AlphaOnFrozen(dim, 1, a))
            mdx, _ = gf._train_once(
                T(A_tr[:, :dim]), T(A_tr[:, dim:]), T(np.zeros((len(tr), 1))),
                T(y[tr].astype(np.float64)), y[tr],
                np.zeros(len(tr), dtype=np.int64), np.arange(len(tr)),
                None, [], dim, 1, 0, seed, wdx, lrx, True,
                wd_head=wdx, n_epochs=epx, make_model=mkX,
                make_groups=gate_head_groups,
            )
            with torch.no_grad():
                x_te, _ = mdx(T(A_te[:, :dim]), T(A_te[:, dim:]),
                              T(np.zeros((len(te), 1))))
            x_te = x_te.numpy()
            a_fix.append(a_sel)
            oof_x[te] = x_te
            auc_x.append(roc_auc_score(y[te], x_te))
            ap_x.append(average_precision_score(y[te], x_te))
            ce_x.append(1.0 - a_sel)
            cp_x.append(a_sel)

            # ── rows: the three GATED forms ──
            # One loop, three modes, so they share the fold, the packs, the
            # gate input, the grid and the refit path, and differ ONLY in the
            # fusion form. See AlphaOnFrozen for what each one is.
            for mode, acc in (('convex', G[0]), ('free', G[1]),
                              ('residual', G[2])):
                rows_a = []
                for _, wh, lr, wp in cands:
                    vas, ee = [], []
                    for pk in packs_v:
                        mkA = (lambda gd=pk['G'].shape[1], md=mode:
                               AlphaOnFrozen(dim, gd, mode=md))
                        _, va, ep = gf._train_once(
                            T(pk['F'][:, :dim]), T(pk['F'][:, dim:]),
                            T(pk['G']),
                            T(pk['y'].astype(np.float64)), pk['y'],
                            np.zeros(len(pk['y']), dtype=np.int64),
                            np.arange(pk['ntr']), pk['vi'], [0],
                            dim, pk['G'].shape[1], 0, seed, wp, lr, True,
                            wd_head=wh, n_epochs=args.epochs,
                            return_epoch=True, eval_every=args.eval_every,
                            make_model=mkA, make_groups=gate_head_groups,
                        )
                        if va is not None:
                            vas.append(va)
                            ee.append(ep)
                    if vas:
                        rows_a.append({'mean': float(np.mean(vas)),
                                       'epoch': int(np.median(ee)),
                                       'wh': wh, 'lr': lr, 'wp': wp})
                if rows_a:
                    rows_a.sort(key=lambda r: -r['mean'])
                    ba = rows_a[0]
                    wha, lra, epa, wpa = (ba['wh'], ba['lr'], ba['epoch'],
                                          ba['wp'])
                else:
                    wha, lra, epa, wpa = (gf.DEFAULT_WD, gf.DEFAULT_LR,
                                          args.epochs, args.wd_proj[-1])
                acc['sel'].append((wpa, wha, lra, epa))
                mkA = (lambda md=mode: AlphaOnFrozen(dim, gm_tr.shape[1],
                                                     mode=md))
                mdl, _ = gf._train_once(
                    T(A_tr[:, :dim]), T(A_tr[:, dim:]),
                    T(scg.transform(gm_tr)),
                    T(y[tr].astype(np.float64)), y[tr],
                    np.zeros(len(tr), dtype=np.int64), np.arange(len(tr)),
                    None, [], dim, gm_tr.shape[1], 0, seed, wpa, lra, True,
                    wd_head=wha, n_epochs=epa, make_model=mkA,
                    make_groups=gate_head_groups,
                )
                with torch.no_grad():
                    sa, cf = mdl(T(A_te[:, :dim]), T(A_te[:, dim:]),
                                 T(scg.transform(gm_te)))
                sa, cf = sa.numpy(), cf.numpy()
                acc['oof'][te] = sa
                acc['auc'].append(roc_auc_score(y[te], sa))
                acc['ap'].append(average_precision_score(y[te], sa))
                # cf is (n,2) = [sEHR coefficient | PPG coefficient] for every
                # form, so the table columns mean one thing across all rows.
                acc['ce'].append(float(cf[:, 0].mean()))
                acc['cp'].append(float(cf[:, 1].mean()))
                acc['sd'].append(float(cf[:, 1].std()))

    def pack(tag, auc, ap, extra=None, oof_arr=None):
        r = {
            # Popped in main before the DataFrame is built. Passed in rather
            # than looked up by tag so a row can never silently inherit
            # another row's predictions -- the same reason sel_desc is passed
            # explicitly through extra.
            '_oof': oof_arr,
            'probe': tag, 'setting': tag,
            'auroc_mean': float(np.mean(auc)) if auc else np.nan,
            'auroc_std_fold': float(np.std(auc)) if auc else np.nan,
            'auprc_mean': float(np.mean(ap)) if ap else np.nan,
            'auprc_std_fold': float(np.std(ap)) if ap else np.nan,
            'auroc_per_fold': af._pf(auc), 'auprc_per_fold': af._pf(ap),
            # No gate and no mixture in either row. NaN, not 0.0: a printed
            # 0.000 would read as "alpha was selected and came out zero".
            'alpha_mean': np.nan, 'alpha_std_within': np.nan,
            'alpha_std_across': np.nan,
            'n_valid_folds': len(auc), 'folds_without_selection': n_no_sel,
            'sel_wd_head': ','.join(f'{b:g}' for b, _ in sel),
            'sel_lr': ','.join(f'{c:g}' for _, c in sel),
            'sel_wd_proj': ('' if frozen else
                            ','.join(f'{v:g}' for v in was)),
            'sel_epoch': ','.join(str(e) for e in eps),
            'w_proj_norm': float(np.mean(wn)) if wn else np.nan,
            'w_head_norm': float(np.mean(hn)) if hn else np.nan,
            'h_norm': float(np.mean(hnorm)) if hnorm else np.nan,
            'h_rownorm_mean': float(np.mean(hrn)) if hrn else np.nan,
            'h_rownorm_sd': float(np.mean(hrs)) if hrs else np.nan,
            'zp_rownorm_mean': float(np.mean(prn)) if prn else np.nan,
            'zp_rownorm_sd': float(np.mean(prs)) if prs else np.nan,
            'wd_median': np.nan, 'alpha_per_fold': '',
            'coef_ehr_per_fold': '', 'coef_ppg_per_fold': '',
            'coef_ehr_mean': np.nan, 'coef_ppg_mean': np.nan,
            # Describes the MLP / frozen-control row, the only one that writes
            # sel/eps/was. Every other row OVERRIDES this through extra with
            # its own -- silently inheriting it is the bug this replaced.
            'sel_desc': sel_desc(
                **({'wd': [q[1] for q in sv_mlp]} if frozen else
                   {'wd_proj': [q[0] for q in sv_mlp],
                    'wd_head': [q[1] for q in sv_mlp]}),
                lr=[q[2] for q in sv_mlp], epoch=[q[3] for q in sv_mlp]),
            'folds_alpha_defaulted': 0,
        }
        r.update(extra or {})
        return r

    if frozen:
        return [pack('N_randproj_frozen', auc_e, ap_e, oof_arr=oof_e)]
    def coef(ce, cp):
        """The twelve coefficient fields every fusion row fills.

        ce is the sEHR-side coefficient per fold, cp the PPG-side. Every
        fusion form has both, and they differ only in how they are
        constrained -- convex sums to one, residual pins ce at 1, free
        constrains neither -- so reporting them the same way puts the
        constraint on the page instead of in the row's name.
        """
        return {'coef_ehr_per_fold': ','.join(f'{v:.4f}' for v in ce),
                'coef_ppg_per_fold': ','.join(f'{v:.4f}' for v in cp),
                'coef_ehr_mean': float(np.mean(ce)) if ce else np.nan,
                'coef_ppg_mean': float(np.mean(cp)) if cp else np.nan}

    def gate_row(tag, acc):
        a = acc
        return pack(tag, a['auc'], a['ap'], oof_arr=a['oof'], extra={
            # alpha_mean is the PPG coefficient's per-fold mean, averaged.
            # For the free form it is beta_ppg, which is NOT a fraction and
            # may leave [0,1]; the coef_* columns are where it should be read.
            'alpha_mean': float(np.mean(a['cp'])) if a['cp'] else np.nan,
            # WITHIN-fold spread: how much the gate moves the PPG
            # coefficient from patient to patient.
            'alpha_std_within': float(np.mean(a['sd'])) if a['sd'] else np.nan,
            # ACROSS-fold spread of the per-fold means: whether the inner CV
            # identifies the mixing weight at all. In every earlier run of
            # this project it did not.
            'alpha_std_across': float(np.std(a['cp'])) if a['cp'] else np.nan,
            'sel_desc': sel_desc(wd_gate=[q[0] for q in a['sel']],
                                 wd_head=[q[1] for q in a['sel']],
                                 lr=[q[2] for q in a['sel']],
                                 epoch=[q[3] for q in a['sel']]),
            'sel_wd_proj': ','.join(f'{q[0]:g}' for q in a['sel']),
            'sel_wd_head': ','.join(f'{q[1]:g}' for q in a['sel']),
            'sel_lr': ','.join(f'{q[2]:g}' for q in a['sel']),
            'sel_epoch': ','.join(str(q[3]) for q in a['sel']),
            **coef(a['ce'], a['cp'])})

    return [pack('N_mlp_320_128_1', auc_e, ap_e, oof_arr=oof_e),
            pack('N_probe_on_128', auc_p, ap_p, oof_arr=oof_p, extra=
                 {'wd_median': float(np.median(Cs)) if Cs else np.nan,
                  'sel_desc': sel_desc(wd=[q[0] for q in sv_probe],
                                       lr=[q[1] for q in sv_probe],
                                       epoch=[q[2] for q in sv_probe]),
                  'sel_wd_head': ','.join(f'{q[0]:g}' for q in sv_probe),
                  'sel_lr': ','.join(f'{q[1]:g}' for q in sv_probe),
                  'sel_wd_proj': '',
                  'sel_epoch': ','.join(str(q[2]) for q in sv_probe)}),
            pack('N_fixed_alpha', auc_x, ap_x, oof_arr=oof_x, extra=
                 {'alpha_mean': float(np.mean(a_fix)) if a_fix else np.nan,
                  # A frozen alpha has no within-fold spread BY CONSTRUCTION,
                  # so the number that carries information is the ACROSS-fold
                  # spread of the SELECTED value.
                  'alpha_std_within': 0.0,
                  'alpha_std_across': (float(np.std(a_fix)) if a_fix
                                       else np.nan),
                  'alpha_per_fold': ','.join(f'{x:g}' for x in a_fix),
                  'wd_median': float(np.median(CXs)) if CXs else np.nan,
                  'sel_desc': sel_desc(alpha=[q[0] for q in sv_fix],
                                       wd=[q[1] for q in sv_fix],
                                       lr=[q[2] for q in sv_fix],
                                       epoch=[q[3] for q in sv_fix]),
                  'sel_wd_head': ','.join(f'{q[1]:g}' for q in sv_fix),
                  'sel_lr': ','.join(f'{q[2]:g}' for q in sv_fix),
                  'sel_wd_proj': '',
                  'sel_epoch': ','.join(str(q[3]) for q in sv_fix),
                  'folds_alpha_defaulted': n_fix_nosel,
                  **coef(ce_x, cp_x)}),
            *[gate_row(t, a) for t, a in zip(G_TAGS, G)]]


# A320_sEHR_pre_selC FIRST: it is the comparison baseline, the only row in
# the table whose hyperparameters were chosen inside the nested CV on the
# same budget the method rows get. A320_untuned_default answers a different
# question -- what an untuned model does -- and anchors nothing.
ROW_ORDER = ['A320_sEHR_pre_selC', 'A320_untuned_default',
             'N_mlp_320_128_1', 'N_randproj_frozen',
             'N_probe_on_128',
             'PPG512_pre', 'PPG128_post',
             'N_fixed_alpha',
             'N_gate_alpha', 'N_gate_beta_free', 'N_gate_residual']


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--features', default=FEATURES_PATH)
    ap_.add_argument('--metadata', default=METADATA_CSV)
    ap_.add_argument('--out-dir', default=reg.OUT_DIR)
    ap_.add_argument('--disease', choices=list(DISEASES) + ['all'],
                     default='all')
    ap_.add_argument('--gate-input', choices=['scalars', 'emb'],
                     default='scalars',
                     help=('unused -- neither row has a gate. Kept only so '
                           'af._adapter_pack can be called with the same '
                           'signature the M_ and H_ rows use.'))
    # These three default to the CONFIGURATION THE OTHER ROWS ARE ON, not to
    # None. In alpha_fusion and linear_fusion they default off to preserve
    # older behaviour; this file has no older behaviour to preserve, and
    # having to remember four flags is exactly how a row ends up on a
    # different configuration from the rows it is meant to sit beside.
    ap_.add_argument('--wd-proj', default='0.01,0.1,1,10,30,100',
                     help='decays for the 320->128 weights (SEPARATE from '
                          'wd_head, which comes from alpha_fusion '
                          'WD_HEAD_GRID and is searched jointly with this).')
    ap_.add_argument('--lr', default='0.001,0.003,0.01,0.03',
                     help=('comma-separated rates, SEARCHED jointly with the '
                           'two decays. A single value pins it, which is what '
                           'this defaulted to before: 0.003, chosen by hand '
                           'because rho = lr*wd caps the reachable decay and '
                           '0.003 is what lets the grid reach 100. That was a '
                           'hand-picked value defended by a cross-run '
                           'comparison of outer-fold results (AF came out '
                           '-0.014 at 0.01 and -0.007 at 0.003), which is '
                           'selection on the test folds and cannot go in a '
                           'methods section. Searching it puts the choice '
                           'back inside the nested CV. 0.001 is included '
                           'because every signal so far points DOWN -- the '
                           'decays pin to the top of their grid and the '
                           'early-stopped epoch sits at 3-9 out of 300 -- so '
                           'the informative direction is a smaller step, not '
                           'a larger one. NOTE the grid is ragged by '
                           'construction: rho caps wd at 300/100/30/10 as lr '
                           'goes 0.001/0.003/0.01/0.03, so which decays are '
                           'reachable depends on which rate is being tried. '
                           'That is unavoidable when both are searched, and '
                           'the reachable set is printed at startup.'))
    ap_.add_argument('--lr-adapter', type=float, default=None,
                     help=('separate lr for the 320->128 group. Defaults to '
                           'None, so BOTH layers train at --lr. The 0.003 '
                           'the adapter rows use exists for reasons that are '
                           'specific to a residual -- reaching an '
                           '"encoder almost off" state that falls back on a '
                           'frozen base, and getting under Adam\'s '
                           'lr*sqrt(n) floor to do it. There is no frozen '
                           'base here and no off state, so two plain layers '
                           'of one MLP train at one rate, and both axes see '
                           'the same rho cap, so neither stops short of the '
                           'other at any rate in the grid.'))
    ap_.add_argument('--alpha-gate', choices=['scalars', 'emb'],
                     default='scalars',
                     help=('what the N_gate_alpha rows\' gate reads. '
                           'scalars: the 4 SCALAR_FIELDS only, 5 gate '
                           'parameters. emb: those plus the two frozen 128-d '
                           'blocks the gate is weighing, 261 parameters '
                           'against 52-155 positives. emb is worth running to '
                           'show that more gate input does not help, not '
                           'because it is expected to win.'))
    ap_.add_argument('--alpha-points', type=int, default=21,
                     help=('how many uniform points on [0,1] the fixed-alpha '
                           'rows search. 21 (step 0.05) matches '
                           'alpha_fusion.ADAPTER_ALPHA_GRID so the selected '
                           'values are directly comparable. Each point costs '
                           'one full (wd, lr) sweep on every inner fold, so '
                           'this is the knob to turn if the run does not fit '
                           'the wall clock -- 11 points is step 0.1.'))
    ap_.add_argument('--wd-a320', type=float, default=1e-2,
                     help=("torch.optim.AdamW's OWN default weight decay. "
                           'This row answers one question -- what do you get '
                           'without tuning -- so the value has to be the '
                           'library default or it answers a different one. It '
                           'was 30 before, defended by lambda = 1/(2Cn) at '
                           'DEFAULT_C and by 30 being the modal decay the '
                           'searched rows picked. The second is selection on '
                           'outer-fold results; the first traces back to '
                           'DEFAULT_C, itself set as the "plateau of the '
                           'table above". Both are hindsight. The library '
                           'default is the only value nobody chose by looking '
                           'at this data, and its being weak is the answer, '
                           'not a problem with it.'))
    ap_.add_argument('--lr-a320', type=float, default=1e-3,
                     help=("torch.optim.AdamW's OWN default learning rate. "
                           'rho = lr*wd = 1e-5, far under MAX_RHO. The row '
                           'runs the full epoch budget with no inner CV and '
                           'no early stopping: untuned means untuned. It is '
                           'NOT the comparison baseline -- vs points at '
                           'A320_sEHR_pre_selC, which is searched on the same '
                           'budget every method row gets.'))
    ap_.add_argument('--wd-mode', choices=['shared', 'split'],
                     default='shared',
                     help=('shared: ONE decay for both layers of '
                           'N_mlp_320_128_1, searched over --wd-proj. split: '
                           'the projection and the head get separate decays '
                           'searched jointly, which is what this did before. '
                           'Shared is the default because the two layers are '
                           'not regularising different objects in a plain '
                           'MLP, and because the split was not identifiable '
                           '-- one cell selected wd_proj across four orders '
                           'of magnitude in its five folds. Only the trained '
                           'row is affected: the frozen control has one '
                           'decay either way, and the probe rows inherit '
                           'whatever it selected.'))
    ap_.add_argument('--epochs', type=int, default=300)
    ap_.add_argument('--eval-every', type=int, default=3)
    args = ap_.parse_args()
    args.wd_proj = [float(v) for v in str(args.wd_proj).split(',') if v]
    global ALPHA_GRID
    ALPHA_GRID = _alpha_grid(int(args.alpha_points))
    args.lr = [float(v) for v in str(args.lr).split(',') if v]
    af.LR_GRID_E = list(args.lr)
    # This file's decay grid, not alpha_fusion's. Rebinding af.WD_HEAD_GRID
    # for this process is the same mechanism the line above uses on
    # LR_GRID_E: adapter_candidates resolves both module globals at call
    # time, and nothing else runs in this process. Kept local because the top
    # value only clears the rho guard at lr <= 0.003, and alpha_fusion and
    # linear_fusion run at 0.01, where it would be unreachable for their head
    # and reachable for their encoder -- the split-axis asymmetry a shared
    # grid exists to prevent. Their grids are theirs to change, in a run
    # aimed at them.
    af.WD_HEAD_GRID = list(args.wd_proj)
    af.WD_GATE_GRID = af.WD_HEAD_GRID

    print('=' * 100)
    print('PLAIN 320 -> 128 -> 1, NO NORMALISATION')
    print('=' * 100)

    feat, df = load_joined(args.features, args.metadata)
    both = df[(df['ppg_mask'] == 1)
              & (df['ehr_mask'] == 1)].reset_index(drop=True)
    idx = both['_idx'].to_numpy()
    VE_pre = np.stack([feat[i]['v_ehr_pre_mlp'] for i in idx]).astype(np.float64)
    VE_post = np.stack([feat[i]['v_ehr_post_mlp'] for i in idx]).astype(np.float64)
    VP = np.stack([feat[i]['v_ppg_post_mlp'] for i in idx]).astype(np.float64)
    # The raw 512-d PPG embedding, before the alignment projection head.
    # Averaged over the admission's blocks and NOT renormalised, unlike
    # v_ppg_post -- which matters nowhere here, since this only ever feeds a
    # probe with a scaler in front of it.
    VP_pre = np.stack([feat[i]['v_ppg_pre_mlp'] for i in idx]).astype(
        np.float64)
    M = build_scalar_matrix(both)
    groups = both['patient_id'].astype(str).to_numpy()
    args.pre_dim = VE_pre.shape[1]

    print(f"  both-modality cohort: {len(both)} admissions, "
          f"{both['patient_id'].nunique()} patients")
    print(f"  N_mlp_320_128_1 : Linear({args.pre_dim},{N_DIM}) -> "
          f"Linear({N_DIM},1), {args.pre_dim * N_DIM + N_DIM} projection "
          f"weights, NO normalisation, NO PPG")
    print("  N_randproj_frozen: the SAME architecture with P left at its "
          "random init, only the head trains -- did training the "
          "projection do anything?")
    print(f"  N_probe_on_128  : that model's {N_DIM}-d layer frozen, "
          f"StandardScaler -> Linear({N_DIM},1), wd and lr by inner "
          f"{af.N_INNER_E}-fold")
    # lr_adapter=None means the projection group takes --lr, which is the
    # default here; print the EFFECTIVE rate rather than the flag, so the log
    # says what the run did and not what was typed.
    if args.wd_mode == 'shared':
        print(f"  ONE shared wd {args.wd_proj} for both layers  x  "
              f"lr {af.LR_GRID_E}")
    else:
        print(f"  wd_proj {args.wd_proj}  x  wd_head {af.WD_HEAD_GRID}  x  "
              f"lr {af.LR_GRID_E}")
    # The grid is ragged whenever more than one rate is searched. Print the
    # reachable set rather than leaving it to be inferred from MAX_RHO.
    print(f"  rho = lr*wd capped at {af.MAX_RHO}, so reachable decays are:")
    for _lr in af.LR_GRID_E:
        _ok = [w for w in args.wd_proj if _lr * w <= af.MAX_RHO]
        print(f"      lr {_lr:<8g} wd {_ok}")
    if args.lr_adapter is not None:
        print(f"  projection group overrides lr to {args.lr_adapter:g}")
    if args.lr_adapter is None:
        print("  both layers train at one rate; --lr-adapter not given")
    print("  PPG512_pre / PPG128_post: the same probe on the PPG side, raw "
          "512-d and aligned 128-d.")
    print("                        Single-modality references; their vs "
          "column is PPG against sEHR, cross-modality.")
    print("  A320_sEHR_pre_selC  : THE BASELINE. Linear(320,1), wd and lr "
          "SEARCHED inside the nested CV on the")
    print("                        same budget every method row gets -- the "
          "only row here with no hand-picked value.")
    print(f"  A320_untuned_default: the same probe at torch.optim.AdamW's OWN "
          f"defaults (wd={args.wd_a320:g}, lr={args.lr_a320:g}),")
    print(f"                        full {args.epochs}-epoch budget, no "
          f"selection. Answers 'what without tuning'. Anchors nothing.")
    print(f"  candidates: {len(candidates(args))} for the trained row, "
          f"{len(candidates(args, frozen=True))} for the frozen control, "
          f"{len(wd_lr_candidates(args))} for each linear probe,")
    print(f"              {len(ALPHA_GRID) * len(wd_lr_candidates(args))} "
          f"for each fixed-alpha row; each on {af.N_INNER_E} inner folds")
    print()
    print(f"  N_fixed_alpha       : (1-a)*h + a*zp, a a HYPERPARAMETER, "
          f"{len(ALPHA_GRID)} points 0..1 searched jointly with C.")
    print("                        a=0 is N_probe_on_128 exactly, so that "
          "end of the grid is its own control.")
    print("  N_gate_alpha        : same blocks, a_i = sigmoid(g . gate "
          "input), per sample; only gate and head train.")
    print(f"                        gate reads "
          f"{len(gf.SCALAR_FIELDS)} scalars {gf.SCALAR_FIELDS}"
          f"{' plus both 128-d blocks' if args.alpha_gate == 'emb' else ''}.")
    print("  N_gate_beta_free    : N_gate_alpha with the GATE changed only "
          "-- Linear(gate_in, 2), no activation,")
    print("                        so (b1,b2) = g.x and fused = b1*h + "
          "b2*zp. Per patient, unconstrained.")
    print("                        Strictly CONTAINS the convex mixture, so "
          "a loss here is selection noise;")
    print("                        a gain says sum-to-one or non-negativity "
          "was binding.")
    print("  N_gate_residual     : N_gate_alpha with the FUSION LINE changed "
          "only -- h + a*zp, sEHR uncoefficiented.")
    print("                        At a FIXED alpha this is the convex form "
          "reparametrised; per-sample it is NOT,")
    print("                        because the convex form's leading (1-a(x)) "
          "rescales the whole vector per patient.")
    print("                        The residual caps the PPG:sEHR ratio at "
          "1; the convex form's a/(1-a) is unbounded.")
    print("  N_probe_on_128 is NOT comparable to A_sEHR_pca128 as a "
          "statement about a representation:")
    print("  its 128 dims were chosen BY fitting the downstream label. "
          "See the module docstring.")
    print()

    diseases = DISEASES if args.disease == 'all' else (args.disease,)
    rows = []
    oof_records = []
    for disease in diseases:
        print('#' * 100)
        print(f'DISEASE: {disease}')
        print('#' * 100)
        fold_of = assign_folds(both, disease)
        for horizon in HORIZONS:
            s = pd.to_numeric(both[label_col(disease, horizon)],
                              errors='coerce')
            usable = s.notna().to_numpy()
            y = s[usable].to_numpy(dtype=np.int64)
            g_h = groups[usable]
            folds_h = subset_folds(fold_of, usable)
            E_h, Ze_h, P_h = (VE_pre[usable], VE_post[usable], VP[usable])
            Ppre_h = VP_pre[usable]
            M_h = M[usable]
            # Same mask, same order as y -- the OOF table needs the full
            # (dataset, patient_id, hadm_id) key, not just the patient, so a
            # row can be joined back to the label and metadata files.
            key_h = both.loc[usable, ['dataset', 'patient_id',
                                      'hadm_id']].reset_index(drop=True)
            fold_h = fold_of[usable]
            print(f"  {horizon}d usable={len(y)}, pos={int(y.sum())}, "
                  f"censored={int((~usable).sum())}")

            def finish(r):
                """Attach pooled OOF + bootstrap CI, and bank the predictions.

                Applied to EVERY row before it is appended, so no row can
                reach the CSV carrying fold metrics but no pooled ones.
                """
                arr = r.pop('_oof', None)
                if arr is None:
                    return r
                r.update(pooled_oof_ci(y, arr, g_h))
                ok = np.isfinite(arr)
                if ok.any():
                    k = key_h[ok]
                    oof_records.append(pd.DataFrame({
                        'disease': disease, 'horizon': horizon,
                        'setting': r.get('setting') or r.get('probe'),
                        'dataset': k['dataset'].to_numpy(),
                        'patient_id': k['patient_id'].to_numpy(),
                        'hadm_id': k['hadm_id'].to_numpy(),
                        'fold': fold_h[ok],
                        'y_true': y[ok],
                        'y_pred': arr[ok],
                    }))
                return r

            r = run_torch_baseline(E_h, y, g_h, folds_h, args,
                                    'A320_untuned_default',
                                    fixed=(args.wd_a320, args.lr_a320)) # yeogi

            # The same probe with its decay SELECTED. Kept beside the pinned
            # row rather than replacing it: a pinned value makes the reference
            # stable across runs, a selected one makes it the strongest the
            # baseline can be, and only the second is a fair thing to measure
            # a method against. The gap between the two rows says how much of
            # any "beats A320" is the baseline being handicapped -- on the
            # sklearn run that gap was 0.017.
            r2 = run_torch_baseline(E_h, y, g_h, folds_h, args,
                                    'A320_sEHR_pre_selC')
            r2.update(setting='A320_sEHR_pre_selC', disease=disease,
                      horizon=horizon, n_samples=len(y),
                      n_pos=int(y.sum()))
            rows.append(finish(r2))
            r.update(setting='A320_untuned_default', disease=disease,
                     horizon=horizon, n_samples=len(y), n_pos=int(y.sum()))
            rows.append(finish(r))

            # Single-modality PPG references, in exactly the form
            # A320_sEHR_pre_selC takes: StandardScaler -> Linear(d,1) ->
            # AdamW, decay and rate searched inside the nested CV on the same
            # 21-candidate budget. Only the input differs.
            #
            # Both, because they answer different questions. PPG128_post is
            # the block the fusion rows actually mix, so it is the reference
            # for "what does the PPG side bring on its own". PPG512_pre is
            # what was in the PPG embedding BEFORE the alignment projection,
            # so post - pre is what that projection did to the PPG side --
            # the strongest signal on record in this project (+0.022, 11/12).
            #
            # Their vs column is a CROSS-MODALITY difference, PPG against the
            # sEHR baseline, not "method against its own control" as it is
            # for every other row. Left in rather than blanked: the number is
            # real and blanking it would hide it.
            for _X, _t in ((Ppre_h, 'PPG512_pre'), (P_h, 'PPG128_post')):
                rp = run_torch_baseline(_X, y, g_h, folds_h, args, _t)
                rp.update(setting=_t, disease=disease, horizon=horizon,
                          n_samples=len(y), n_pos=int(y.sum()))
                rows.append(finish(rp))

            Xa = np.hstack([E_h, Ze_h, P_h])
            for fz in (False, True):
                for r in run_plain(Xa, M_h, y, g_h, folds_h,
                                   args.gate_input, args, frozen=fz):
                    r.update(disease=disease, horizon=horizon,
                             n_samples=len(y), n_pos=int(y.sum()))
                    rows.append(finish(r))

    res = pd.DataFrame(rows)

    print()
    print('=' * 160)
    print('RESULTS')
    print('=' * 160)
    # Twelve coefficient columns: the sEHR-side coefficient per fold and its
    # mean, then the PPG-side. Every fusion row fills all twelve and they
    # differ only in what constrains them -- convex sums to one across each
    # column pair, residual pins e at 1.00, free constrains neither -- so the
    # constraint is visible instead of being something you have to know.
    print(f"  {'cell':<14}{'row':<22}{'AUROC':>16}{'AUPRC':>16}"
          f"{'vs A320':>9}   "
          + ''.join(f"{'e_f'+str(k+1):>7}" for k in range(5))
          + f"{'e_avg':>8}  "
          + ''.join(f"{'p_f'+str(k+1):>7}" for k in range(5))
          + f"{'p_avg':>8}")
    for disease in diseases:
        for horizon in HORIZONS:
            sub = res[(res.disease == disease) & (res.horizon == horizon)]
            base = sub[sub.setting == 'A320_sEHR_pre_selC']
            b = float(base['auroc_mean'].iloc[0]) if len(base) else np.nan
            for tag in ROW_ORDER:
                m = sub[sub.setting == tag]
                if not len(m):
                    continue
                r = m.iloc[0]
                d = ('' if tag == 'A320_sEHR_pre_selC'
                     else f"{r['auroc_mean'] - b:+.3f}")
                def _cols(key, mean_key):
                    raw = r.get(key, '')
                    if not isinstance(raw, str):
                        raw = ''      # a missing column arrives as NaN
                    vals = [v for v in raw.split(',') if v]
                    cells = ''.join(f"{float(v):>7.2f}" for v in vals[:5])
                    cells += ' ' * (7 * (5 - len(vals[:5])))
                    mu = r.get(mean_key, np.nan)
                    return cells + (f"{mu:>8.2f}" if np.isfinite(mu)
                                    else ' ' * 8)
                print(f"  {disease:<8}{horizon:>4}d "
                      f"{tag:<22}"
                      f"{r['auroc_mean']:>9.3f}+-{r['auroc_std_fold']:.3f}"
                      f"{r['auprc_mean']:>9.3f}+-{r['auprc_std_fold']:.3f}"
                      f"{d:>9}   "
                      + _cols('coef_ehr_per_fold', 'coef_ehr_mean') + '  '
                      + _cols('coef_ppg_per_fold', 'coef_ppg_mean'))
            # POOLED OOF + PATIENT-BOOTSTRAP CI. A separate block, not extra
            # columns on the table above: the table's format is what every
            # earlier log used and what the audit file quotes, and widening
            # it would make the two un-diffable.
            print("      pooled OOF (all 5 outer folds concatenated, ONE "
                  "metric) with patient-cluster 95% CI:")
            print(f"        {'row':<22s}{'AUROC_oof':>10s}"
                  f"{'95% CI':>18s}{'AUPRC_oof':>11s}{'95% CI':>18s}"
                  f"{'n':>7s}{'boot':>10s}")
            for tag in ROW_ORDER:
                m_ = sub[sub.setting == tag]
                if not len(m_):
                    continue
                q = m_.iloc[0]
                if not np.isfinite(q.get('auroc_oof', np.nan)):
                    continue
                nb = int(q.get('n_boot_valid', 0) or 0)
                nd = int(q.get('n_boot_draws', 0) or 0)
                # nd > nb means replicates were rejected for being
                # single-class; the accepted set is then conditioned on
                # containing a positive, which narrows the interval.
                bt = f"{nb}" + (f"/{nd}" if nd > nb else "")
                # Built outside the f-string: this file runs on Python 3.10
                # on the cluster, where a replacement field cannot span lines.
                ci_auc = (f"({q['auroc_ci_low']:.3f}-"
                          f"{q['auroc_ci_high']:.3f})")
                ci_ap = (f"({q['auprc_ci_low']:.3f}-"
                         f"{q['auprc_ci_high']:.3f})")
                print(f"        {tag:<22s}{q['auroc_oof']:>10.3f}"
                      f"{ci_auc:>18s}"
                      f"{q['auprc_oof']:>11.3f}"
                      f"{ci_ap:>18s}"
                      f"{int(q.get('n_oof', 0) or 0):>7d}{bt:>10s}")

            # WHAT EACH ROW SELECTED, per fold, printed by the row itself.
            # One block per cell listing all nine, because a single shared
            # line used to print the MLP's hyperparameters under every row.
            print("      selected per fold (5 outer folds, in order):")
            for tag in ROW_ORDER:
                m_ = sub[sub.setting == tag]
                if not len(m_):
                    continue
                d_s = str(m_.iloc[0].get('sel_desc', '') or '')
                print(f"        {tag:<22s} "
                      + (d_s if d_s else "(nothing to select)"))
            det = sub[sub.setting == 'N_mlp_320_128_1']
            if len(det):
                r = det.iloc[0]
                fx = sub[sub.setting == 'N_fixed_alpha']
                if len(fx) and int(fx.iloc[0].get(
                        'folds_alpha_defaulted', 0) or 0):
                    print(f"      !! "
                          f"{int(fx.iloc[0]['folds_alpha_defaulted'])} "
                          f"fold(s) forced alpha=0, too few positives")
                # The WITHIN-fold spread of a per-patient coefficient. The
                # five per-fold values are in the table above; this is the
                # other quantity -- how much the gate moves the PPG
                # coefficient BETWEEN patients inside a fold. Near zero means
                # the gate found a constant.
                for _t in ('N_gate_alpha', 'N_gate_beta_free',
                           'N_gate_residual'):
                    al = sub[sub.setting == _t]
                    if len(al) and np.isfinite(al.iloc[0]['alpha_std_within']):
                        print(f"      {_t:<20s} PPG coef: sd within-fold "
                              f"{al.iloc[0]['alpha_std_within']:.3f}, "
                              f"across folds "
                              f"{al.iloc[0]['alpha_std_across']:.3f}")
                un_ = sub[sub.setting == 'A320_untuned_default']
                if len(un_) and np.isfinite(b):
                    d_ = b - float(un_.iloc[0]['auroc_mean'])
                    print(f"      tuning the baseline is worth {d_:+.3f} "
                          f"over AdamW's own defaults (wd={args.wd_a320:g}, "
                          f"lr={args.lr_a320:g}); vs is against the TUNED "
                          f"one")
                if np.isfinite(r.get('h_rownorm_sd', np.nan)):
                    print(f"      after scaling, row norms:  h "
                          f"{r['h_rownorm_mean']:.2f}+-"
                          f"{r['h_rownorm_sd']:.2f}   zp "
                          f"{r['zp_rownorm_mean']:.2f}+-"
                          f"{r['zp_rownorm_sd']:.2f}   "
                          f"(sqrt(dim)={np.sqrt(N_DIM):.2f}; the two SDs are "
                          f"the question, not the means)")
                print(f"      ||W_proj|| {r['w_proj_norm']:.2f}  "
                      f"||w_head|| {r['w_head_norm']:.2f}  "
                      f"mean ||h|| {r['h_norm']:.2f}")
                if int(r.get('folds_without_selection', 0) or 0):
                    print(f"      !! {int(r['folds_without_selection'])} "
                          f"fold(s) ran with NO selection")
            ctl = sub[sub.setting == 'N_randproj_frozen']
            if len(ctl) and len(det):
                # sqrt(dim/3) is E||W|| for nn.Linear's kaiming_uniform init
                # at this shape. The control must sit ON it -- if it does
                # not, P moved and the row is not the control it claims.
                init = float(np.sqrt(N_DIM / 3.0))
                print(f"      randproj ||W_proj|| {ctl.iloc[0]['w_proj_norm']:.2f}"
                      f" (init {init:.2f}, must match)  vs trained "
                      f"{det.iloc[0]['w_proj_norm']:.2f}")
        print()

    # The gate mode goes in the FILENAME: without it a scalars run and an
    # emb run of the same disease write the same path, and submitting both
    # as one array has them racing on it.
    tag_out = (('' if args.alpha_gate == 'scalars' else '_embgate')
               + ('' if args.disease == 'all' else f'_{args.disease}'))
    os.makedirs(args.out_dir, exist_ok=True)
    f_res = f'plain_mlp_torch_postdisch_results{tag_out}.csv'
    res.to_csv(os.path.join(args.out_dir, f_res), index=False)
    print(f'Saved {f_res}')

    # The per-admission OOF predictions. Saved so that a paired delta-AUROC
    # between two rows, a calibration curve, or an error-case listing can be
    # produced later WITHOUT retraining -- all of those need the individual
    # predictions, and recomputing them costs a full nested CV.
    if oof_records:
        f_oof = f'plain_mlp_torch_postdisch_oof{tag_out}.csv'
        oof_df = pd.concat(oof_records, ignore_index=True)
        oof_df.to_csv(os.path.join(args.out_dir, f_oof), index=False)
        print(f'Saved {f_oof}  ({len(oof_df)} rows)')
    print()
    print('  vs A320  : against A320_sEHR_pre_selC, the TUNED 320-d probe. '
          'Did a supervised')
    print('             320->128 reach it? The prerequisite -- if N_mlp is '
          'below it, nothing')
    print('             built on it can attribute a later gain to anything '
          'but the handicap.')
    print('             NOT against A320_untuned_default: measuring a method '
          'against an')
    print('             untuned baseline inflates it, which is the whole '
          'reason that row')
    print('             uses the library default instead of a value picked '
          'with hindsight.')
    print('  N_mlp - N_randproj : did TRAINING the projection buy anything? '
          'Same')
    print('             architecture, same init, same folds; only whether P '
          'moves differs.')
    print('             A gap near zero says the trained projection is doing '
          'no more')
    print('             than a random one, which Johnson-Lindenstrauss says '
          'already')
    print('             preserves most of a linear signal at 320 -> 128.')
    print('  alpha - N_probe : the PPG increment. Identical sEHR features, '
          'identical')
    print('             folds; the PPG block and the gate are the only '
          'difference, so a')
    print('             gain is PPG plus whatever the mixture form buys, not '
          'PPG alone.')
    print('             Read the across-fold alpha spread first: if the '
          'five folds')
    print('             disagree, the mixing weight is not identified and '
          'its mean')
    print('             is not a quantity to report.')
    print('  N_probe - N_mlp : same 128 features, a linear head on them '
          'instead of')
    print('             the jointly-trained one. Both AdamW now, so a gap '
          'here is')
    print('             co-adaptation of the projection to its own head, '
          'not a')
    print('             framework difference.')
    print('  vs plain_mlp.py : identical rows, folds and grids, sklearn '
          'models')
    print('             swapped for torch ones. Row for row, that gap IS '
          'the')
    print('             framework effect.')


if __name__ == '__main__':
    main()
