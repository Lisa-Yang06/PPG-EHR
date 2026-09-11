"""
PER-HORIZON LEARNED-ALPHA EMBEDDING FUSION -- post-discharge cohort.

WHAT THIS EXISTS TO ANSWER
--------------------------
Does an embedding-level fusion of sEHR and PPG beat the plain sEHR probe on
the ORIGINAL 320-d BERT vector (0a_EHR_pre), when the two are given exactly
the same shot?

gated_fusion.py's setting 4 could not answer it. Two structural gaps:

  1  its classifier was a torch Linear trained at a fixed epoch budget, and
     4b -- the SAME gate, the SAME per-admission alpha, only the classifier
     swapped for the inner-CV L2 logistic that 0a uses -- beat it in every
     cell, by a wide margin. The head, not the representation, was binding.

  2  setting 4 fits ONE model across the four horizons while 0a fits four
     independent ones. A shared model also sees all four horizons' labels,
     so at 30d it trains on information 0a never gets. Not test leakage, but
     not the same contest either.

Here everything is per horizon, end to end: its own PCA, its own gate, its
own classifier, its own C. Nothing is shared across horizons and the gate has
no horizon input -- inside one horizon that input is constant, so it would
survive StandardScaler only as a zeroed column and a wasted parameter.

THE sEHR SIDE
-------------
PCA(128) of v_ehr_pre, block-scaled, fitted inside the fold. 0e matches 0a in
every cell, so this compression is free and the fusion is not being handed a
degraded sEHR side the way v_ehr_post is -- the aligned 128-d vector sits
below 0a_EHR_pre, worst on eMACE. Not whitened: whitening measurably cost
AUROC in every cell (see gated_fusion.py).

WHAT MAKES THE COMPARISON FAIR, AND THE ONE THING THAT STILL DIFFERS
--------------------------------------------------------------------
Shared with A320_sEHR_pre: the outer folds (assign_folds, per disease, patient
-grouped, 365d-status stratified), the per-horizon censoring, the classifier
family, C_GRID, and the inner 3-fold StratifiedGroupKFold that selects C.

Still different, and it cannot be removed: the gate is fitted by gradient
descent with its own single inner split for early stopping and (wd, lr)
selection, while every sklearn row selects C by 3-fold inner CV. Two
selection protocols, same outer folds. E's advantage is that it gets to see
the PPG block at all; that is the hypothesis, not a confound.

GATE SIZE IS THE BINDING CONSTRAINT, and it decides which variant can work
------------------------------------------------------------------------
Per horizon there are ~1480 training rows and 42-124 training positives.

    gate input          params    verdict
    4 scalars              5      fits comfortably
    ze + zp + scalars    261      261 parameters against ~42 positives at 30d

So --gate-input scalars is the variant with a real chance, and it is the one
to read. The emb variant is run as the control: the randmlp run already showed
setting 4 degrading in almost every cell while setting 3 -- which MIXES the
embedding but does not READ it -- improved in 8/12, i.e. the gate struggles to
learn from 128 dimensions with no semantic structure. This tests that again
under a budget where it is even less plausible.

THE ROWS
--------
    A320_sEHR_pre         v_ehr_pre (320d)          <- the target to beat (= 0a)
    A_sEHR_pca128         ze                        <- the mixture's alpha=0 end
    B_PPG_post            zp                        <- the mixture's alpha=1 end
    F_sEHR_plus_scalars   [ze | scalars]            <- same side info, no fusion
    C_concat              [ze | zp]                 <- the mixture UNRESTRICTED
    D_fixed_alpha         (1-a)ze + a*zp, a by CV   <- did LEARNING alpha help?
    E_learned_alpha       a_i = sigmoid(g(x_i))     <- the proposal
    E_sklearnhead         E's gate, E's alpha, sklearn classifier -- the
                          control that says whether one end-to-end scheme is
                          enough

C_concat is not decoration. The convex mixture is the concatenated probe with
the two coefficient vectors constrained PARALLEL:

    w . ((1-a) ze + a zp)  =  [ (1-a)w | a w ] . [ ze | zp ]

so every fixed mixture lies inside C_concat's hypothesis class. That nesting
is about the CLASS, not about the test AUROC -- C_concat carries 256 free
coefficients against 42-124 training positives, and the same function is
penalised differently in the two models (L2 costs ((1-a)^2 + a^2)||w||^2 there
against ||w||^2 here), so C_concat scoring BELOW D_fixed_alpha is entirely
possible and is itself the finding: the constraint is buying generalisation.
Read it the other way too -- if C_concat beats E, the recommendation is plain
concatenation and the gate buys nothing. Either way it has to be
measured before the gate is proposed, not after.

D_fixed_alpha is the control that isolates the LEARNING. E and D mix the same
two blocks with the same classifier; only "alpha per admission, by gradient"
versus "one alpha, by inner CV" differs.

E's stage-2 C is selected on fused training rows whose alpha came from a gate
early-stopped on a fraction of those same rows; the test fold is untouched.
--oof-alpha rotates the inner split so every training row gets an alpha from a
gate that never saw it, at k times the stage-1 cost. Off by default.

Usage:
    python alpha_fusion.py --disease AF
    python alpha_fusion.py --disease AF --gate-input emb
    python alpha_fusion.py --disease AF --oof-alpha
"""
import os

import argparse
import sys

import shutil

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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
import gated_fusion as gf

from gated_fusion import (
    DISEASES, HORIZONS, FEATURES_PATH, METADATA_CSV,
    label_col, load_joined, assign_folds, horizon_valid_folds, subset_folds,
    build_scalar_matrix, ConvexMixer,
)

N_COMPONENTS = 128
# D_fixed_alpha's convex grid. Kept identical to ADAPTER_ALPHA_GRID so that
# D and G_adapter_fixed_alpha differ ONLY in the sEHR side -- both are "convex
# mixture with alpha chosen by inner CV", and a 0.1 grid against a 0.05 one
# would put a second variable into that comparison. D is a sklearn row, so the
# extra points cost seconds.
ALPHA_GRID = [round(0.05 * i, 2) for i in range(21)]

# gated_fusion's setting 4 ran 300 full-batch steps with lr <= 1e-2, so a
# weight coordinate could move at most 3.0 over the whole run (Adam's step is
# gradient-magnitude independent, so |dw| per step ~ lr). Two independent
# signs that this was binding: 4b -- the same mixture under an inner-CV
# logistic -- beat setting 4 in every cell, and nearly every outer fold
# selected the LARGEST lr on offer. Both say "the head had not
# converged", not "the representation is weak".
#
# These override gated_fusion's module globals for THIS PROCESS ONLY. That
# module is imported for its optimizer, not run, so gated_fusion.py's own runs
# are unaffected -- and the override is stated in the header for that
# reason.
# 300 steps, not more. The head's only real
# regulariser today is the step budget plus early stopping -- pinned at
# wd=1e-4 the decoupled decay removes 1e-4 of its norm over a whole run --
# so raising the budget without also giving the head a selected decay would
# have REMOVED regularisation, not added capacity. The decay is selected now
# (WD_HEAD_GRID), which is the change that had to come first.
N_EPOCHS_DEFAULT = 300


class BlockPrep(BaseEstimator, TransformerMixin):
    """[v_ehr_pre | v_ppg_post | extra] -> the standardised blocks, per fold.

        ze = PCA(128) of StandardScaler(v_ehr_pre), divided by sqrt of the mean
             retained eigenvalue
        zp = StandardScaler(v_ppg_post)
        ex = StandardScaler(extra)

    Everything is fitted here rather than up front so that it lands inside the
    Pipeline and therefore inside each outer fold. PCA reads the feature
    matrix, so a cohort-wide fit would let the test fold shape the principal
    axes -- unsupervised, but still leakage.

    The block scaling is ONE scalar. Relative variance between components is
    untouched (whitening destroys it, at a measured cost in AUROC), while the
    block's mean column variance becomes 1, matching what StandardScaler leaves
    on the PPG side. That is what makes alpha a block-level weight rather than
    an accident of the two blocks' scales.

    svd_solver='full' for determinism -- the default picks the randomized
    solver at this shape.
    """

    def __init__(self, ehr_dim=320, ppg_dim=128, n_components=N_COMPONENTS,
                 out='both'):
        self.ehr_dim = ehr_dim
        self.ppg_dim = ppg_dim
        self.n_components = n_components
        self.out = out

    def _split(self, X):
        a, b = self.ehr_dim, self.ehr_dim + self.ppg_dim
        return X[:, :a], X[:, a:b], X[:, b:]

    def fit(self, X, y=None):
        E, P, Xtra = self._split(X)
        self.sc_e_ = StandardScaler().fit(E)
        self.pca_ = PCA(n_components=self.n_components,
                        svd_solver='full').fit(self.sc_e_.transform(E))
        blk = float(np.sqrt(np.mean(self.pca_.explained_variance_)))
        self.blk_ = blk if blk > 0 else 1.0
        self.sc_p_ = StandardScaler().fit(P)
        self.sc_x_ = StandardScaler().fit(Xtra) if Xtra.shape[1] else None
        return self

    def blocks(self, X):
        E, P, Xtra = self._split(X)
        ze = self.pca_.transform(self.sc_e_.transform(E)) / self.blk_
        zp = self.sc_p_.transform(P)
        ex = self.sc_x_.transform(Xtra) if self.sc_x_ is not None else Xtra
        return ze, zp, ex

    def transform(self, X):
        ze, zp, ex = self.blocks(X)
        if self.out == 'ehr':
            return ze
        if self.out == 'ppg':
            return zp
        if self.out == 'both':
            return np.hstack([ze, zp])
        if self.out == 'ehr+extra':
            return np.hstack([ze, ex])
        raise ValueError(self.out)


# GridSearchCV clones the whole Pipeline for every (alpha, C), so without a
# cache the 320->128 PCA is refitted 21 times for a C grid alone and 231 times
# for D_fixed_alpha's (alpha, C) grid -- on IDENTICAL training rows every time,
# since only the steps AFTER 'prep' vary. Pipeline(memory=...) keys each
# transformer's fit_transform on (transformer, params, X, y), so within one
# GridSearchCV there are only N_FOLDS_INNER + 1 distinct prep entries per outer
# fold. Measured: A_sEHR_pca128 went from 11.7s to 1.4s per horizon.
#
# Set by main(); left None so the estimators stay usable without one.
_MEMORY = None


def _lr():
    return LogisticRegression(penalty='l2', solver='lbfgs',
                              class_weight='balanced', max_iter=5000)


def est_plain():
    return Pipeline([('sc', StandardScaler()), ('lr', _lr())],
                    memory=_MEMORY)


def est_block(out):
    return Pipeline([('prep', BlockPrep(out=out)), ('lr', _lr())],
                    memory=_MEMORY)


def est_mixed():
    """BlockPrep -> ConvexMixer -> logistic. 'mix' is the name run_cv addresses.

    The mixer sits AFTER the block preparation, so the mixture is a restriction
    of the C_concat probe with the two 128-d coefficient vectors constrained
    parallel -- which is why C_concat >= D_fixed_alpha is structural rather
    than empirical.
    """
    return Pipeline([
        ('prep', BlockPrep(out='both')),
        ('mix', ConvexMixer(dim=N_COMPONENTS)),
        ('lr', _lr()),
    ], memory=_MEMORY)


def _pf(vals):
    return ','.join(f'{v:.6f}' for v in vals)


def select_C(F_tr, y_tr, g_tr):
    """The same C selection every sklearn row in this table uses.

    Returns (fitted_estimator, C). Falls back to DEFAULT_C when the fold holds
    too few positives for a 3-fold inner split, exactly as run_cv does, so E is
    not quietly given a selection the baselines were denied.
    """
    if int(y_tr.sum()) >= reg.MIN_POS_FOR_INNER_CV:
        try:
            inner = StratifiedGroupKFold(n_splits=reg.N_FOLDS_INNER,
                                         shuffle=True,
                                         random_state=reg.RANDOM_SEED)
            gs = GridSearchCV(_lr(), {'C': list(reg.C_GRID)}, scoring='roc_auc',
                              cv=inner, n_jobs=-1, error_score=np.nan,
                              refit=True)
            gs.fit(F_tr, y_tr, groups=g_tr)
            if np.isfinite(gs.best_score_):
                return gs.best_estimator_, float(gs.best_params_['C'])
        except (ValueError, AttributeError):
            pass
    m = _lr().set_params(C=reg.DEFAULT_C).fit(F_tr, y_tr)
    return m, reg.DEFAULT_C



# ══════════════════════════════════════════════════════════════════════════
# E: one end-to-end AdamW model per horizon
# ══════════════════════════════════════════════════════════════════════════
# THREE AXES, SEARCHED JOINTLY. gate and head get SEPARATE decays because they
# solve different problems -- ||w_gate|| controls how much alpha varies with
# its inputs, ||w_head|| controls how hard the classifier is shrunk -- and
# they are searched TOGETHER because they interact: d(logit)/d(alpha) =
# w_head . (zp - ze), so shrinking the head also shrinks the gradient reaching
# the gate. Tuning one and then the other would miss that.
#
# The grid is deliberately wide and un-derived. An earlier version set the
# values by inverting ||w|| ~ sqrt(n_inputs)/wd to hit a target alpha spread;
# that is choosing hyperparameters with a desired answer in mind, and the
# sqrt(n) scaling had only ever been measured at one width. Here the grid just
# spans four decades, inner CV picks, and the log reports whether any axis was
# selected at an edge -- widen and rerun if so.
# ONE decay grid for every weight-decay axis in this file -- gate, head, and
# the representation side (adapter / MLP / projection). They were different
# once, and the reason was the residual: an adapter CAN be driven to zero, at
# which point the model falls back on the frozen aligned vector, so very
# strong decay is a meaningful endpoint there and the grid was pushed up to
# 1..100. That argument does not survive contact with H_mlp, M_ or N_mlp,
# none of which is a residual: driving their projection to zero does not turn
# a correction off, it erases the representation. Keeping the encoder axis
# two decades above the head axis was therefore an assumption, not a result.
#
# 30 is the ceiling, not 100, and the constraint is rho. The guard
# is lr*wd <= MAX_RHO with each group's OWN lr: head and gate run at 0.01 so
# they stop at 30, the encoder runs at 0.003 so it could reach 100. A shared
# grid ending at 100 would be reachable by the encoder and not by the head --
# the same ragged-grid problem --lr-adapter was added to remove.
#
# If a residual genuinely needs more decay than a head does, the way to learn
# that is to watch G_adapter pin itself to 30 here. That is evidence. The old
# split assumed the answer.
WD_GRID = [1e-2, 1e-1, 1.0, 10.0, 30.0]
WD_GATE_GRID = WD_GRID
WD_HEAD_GRID = WD_GRID
# A tenth was added because lr sat at the grid edge in every fold of several
# cells, and then REMOVED because a later run showed it was harmful. rho =
# lr*wd caps the per-step shrinkage but does nothing about total travel: a
# larger rate paired with a smaller decay shrinks exactly as fast per step
# while taking proportionally larger gradient steps, so the adapter runs much
# further before decay catches it. The cells that lost the most AUROC were
# precisely the ones that selected the largest rate, and their diagnostics
# show it: the adapter's output norm dwarfs the frozen vector it corrects and
# the cosine to that vector collapses, i.e. the pretrained representation has
# been overwritten rather than adjusted. A grid edge being selected is a
# request to widen; it is not a promise that widening helps.
LR_GRID_E = [3e-3, 1e-2, 3e-2]

# AdamW's decoupled decay multiplies the old weight by (1 - lr*wd) every step,
# so lr*wd is the per-step shrinkage and it is what has to stay sane -- the
# same wd means very different things at lr=3e-3 and lr=3e-2. Above 1 the
# multiplier goes NEGATIVE (the weight flips sign each step) and above 2 it
# also grows; gated_fusion's MAX_LR_TIMES_WD=2 only rules out the second,
# which is the divergence boundary, not a sensible regularisation range.
# 0.3 means at most a 30% cut per step. Nothing in the grid above trips it --
# the largest product is 3e-2 * 10 = 0.3 exactly -- so it is a guard for when
# the grid is widened, not an active filter today.
MAX_RHO = 0.3

N_INNER_E = 3


def candidates():
    return [(wg, wh, lr)
            for wg in WD_GATE_GRID
            for wh in WD_HEAD_GRID
            for lr in LR_GRID_E
            if lr * wg <= MAX_RHO and lr * wh <= MAX_RHO]


def _gate_matrix(ze, zp, M, gate_input):
    """scalars: 4 columns. emb: + both 128-d blocks. No horizon column --
    inside one horizon it is constant, so StandardScaler would emit zeros and
    the gate would carry a parameter that cannot move the loss."""
    return M if gate_input == 'scalars' else np.hstack([ze, zp, M])


def _pack(X, M, y, i_tr, i_va, gate_input):
    """Prepare ONE inner fold, fitting every data-driven transform on i_tr only.

    This is the whole reason the selection loop lives here instead of calling
    gated_fusion.fit_gate: BlockPrep (StandardScaler + PCA) and the gate's
    scaler have to be fitted INSIDE the inner split, not on the full outer-
    training fold before splitting. Fitting first and splitting after lets the
    inner-validation rows shape the principal axes, which is what fitting
    inside the split exists to prevent.

    It also matters for symmetry, which is the point of this whole script:
    every sklearn row here carries BlockPrep as a Pipeline step, so
    GridSearchCV already refits it per inner fold. Leaving E on the looser
    protocol would put the two sides of the comparison on different standards.

    Returned as one concatenated array plus index arrays, because that is the
    interface _train_once takes. None when the validation slice is single-
    class and therefore cannot score.

    The prep does NOT depend on the hyperparameters, so this runs once per
    inner fold and all candidates share it.
    """
    prep = BlockPrep(out='both').fit(X[i_tr])
    ze_a, zp_a, _ = prep.blocks(X[i_tr])
    ze_b, zp_b, _ = prep.blocks(X[i_va])
    g_a = _gate_matrix(ze_a, zp_a, M[i_tr], gate_input)
    g_b = _gate_matrix(ze_b, zp_b, M[i_va], gate_input)
    scg = StandardScaler().fit(g_a)
    yy = np.concatenate([y[i_tr], y[i_va]])
    va_i = np.arange(len(i_tr), len(yy))
    if len(np.unique(yy[va_i])) < 2:
        return None
    return {
        'ze': torch.tensor(np.vstack([ze_a, ze_b]), dtype=torch.float32),
        'zp': torch.tensor(np.vstack([zp_a, zp_b]), dtype=torch.float32),
        'xg': torch.tensor(np.vstack([scg.transform(g_a), scg.transform(g_b)]),
                           dtype=torch.float32),
        'yt': torch.tensor(yy, dtype=torch.float32),
        'y': yy,
        'h': np.zeros(len(yy), dtype=np.int64),
        'tr_i': np.arange(len(i_tr)),
        'va_i': va_i,
        'dim': ze_a.shape[1],
        'gate_dim': g_a.shape[1],
    }


def select_hparams(X, M, y, groups, gate_input, seed, epochs, eval_every=1):
    """3-fold inner CV over the joint grid. Returns (best_row, all_rows).

    Every candidate trains once per inner fold and is ranked by the MEAN of
    its validation AUROCs -- the same protocol GridSearchCV gives C on every
    sklearn row in this table. Ranking 48 candidates on a single 25% split
    would be argmax over 48 numbers each computed from 10-30 positives, which
    mostly selects whichever candidate got the lucky split.

    best_epoch is the MEDIAN of the three folds' best epochs, not the mean:
    one fold running to the budget because of noise should not drag the refit
    length with it.

    Returns (None, []) when no inner fold can score -- too few positive
    patients to split, or every validation slice single-class. The caller then
    falls back to fixed hyperparameters AND COUNTS THE FOLD, because a fold
    that quietly ran the full budget with no early stopping and no selection
    is exactly the kind of thing that should not be invisible.
    """
    pos = np.unique(groups[y == 1])
    packs = []
    if len(pos) >= N_INNER_E:
        ever = np.isin(groups, pos).astype(int)
        sp = StratifiedGroupKFold(n_splits=N_INNER_E, shuffle=True,
                                  random_state=seed)
        for i_tr, i_va in sp.split(np.zeros((len(y), 1)), ever, groups=groups):
            if len(np.unique(y[i_tr])) < 2:
                continue
            pk = _pack(X, M, y, i_tr, i_va, gate_input)
            if pk is not None:
                packs.append(pk)
    if not packs:
        return None, []

    rows = []
    for wg, wh, lr in candidates():
        vas, eps = [], []
        for pk in packs:
            _, va, ep = gf._train_once(
                pk['ze'], pk['zp'], pk['xg'], pk['yt'], pk['y'], pk['h'],
                pk['tr_i'], pk['va_i'], [0],
                pk['dim'], pk['gate_dim'], 0, seed, wg, lr, True,
                wd_head=wh, n_epochs=epochs, return_epoch=True,
                eval_every=eval_every,
            )
            if va is not None:
                vas.append(va)
                eps.append(ep)
        if vas:
            rows.append({'mean': float(np.mean(vas)), 'sd': float(np.std(vas)),
                         'epoch': int(np.median(eps)), 'n': len(vas),
                         'wd_gate': wg, 'wd_head': wh, 'lr': lr})
    if not rows:
        return None, []
    rows.sort(key=lambda r: -r['mean'])
    return rows[0], rows


def run_learned_alpha(X, M, y, groups, folds, gate_input, args):
    """E and its sklearn-head control, one horizon, over the outer folds."""
    te_t, ap_t, te_s, ap_s = [], [], [], []
    a_mu, a_sd, Cs, sel, wg_n, wh_n, eps = [], [], [], [], [], [], []
    n_no_sel = 0
    gap_top = []

    for fold_id, (tr, te) in enumerate(folds):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        seed = reg.RANDOM_SEED + fold_id

        best, rows = select_hparams(X[tr], M[tr], y[tr], groups[tr],
                                    gate_input, seed, args.epochs,
                                    args.eval_every)
        if best is None:
            n_no_sel += 1
            wd_g, wd_h, lr, ep = (gf.DEFAULT_WD, gf.DEFAULT_WD,
                                  gf.DEFAULT_LR, args.epochs)
        else:
            wd_g, wd_h, lr, ep = (best['wd_gate'], best['wd_head'],
                                  best['lr'], best['epoch'])
            # How separable were the candidates at all? If the top of the
            # ranking sits inside the fold-to-fold spread, the selected value
            # is not identified by the data.
            if len(rows) > 2:
                gap_top.append(rows[0]['mean'] - rows[2]['mean'])
        sel.append((wd_g, wd_h, lr))
        eps.append(ep)

        # ── refit: fresh init, FULL outer-train, the selected epoch count ──
        # The sklearn rows get this for free (GridSearchCV refit=True fits the
        # winner on all of X[tr]). Without it the torch model would be trained
        # on ~67% of the outer-training rows while every row it is compared
        # against used 100% -- and "the sklearn head is better" would be
        # confounded with "the sklearn head saw half again as much data".
        prep = BlockPrep(out='both').fit(X[tr])
        ze_tr, zp_tr, _ = prep.blocks(X[tr])
        ze_te, zp_te, _ = prep.blocks(X[te])
        g_tr = _gate_matrix(ze_tr, zp_tr, M[tr], gate_input)
        g_te = _gate_matrix(ze_te, zp_te, M[te], gate_input)
        scg = StandardScaler().fit(g_tr)
        xg_tr, xg_te = scg.transform(g_tr), scg.transform(g_te)

        T = lambda a: torch.tensor(a, dtype=torch.float32)
        model, _ = gf._train_once(
            T(ze_tr), T(zp_tr), T(xg_tr), T(y[tr].astype(np.float64)),
            y[tr], np.zeros(len(tr), dtype=np.int64),
            np.arange(len(tr)), None, [],
            ze_tr.shape[1], xg_tr.shape[1], 0, seed, wd_g, lr, True,
            wd_head=wd_h, n_epochs=ep,
        )

        with torch.no_grad():
            s_tr_t, a_tr = model(T(ze_tr), T(zp_tr), T(xg_tr))
            s_te_t, a_te = model(T(ze_te), T(zp_te), T(xg_te))
            wg_n.append(float(model.gate.weight.norm()))
            wh_n.append(float(model.head.weight.norm()))
        a_tr, a_te = a_tr.numpy(), a_te.numpy()
        s_te_t = s_te_t.numpy()

        te_t.append(roc_auc_score(y[te], s_te_t))
        ap_t.append(average_precision_score(y[te], s_te_t))
        a_mu.append(float(a_te.mean()))
        a_sd.append(float(a_te.std()))

        # ── control: same alpha, sklearn classifier ──
        if args.oof_alpha:
            a_fit = oof_alpha(X[tr], M[tr], y[tr], groups[tr], gate_input,
                              seed, wd_g, wd_h, lr, ep)
        else:
            a_fit = a_tr
        F_tr = a_fit[:, None] * zp_tr + (1.0 - a_fit[:, None]) * ze_tr
        F_te = a_te[:, None] * zp_te + (1.0 - a_te[:, None]) * ze_te
        clf, C = select_C(F_tr, y[tr], groups[tr])
        s_te_s = clf.decision_function(F_te)
        te_s.append(roc_auc_score(y[te], s_te_s))
        ap_s.append(average_precision_score(y[te], s_te_s))
        Cs.append(C)

    def pack(auc, ap, tag):
        r = {
            'probe': tag, 'setting': tag, 'train_auroc': np.nan,
            'auroc_mean': float(np.mean(auc)) if auc else np.nan,
            'auroc_std_fold': float(np.std(auc)) if auc else np.nan,
            'auprc_mean': float(np.mean(ap)) if ap else np.nan,
            'auprc_std_fold': float(np.std(ap)) if ap else np.nan,
            'auroc_per_fold': _pf(auc), 'auprc_per_fold': _pf(ap),
            'alpha_mean': float(np.mean(a_mu)) if a_mu else np.nan,
            'alpha_std_within': float(np.mean(a_sd)) if a_sd else np.nan,
            'alpha_std_across': float(np.std(a_mu)) if a_mu else np.nan,
            'n_valid_folds': len(auc),
            'folds_without_selection': n_no_sel,
            'sel_wd_gate': ','.join(f'{a:g}' for a, _, _ in sel),
            'sel_wd_head': ','.join(f'{b:g}' for _, b, _ in sel),
            'sel_lr': ','.join(f'{c:g}' for _, _, c in sel),
            'sel_epoch': ','.join(str(e) for e in eps),
            'w_gate_norm': float(np.mean(wg_n)) if wg_n else np.nan,
            'w_head_norm': float(np.mean(wh_n)) if wh_n else np.nan,
            'top_vs_third': (float(np.mean(gap_top)) if gap_top else np.nan),
        }
        r['C_median'] = float(np.median(Cs)) if (Cs and 'sklearn' in tag) \
            else np.nan
        return r

    return (pack(te_t, ap_t, 'E_learned_alpha'),
            pack(te_s, ap_s, 'E_sklearnhead'))


def oof_alpha(X, M, y, groups, gate_input, seed, wd_g, wd_h, lr, ep):
    """Out-of-fold alpha for the training rows, at the SELECTED hyperparameters.

    Only affects E_sklearnhead's C selection: without it, alpha on the
    training rows comes from a model that was fitted on those same rows. The
    outer test fold is untouched either way, so both E rows' reported AUROC is
    clean regardless. Costs one extra training run per inner fold.
    """
    a = np.full(len(y), 0.5, dtype=np.float64)
    pos = np.unique(groups[y == 1])
    if len(pos) < N_INNER_E:
        return a
    ever = np.isin(groups, pos).astype(int)
    sp = StratifiedGroupKFold(n_splits=N_INNER_E, shuffle=True,
                              random_state=seed)
    for i_tr, i_va in sp.split(np.zeros((len(y), 1)), ever, groups=groups):
        if len(np.unique(y[i_tr])) < 2:
            continue
        prep = BlockPrep(out='both').fit(X[i_tr])
        ze_a, zp_a, _ = prep.blocks(X[i_tr])
        ze_b, zp_b, _ = prep.blocks(X[i_va])
        g_a = _gate_matrix(ze_a, zp_a, M[i_tr], gate_input)
        g_b = _gate_matrix(ze_b, zp_b, M[i_va], gate_input)
        scg = StandardScaler().fit(g_a)
        T = lambda z: torch.tensor(z, dtype=torch.float32)
        m, _ = gf._train_once(
            T(ze_a), T(zp_a), T(scg.transform(g_a)),
            T(y[i_tr].astype(np.float64)), y[i_tr],
            np.zeros(len(i_tr), dtype=np.int64), np.arange(len(i_tr)), None,
            [], ze_a.shape[1], g_a.shape[1], 0, seed, wd_g, lr, True,
            wd_head=wd_h, n_epochs=ep,
        )
        with torch.no_grad():
            _, av = m(T(ze_b), T(zp_b), T(scg.transform(g_b)))
        a[i_va] = av.numpy()
    return a



# ══════════════════════════════════════════════════════════════════════════
# G: task-specific residual adapter on the ALIGNED sEHR vector
# ══════════════════════════════════════════════════════════════════════════
# WHY THIS EXISTS, AND WHAT IT IS NOT
# ----------------------------------
# The PCA rows above fixed the sEHR side's QUALITY (0e ~ 0a) but lost the one
# thing the aligned space had: a coordinate system SHARED with the PPG vector.
# The convex mixture ties the two blocks through a single w, so coordinate j
# of ze and of zp had better mean the same thing -- true in the aligned space,
# false for PCA axes. The evidence that this matters is AF, where alignment
# costs the sEHR side nothing and the ALIGNED fixed-alpha mixture beats 0a in
# every cell while the PCA one is flat against it. On HF and eMACE the aligned
# side starts well below 0a and no mixing recovers that.
#
# So: keep the aligned space, repair the deficit. z_E_post stays frozen and a
# small low-rank residual is added to it:
#
#     z_new = normalize( z_E_post + A(v_ehr_pre) ),   A: 320 -> r -> 128
#
# A's second layer is zero-initialised, so at step 0 the model IS the original
# aligned representation and everything learned is a departure from it. This
# is the LoRA pattern: the gradient reaches the down-projection only once the
# up-projection has moved off zero.
#
# THIS IS NOT A NEW PRETRAINED REPRESENTATION. A sees downstream labels, from
# the current training split only, and is refitted inside every outer fold. It
# is a disease- and horizon-specific supervised adapter, and it must be
# reported that way.
#
# WHICH IS WHY G_adapter_only IS NOT OPTIONAL. Comparing the adapted fusion to
# A320 would credit the adapter's label access to the PPG side. The increment
# attributable to PPG is
#
#     G_adapter_fusion  -  G_adapter_only
#
# same adapter architecture, same protocol, same folds, PPG the only
# difference.
#
# SCALING DIFFERS FROM EVERY OTHER ROW HERE, deliberately. The aligned blocks
# arrive L2-normalised from extract_feature and are used RAW: column-
# standardising them and then L2-renormalising after the residual would be an
# odd composition and would distort the very space this row exists to keep.
# Only the adapter's INPUT (v_ehr_pre, a raw BERT vector with arbitrary column
# scales) is standardised, fitted inside the inner split like everything else.
#
# EXPECT A CEILING. The adapter works on the sEHR side, and the sEHR side is
# not what caps the PPG increment: alpha_fusion already ran the fusion with an
# undamaged sEHR side (PCA-128 ~ A320) and did not exceed it. Six methods have now
# put PPG's marginal contribution at the same small positive value. The
# realistic best case here is recovering that increment at the EMBEDDING
# level, which score-level
# fusion already reaches -- worth having because embedding fusion is the
# intended final method, not because more signal is expected.

# The full convex range, 0 to 1 in steps of 0.05. Both endpoints have to be
# reachable: 0 is "decline PPG entirely" and 1 is "decline sEHR entirely", and
# a row that cannot express them reports a property of the grid rather than of
# the data.
#
# The top was 0.5 until a run showed one cell selecting that cap in every
# fold, which is a grid boundary reporting itself rather than a choice.
# That cap had no principled endpoint, unlike the weight-decay ceiling, where
# wd -> infinity converges on the frozen aligned vector and is measurably
# worse. That cell is also
# where PPG alone comes closest to the sEHR baseline -- far closer than on HF
# or eMACE -- so wanting a larger mixing weight there is what the data would
# predict.
#
# Cost: the fixed-alpha rows scale linearly with this list, 220 -> 420
# candidates, so the three G rows go 320 -> 520 and a disease from ~50 to
# ~75 min. Only the fixed-alpha rows grow; _only freezes alpha at 0 and
# _learned searches wd_gate on that axis instead.
ADAPTER_ALPHA_GRID = [round(0.05 * i, 2) for i in range(21)]


class ResidualAdapter(nn.Module):
    """320 -> r -> 128, zero-initialised at the output.

    Zero-init on `up` alone (not `down`) is the LoRA arrangement. At step 0 the
    adapter contributes exactly nothing, so the model starts from the
    pretrained aligned vector; the gradient w.r.t. `down` is zero at that
    instant and becomes non-zero as soon as `up` leaves the origin. Zeroing
    both would freeze the adapter permanently.

    r=16 is 320*16 + 16*128 = 7168 weights against 42-124 training positives.
    That is small for an MLP and still large for this cohort, which is why the
    output is a RESIDUAL on a frozen vector rather than a replacement: the
    model cannot do worse than the frozen aligned representation by leaving
    the adapter at zero, and early stopping can keep it there.
    """

    def __init__(self, in_dim=320, rank=16, out_dim=128, zero_init=True):
        super().__init__()
        self.down = nn.Linear(in_dim, rank)
        self.up = nn.Linear(rank, out_dim)
        if zero_init:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        # zero_init MUST be off when there is no frozen base to fall back on.
        # F.normalize divides by ||v|| clamped at 1e-12, so normalize(0) returns
        # 0 rather than NaN -- but its Jacobian there is I/1e-12 and the first
        # backward pass explodes. With a base, v = z_post + 0 has unit norm and
        # the problem does not arise.

    def forward(self, x):
        return self.up(torch.relu(self.down(x)))


class AdaptedFusion(nn.Module):
    """Adapter + optional gate + head, with GatedFusion's forward signature.

    `ez` arrives as [v_ehr_pre (pre_dim) | z_E_post (dim)] so that _train_once
    -- which passes exactly three tensors -- can drive this module unchanged.

    alpha_fixed=None gives the learned gate; a float freezes alpha, which
    turns this into the fixed-mixture variant and, at 0.0, into the
    adapter-only control. Every run so far has had a fixed alpha beat the
    learned one, so testing the adapter under BOTH is what separates "the
    adapter helps" from "the gate hurts".
    """

    def __init__(self, pre_dim, dim, gate_in_dim, rank, alpha_fixed=None,
                 use_base=True):
        super().__init__()
        self.pre_dim = pre_dim
        self.use_base = use_base
        self.adapter = ResidualAdapter(pre_dim, rank, dim, zero_init=use_base)
        self.alpha_fixed = alpha_fixed
        self.gate = None if alpha_fixed is not None else nn.Linear(gate_in_dim, 1)
        self.head = nn.Linear(dim, 1)

    def forward(self, ez, zp, xg):
        x_pre, z_post = ez[:, :self.pre_dim], ez[:, self.pre_dim:]
        a = self.adapter(x_pre)
        # use_base=False drops the pretrained vector entirely: the sEHR side
        # becomes a task-trained MLP(v_ehr_pre) and nothing of the aligned
        # space survives. Kept as a separate row rather than a replacement,
        # because it answers a different question -- "can a supervised sEHR
        # encoder carry the fusion" -- and forfeits the shared-coordinate
        # argument the residual formulation exists to keep.
        z_new = torch.nn.functional.normalize(
            (z_post + a) if self.use_base else a, dim=-1)
        if self.gate is None:
            a = torch.full((ez.shape[0], 1), float(self.alpha_fixed))
        else:
            a = torch.sigmoid(self.gate(xg))
        fused = (1.0 - a) * z_new + a * zp
        return self.head(fused).squeeze(-1), a.squeeze(-1)


def adapter_groups(wd_adapter, lr_adapter=None):
    """Six groups. Every bias is exempt, and the adapter never shares a decay
    with the gate: one controls how far the representation may move from the
    pretrained one, the other how much alpha may vary. Held apart on purpose."""
    def f(model, wd, wd_head):
        aw = [p for n, p in model.adapter.named_parameters()
              if not n.endswith('bias')]
        ab = [p for n, p in model.adapter.named_parameters()
              if n.endswith('bias')]
        # lr_adapter, when given, applies to the adapter groups ONLY. Two
        # things forced this. The rho cap makes wd_adapter unreachable above
        # MAX_RHO/lr -- at the shared lr=0.01 that is 30, exactly the value the
        # search pinned itself to in most cells. And Adam puts a FLOOR under
        # the adapter that no decay can cross: its step is ~lr per coordinate
        # regardless of gradient, so after any step ||W_up|| >~ lr*sqrt(2048)
        # = 0.45 at lr=0.01, against an observed minimum of 0.82-0.90. The
        # "adapter almost off" endpoint is therefore not reachable by raising
        # the decay; only a smaller step reaches it.
        ex = {} if lr_adapter is None else {'lr': float(lr_adapter)}
        g = [{'params': aw, 'weight_decay': wd_adapter, **ex},
             {'params': ab, 'weight_decay': 0.0, **ex},
             {'params': [model.head.weight], 'weight_decay': wd_head},
             {'params': [model.head.bias], 'weight_decay': 0.0}]
        if model.gate is not None:
            g += [{'params': [model.gate.weight], 'weight_decay': wd},
                  {'params': [model.gate.bias], 'weight_decay': 0.0}]
        return g
    return f


def _adapter_pack(Xa, M, y, i_tr, i_va, gate_input, pre_dim, dim):
    """One inner fold for the adapter rows.

    Xa is [v_ehr_pre | v_ehr_post | v_ppg_post]. Only v_ehr_pre is scaled, and
    its scaler is fitted on i_tr alone -- same discipline as BlockPrep, for the
    same reason: fitting before splitting lets the validation rows shape the
    statistics that hyperparameter selection is then read off.

    In `emb` gate mode the gate reads the FROZEN aligned blocks, not the
    adapted one. The adapted vector changes every step, so a scaler fitted
    once on it would be stale by the second epoch.
    """
    E, Zp = Xa[:, :pre_dim], Xa[:, pre_dim + dim:]
    Ze = Xa[:, pre_dim:pre_dim + dim]
    sc = StandardScaler().fit(E[i_tr])

    def build(idx):
        ez = np.hstack([sc.transform(E[idx]), Ze[idx]])
        g = (M[idx] if gate_input == 'scalars'
             else np.hstack([Ze[idx], Zp[idx], M[idx]]))
        return ez, Zp[idx], g

    ez_a, zp_a, g_a = build(i_tr)
    ez_b, zp_b, g_b = build(i_va)
    scg = StandardScaler().fit(g_a)
    yy = np.concatenate([y[i_tr], y[i_va]])
    va_i = np.arange(len(i_tr), len(yy))
    if len(np.unique(yy[va_i])) < 2:
        return None
    return {
        'ez': torch.tensor(np.vstack([ez_a, ez_b]), dtype=torch.float32),
        'zp': torch.tensor(np.vstack([zp_a, zp_b]), dtype=torch.float32),
        'xg': torch.tensor(np.vstack([scg.transform(g_a), scg.transform(g_b)]),
                           dtype=torch.float32),
        'yt': torch.tensor(yy, dtype=torch.float32),
        'y': yy, 'h': np.zeros(len(yy), dtype=np.int64),
        'tr_i': np.arange(len(i_tr)), 'va_i': va_i,
        'gate_dim': g_a.shape[1],
    }


def adapter_candidates(mode, wd_adapter_grid, lr_repr=None):
    """(alpha_or_wd_gate, wd_head, lr) per mode, with the same rho guard.

    only    alpha frozen at 0 -- 12 candidates, since there is no gate and no
            alpha to search
    fixed   alpha searched over ADAPTER_ALPHA_GRID; every selected alpha in
            every run so far has landed in 0.00-0.40, so the grid is targeted
            rather than uniform on [0,1]
    learned wd_gate searched, exactly as for E
    """
    out = []
    first = ([0.0] if mode == 'only' else
             ADAPTER_ALPHA_GRID if mode == 'fixed' else WD_GATE_GRID)
    for a in first:
        for wh in WD_HEAD_GRID:
            for wa in wd_adapter_grid:
                for lr in LR_GRID_E:
                    # The adapter's rho uses ITS lr, not the shared one, so a
                    # smaller lr_repr is what makes a larger wd_adapter
                    # reachable at all.
                    lr_a = lr if lr_repr is None else lr_repr
                    if lr * wh > MAX_RHO or lr_a * wa > MAX_RHO:
                        continue
                    if mode == 'learned' and lr * a > MAX_RHO:
                        continue
                    out.append((a, wh, lr, wa))
    return out


def run_adapter(Xa, M, y, groups, folds, gate_input, args, mode,
                use_base=True):
    """One adapter row: 3-fold inner CV, refit on full outer-train, score test.

    Identical fold logic to run_learned_alpha -- the outer folds, the inner
    3-fold, the mean-AUROC ranking, the median epoch, the discard-and-refit --
    so the rows are comparable to each other and to every sklearn row.
    """
    pre_dim, dim = args.pre_dim, N_COMPONENTS
    rank = args.adapter_rank if use_base else args.mlp_rank
    wd_repr_grid = args.wd_adapter if use_base else args.wd_mlp
    pre = 'G_adapter' if use_base else 'H_mlp'
    tag = f"{pre}_" + {'only': 'only', 'fixed': 'fixed_alpha',
                       'learned': 'learned_alpha'}[mode]
    auc, ap, a_mu, a_sd, sel, eps, an, hn, was = ([], [], [], [], [], [],
                                                   [], [], [])
    rn, cs = [], []
    n_no_sel = 0
    cands = adapter_candidates(mode, wd_repr_grid, args.lr_adapter)

    for fold_id, (tr, te) in enumerate(folds):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        seed = reg.RANDOM_SEED + fold_id
        Xtr, Mtr, ytr, gtr = Xa[tr], M[tr], y[tr], groups[tr]

        # ── inner CV ──
        packs = []
        pos = np.unique(gtr[ytr == 1])
        if len(pos) >= N_INNER_E:
            ever = np.isin(gtr, pos).astype(int)
            sp = StratifiedGroupKFold(n_splits=N_INNER_E, shuffle=True,
                                      random_state=seed)
            for i_tr, i_va in sp.split(np.zeros((len(ytr), 1)), ever,
                                       groups=gtr):
                if len(np.unique(ytr[i_tr])) < 2:
                    continue
                pk = _adapter_pack(Xtr, Mtr, ytr, i_tr, i_va, gate_input,
                                   pre_dim, dim)
                if pk is not None:
                    packs.append(pk)

        best = None
        if packs:
            rows = []
            for a1, wh, lr, wa in cands:
                vas, ee = [], []
                for pk in packs:
                    af = None if mode == 'learned' else a1
                    wg = a1 if mode == 'learned' else 0.0
                    mk = (lambda gd=pk['gate_dim'], af=af:
                          AdaptedFusion(pre_dim, dim, gd, rank, af,
                                        use_base))
                    _, va, ep = gf._train_once(
                        pk['ez'], pk['zp'], pk['xg'], pk['yt'], pk['y'],
                        pk['h'], pk['tr_i'], pk['va_i'], [0],
                        dim, pk['gate_dim'], 0, seed, wg, lr, True,
                        wd_head=wh, n_epochs=args.epochs, return_epoch=True,
                        eval_every=args.eval_every, make_model=mk,
                        make_groups=adapter_groups(wa, args.lr_adapter),
                    )
                    if va is not None:
                        vas.append(va)
                        ee.append(ep)
                if vas:
                    rows.append({'mean': float(np.mean(vas)),
                                 'epoch': int(np.median(ee)),
                                 'a1': a1, 'wd_head': wh, 'lr': lr,
                                 'wa': wa})
            if rows:
                rows.sort(key=lambda r: -r['mean'])
                best = rows[0]
        if best is None:
            n_no_sel += 1
            a1, wh, lr, ep = (0.0 if mode != 'learned' else gf.DEFAULT_WD,
                              gf.DEFAULT_WD, gf.DEFAULT_LR, args.epochs)
            wa = wd_repr_grid[-1]
        else:
            a1, wh, lr, ep, wa = (best['a1'], best['wd_head'], best['lr'],
                                  best['epoch'], best['wa'])
        sel.append((a1, wh, lr))
        was.append(wa)
        eps.append(ep)

        # ── refit on the FULL outer-train, fresh init ──
        # Transforms rebuilt on ALL of outer-train here (the inner split is
        # gone), then applied to the test fold without refitting.
        E, Ze, Zp = (Xa[:, :pre_dim], Xa[:, pre_dim:pre_dim + dim],
                     Xa[:, pre_dim + dim:])
        sc = StandardScaler().fit(E[tr])
        ez_tr = np.hstack([sc.transform(E[tr]), Ze[tr]])
        ez_te = np.hstack([sc.transform(E[te]), Ze[te]])
        g_tr = (M[tr] if gate_input == 'scalars'
                else np.hstack([Ze[tr], Zp[tr], M[tr]]))
        g_te = (M[te] if gate_input == 'scalars'
                else np.hstack([Ze[te], Zp[te], M[te]]))
        scg = StandardScaler().fit(g_tr)
        T = lambda z: torch.tensor(z, dtype=torch.float32)
        af = None if mode == 'learned' else a1
        wg = a1 if mode == 'learned' else 0.0
        mk = lambda: AdaptedFusion(pre_dim, dim, g_tr.shape[1], rank, af,
                                   use_base)
        model, _ = gf._train_once(
            T(ez_tr), T(Zp[tr]), T(scg.transform(g_tr)),
            T(y[tr].astype(np.float64)), y[tr],
            np.zeros(len(tr), dtype=np.int64), np.arange(len(tr)), None, [],
            dim, g_tr.shape[1], 0, seed, wg, lr, True,
            wd_head=wh, n_epochs=ep, make_model=mk,
            make_groups=adapter_groups(wa, args.lr_adapter),
        )
        with torch.no_grad():
            s_te, a_te = model(T(ez_te), T(Zp[te]), T(scg.transform(g_te)))
            an.append(float(model.adapter.up.weight.norm()))
            hn.append(float(model.head.weight.norm()))
            # ||W_up|| cannot tell "the adapter barely moved" from "the
            # adapter rewrote the vector" -- the output also depends on the
            # down-projection. These two can. z_post is L2-normalised at
            # extraction and z_new is normalised here, so the cosine is a
            # plain dot product and ||A(x)|| is directly comparable to the
            # base's 1.0:
            #   ||A(x)|| ~ 0.1, cos ~ 0.995  a genuine small correction
            #   ||A(x)|| ~ 1,   cos ~ 0.7    half the direction is new
            #   ||A(x)|| >> 1,  cos ~ 0      the pretrained vector no
            #                                longer dominates and the module
            #                                is a task-trained encoder rather
            #                                than a residual
            zpost_t = T(Ze[te])
            a_vec = model.adapter(T(ez_te)[:, :pre_dim])
            z_new = torch.nn.functional.normalize(
                (zpost_t + a_vec) if use_base else a_vec, dim=-1)
            rn.append(float(a_vec.norm(dim=-1).mean()))
            cs.append(float((z_new * zpost_t).sum(-1).mean()))
        s_te, a_te = s_te.numpy(), a_te.numpy()
        auc.append(roc_auc_score(y[te], s_te))
        ap.append(average_precision_score(y[te], s_te))
        a_mu.append(float(a_te.mean()))
        a_sd.append(float(a_te.std()))

    return {
        'probe': tag, 'setting': tag, 'train_auroc': np.nan,
        'auroc_mean': float(np.mean(auc)) if auc else np.nan,
        'auroc_std_fold': float(np.std(auc)) if auc else np.nan,
        'auprc_mean': float(np.mean(ap)) if ap else np.nan,
        'auprc_std_fold': float(np.std(ap)) if ap else np.nan,
        'auroc_per_fold': _pf(auc), 'auprc_per_fold': _pf(ap),
        'alpha_mean': float(np.mean(a_mu)) if a_mu else np.nan,
        'alpha_std_within': float(np.mean(a_sd)) if a_sd else np.nan,
        # alpha_std_within is the WITHIN-fold spread averaged over folds, so
        # for a frozen alpha it is identically zero BY CONSTRUCTION and a
        # printed "+-0.000" reads as stability when it is zero by
        # construction. The
        # spread that exists is ACROSS folds: a single cell can select alpha
        # anywhere across the grid in its five folds, one of them declining
        # PPG entirely. That is the inner CV failing to identify the mixing
        # weight -- the same fact the flat best-vs-third gap reports.
        'alpha_std_across': float(np.std(a_mu)) if a_mu else np.nan,
        'n_valid_folds': len(auc), 'folds_without_selection': n_no_sel,
        'sel_wd_gate': ','.join(f'{a:g}' for a, _, _ in sel),
        'sel_wd_head': ','.join(f'{b:g}' for _, b, _ in sel),
        'sel_lr': ','.join(f'{c:g}' for _, _, c in sel),
        'sel_epoch': ','.join(str(e) for e in eps),
        # For 'only'/'fixed' the first column is the FROZEN alpha, not a decay.
        'sel_wd_adapter': ','.join(f'{v:g}' for v in was),
        'adapter_norm': float(np.mean(an)) if an else np.nan,
        'resid_norm': float(np.mean(rn)) if rn else np.nan,
        'cos_to_base': float(np.mean(cs)) if cs else np.nan,
        'grid_first': ','.join(f'{v:g}' for v in (
            [0.0] if mode == 'only' else
            ADAPTER_ALPHA_GRID if mode == 'fixed' else WD_GATE_GRID)),
        'grid_wd_repr': ','.join(f'{v:g}' for v in wd_repr_grid),
        'w_head_norm': float(np.mean(hn)) if hn else np.nan,
        'w_gate_norm': np.nan, 'top_vs_third': np.nan, 'C_median': np.nan,
    }


ROW_ORDER = [
    'A320_sEHR_pre',
    '1_EHR_post_frozen',
    'A_sEHR_pca128',
    'B_PPG_post',
    'F_sEHR_plus_scalars',
    'C_concat',
    'D_fixed_alpha',
    'E_learned_alpha',
    'E_sklearnhead',
    'G_adapter_only',
    'G_adapter_fixed_alpha',
    'G_adapter_learned_alpha',
    'H_mlp_only',
    'H_mlp_fixed_alpha',
    'H_mlp_learned_alpha',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default=FEATURES_PATH)
    ap.add_argument('--metadata', default=METADATA_CSV)
    ap.add_argument('--out-dir', default=reg.OUT_DIR)
    ap.add_argument('--disease', choices=list(DISEASES) + ['all'], default='all')
    ap.add_argument('--gate-input', choices=['scalars', 'emb'],
                    default='scalars',
                    help=('what the gate reads. "scalars" is 5 parameters and '
                          'is the variant with a real chance against 42-124 '
                          'training positives; "emb" adds both 128-d blocks '
                          'for 261 parameters and is run as the control.'))
    ap.add_argument('--epochs', type=int, default=N_EPOCHS_DEFAULT,
                    help='max AdamW steps per candidate (full batch)')
    ap.add_argument('--cache-dir', default=None,
                    help=('directory for the Pipeline transformer cache '
                          '(default: <out-dir>/.alpha_fusion_cache). It is '
                          'wiped at startup so a stale entry can never be '
                          'read; pass "none" to disable caching entirely.'))
    ap.add_argument('--eval-every', type=int, default=3,
                    help=('score the inner-validation split every k-th step '
                          'during candidate selection. Full-batch updates '
                          'barely differ between consecutive steps, and the '
                          'validation pass is 57%% of the wall clock at k=1; '
                          'k=3 cuts a per-disease run from ~31 to ~20 min. '
                          'PATIENCE stays measured in epochs.'))
    ap.add_argument('--adapter', action='store_true',
                    help=('add the residual-adapter rows: a low-rank, zero-'
                          'initialised residual on the FROZEN aligned '
                          'v_ehr_post, trained jointly with the gate and head '
                          'on downstream labels. Off by default so an '
                          'unflagged run reproduces the existing table '
                          'exactly. The adapter sees labels, so the PPG '
                          'increment has to be G_adapter_fusion minus '
                          'G_adapter_only -- never against A320.'))
    ap.add_argument('--mlp-sehr', action='store_true',
                    help=('add the H_ rows: the SAME architecture with the '
                          'frozen aligned vector removed, so the sEHR side is '
                          'a task-trained MLP(v_ehr_pre) alone. Answers "can a '
                          'supervised sEHR encoder carry the fusion", and '
                          'forfeits the shared-coordinate argument. '
                          'H_mlp_only is again the matched control -- the MLP '
                          'sees labels, so the PPG increment is fusion minus '
                          'it.'))
    ap.add_argument('--mlp-rank', type=int, default=32,
                    help=('bottleneck for the H_ rows. NOTE this is far more '
                          'binding than --adapter-rank: as a residual, rank r '
                          'only has to carry the CORRECTION to a 128-d base, '
                          'while here it carries the entire sEHR '
                          'representation. A_sEHR_pca128 ~ A320 says 128 '
                          'components lose nothing, so this defaults to 32 '
                          "rather than the adapter's 16. If H_mlp_only still "
                          'comes in well below A320, raise it to 64 before '
                          'concluding anything about supervised training.'))
    ap.add_argument('--adapter-rank', type=int, default=16,
                    help='bottleneck width; 16 is 7168 adapter weights')
    ap.add_argument('--wd-adapter', default='0.01,0.1,1,10,30',
                    help=('decays for the adapter weights, searched jointly '
                          'with the rest. Defaults to WD_GRID, the SAME grid '
                          'the gate and head use -- see the comment on '
                          'WD_GRID for why a separate, wider grid on this '
                          'axis would be an assumption rather than a result. '
                          'Still a separate '
                          'flag, because the adapter is the one encoder here '
                          'that CAN be driven to zero (the model then falls '
                          'back on the frozen aligned vector), so if it pins '
                          'itself to 30 while the head does not, that is a '
                          'real asymmetry worth overriding this with.'))
    ap.add_argument('--wd-mlp', default='0.01,0.1,1,10,30',
                    help=('decays for the H_ MLP. Defaults to WD_GRID, the '
                          'same grid every other decay axis uses. It was '
                          'deliberately weaker than --wd-adapter once, on the '
                          'argument that an adapter bounds a CORRECTION while '
                          'this MLP IS the representation. That distinction '
                          'is real but it does not say the two grids should '
                          'differ, and holding the encoder axis two decades '
                          'above the head axis made G_ vs H_ vs M_ vs N_ '
                          'differ by regularisation as well as by '
                          'architecture. Separate flag kept for overrides.'))
    ap.add_argument('--lr', type=float, default=None,
                    help=('fix the learning rate instead of selecting it, '
                          'collapsing LR_GRID_E to this single value. Three '
                          'reasons, all measured. (1) The inner-validation gap '
                          'between the best and third-best candidate is an '
                          'order of magnitude smaller than the fold-to-fold '
                          'spread, so an argmax over hundreds of candidates is '
                          'mostly selecting noise, and removing an axis '
                          'divides the count by three. (2) A run exists in '
                          'which lr selection picked a value that generalises '
                          'worse: the cells that collapsed are exactly the '
                          'ones whose inner CV chose the largest rate. (3) lr '
                          'is an optimisation detail, not a quantity this '
                          'analysis reports, unlike the decays and alpha. The '
                          'value to pin is the modal choice across folds on '
                          'the clean run, which the log prints. '
                          'Left unset the grid is searched as before, so the '
                          'choice is visible in the command and the log rather '
                          'than fixed in the code.'))
    ap.add_argument('--lr-adapter', type=float, default=None,
                    help=('separate learning rate for the ADAPTER / MLP '
                          'parameter groups only; gate and head keep --lr. '
                          'Needed because two constraints, not one, keep the '
                          'encoder from being small. The rho cap makes '
                          'wd_adapter unreachable above MAX_RHO/lr, which at '
                          'the shared lr=0.01 is exactly the 30 the search '
                          'pinned itself to. And Adam puts a floor under the '
                          'encoder that no decay can cross: its step is ~lr '
                          'per coordinate regardless of gradient, so '
                          '||W_up|| >~ lr*sqrt(2048), which at the shared lr '
                          'is the same order as the smallest norm any fold '
                          'actually reached. Raising the '
                          'decay cannot reach "adapter almost off"; only a '
                          'smaller step can.'))
    ap.add_argument('--skip-e', action='store_true',
                    help=('skip E_learned_alpha and E_sklearnhead. They are '
                          'the only expensive base rows (~12 min of the ~16) '
                          'and they do not beat A320. Their sEHR side is '
                          'PCA-128 of '
                          'v_ehr_pre and shares nothing with the adapter '
                          'rows, so dropping them changes no other number. '
                          'OFF by default so an unflagged run still '
                          'reproduces the full table.'))
    ap.add_argument('--oof-alpha', action='store_true',
                    help=("out-of-fold alpha for the training rows, so stage "
                          "2's C is not selected on rows the gate early-"
                          "stopped on. Costs one extra gate fit per inner "
                          "fold; the reported AUROC is clean either way."))
    args = ap.parse_args()
    args.wd_adapter = [float(v) for v in str(args.wd_adapter).split(',') if v]
    if args.lr is not None:
        global LR_GRID_E
        LR_GRID_E = [float(args.lr)]
    args.wd_mlp = [float(v) for v in str(args.wd_mlp).split(',') if v]


    global _MEMORY
    if str(args.cache_dir).lower() != 'none':
        cdir = args.cache_dir or os.path.join(args.out_dir,
                                              '.alpha_fusion_cache')
        # Wiped rather than reused across runs: the cache is keyed on the
        # transformer's params and the input array, so a stale entry cannot be
        # silently wrong -- but parallel per-disease jobs sharing one directory
        # can race on eviction, and a fresh directory costs one refit.
        cdir = f'{cdir}_{os.getpid()}'
        shutil.rmtree(cdir, ignore_errors=True)
        _MEMORY = joblib.Memory(location=cdir, verbose=0)
    else:
        cdir = None

    print('=' * 112)
    print('PER-HORIZON LEARNED-ALPHA EMBEDDING FUSION')
    print('=' * 112)

    feat, df = load_joined(args.features, args.metadata)
    both = df[(df['ppg_mask'] == 1) & (df['ehr_mask'] == 1)].reset_index(drop=True)
    print(f"  both-modality cohort: {len(both)} admissions, "
          f"{both['patient_id'].nunique()} patients")

    idx = both['_idx'].to_numpy()
    VE_pre = np.stack([feat[i]['v_ehr_pre_mlp'] for i in idx]).astype(np.float64)
    VE_post = np.stack([feat[i]['v_ehr_post_mlp'] for i in idx]).astype(np.float64)
    VP = np.stack([feat[i]['v_ppg_post_mlp'] for i in idx]).astype(np.float64)
    M = build_scalar_matrix(both)
    groups = both['patient_id'].astype(str).to_numpy()

    print(f"  sEHR: PCA({N_COMPONENTS}) of v_ehr_pre ({VE_pre.shape[1]}d), "
          f"block-scaled, fitted per outer fold")
    print(f"  PPG : v_ppg_post_mlp ({VP.shape[1]}d), standardised per outer fold")
    print(f"  gate: per horizon, NO horizon input, reads "
          f"{'the 4 scalars (5 params)' if args.gate_input == 'scalars' else 'ze+zp+scalars (261 params)'}")
    print("  E: ONE end-to-end AdamW model per horizon (gate + head together)")
    print(f"     wd_gate {WD_GATE_GRID}")
    print(f"     wd_head {WD_HEAD_GRID}   (head BIAS never decayed, as in sklearn)")
    print(f"     lr      {LR_GRID_E}"
          + ('   (FIXED by --lr, not selected)' if args.lr is not None else ''))
    print(f"     -> {len(candidates())} candidates, each trained on "
          f"{N_INNER_E} inner folds, ranked by mean inner-val AUROC;")
    print(f"        winner refitted from scratch on the FULL outer-train for "
          f"the median best epoch, max {args.epochs}")
    print(f"  every sklearn row: C from C_GRID by inner {reg.N_FOLDS_INNER}-fold"
          f" -- the same protocol, which is what makes 'vs A320' like-for-like")
    print(f"  training-row alpha: {'out-of-fold' if args.oof_alpha else 'in-fold (C selection mildly optimistic)'}")
    if args.skip_e:
        print("  --skip-e: E_learned_alpha / E_sklearnhead NOT computed")
    print(f"  inner-val scored every {args.eval_every} step(s); "
          f"patience {gf.PATIENCE} epochs")
    print(f"  pipeline cache: {cdir or 'disabled'}")
    print()

    args.pre_dim = VE_pre.shape[1]
    if args.lr_adapter is not None:
        print(f"  adapter/MLP groups use lr={args.lr_adapter:g} "
              f"(gate and head keep {LR_GRID_E})")
    if args.adapter:
        print(f"  ADAPTER rows on: residual {args.pre_dim}->"
              f"{args.adapter_rank}->{N_COMPONENTS} on FROZEN v_ehr_post, "
              f"zero-init output, wd_adapter from {args.wd_adapter}")
        print( "     aligned blocks used RAW (already L2-normalised); only the "
               "adapter's v_ehr_pre input is standardised")
        print( "     read G_adapter_fusion - G_adapter_only, NOT vs A320: the "
               "adapter sees downstream labels")
    if args.mlp_sehr:
        print(f"  MLP-sEHR rows on: {args.pre_dim}->{args.mlp_rank}->"
              f"{N_COMPONENTS} on v_ehr_pre, NO aligned base, default init")
        print( "     the aligned space is NOT used on the sEHR side here; the "
               "shared-coordinate argument does not apply")
        print(f"     rank {args.mlp_rank} carries the WHOLE sEHR "
               "representation, not just a correction -- if H_mlp_only lands "
               "well below\n     A320, raise --mlp-rank before concluding")
    print()

    diseases = DISEASES if args.disease == 'all' else (args.disease,)
    all_rows = []

    for disease in diseases:
        print('\n' + '#' * 112)
        print(f'DISEASE: {disease}')
        print('#' * 112)
        fold_of = assign_folds(both, disease)
        valid = horizon_valid_folds(both, fold_of, disease)

        for horizon in HORIZONS:
            s = pd.to_numeric(both[label_col(disease, horizon)], errors='coerce')
            usable = s.notna().to_numpy()
            y = s[usable].to_numpy(dtype=np.int64)
            g_h = groups[usable]
            E_h, P_h, M_h = VE_pre[usable], VP[usable], M[usable]
            folds_h = subset_folds(fold_of, usable)
            skipped = sorted(set(range(reg.N_FOLDS_OUTER)) - valid[horizon])
            print(f"  {horizon}d usable={len(y)}, pos={int(y.sum())}, "
                  f"censored={int((~usable).sum())}"
                  + (f", folds {skipped} unusable" if skipped else ""))

            Zpost_h = VE_post[usable]
            X_ep = np.hstack([E_h, P_h])              # BlockPrep input
            X_epx = np.hstack([E_h, P_h, M_h])        # + scalars

            jobs = [
                (E_h, est_plain, None, 'A320_sEHR_pre'),
                (X_ep, lambda: est_block('ehr'), None, 'A_sEHR_pca128'),
                (X_ep, lambda: est_block('ppg'), None, 'B_PPG_post'),
                (X_epx, lambda: est_block('ehr+extra'), None,
                 'F_sEHR_plus_scalars'),
                (X_ep, lambda: est_block('both'), None, 'C_concat'),
                (X_ep, est_mixed, ALPHA_GRID, 'D_fixed_alpha'),
                # The frozen aligned sEHR probe, so the table shows how much
                # of the 1_EHR_post deficit the adapter actually repairs.
                (Zpost_h, est_plain, None, '1_EHR_post_frozen'),
            ]
            for X, mk, agrid, tag in jobs:
                r = reg.run_cv(X, y, g_h, tag, folds=folds_h,
                               make_est=mk,
                               fusion=agrid is not None,
                               dim=N_COMPONENTS,
                               alpha_grid=agrid)
                r.update(setting=tag, disease=disease, horizon=horizon,
                         n_samples=len(y), n_pos=int(y.sum()))
                all_rows.append(r)

            if args.skip_e:
                extra = []
            else:
                rE, rEt = run_learned_alpha(X_ep, M_h, y, g_h, folds_h,
                                            args.gate_input, args)
                extra = [rE, rEt]
            if args.adapter or args.mlp_sehr:
                X_ad = np.hstack([E_h, Zpost_h, P_h])
                for ub in ([True] if args.adapter else []) + \
                          ([False] if args.mlp_sehr else []):
                    for mode in ('only', 'fixed', 'learned'):
                        extra.append(run_adapter(X_ad, M_h, y, g_h, folds_h,
                                                 args.gate_input, args, mode,
                                                 ub))
            for r in extra:
                r.update(disease=disease, horizon=horizon,
                         n_samples=len(y), n_pos=int(y.sum()))
                all_rows.append(r)

    res = pd.DataFrame(all_rows)

    # ── tables ──
    for disease in diseases:
        for horizon in HORIZONS:
            sub = res[(res.disease == disease) & (res.horizon == horizon)]
            if sub.empty:
                continue
            print('=' * 112)
            print(f"{disease} {horizon}d   n={int(sub.iloc[0]['n_samples'])}  "
                  f"n_pos={int(sub.iloc[0]['n_pos'])}")
            print('=' * 112)
            print(f"  {'setting':24s} {'AUROC':>14s} {'AUPRC':>14s} "
                  f"{'alpha':>14s} {'vs A320':>9s} {'folds':>6s}")
            ref = sub[sub.setting == 'A320_sEHR_pre']
            base = float(ref.iloc[0]['auroc_mean']) if len(ref) else np.nan
            for tag in ROW_ORDER:
                rr = sub[sub.setting == tag]
                if rr.empty:
                    continue
                r = rr.iloc[0]
                if np.isfinite(r.get('alpha_mean', np.nan)):
                    # Show the spread that carries information. On a learned
                    # alpha that is the WITHIN-fold spread across admissions --
                    # it answers whether the gate adapted at all. On a frozen
                    # alpha the within-fold spread is zero by construction and
                    # tells the reader nothing, so the ACROSS-fold spread of
                    # the SELECTED value is shown instead, suffixed 'f'. That
                    # number is the one that matters there: a single cell can
                    # select alpha anywhere across the grid in its five folds.
                    w = r.get('alpha_std_within', np.nan)
                    ac = r.get('alpha_std_across', np.nan)
                    if np.isfinite(w) and w > 1e-9:
                        a = f"{r['alpha_mean']:.3f}±{w:.3f}"
                    elif np.isfinite(ac):
                        a = f"{r['alpha_mean']:.3f}±{ac:.3f}f"
                    else:
                        a = f"{r['alpha_mean']:.3f}±{w:.3f}"
                elif np.isfinite(r.get('alpha_median', np.nan)):
                    a = f"{r['alpha_median']:.2f} sel"
                else:
                    a = '--'
                print(f"  {tag:24s} {r['auroc_mean']:>7.3f}±"
                      f"{r['auroc_std_fold']:.3f} {r['auprc_mean']:>7.3f}±"
                      f"{r['auprc_std_fold']:.3f} {a:>14s} "
                      f"{r['auroc_mean'] - base:>+9.3f} "
                      f"{int(r['n_valid_folds']):>4d}/{reg.N_FOLDS_OUTER}")
            print()

    # ── paired deltas against A320, per fold ──
    print('=' * 112)
    print('PAIRED vs A320_sEHR_pre  (same outer folds, so difference per fold)')
    print('=' * 112)
    print('  A fold is hard or easy for every probe alike, so the spread of the')
    print('  per-fold DIFFERENCE is much smaller than the difference of the two')
    print('  fold spreads. wins counts folds, not cells.\n')
    print(f"  {'setting':24s} {'mean d':>8s} {'sd d':>8s} {'wins':>8s} "
          f"{'cells+':>8s}")
    for tag in ROW_ORDER[1:]:
        diffs = []
        cells = 0
        n_cells = 0
        for disease in diseases:
            for horizon in HORIZONS:
                sub = res[(res.disease == disease) & (res.horizon == horizon)]
                a = sub[sub.setting == 'A320_sEHR_pre']
                b = sub[sub.setting == tag]
                if a.empty or b.empty:
                    continue
                fa = str(a.iloc[0].get('auroc_per_fold', '') or '')
                fb = str(b.iloc[0].get('auroc_per_fold', '') or '')
                if not fa or not fb:
                    continue
                va = np.array([float(v) for v in fa.split(',')])
                vb = np.array([float(v) for v in fb.split(',')])
                if len(va) != len(vb):
                    continue
                diffs.extend(vb - va)
                n_cells += 1
                cells += int(vb.mean() > va.mean())
        if not diffs:
            continue
        d = np.array(diffs)
        print(f"  {tag:24s} {d.mean():>+8.3f} {d.std():>8.3f} "
              f"{int((d > 0).sum()):>4d}/{len(d):<3d} {cells:>4d}/{n_cells:<3d}")
    print()

    # ── what got selected, and whether the grid was wide enough ──
    # An axis pinned to a grid edge in EVERY fold does not mean that value is
    # right; it means the search wanted to keep going, and an undertrained
    # model is indistinguishable from a weak one until the selection is
    # inspected.
    # These are diagnostics only -- none of them feeds back into the grid
    # automatically.
    # Skipped entirely when --skip-e removed the only rows it describes,
    # rather than printing a header over nothing.
    if not args.skip_e:
      print('=' * 112)
      print('SELECTED HYPERPARAMETERS AND DIAGNOSTICS  (inner CV only, never '
            'test)')
      print('=' * 112)
      edges = {'wd_gate': 0, 'wd_head': 0, 'lr': 0}
      hi = {'wd_gate': max(WD_GATE_GRID), 'wd_head': max(WD_HEAD_GRID),
            'lr': max(LR_GRID_E)}
      lo = {'wd_gate': min(WD_GATE_GRID), 'wd_head': min(WD_HEAD_GRID),
            'lr': min(LR_GRID_E)}
      lo_edges = {'wd_gate': 0, 'wd_head': 0, 'lr': 0}
      n_cells = n_nosel = 0
      print(f"  {'cell':12s} {'alpha':>13s} {'|w_gate|':>9s} {'|w_head|':>9s} "
            f"{'epoch':>16s}")
      for disease in diseases:
          for horizon in HORIZONS:
              rr = res[(res.disease == disease) & (res.horizon == horizon)
                       & (res.setting == 'E_learned_alpha')]
              if rr.empty:
                  continue
              r = rr.iloc[0]
              got = {k: [float(v) for v in str(r[c]).split(',') if v]
                     for k, c in [('wd_gate', 'sel_wd_gate'),
                                  ('wd_head', 'sel_wd_head'),
                                  ('lr', 'sel_lr')]}
              if not got['lr']:
                  continue
              n_cells += 1
              n_nosel += int(r.get('folds_without_selection', 0) or 0)
              marks = []
              for k in edges:
                  if all(v == hi[k] for v in got[k]):
                      edges[k] += 1
                      marks.append(f'{k}@max')
                  elif all(v == lo[k] for v in got[k]):
                      lo_edges[k] += 1
                      marks.append(f'{k}@min')
              print(f"  {disease:6s}{horizon:>4d}d "
                    f"{r['alpha_mean']:>6.3f}±{r['alpha_std_within']:.3f} "
                    f"{r['w_gate_norm']:>9.3f} {r['w_head_norm']:>9.3f} "
                    f"{str(r['sel_epoch']):>16s}"
                    + ('   <-- ' + ', '.join(marks) if marks else ''))
              print(f"      wd_gate {r['sel_wd_gate']}   "
                    f"wd_head {r['sel_wd_head']}   lr {r['sel_lr']}")
      print()
      for k in edges:
          for cnt, side in ((edges[k], 'max'), (lo_edges[k], 'min')):
              if cnt:
                  print(f"  !! {k} sat at the {side} of its grid in every fold of "
                        f"{cnt}/{n_cells} cells -- widen that axis and rerun "
                        f"before reading E.")
      if n_nosel:
          print(f"  !! {n_nosel} outer fold(s) had too few positive patients for "
                f"{N_INNER_E} usable inner folds.\n"
                f"     Those ran the full budget at fixed hyperparameters with NO "
                f"early stopping and NO selection.")
      gaps = res[res.setting == 'E_learned_alpha']['top_vs_third'].dropna()
      if len(gaps):
          print(f"\n  best candidate vs 3rd best, mean inner-val AUROC gap: "
                f"{gaps.mean():+.4f}")
          print( "  If that is small next to the fold-to-fold spread, the 48 "
                 "candidates were not\n  distinguishable and the selected values "
                 "are not identified by the inner split.")
      print()

    # ── the adapter's own increment: fusion minus the adapter alone ──
    for pre, on, what in (('G_adapter', args.adapter,
                           'RESIDUAL on the frozen aligned vector'),
                          ('H_mlp', args.mlp_sehr,
                           'task-trained MLP(v_ehr_pre), NO aligned base')):
      if on:
        print('=' * 112)
        print(f'{pre} ROWS -- {what}')
        print('  (it sees downstream labels, so read the LAST column)')
        print('=' * 112)
        print(f'  {pre}_only is the matched control: same encoder, same '
              'protocol, same folds,\n  no PPG. BOTH comparisons are '
              'valid and leakage-free -- the encoder is fitted\n  inside the '
              'outer-training fold and the test fold is never touched -- but '
              'they\n  answer different questions:\n\n'
              '    vs A320    does this multimodal model beat the sEHR '
              'baseline? A headline.\n'
              '    vs _only   did PPG contribute? The fusion differs from '
              'A320 in TWO ways,\n'
              '               a supervised encoder AND PPG, so only this one '
              'attributes.\n\n'
              '  It matters for a LOSS too: if _only sits below A320 the '
              'fusion inherits a\n  handicapped sEHR side, and losing to '
              'A320 says nothing about PPG.\n')
        print(f"  {'cell':12s} {'A320':>7s} {'1_post':>7s} {'adapt':>7s} "
              f"{'+PPG fix':>9s} {'+PPG lrn':>9s} {'fix-only':>9s} "
              f"{'lrn-only':>9s}")
        for disease in diseases:
            for horizon in HORIZONS:
                sub = res[(res.disease == disease) & (res.horizon == horizon)]

                def g(t):
                    rr = sub[sub.setting == t]
                    return (float(rr.iloc[0]['auroc_mean'])
                            if len(rr) else np.nan)
                a0, fz = g('A320_sEHR_pre'), g('1_EHR_post_frozen')
                on = g(f'{pre}_only')
                fx, ln = g(f'{pre}_fixed_alpha'), g(f'{pre}_learned_alpha')
                print(f"  {disease:6s}{horizon:>4d}d {a0:>7.3f} {fz:>7.3f} "
                      f"{on:>7.3f} {fx:>9.3f} {ln:>9.3f} "
                      f"{fx - on:>+9.3f} {ln - on:>+9.3f}")
        print()
        print('  adapt - 1_post  is how much of the alignment deficit was '
              'repaired.')
        print('  adapt - A320    is whether the sEHR side reached the '
              'baseline at all.')
        print('  fix-only / lrn-only is the PPG increment. THAT is the result.')
        print()

        # ── the headline, split into the two things that produce it ──
        # fusion - A320 is what a reader wants; it is also the sum of two
        # effects that pull in opposite directions and have different fixes:
        #
        #   fusion - A320  =  (encoder - A320)  +  (fusion - encoder)
        #                      sEHR side           PPG increment
        #
        # A negative headline with a positive PPG term means the sEHR side is
        # the problem and the fusion is not; a positive sEHR term with a zero
        # PPG term means the opposite. Printed because reading it off two
        # columns by hand is where transcription errors come from, and because
        # the identity is an arithmetic self-check on the rows above.
        print(f"  {'DECOMPOSITION':13s} fusion - A320  =  (encoder - A320)  +"
              f"  (fusion - encoder)")
        print(f"  {'cell':11s} {'sEHR side':>10s} | {'PPG fix':>8s} "
              f"{'= fix vs A320':>14s} | {'PPG lrn':>8s} {'= lrn vs A320':>14s}")
        for disease in diseases:
            for horizon in HORIZONS:
                sub = res[(res.disease == disease) & (res.horizon == horizon)]

                def g(t):
                    rr = sub[sub.setting == t]
                    return (float(rr.iloc[0]['auroc_mean'])
                            if len(rr) else np.nan)
                a0 = g('A320_sEHR_pre')
                on = g(f'{pre}_only')
                fx, ln = g(f'{pre}_fixed_alpha'), g(f'{pre}_learned_alpha')
                print(f"  {disease:6s}{horizon:>4d}d {on - a0:>+10.3f} | "
                      f"{fx - on:>+8.3f} {fx - a0:>+14.3f} | "
                      f"{ln - on:>+8.3f} {ln - a0:>+14.3f}")
        print()

        # ── what the encoder actually did, and what was selected ──
        # ||W_up|| alone cannot separate "the encoder barely moved" from "the
        # encoder rewrote the vector": the output also depends on the down-
        # projection. ||A(x)|| and cos(z_new, z_post) can, and for the G_ rows
        # they decide whether "residual on the pretrained space" is a
        # description or a slogan. A grid edge taken in EVERY fold is the
        # other thing worth seeing -- it says the search wanted to keep going.
        print(f"  {'cell':11s} {'variant':14s} {'|A(x)|':>7s} {'cos':>6s} "
              f"{'|W_up|':>7s} {'|w_h|':>6s} {'a sd/fold':>10s}")
        grids = {'wd_head': WD_HEAD_GRID, 'lr': LR_GRID_E}
        for disease in diseases:
            for horizon in HORIZONS:
                for mode in ('only', 'fixed_alpha', 'learned_alpha'):
                    rr = res[(res.disease == disease)
                             & (res.horizon == horizon)
                             & (res.setting == f'{pre}_{mode}')]
                    if rr.empty:
                        continue
                    r = rr.iloc[0]
                    g = dict(grids)
                    for k, c in (('first', 'grid_first'),
                                 ('wd_repr', 'grid_wd_repr')):
                        g[k] = [float(v) for v in str(r[c]).split(',') if v]
                    first_nm = 'alpha' if mode != 'learned_alpha' else 'wd_gate'
                    marks = []
                    for key, col, nm in (('first', 'sel_wd_gate', first_nm),
                                         ('wd_head', 'sel_wd_head', 'wd_head'),
                                         ('lr', 'sel_lr', 'lr'),
                                         ('wd_repr', 'sel_wd_adapter',
                                          'wd_repr')):
                        got = [float(v) for v in str(r[col]).split(',') if v]
                        if not got or len(g[key]) < 2:
                            continue
                        if all(v == max(g[key]) for v in got):
                            marks.append(f'{nm}@max')
                        elif all(v == min(g[key]) for v in got):
                            marks.append(f'{nm}@min')
                    print(f"  {disease:6s}{horizon:>4d}d {mode:14s} "
                          f"{r['resid_norm']:>7.3f} {r['cos_to_base']:>6.3f} "
                          f"{r['adapter_norm']:>7.3f} "
                          f"{r['w_head_norm']:>6.3f} "
                          f"{r.get('alpha_std_across', np.nan):>10.3f}"
                          + ('   <-- ' + ', '.join(marks) if marks else ''))
                    print(f"      wd_repr {r['sel_wd_adapter']}"
                          f" | wd_head {r['sel_wd_head']}"
                          f" | lr {r['sel_lr']}"
                          f" | {first_nm} {r['sel_wd_gate']}"
                          f" | epoch {r['sel_epoch']}")
                    if int(r.get('folds_without_selection', 0) or 0):
                        print(f"      !! {int(r['folds_without_selection'])} "
                              f"fold(s) ran with NO selection and NO early "
                              f"stopping")
        print()

    tag_out = (f"_{args.gate_input}"
               + ('_adapter' if args.adapter else '')
               # rank in the name: otherwise a rank-64 run silently
               # overwrites the rank-32 CSV and the comparison the rerun
               # exists to make is destroyed.
               + (f'_mlp{args.mlp_rank}' if args.mlp_sehr else '')
               + ('_oof' if args.oof_alpha else '')
               + ('' if args.disease == 'all' else f'_{args.disease}'))
    os.makedirs(args.out_dir, exist_ok=True)
    f_res = f'alpha_fusion_postdisch_results{tag_out}.csv'
    res.to_csv(os.path.join(args.out_dir, f_res), index=False)
    print(f'Saved {f_res}\n')
    if cdir:
        shutil.rmtree(cdir, ignore_errors=True)

    print('=' * 112)
    print('HOW TO READ')
    print('=' * 112)
    print('  The alpha column shows the spread that carries information.')
    print('  On LEARNED-alpha rows it is the WITHIN-fold spread across')
    print('  admissions: does the gate adapt per admission at all. On rows')
    print('  where alpha is frozen that spread is zero by construction, so')
    print('  the ACROSS-fold spread of the SELECTED value is shown instead,')
    print('  suffixed "f", and repeated as "a sd/fold" in the diagnostics.')
    print('  It is large -- a cell can select alpha anywhere across the grid')
    print('  in its five folds. The inner CV cannot identify the mixing')
    print('  weight.')
    print()
    print('  A320_sEHR_pre is the opponent. Everything here is fitted per')
    print('  horizon on the same outer folds with the same C selection, so')
    print('  "vs A320" is a like-for-like difference.')
    print()
    print('  C_concat contains every fixed mixture as a special case (parallel')
    print('  coefficient vectors), but only in the hypothesis class -- it also')
    print('  carries 256 free coefficients against 42-124 positives, so it can')
    print('  and does score BELOW the mixture. C_concat < D means the parallel')
    print('  constraint is buying generalisation; C_concat > E means plain')
    print('  concatenation is the recommendation and the gate is not earning')
    print('  its place.')
    print()
    print('  E vs D isolates LEARNING alpha: same blocks, same classifier, the')
    print('  only difference is per-admission by gradient vs one alpha by CV.')
    print()
    print('  n_features in the CSV is the INPUT width run_cv was handed (448 =')
    print('  320 sEHR + 128 PPG), not the width the probe actually fits on --')
    print(f'  BlockPrep reduces the sEHR block to {N_COMPONENTS} inside the fold.')
    print()
    print('  E_learned_alpha is ONE scheme end to end: gate and head trained')
    print('  together by AdamW, both decays selected, refitted on the full')
    print('  outer-train. E_sklearnhead takes that same alpha and swaps only')
    print('  the classifier. In gated_fusion that swap alone was worth a wide')
    print('  margin, but that comparison also gave the sklearn side 33% more')
    print('  training rows and left the torch head effectively unregularised.')
    print('  Both are fixed here, so a gap that persists is a real property')
    print('  of the recipe.')


if __name__ == '__main__':
    main()
