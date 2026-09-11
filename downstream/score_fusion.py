"""
SCORE-LEVEL FUSION -- mix two models' OPINIONS, not their feature vectors.

    s_e,i = logit of an EHR probe on v_ehr_pre_mlp   (320-d, native)
    s_p,i = logit of a PPG probe on v_ppg_post_mlp   (128-d, aligned)

    logit_i = (1 - alpha) * s_e,i  +  alpha * s_p,i

WHY, GIVEN WHAT THE OTHER RUNS SHOWED
-------------------------------------
Three findings drive this design, none of them speculative:

  1. v_ehr_pre (320-d) is the strongest sEHR representation, but it cannot
     enter a 128-d mixer. At score level a model's output is one number
     whatever its input dimension, so 320-d and 128-d combine freely.

  2. concat setting 6 beat its own baseline only once the two blocks got
     SEPARATE penalties, and the selected penalty shrank the PPG block
     several-fold harder than the sEHR one in every fold. Score level takes
     that to the limit: C_EHR and C_PPG are two independent hyperparameters,
     each selected on its own modality's inner CV.

  3. A 128-d gate head is more parameters than these positive counts
     support. Mixing two scalars needs no such head, so the gate here is far
     smaller -- a k -> 8 -> 1 MLP on the scalars, printed at startup.

A fourth reason is structural. The embedding-level mixer computes
w'[(1-a)v_e + a v_p], one shared w, which forces both modalities onto the
same discriminant direction (train_probe_regularized.py:150-156) and cannot
express per-modality shrinkage at all. Fitting w_e and w_p separately removes
both restrictions, and no shared coordinate system is needed: two risk scores
are comparable by construction, two embeddings only if alignment made them so.

WHY OUT-OF-FOLD SCORES ARE MANDATORY
------------------------------------
The combiner trains on (s_e, s_p) pairs. Those must NOT be the base models'
in-sample predictions. In sample both towers score far above their test
values, and the sEHR tower inflates more, because it has 320 columns to
memorise with and PPG has 128. A combiner fed those pairs concludes "trust
sEHR almost entirely" for a reason that is about column count, not
information. Out-of-fold scores remove that inflation, so the learned weight
reflects information rather than input width.

KNOWN CAVEATS, both inherited from stacking_fusion.py
-----------------------------------------------------
  * C_EHR / C_PPG are selected on the same inner folds that then produce the
    OOF scores. A stricter design nests one more level; at 52-155 positives
    that is unaffordable. The optimism is small but real.
  * OOF scores come from models fitted on 2/3 of the training fold, while the
    test scores come from models refit on all of it, so the two are on
    slightly different scales. Standardising the score columns on the TRAIN
    OOF statistics absorbs most of it, not all. This is the standard stacking
    compromise, stated rather than hidden.

DELIBERATE DIFFERENCE FROM gated_fusion SETTING 4
-------------------------------------------------
Setting 4 shares one gate across the four horizons and feeds horizon in as an
input. Here each horizon is fitted separately, so horizon is not a gate input
and the long-format censoring expansion is not needed. Sharing existed to let
one model serve four horizons; fitting per horizon removes the reason for it,
and at this gate's size there is no capacity argument for pooling.

THE EXPERT-SUBSET ABLATION
--------------------------
Three experts over the same admission, and every subset of them, so which
experts earn their place is read off a lattice rather than asserted:

    subset   row                   experts mixed
    E        A_EHR_pre             sEHR alone
    P        B_PPG_post            aligned PPG alone
    J        C_joint_concat        one head over [sEHR | PPG] concatenated
    E+P      H_gate_EP
    E+J      H_gate_EJ
    P+J      H_gate_PJ
    E+P+J    H_gate_with_logits    the model this file reports

Every member is fitted the same way -- same towers, same OOF construction,
same standardisation, same candidate grid, inner splits, seed and stopping
rule -- so a difference between subsets is the expert list and nothing else.
The singletons need no gate: with one expert the softmax is 1.0 by
construction, so the mixture equals that expert's own score and A/B/C already
are those members.

One residual confound, unavoidable while the gate reads the expert logits:
its input is 4+k wide, so going from two experts to three also grows the gate
from 74 parameters to 91. A scalars-only gate would hold the input fixed but
is not the model being reported.

PPG_pre is carried alongside but is NOT in the lattice -- it is the same PPG
side before alignment, there to say what alignment bought.

A and B are not fitted separately from the fusion: they ARE the towers the
mixtures read, scored on the outer test rows. Everything that defines how a
linear model is fitted and selected here -- the grids, the rho cap, the inner
split, the per-outer-fold seed, biases exempt from decay -- is imported from
plain_mlp_torch, so A and B should reproduce its A320_sEHR_pre_selC and
PPG128_post rows fold for fold, not merely on average. If they do not, the
two files have drifted and the comparison is not on the same footing as the
rest of the project.

WHAT WAS REMOVED, AND WHAT IT COST
----------------------------------
F_fixed_alpha, S_stacking and G_learned_alpha are gone, along with their
alpha_per_fold, alpha_sd_within and beta_per_fold columns.

F was the fixed-weight control for E+P: with it gone, this file answers "how
many experts" but no longer answers "learned weighting versus one global
weight". If that question comes back, F is the row to restore.

G was the same mixture with a gate that saw only the four scalars, so its
absence removes the "does the gate need the logits" contrast.

Two diagnostics they carried are kept, recomputed from the reported 3-expert
mixture: w_sd_within, the within-fold spread of the PPG weight, near zero
when the gate has collapsed to a constant; and corr_wppg_*, the association
between that weight and each scalar.

Usage:
    python score_fusion.py --disease AF
"""
import os

import argparse
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

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
    DISEASES, HORIZONS, FEATURES_PATH, METADATA_CSV, SCALAR_FIELDS,
    label_col, load_joined, assign_folds, horizon_valid_folds, subset_folds,
    build_scalar_matrix,
)

# Imported, not re-implemented. Every rule that defines "how a linear model is
# fitted and selected in this project" -- the wd grid, the lr grid, the rho
# cap, biases exempt from decay, the inner split, the epoch budget and its
# early stop -- lives in plain_mlp_torch and is used from there, so the two
# files cannot drift apart. A copy would.
import alpha_fusion as af
import plain_mlp_torch as pmt

N_INNER = reg.N_FOLDS_INNER
SEED = reg.RANDOM_SEED

# The gate's decay and rate come from the SAME grids every other torch model
# in this project searches, under the same rho = lr*wd cap, so nothing here is
# tuned on a private schedule.
#
# The grid must reach high enough to crush the gate weights to zero. That
# matters structurally, not just numerically: with the weights at zero
# alpha_i = sigmoid(bias) is a CONSTANT, so setting G degenerates exactly to
# setting F. Only with F inside G's reachable set does "the learned gate does
# not beat a fixed alpha" mean anything -- otherwise it only means the gate
# was never allowed to stop varying.


def inner_splits(y, groups, seed):
    """The inner 3-fold every model in this project selects on.

    pmt.inner_folds, not a local construction: it stratifies on "this patient
    is ever positive" and groups by patient.

    `seed` is the OUTER FOLD's seed, reg.RANDOM_SEED + fold_id, not a module
    constant. Passing a fixed 42 would give every outer fold the same inner
    partition, which is not what plain_mlp_torch does -- and the base towers
    here are meant to reproduce its rows fold for fold, which they cannot if
    they select on different splits.
    """
    return pmt.inner_folds(np.asarray(y), np.asarray(groups), seed)


def _refit_predict(X_fit, y_fit, X_pred, wd, lr, ep, seed):
    """Fit at fixed (wd, lr, epoch) and score X_pred.

    The refit half of pmt.torch_linear_cv, split out because the OOF logits
    below need it at hyperparameters already selected, without re-running the
    selection inside every inner split. Scaler fitted on X_fit only.
    """
    d = X_fit.shape[1]
    T = lambda z: torch.tensor(z, dtype=torch.float32)
    Z = lambda n: T(np.zeros((n, 1)))
    sc = StandardScaler().fit(X_fit)
    Xs_fit, Xs_pred = sc.transform(X_fit), sc.transform(X_pred)
    model, _ = gf._train_once(
        T(Xs_fit), Z(len(X_fit)), Z(len(X_fit)),
        T(y_fit.astype(np.float64)), y_fit,
        np.zeros(len(X_fit), dtype=np.int64), np.arange(len(X_fit)), None, [],
        d, 1, 0, seed, wd, lr, True,
        wd_head=wd, n_epochs=ep, make_model=lambda: pmt.LinearHead(d),
        make_groups=pmt.head_only_groups)
    with torch.no_grad():
        s, _ = model(T(Xs_pred), Z(len(X_pred)), Z(len(X_pred)))
    return s.numpy(), model


def base_tower(X_tr, X_te, y_tr, splits, cands, seed, args, surface=None):
    """One side's base model: test scores, out-of-fold train scores, selection.

    Replaces the sklearn LogisticRegression whose C was chosen by GridSearchCV.
    What played C's role is weight decay on the head -- see pmt.LinearHead for
    why the two are the same knob under different parameterisations -- searched
    jointly with the learning rate on the shared grid.

    Two sets of scores come out, and they are not interchangeable:
      s_te   from a model refit on the whole outer-training fold, for scoring
             the outer test rows;
      oof    one score per training row, each from a model that never saw that
             row, for fitting the fusion on top without leakage.

    The OOF pass reuses the (wd, lr, epoch) selected here rather than
    re-selecting inside every inner split: re-selecting would be a nested
    search whose cost buys nothing, and would leave the fusion reading scores
    produced under different hyperparameters than the test scores it is
    evaluated against.
    """
    s_te, wd, lr, ep = pmt.torch_linear_cv(X_tr, X_te, y_tr, splits, cands,
                                           seed, args, surface=surface)
    oof = np.full(len(y_tr), np.nan)
    for i_tr, i_va in splits:
        if len(np.unique(y_tr[i_tr])) < 2:
            continue
        oof[i_va], _ = _refit_predict(X_tr[i_tr], y_tr[i_tr], X_tr[i_va],
                                      wd, lr, ep, seed)
    return s_te, oof, (wd, lr, ep)


def fit_gate(S, M, y, wd, lr, n_epochs, seed, va=None, eval_every=1):
    """w_i = softmax(g(scalars_i));  logit_i = w_i . (s_e, s_p, s_joint)_i.

    S is (n, 3) standardised OOF scores -- sEHR, PPG, and the joint model over
    both feature blocks -- and M is (n, k) standardised scalars. Returns
    (apply, best_va_auc, best_epoch); the last two are None unless `va`
    supplies held-out (S_va, M_va, y_va).

    The gate is an MLP, k -> 8 -> GELU -> 3, so the weights can depend on
    interactions between the scalars rather than on a linear score of them.
    Softmax generalises the two-expert sigmoid: it keeps the weights
    non-negative and summing to one, so this is still a convex combination of
    opinions and z is still a score-level mixture. The FUSION stays at score
    level -- z is itself the classification logit, with no head after it.

    INITIALISATION. The final layer starts at zero, so it outputs a zero
    vector and every sample begins at w = (1/3, 1/3, 1/3): all three experts
    weighted equally, the neutral start the comparison needs. The FIRST layer
    is Xavier-initialised and NOT zeroed: zeroing both layers also starts at
    uniform weights, but then d(out)/dW2 = 0 and d(out)/dW1 propto W2 = 0, so
    only the final bias ever receives gradient and the gate is frozen at
    constant weights for the whole run. It would not error -- it would quietly
    reduce to a fixed mixture at every weight decay.

    THE BIASES ARE EXCLUDED FROM WEIGHT DECAY. The output biases are what
    encode a constant weighting, so decaying them pulls the mixture toward
    uniform wherever the optimum actually is. Same rule as
    pmt.head_only_groups applies to the linear rows.

    Early stopping mirrors gf._train_once exactly: score the held-out rows
    every eval_every steps, keep the best state, require a 1e-5 improvement
    to count as better, and stop after gf.PATIENCE epochs without one.

    Patience is a real restriction, not just a saving. Running the full budget
    and taking the best step searches strictly more: a validation curve that
    dips and recovers beyond the patience window is found without it and
    missed with it. It is adopted anyway so the gate's epoch is chosen under
    the same rule as the base towers' -- an advantage that comes from
    searching harder than the baseline is not an advantage of the gate.
    """
    Xg = torch.tensor(M, dtype=torch.float32)
    St = torch.tensor(S, dtype=torch.float32)          # (n, 3)
    yt = torch.tensor(y, dtype=torch.float32)
    n_exp = S.shape[1]

    # Seeded per outer fold, as every other torch model here is. The first
    # layer is randomly initialised, so without this a candidate's
    # initialisation would depend on how many candidates ran before it and
    # the selection would move if the loop order changed.
    torch.manual_seed(seed)
    g = nn.Sequential(
        nn.Linear(M.shape[1], GATE_HIDDEN),
        nn.GELU(),
        nn.Linear(GATE_HIDDEN, n_exp),
    )
    nn.init.xavier_uniform_(g[0].weight)
    nn.init.zeros_(g[0].bias)
    nn.init.zeros_(g[2].weight)
    nn.init.zeros_(g[2].bias)       # weights start at exactly uniform

    n_pos = float(yt.sum())
    n_neg = float(len(yt) - n_pos)
    pw = torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    # By name, so BOTH layers' biases are exempt; naming the two tensors
    # directly would have silently decayed the second layer's bias.
    opt = torch.optim.AdamW(
        [{'params': [q for n, q in g.named_parameters()
                     if n.endswith('weight')], 'weight_decay': wd},
         {'params': [q for n, q in g.named_parameters()
                     if n.endswith('bias')], 'weight_decay': 0.0}],
        lr=lr,
    )

    def _apply(S_new, M_new):
        with torch.no_grad():
            w = torch.softmax(
                g(torch.tensor(M_new, dtype=torch.float32)), dim=-1
            ).numpy()
        return (w * S_new).sum(axis=1), w

    # Stopping rule copied from gf._train_once, including the 1e-5 threshold
    # and counting `bad` in epochs rather than evaluations, so PATIENCE means
    # the same number of epochs on both sides.
    best_auc, best_ep, best_state = -np.inf, None, None
    bad = 0
    for step in range(1, n_epochs + 1):
        opt.zero_grad()
        w = torch.softmax(g(Xg), dim=-1)
        loss = lossf((w * St).sum(dim=1), yt)
        loss.backward()
        opt.step()
        if va is not None and step % eval_every == 0:
            S_va, M_va, y_va = va
            z, _ = _apply(S_va, M_va)
            try:
                v = float(roc_auc_score(y_va, z))
            except ValueError:
                # No score, so nothing to judge -- does not count as a step
                # without improvement, matching _train_once.
                continue
            if v > best_auc + 1e-5:
                best_auc, bad, best_ep = v, 0, step
                best_state = {k: t.detach().clone()
                              for k, t in g.state_dict().items()}
            else:
                bad += eval_every
                if bad >= gf.PATIENCE:
                    break
    if best_state is not None:
        g.load_state_dict(best_state)
    return _apply, (best_auc if best_state is not None else None), best_ep


# The expert-subset ablation. Column indices into S = (s_E, s_P, s_J), so a
# subset is just a column slice -- fit_gate is already generic in the number
# of experts (it reads S.shape[1] and sizes its softmax to match), which is
# why widening or narrowing the mixture needs no architecture change.
#
# The three singletons are NOT re-fitted here: with one expert the softmax is
# 1.0 by construction, so the mixture equals that expert's own score and the
# existing A/B/C rows already are those members of the lattice.
GATE_HIDDEN = 8         # the (scalars + logits) -> GATE_HIDDEN -> k gate


def n_experts_of(row):
    """How many experts a row mixes, read off SUBSET_OF.

    Derived rather than listed a second time, so the lattice has one source of
    truth. NaN for rows outside the lattice, which is only PPG_pre.
    """
    sset = SUBSET_OF.get(row, '')
    if not sset or sset.startswith('('):
        return np.nan
    return len(sset.split('+'))


def gate_params(n_scalars, n_exp, hidden=GATE_HIDDEN):
    """Parameters in the gate that mixes n_exp experts.

    Derived from the same constant fit_gate builds with, so this cannot go
    stale the way a hard-coded count did: the printed number was right for a
    gate reading only the scalars and silently wrong once the reported gate
    started reading the logits too.
    """
    d_in = n_scalars + n_exp      # the gate reads the scalars AND the logits
    return d_in * hidden + hidden + hidden * n_exp + n_exp


ABLATION_PAIRS = (
    ('H_gate_EP', (0, 1)),
    ('H_gate_EJ', (0, 2)),
    ('H_gate_PJ', (1, 2)),
)

# Which subset each reported row is, so the lattice is readable in the CSV
# whatever the row happens to be named.
SUBSET_OF = {
    'A_EHR_pre': 'E', 'B_PPG_post': 'P', 'C_joint_concat': 'J',
    'H_gate_EP': 'E+P', 'H_gate_EJ': 'E+J', 'H_gate_PJ': 'P+J',
    'H_gate_with_logits': 'E+P+J',
    # The two representations the fusion does NOT use, reported so the
    # {sEHR, PPG} x {pre, post} square is complete. Outside the lattice: the
    # mixture reads sEHR_pre and PPG_post, and adding a fourth expert would
    # change the ablation rather than describe it.
    'PPG_pre': '(baseline)', 'sEHR_post': '(baseline)',
}

# The rows that carry a confidence interval: the seven-member ablation
# lattice plus the unaligned-PPG baseline.
CI_ROWS = ('A_EHR_pre', 'sEHR_post', 'PPG_pre', 'B_PPG_post',
           'C_joint_concat',
           'H_gate_EP', 'H_gate_EJ', 'H_gate_PJ', 'H_gate_with_logits')

# The unimodal representations, reported on identical footing: same
# base_tower, same inner splits, same grid, same seed, same scaling. Two are
# also lattice experts; the other two exist so alignment can be read on each
# side separately.
UNIMODAL = ('A_EHR_pre', 'sEHR_post', 'PPG_pre', 'B_PPG_post')

# The expert-subset lattice, and ONLY it. The paired delta below compares
# subsets of the mixture; a unimodal baseline that the mixture never reads has
# no place in that comparison, and filtering by exclusion let sEHR_post in.
ABLATION_ROWS = ('A_EHR_pre', 'B_PPG_post', 'C_joint_concat',
                 'H_gate_EP', 'H_gate_EJ', 'H_gate_PJ',
                 'H_gate_with_logits')


def gate_input(M_z, S_z, use_logits):
    """What the gate reads: the scalars, optionally with the OOF logits.

    The logits are ALREADY out-of-fold -- each is a score for a row from a
    model that never saw it -- so feeding them to the gate adds no leakage.
    What it adds is self-assessment: the gate can learn "when the sEHR score
    is extreme, trust the sEHR expert", which a gate that sees only
    acquisition metadata cannot express.

    They are standardised by the same scaler that standardises the columns
    being mixed, so the gate reads the same numbers the fusion does.
    """
    return np.hstack([M_z, S_z]) if use_logits else M_z


def select_gate(S_raw, M_raw, y, groups, cands, args, seed, use_logits=False,
                surface=None):
    """Pick the gate's (wd, lr, epoch) on an inner split of the OOF rows.

    Takes RAW scores and scalars, not standardised ones, and fits a fresh
    scaler on each inner-TRAINING subset. Standardising the whole outer-train
    block first and splitting afterwards would let the inner-validation rows
    contribute to the mean and variance the selection is then read off, which
    is why plain_mlp_torch also refits the scaler inside every inner split.
    The refit that follows selection uses the outer-train scaler,
    which is correct: those rows are all training data by then.

    Selected rather than pinned so the gate and the fixed alpha it is measured
    against are tuned on the same budget; otherwise G can lose for a reason
    that has nothing to do with adaptivity. The candidate list is the shared
    one, so wd, lr and the rho cap are identical to every other torch model
    in this project.
    """
    splits = inner_splits(y, groups, seed)
    if not splits:
        return gf.DEFAULT_WD, gf.DEFAULT_LR, args.epochs
    rows = []
    for wd_c, lr_c in cands:
        vs, es = [], []
        for itr, iva in splits:
            if len(np.unique(y[itr])) < 2 or len(np.unique(y[iva])) < 2:
                continue
            sc_s = StandardScaler().fit(S_raw[itr])
            sc_m = StandardScaler().fit(M_raw[itr])
            S_i, S_v = sc_s.transform(S_raw[itr]), sc_s.transform(S_raw[iva])
            M_i, M_v = sc_m.transform(M_raw[itr]), sc_m.transform(M_raw[iva])
            G_i, G_v = gate_input(M_i, S_i, use_logits), \
                gate_input(M_v, S_v, use_logits)
            _, v, e = fit_gate(
                S_i, G_i, y[itr], wd_c, lr_c, args.epochs, seed,
                va=(S_v, G_v, y[iva]),
                eval_every=args.eval_every)
            if v is not None:
                vs.append(v)
                es.append(e)
        if vs:
            rows.append({'mean': float(np.mean(vs)),
                         'epoch': int(np.median(es)),
                         'wd': wd_c, 'lr': lr_c})
    if surface is not None:
        surface.extend(rows)
    if not rows:
        return gf.DEFAULT_WD, gf.DEFAULT_LR, args.epochs
    rows.sort(key=lambda r: -r['mean'])
    b = rows[0]
    return b['wd'], b['lr'], b['epoch']


def bootstrap_ci(pred, y, groups, fold_h, ci_rows, auc, ap):
    """Fold-preserving patient-cluster 95% CIs for the rows in ci_rows.

    The point estimate is NOT recomputed here: it stays the mean of the five
    per-fold metrics the fold loop already produced, and the bootstrap only
    supplies the interval around it.

    Each replicate resamples patients WITH REPLACEMENT inside each outer test
    fold separately, keeping a drawn patient's admissions together and
    duplicating the whole cluster when the patient is drawn more than once,
    recomputes the metric within each fold, and averages the five folds with
    equal weight. Predictions are never pooled across folds, and no model is
    refitted.

    The machinery is pmt's, not a second copy: pmt._fold_boot_indices already
    does the per-fold cluster draw and the single-class redraw, and
    pmt.fold_mean_boot already averages the folds. Sharing them keeps this
    interval identical in method to the one plain_mlp_torch reports.

    Valid folds are taken PER MODEL. A fold can be scored for the towers and
    skipped for H, which is scored only when the out-of-fold columns survive
    the coverage guard, so a single shared fold set would silently borrow
    folds H does not have.
    """
    out = {}
    cache = {}
    for k in ci_rows:
        p = pred[k]
        ok = np.isfinite(p)
        if not ok.any():
            continue
        yo, po, go, fo = y[ok], p[ok], groups[ok], fold_h[ok]
        valid = sorted(int(v) for v in np.unique(fo))
        # Only folds where both classes are present can be scored at all.
        valid = [v for v in valid if len(np.unique(yo[fo == v])) >= 2]
        if not valid:
            continue
        # Cached on everything the draws depend on, so rows sharing a fold set
        # share one replicate set and their intervals are comparable draw for
        # draw.
        ck = (ok.tobytes(), tuple(valid))
        if ck not in cache:
            cache[ck] = pmt._fold_boot_indices(yo, go, fo, valid)
        boots, n_redraw = cache[ck]
        aucs, aps = pmt.fold_mean_boot(yo, po, boots, valid)
        lo_a, hi_a = pmt._pct_ci(aucs)
        lo_p, hi_p = pmt._pct_ci(aps)
        out[k] = {
            'auroc_mean': float(np.mean(auc[k])) if auc[k] else np.nan,
            'auroc_ci_low': lo_a, 'auroc_ci_high': hi_a,
            'auprc_mean': float(np.mean(ap[k])) if ap[k] else np.nan,
            'auprc_ci_low': lo_p, 'auprc_ci_high': hi_p,
            'n_valid_folds': len(valid), 'n_boot': len(boots),
            'n_fold_redraws': n_redraw,
        }
    return out


def run_cell(Xe, Xp, Xpp, Xep, M, y, groups, folds, args):
    """One disease x horizon. Returns a dict of per-setting per-fold AUROCs."""
    # A and B are the SAME two towers F, S and G read, scored on the outer
    # test rows. Reported from here rather than from a separate reg.run_cv so
    # the baselines in the table are the experts the fusion actually used.
    keys = ['A_EHR_pre', 'sEHR_post', 'PPG_pre', 'B_PPG_post',
            'C_joint_concat',
            'H_gate_with_logits'] + [t for t, _ in ABLATION_PAIRS]
    auc = {k: [] for k in keys}
    ap = {k: [] for k in keys}
    sels = []
    corrs = {f: [] for f in SCALAR_FIELDS}
    w_sd_within = []   # per fold, the within-fold sd of the PPG weight
    # The inner-validation surface: one record per (stage, model, fold,
    # candidate). Already computed by the selection and previously thrown
    # away, so this costs nothing but the bookkeeping.
    surface = []

    def _grab(stage, model):
        """A fresh list that, once filled, is tagged and appended to surface."""
        buf = []
        surf_bufs.append((stage, model, buf))
        return buf
    # Outer-test predictions kept for the four rows that get a confidence
    # interval. These are the SAME untouched scores the point estimates were
    # computed from -- nothing is refitted for the bootstrap. NaN marks a row
    # whose fold that model was not scored on.
    pred = {k: np.full(len(y), np.nan) for k in CI_ROWS}
    fold_h = np.full(len(y), -1, dtype=int)
    # Selection and mean weights for every gated row, keyed by row name, so
    # each reports its own rather than borrowing another's.
    gate_diag = {t: {'sel': [], 'wmix': []}
                 for t in ('H_gate_with_logits',)
                 + tuple(t for t, _ in ABLATION_PAIRS)}
    cands = pmt.wd_lr_candidates(args)

    for fold_id, (tr, te) in enumerate(folds):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        # Per-fold seed, as every other torch model in this project uses, so a
        # fold's initialisation does not depend on how many folds precede it.
        seed = reg.RANDOM_SEED + fold_id

        sp_in = inner_splits(y[tr], groups[tr], seed)
        surf_bufs = []          # (stage, model, buf) filled during this fold
        se_te, oe, sel_e = base_tower(Xe[tr], Xe[te], y[tr], sp_in, cands,
                                      seed, args,
                                      surface=_grab('expert', 'E'))
        sp_te, op, sel_p = base_tower(Xp[tr], Xp[te], y[tr], sp_in, cands,
                                      seed, args,
                                      surface=_grab('expert', 'P'))
        # THIRD EXPERT: one model over the two feature blocks concatenated,
        # so the gate has a "look at both together" opinion to weigh against
        # the two single-modality ones. Identical treatment -- same inner
        # splits, same candidate grid, same seed, same OOF construction --
        # so a difference between it and the other two is the input, not the
        # protocol. The sEHR block enters at its native 320 and the PPG block
        # at 128, giving 448 in: no bottleneck is introduced, because any
        # bottleneck would have to be fitted and would hand this expert
        # capacity the other two do not have.
        Xj_tr = np.hstack([Xe[tr], Xp[tr]])
        Xj_te = np.hstack([Xe[te], Xp[te]])
        sj_te, oj, sel_j = base_tower(Xj_tr, Xj_te, y[tr], sp_in, cands,
                                      seed, args,
                                      surface=_grab('expert', 'J'))
        # PPG before alignment, through the SAME base_tower with the same
        # inner splits, grid, seed and scaling as A_EHR_pre. Its OOF column is
        # discarded: this is a baseline, not a fourth expert.
        spp_te, _, sel_pp = base_tower(Xpp[tr], Xpp[te], y[tr], sp_in, cands,
                                       seed, args,
                                       surface=_grab('expert', 'PPG_pre'))
        # sEHR after alignment, the fourth corner of the pre/post square.
        # Same base_tower, same everything; its OOF column is discarded for
        # the same reason PPG_pre's is -- a baseline, not a fourth expert.
        sep_te, _, sel_ep = base_tower(Xep[tr], Xep[te], y[tr], sp_in, cands,
                                       seed, args,
                                       surface=_grab('expert', 'sEHR_post'))
        # Each selection carries its OWN tower name. Duplicating the order
        # in the print caption instead lets the caption drift out of step
        # with this tuple and attribute every tower's hyperparameters to the
        # wrong model. A name cannot drift from the value beside it.
        sels.append((('E', sel_e), ('P', sel_p), ('J', sel_j),
                     ('PPGpre', sel_pp), ('sEHRpost', sel_ep)))

        # The three towers on their own, before any fusion.
        auc['A_EHR_pre'].append(roc_auc_score(y[te], se_te))
        ap['A_EHR_pre'].append(average_precision_score(y[te], se_te))
        fold_h[te] = fold_id
        pred['A_EHR_pre'][te] = se_te
        pred['PPG_pre'][te] = spp_te
        pred['B_PPG_post'][te] = sp_te
        pred['C_joint_concat'][te] = sj_te
        pred['sEHR_post'][te] = sep_te
        auc['sEHR_post'].append(roc_auc_score(y[te], sep_te))
        ap['sEHR_post'].append(average_precision_score(y[te], sep_te))
        auc['PPG_pre'].append(roc_auc_score(y[te], spp_te))
        ap['PPG_pre'].append(average_precision_score(y[te], spp_te))
        auc['B_PPG_post'].append(roc_auc_score(y[te], sp_te))
        ap['B_PPG_post'].append(average_precision_score(y[te], sp_te))
        auc['C_joint_concat'].append(roc_auc_score(y[te], sj_te))
        ap['C_joint_concat'].append(average_precision_score(y[te], sj_te))

        # All three towers share sp_in, so a row is out-of-fold in all three
        # or in none: adding the third cannot change which rows survive here,
        # and F and S therefore see exactly the rows they saw before.
        ok = np.isfinite(oe) & np.isfinite(op) & np.isfinite(oj)
        if ok.sum() < 20 or len(np.unique(y[tr][ok])) < 2:
            continue

        S_tr = np.column_stack([oe[ok], op[ok], oj[ok]])
        y_tr = y[tr][ok]
        M_tr = M[tr][ok]

        # Score columns are standardised on the TRAIN OOF statistics so the
        # gate weights are relative WEIGHTS rather than a reflection of the
        # three models' differing logit scales. StandardScaler is per column,
        # so the first two columns are scaled exactly as they were before the
        # third expert existed -- F and S below are unaffected.
        sc_s = StandardScaler().fit(S_tr)
        sc_m = StandardScaler().fit(M_tr)
        S_trz, M_trz = sc_s.transform(S_tr), sc_m.transform(M_tr)

        # Test scores come from base_tower's refit on the FULL outer-training
        # fold, standardised on the same TRAIN OOF statistics.
        S_te = sc_s.transform(np.column_stack([se_te, sp_te, sj_te]))
        M_te = sc_m.transform(M[te])
        y_te = y[te]

        # ── the full 3-expert mixture: the model this file reports ──
        # The gate reads the 4 scalars plus the 3 OOF logits, so it can key on
        # how confident each expert is on this admission as well as on how the
        # admission was acquired.
        h_wd, h_lr, h_ep = select_gate(S_tr, M_tr, y_tr, groups[tr][ok],
                                       cands, args, seed, use_logits=True,
                                       surface=_grab('gate', 'E+P+J'))
        apply_h, _, _ = fit_gate(S_trz, gate_input(M_trz, S_trz, True), y_tr,
                                 h_wd, h_lr, h_ep, seed)
        z_h, W_h = apply_h(S_te, gate_input(M_te, S_te, True))
        pred['H_gate_with_logits'][te] = z_h
        auc['H_gate_with_logits'].append(roc_auc_score(y_te, z_h))
        ap['H_gate_with_logits'].append(average_precision_score(y_te, z_h))
        gate_diag['H_gate_with_logits']['sel'].append((h_wd, h_lr, h_ep))
        gate_diag['H_gate_with_logits']['wmix'].append(
            tuple(float(v) for v in W_h.mean(axis=0)))
        # Within-fold spread of the PPG weight: near zero means the gate
        # collapsed to a constant mixture, which no AUROC reveals.
        w_sd_within.append(float(W_h[:, 1].std()))

        # ── the 2-expert members of the same H family ──
        # Each drops one expert and keeps everything else: the same gate
        # architecture and inputs (scalars plus the surviving logits), the
        # same candidate grid, inner splits, seed and stopping rule. So a
        # difference against the 3-expert row is the missing expert.
        #
        # Slicing the columns of an already-fitted StandardScaler's output is
        # exact, because the scaler is per column -- the surviving columns are
        # scaled as they were with three present.
        #
        # One residual confound, unavoidable while the gate reads the logits:
        # its input is 4+k wide, so going from two experts to three also grows
        # the gate from 74 parameters to 91. The alternative -- a gate that
        # sees only the scalars -- has a fixed input but is not the model
        # being reported.
        for tag, cols in ABLATION_PAIRS:
            cs = list(cols)
            p_wd, p_lr, p_ep = select_gate(S_tr[:, cs], M_tr, y_tr,
                                           groups[tr][ok], cands, args, seed,
                                           use_logits=True,
                                           surface=_grab(
                                               'gate', SUBSET_OF[tag]))
            S_trz_s, S_te_s = S_trz[:, cs], S_te[:, cs]
            apply_p, _, _ = fit_gate(
                S_trz_s, gate_input(M_trz, S_trz_s, True), y_tr,
                p_wd, p_lr, p_ep, seed)
            z_p, W_p = apply_p(S_te_s, gate_input(M_te, S_te_s, True))
            pred[tag][te] = z_p
            auc[tag].append(roc_auc_score(y_te, z_p))
            ap[tag].append(average_precision_score(y_te, z_p))
            gate_diag[tag]['sel'].append((p_wd, p_lr, p_ep))
            gate_diag[tag]['wmix'].append(
                tuple(float(v) for v in W_p.mean(axis=0)))

        # What the gate actually LEARNED, which AUROC does not reveal.
        # On a synthetic control where PPG was informative only when a quality
        # scalar q was high, the gate recovered corr(alpha, q) = +0.98 while
        # scoring BELOW the best fixed alpha on AUROC. So a null AUROC delta
        # does not license "the gate found nothing" -- these correlations are
        # the direct readout of the advisor's question, and they are reported
        # whatever the AUROC does.
        a_te = W_h[:, 1]          # the PPG weight, from the reported mixture
        for j, f in enumerate(SCALAR_FIELDS):
            v = M_te[:, j]
            corrs[f].append(
                float(np.corrcoef(a_te, v)[0, 1])
                if np.std(a_te) > 1e-9 and np.std(v) > 1e-9 else np.nan
            )

        # Flush this fold's candidate records, tagged. Done at the end of the
        # fold so a fold that fell through the coverage guard contributes its
        # tower surfaces and not its gate ones -- which is also why the gate
        # panel can rest on fewer folds than the expert panel, and why the
        # aggregation below counts cells per (stage, model).
        for stage, model, buf in surf_bufs:
            for r in buf:
                surface.append({'stage': stage, 'model': model,
                                'fold': fold_id, 'wd': r['wd'], 'lr': r['lr'],
                                'rho': r['wd'] * r['lr'],
                                'val_auroc': r['mean'], 'epoch': r['epoch']})

    ci = bootstrap_ci(pred, y, groups, fold_h, CI_ROWS, auc, ap)

    return auc, ap, sels, corrs, w_sd_within, ci, gate_diag, surface


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--features', default=FEATURES_PATH)
    ap_.add_argument('--metadata', default=METADATA_CSV)
    ap_.add_argument('--out-dir', default=reg.OUT_DIR)
    ap_.add_argument('--disease', choices=list(DISEASES) + ['all'], default='all')
    # Kept byte-identical to plain_mlp_torch's defaults. Every torch model in
    # this file searches these two grids under the rho cap, so a difference
    # between the two files cannot come from a difference in budget.
    ap_.add_argument('--wd-proj', default='0.01,0.1,1,10,30,100',
                     help='weight decays, searched jointly with --lr under '
                          'the rho = lr*wd cap.')
    ap_.add_argument('--lr', default='0.001,0.003,0.01,0.03',
                     help='learning rates, searched jointly with --wd-proj.')
    ap_.add_argument('--epochs', type=int, default=300)
    ap_.add_argument('--eval-every', type=int, default=3)
    ap_.add_argument(
        '--horizon', type=int, choices=list(HORIZONS) + [0], default=0,
        help=('run one horizon (0 = all). Horizons are independent -- separate '
              'labels, separate fits, nothing shared but the CSV -- so this '
              'changes no number, only which rows land in which file. With '
              '--disease it gives 12 parallel jobs instead of 3.'),
    )
    args = ap_.parse_args()
    # Same parsing and the same rebinding plain_mlp_torch does, so
    # pmt.wd_lr_candidates and the gate below read one grid, not two.
    args.wd_proj = [float(v) for v in str(args.wd_proj).split(',') if v]
    args.lr = [float(v) for v in str(args.lr).split(',') if v]
    af.LR_GRID_E = list(args.lr)
    af.WD_HEAD_GRID = list(args.wd_proj)
    af.WD_GATE_GRID = af.WD_HEAD_GRID

    print('=' * 104)
    print('SCORE-LEVEL FUSION -- mix opinions, not feature vectors')
    print('=' * 104)

    feat, df = load_joined(args.features, args.metadata)
    both = df[(df['ppg_mask'] == 1) & (df['ehr_mask'] == 1)].reset_index(drop=True)
    idx = both['_idx'].to_numpy()
    Xe = np.stack([feat[i]['v_ehr_pre_mlp'] for i in idx]).astype(np.float64)
    Xp = np.stack([feat[i]['v_ppg_post_mlp'] for i in idx]).astype(np.float64)
    # The 512-d PaPaGei representation BEFORE the alignment projection. A
    # reported baseline only -- not an expert, never in the fusion -- so the
    # table can say what the PPG side is worth unaligned, which B_PPG_post
    # (the aligned 128-d) cannot.
    Xpp = np.stack([feat[i]['v_ppg_pre_mlp'] for i in idx]).astype(np.float64)
    # The aligned sEHR side, completing {sEHR, PPG} x {pre, post}. Also a
    # reported baseline only.
    #
    #   Xe   v_ehr_pre_mlp   320   sEHR before alignment   -- expert E
    #   Xep  v_ehr_post_mlp  128   sEHR after alignment    -- baseline
    #   Xpp  v_ppg_pre_mlp   512   PPG before alignment    -- baseline
    #   Xp   v_ppg_post_mlp  128   PPG after alignment     -- expert P
    Xep = np.stack([feat[i]['v_ehr_post_mlp'] for i in idx]).astype(np.float64)
    M = build_scalar_matrix(both)
    groups = both['patient_id'].astype(str).to_numpy()

    print(f"  cohort {len(both)} admissions, {both['patient_id'].nunique()} patients")
    print(f"  EHR base: v_ehr_pre_mlp {Xe.shape[1]}d   "
          f"PPG base: v_ppg_post_mlp {Xp.shape[1]}d")
    print(f"  gate scalars ({M.shape[1]}): {SCALAR_FIELDS}")
    nsc = M.shape[1]
    print(f"  gate parameters: {gate_params(nsc, 3)} for the full E+P+J gate "
          f"({nsc}+3 -> {GATE_HIDDEN} -> 3), "
          f"{gate_params(nsc, 2)} for each pair ({nsc}+2 -> {GATE_HIDDEN} "
          f"-> 2).")
    print(f"  Going from two experts to three grows the gate as well as the "
          f"expert list; that confound is in the ablation and cannot be "
          f"removed while the gate reads the logits.\n")

    diseases = DISEASES if args.disease == 'all' else (args.disease,)
    horizons = HORIZONS if args.horizon == 0 else (args.horizon,)
    if args.disease != 'all' or args.horizon != 0:
        print(f"  running disease={args.disease}  horizon={args.horizon or 'all'}"
              f"  -- concatenate the CSVs afterwards\n")
    rows = []
    ci_rows = []
    surf_rows = []

    for disease in diseases:
        print('\n' + '#' * 104)
        print(f'DISEASE: {disease}')
        print('#' * 104)
        fold_of = assign_folds(both, disease)
        valid = horizon_valid_folds(both, fold_of, disease)

        for horizon in horizons:
            skipped = sorted(set(range(reg.N_FOLDS_OUTER)) - valid[horizon])
            if skipped:
                print(f"  {horizon}d: fold(s) {skipped} unusable; excluded")

            c = label_col(disease, horizon)
            s = pd.to_numeric(both[c], errors='coerce')
            usable = s.notna().to_numpy()
            y = s[usable].to_numpy(dtype=np.int64)
            g = groups[usable]
            folds = subset_folds(fold_of, usable)
            Xe_h, Xp_h, M_h = Xe[usable], Xp[usable], M[usable]
            Xpp_h, Xep_h = Xpp[usable], Xep[usable]
            n, n_pos = len(y), int(y.sum())
            print(f"\n  {disease} {horizon}d  n={n}  pos={n_pos}")

            # A and B now come out of run_cell: they are the two towers F,
            # S and G actually consume, so the baselines in this table are
            # the experts the fusion used rather than a separately fitted
            # pair. They should match plain_mlp_torch's A320_sEHR_pre_selC
            # and PPG128_post fold for fold -- same features, folds, inner
            # splits, seeds, grid and estimator.
            (auc, apr, sels, corrs, w_sd, ci, gdiag, surf) = run_cell(
                Xe_h, Xp_h, Xpp_h, Xep_h, M_h, y, g, folds, args
            )
            for r in surf:
                surf_rows.append({'disease': disease, 'horizon': horizon,
                                  **r})
            for k, d in ci.items():
                ci_rows.append({'disease': disease, 'horizon': horizon,
                                'setting': k, 'n_samples': n, 'n_pos': n_pos,
                                **d})
            for k in auc:
                if not auc[k]:
                    continue
                rows.append({
                    'setting': k, 'probe': k, 'disease': disease,
                    'horizon': horizon, 'n_samples': n, 'n_pos': n_pos,
                    # Representation width of the expert this row scores.
                    # NaN for the gated rows: they mix logits rather than read
                    # a feature block, and reporting 2 there mixed two
                    # different meanings into one column.
                    'n_features': (Xe_h.shape[1] if k == 'A_EHR_pre' else
                                   Xep_h.shape[1] if k == 'sEHR_post' else
                                   Xpp_h.shape[1] if k == 'PPG_pre' else
                                   Xp_h.shape[1] if k == 'B_PPG_post' else
                                   Xe_h.shape[1] + Xp_h.shape[1]
                                   if k == 'C_joint_concat' else np.nan),
                    # How many experts are mixed, and how wide the gate that
                    # mixes them is -- the two things n_features was being
                    # asked to mean at once.
                    'n_experts': n_experts_of(k),
                    'gate_input_dim': (M_h.shape[1] + n_experts_of(k)
                                       if k in gdiag else np.nan),
                    'gate_params': (gate_params(M_h.shape[1], n_experts_of(k))
                                    if k in gdiag else np.nan),
                    'train_auroc': np.nan,
                    'auroc_mean': float(np.mean(auc[k])),
                    'auroc_std_fold': float(np.std(auc[k])),
                    'auprc_mean': float(np.mean(apr[k])),
                    'auprc_std_fold': float(np.std(apr[k])),
                    'n_valid_folds': len(auc[k]),
                    'auroc_per_fold': ','.join(f'{v:.6f}' for v in auc[k]),
                    'auprc_per_fold': ','.join(f'{v:.6f}' for v in apr[k]),
                    # Within-fold sd of the PPG weight, from the reported
                    # 3-expert mixture. Near zero means the gate collapsed to
                    # a constant, which no AUROC reveals.
                    'w_sd_within': (
                        float(np.mean(w_sd)) if k == 'H_gate_with_logits'
                        else np.nan
                    ),
                    # wd/lr/epoch chosen per fold for the two base towers,
                    # replacing the single C the sklearn version reported.
                    # name:wd,lr,epoch per tower, towers split by '|',
                    # folds by ';'. The name is in the value, so no caption
                    # can disagree with it.
                    'base_sel_per_fold': '; '.join(
                        '|'.join(f'{nm}:{t[0]:g},{t[1]:g},{t[2]}'
                                 for nm, t in tw)
                        for tw in sels),
                    # Which subset of the expert lattice this row is, so the
                    # ablation is readable from the CSV whatever the rows are
                    # named.
                    'subset': SUBSET_OF.get(k, ''),
                    # From gate_diag, keyed by row: each gated row reports its
                    # OWN weights and selection. A shared column would print
                    # one row's numbers under another's name, and with five
                    # gated rows now that would be silent.
                    'gate_weights_per_fold': (
                        '; '.join('/'.join(f'{v:.3f}' for v in t)
                                  for t in gdiag[k]['wmix'])
                        if k in gdiag else ''),
                    'gate_sel_per_fold': (
                        '; '.join(f'{w:g},{l:g},{n}'
                                  for w, l, n in gdiag[k]['sel'])
                        if k in gdiag else ''),
                    **({f'corr_wppg_{f}': float(np.nanmean(corrs[f]))
                        for f in SCALAR_FIELDS}
                       if k == 'H_gate_with_logits' else {}),
                })

    res = pd.DataFrame(rows)
    order = ['A_EHR_pre', 'sEHR_post', 'PPG_pre', 'B_PPG_post',
             'C_joint_concat',
             'H_gate_EP', 'H_gate_EJ', 'H_gate_PJ', 'H_gate_with_logits']

    for disease in diseases:
        for horizon in horizons:
            sub = res[(res.disease == disease) & (res.horizon == horizon)]
            if sub.empty:
                continue
            print('=' * 104)
            print(f"{disease} {horizon}d   n={int(sub.iloc[0]['n_samples'])}  "
                  f"n_pos={int(sub.iloc[0]['n_pos'])}")
            print('=' * 104)
            print(f"  {'setting':18s} {'AUROC':>14s} {'AUPRC':>14s} "
                  f"{'alpha':>16s} {'folds':>6s}")
            for st in order:
                rr = sub[sub.setting == st]
                if rr.empty:
                    continue
                r = rr.iloc[0]
                a = ''
                if st == 'H_gate_with_logits':
                    a = f"w_ppg sd~{r['w_sd_within']:.3f}"
                print(f"  {st:18s} "
                      f"{r['auroc_mean']:>7.3f}±{r['auroc_std_fold']:.3f} "
                      f"{r['auprc_mean']:>7.3f}±{r['auprc_std_fold']:.3f} "
                      f"{a:>16s} {int(r['n_valid_folds']):>4d}/"
                      f"{reg.N_FOLDS_OUTER}")
            for gk in ('H_gate_EP', 'H_gate_EJ', 'H_gate_PJ',
                       'H_gate_with_logits'):
                gr = sub[sub.setting == gk]
                if gr.empty or not gr.iloc[0].get('gate_weights_per_fold'):
                    continue
                # The weights are labelled by the row's OWN experts. A shared
                # "EHR/PPG/joint" caption would read E+J's second weight as a
                # PPG weight, which it is not.
                sset = SUBSET_OF.get(gk, gk)
                lbl = f'H {sset}'
                print(f"    {lbl:<17s} weights {sset.replace('+', '/')} "
                      f"per fold: {gr.iloc[0]['gate_weights_per_fold']}")
                print(f"    {'':<17s} wd,lr,epoch per fold: "
                      f"{gr.iloc[0]['gate_sel_per_fold']}")
            print("    (top of the wd grid drives the weights toward a "
                  "constant mixture)")
            # Read off any row: base_sel_per_fold records all four towers'
            # selections and is written identically onto every row.
            if not sub.empty:
                print(f"    towers name:wd,lr,epoch per fold: "
                      f"{sub.iloc[0]['base_sel_per_fold']}")
            print()

    # ── the ablation: does each expert earn its place? ──
    print('=' * 104)
    print('EXPERT-SUBSET ABLATION  -- paired against the full mixture, '
          'per outer fold')
    print('=' * 104)
    print(
        "  Every subset is fitted the same way and scored on the same folds,\n"
        "  so the delta against E+P+J isolates the missing expert. Paired per\n"
        "  fold, because the folds are shared and their difficulty cancels.\n"
        "\n"
        "  A subset that MATCHES the full mixture says the dropped expert adds\n"
        "  nothing; one that BEATS it says the expert is noise the gate has to\n"
        "  spend capacity suppressing. Both are answers.\n"
        "\n"
        "  Read the 95% CI table too: a delta smaller than the intervals is\n"
        "  not resolved by this cohort either way.\n"
    )
    FULL = 'H_gate_with_logits'
    print(f"  {'disease':7s} {'hz':>5s} {'subset':<8s} {'AUROC':>7s} "
          f"{'vs full':>8s} {'up/dn':>7s}  per-fold delta")
    for disease in diseases:
        for horizon in horizons:
            sub = res[(res.disease == disease) & (res.horizon == horizon)]
            fu = sub[sub.setting == FULL]
            if fu.empty:
                continue
            vfull = np.array([float(x) for x in
                              fu.iloc[0]['auroc_per_fold'].split(',')])
            for k in ABLATION_ROWS:
                r = sub[sub.setting == k]
                if r.empty:
                    continue
                v = np.array([float(x) for x in
                              r.iloc[0]['auroc_per_fold'].split(',')])
                lbl = SUBSET_OF.get(k, k)
                if k == FULL or len(v) != len(vfull):
                    print(f"  {disease:7s} {horizon:>5d} {lbl:<8s} "
                          f"{v.mean():>7.3f} {'--':>8s} {'':>7s}")
                    continue
                d = v - vfull
                print(f"  {disease:7s} {horizon:>5d} {lbl:<8s} "
                      f"{v.mean():>7.3f} {d.mean():>+8.3f} "
                      f"{f'{int((d>0).sum())}/{int((d<0).sum())}':>7s}  "
                      + ' '.join(f'{x:+.3f}' for x in d))
    print()

    print('=' * 104)
    print('WHAT THE GATE LEARNED -- mean corr(alpha_i, scalar) over outer folds')
    print('=' * 104)
    print(
        "  This, not AUROC, is the direct readout of 'does trust in PPG vary\n"
        "  by signal quality / timing / code volume'. On a synthetic control\n"
        "  where PPG was informative only at high quality, the gate recovered\n"
        "  that dependence almost perfectly in this correlation while still\n"
        "  LOSING to the best fixed alpha on AUROC -- so a null AUROC delta\n"
        "  does not mean the gate learned nothing.\n"
        "\n"
        "  Sign convention: alpha is the PPG weight, so a POSITIVE correlation\n"
        "  with a quality field means 'trust PPG more when the signal is\n"
        "  cleaner', which is the mechanistically expected direction.\n"
        "\n"
        "  These are associations in the gate's own fitted output, not causal\n"
        "  claims, and any scalar that predicts the outcome ON ITS OWN is a\n"
        "  free predictor -- run it through audit_single_variable before\n"
        "  reading a correlation here as evidence about PPG.\n"
    )
    hdr = '  '.join(f'{f[:14]:>14s}' for f in SCALAR_FIELDS)
    print(f"  {'disease':7s} {'hz':>5s}  {hdr}")
    for disease in diseases:
        for horizon in horizons:
            sub = res[(res.disease == disease) & (res.horizon == horizon)]
            gl = sub[sub.setting == 'H_gate_with_logits']
            if gl.empty:
                continue
            r = gl.iloc[0]
            cells = '  '.join(
                f"{r.get(f'corr_wppg_{f}', np.nan):>+14.3f}"
                for f in SCALAR_FIELDS
            )
            print(f"  {disease:7s} {horizon:>5d}  {cells}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = '' if args.disease == 'all' else f'_{args.disease}'
    tag += '' if args.horizon == 0 else f'_{args.horizon}d'
    f_res = f'score_fusion_postdisch_results{tag}.csv'
    res.to_csv(os.path.join(args.out_dir, f_res), index=False)
    print(f'Saved {f_res}')

    # ── confidence intervals, a SEPARATE table and file ──
    # The results table above keeps its columns and rows unchanged; this adds
    # a second view for the four rows that carry an interval.
    if ci_rows:
        cidf = pd.DataFrame(ci_rows)
        cidf['subset'] = cidf['setting'].map(SUBSET_OF).fillna('')
        cols = ['disease', 'horizon', 'setting', 'subset', 'n_samples',
                'n_pos', 'n_valid_folds', 'auroc_mean', 'auroc_ci_low',
                'auroc_ci_high', 'auprc_mean', 'auprc_ci_low',
                'auprc_ci_high', 'n_boot', 'n_fold_redraws']
        cidf = cidf[[c for c in cols if c in cidf.columns]]
        f_ci = f'score_fusion_postdisch_ci{tag}.csv'
        cidf.to_csv(os.path.join(args.out_dir, f_ci), index=False)

        print()
        print('=' * 104)
        print('95% CI  --  mean of the 5 per-fold metrics, '
              'fold-preserving patient-cluster bootstrap')
        print('=' * 104)
        print("  The point estimate is the SAME mean-of-folds the table above")
        print("  reports; only the interval is bootstrapped. Patients are")
        print("  resampled inside each outer test fold, the 5 fold metrics are")
        print("  averaged with equal weight, and predictions are never pooled")
        print("  across folds.")
        print()
        print(f"  {'cell':<12}{'row':<22}{'subset':<11}{'AUROC':>8}"
              f"{'95% CI':>18}{'AUPRC':>9}{'95% CI':>18}{'folds':>7}")
        for (dis, hor), grp in cidf.groupby(['disease', 'horizon'],
                                            sort=False):
            for k in CI_ROWS:
                r = grp[grp.setting == k]
                if r.empty:
                    continue
                r = r.iloc[0]
                # Built outside the f-string: this runs on Python 3.10, where
                # a replacement field cannot span lines.
                ci_a = f"({r['auroc_ci_low']:.3f}-{r['auroc_ci_high']:.3f})"
                ci_p = f"({r['auprc_ci_low']:.3f}-{r['auprc_ci_high']:.3f})"
                cell = f"{dis} {hor}d"
                print(f"  {cell:<12}{k:<22}{SUBSET_OF.get(k, ''):<11}"
                      f"{r['auroc_mean']:>8.3f}{ci_a:>18}"
                      f"{r['auprc_mean']:>9.3f}{ci_p:>18}"
                      f"{int(r['n_valid_folds']):>7d}")
        nr = int(cidf['n_fold_redraws'].sum())
        if nr:
            print(f"\n  ({nr} single-class fold draws redrawn across all "
                  f"cells)")
        print(f"\nSaved {f_ci}")

    # ── hyperparameter sensitivity, two panels ──
    # The inner-validation AUROC of every candidate was already computed by
    # the selection and discarded; these panels are that surface, and cost
    # nothing but the bookkeeping.
    #
    # Raw AUROC is NOT averaged across experts or tasks -- they differ in
    # difficulty, so a mean over them would be dominated by the easy ones.
    # What is averaged is the REGRET: how much this candidate gives up
    # against the best candidate for that same (model, task, fold). Each
    # group is therefore centred on its own optimum before averaging.
    if surf_rows:
        sdf = pd.DataFrame(surf_rows)
        f_surf = f'score_fusion_postdisch_hpsurface{tag}.csv'
        sdf.to_csv(os.path.join(args.out_dir, f_surf), index=False)

        GROUP = ['stage', 'model', 'disease', 'horizon', 'fold']
        sdf['regret'] = sdf['val_auroc'] - sdf.groupby(GROUP)['val_auroc'] \
            .transform('max')

        def panel(sub, title, note):
            print()
            print('=' * 104)
            print(title)
            print('=' * 104)
            print(note)
            g = sub.groupby(['wd', 'lr'])['regret'].agg(['mean', 'size'])
            lrs = sorted(sub['lr'].unique())
            wds = sorted(sub['wd'].unique())
            hdr = 'wd | lr'
            print(f"\n    {hdr:>10}"
                  + ''.join(f"{l:>12g}" for l in lrs)
                  + f"{'row best':>11}")
            for w in wds:
                cells = ''
                for l in lrs:
                    if (w, l) in g.index:
                        cells += f"{g.loc[(w, l), 'mean']:>12.4f}"
                    else:
                        cells += f"{'--':>12}"   # cut by the rho cap
                have = [g.loc[(w, l), 'mean'] for l in lrs
                        if (w, l) in g.index]
                best = f"{max(have):>11.4f}" if have else f"{'--':>11}"
                print(f"    {w:>10g}{cells}{best}")
            n = int(g['size'].max()) if len(g) else 0
            print(f"\n    cells averaged per (wd, lr): up to {n}"
                  f"    '--' = excluded by the rho = lr*wd <= "
                  f"{af.MAX_RHO} cap")
            if len(g):
                b = g['mean'].idxmax()
                w_ = g['mean'].idxmin()
                print(f"    flattest  wd={b[0]:g} lr={b[1]:g}  "
                      f"regret {g.loc[b, 'mean']:+.4f}")
                print(f"    worst     wd={w_[0]:g} lr={w_[1]:g}  "
                      f"regret {g.loc[w_, 'mean']:+.4f}")
                print(f"    spread across the grid: "
                      f"{g['mean'].max() - g['mean'].min():.4f} AUROC")

        # Every base predictor that is fitted downstream, whether or not the
        # mixture reads it: sensitivity is a property of fitting a linear
        # probe on a representation, so leaving two of the five out would
        # have described the protocol on a subset of the models it governs.
        panel(sdf[(sdf.stage == 'expert')
                  & sdf.model.isin(['E', 'sEHR_post', 'PPG_pre', 'P', 'J'])],
              '(a) BASE PREDICTOR hyperparameter sensitivity',
              "  Mean inner-validation AUROC given up against each\n"
              "  (predictor, task, fold)'s OWN best candidate, averaged over\n"
              "  all five: sEHR_pre, sEHR_post, PPG_pre, PPG_post and joint.\n"
              "  Per-predictor surfaces are in the CSV under `model`.")

        panel(sdf[(sdf.stage == 'gate') & (sdf.model == 'E+P+J')],
              '(b) GATING NETWORK hyperparameter sensitivity',
              "  The same quantity for the gate of the reported 3-expert\n"
              "  mixture, with the experts already selected and their\n"
              "  out-of-fold scores fixed -- so this is the second stage on\n"
              "  its own. The three pair gates are in the CSV under\n"
              "  model = E+P, E+J, P+J.")

        print(f"\n  One run covers one disease, so these panels average over "
              f"4 tasks x folds.")
        print(f"  Concatenate the three diseases' {f_surf} for the "
              f"12-task version.")
        print(f"\nSaved {f_surf}")


if __name__ == '__main__':
    main()
