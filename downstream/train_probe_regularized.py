"""
STAGE 5b: properly regularised probes (replaces the hand-rolled Adam loop).

WHY THIS EXISTS
---------------
Diagnostics on train_probe_admission.py showed every probe saturating train
AUROC on the both-modality subset while test AUROC fell, and the one variant
that did not saturate lost that property as soon as its features were scaled.

Two conclusions:
  1. torch.optim.Adam(weight_decay=...) adds L2 to the GRADIENT, and Adam's
     per-parameter normalisation largely cancels it. The knob is inert: raising
     wd by orders of magnitude still memorises. (This is exactly why AdamW
     exists.)
  2. The only thing that kept ehr_post from memorising was its small magnitude
     -- L2-normalised rows have a per-feature std far below the unit std
     StandardScaler produces. That is accidental implicit regularisation, not a
     modelling choice, and it means the reported EHR_post advantage was an
     artefact of feature scale rather than of the alignment projection.

So: LogisticRegression with a real L2 penalty, C chosen by inner CV on the
training folds only. Scale-invariant by construction, so StandardScaler is now
safe to apply to everything uniformly.

WHAT IS REPORTED
----------------
- TRAIN auroc next to TEST auroc, always. If train == 1.000 the row is
  memorised and the test number means nothing; this must be visible.
- chosen C per fold (median), so over/under-regularisation is diagnosable.
- a SHUFFLED-LABEL control per row: labels permuted across samples, whole CV
  re-run. A correctly wired pipeline gives ~0.5 here. If it does not, the
  leak is in
  the harness, not the features.

POPULATION
----------
One only: ppg_mask==1 & ehr_mask==1 (953 admissions, 3/7/9/17 positives). Every
row -- single-modality and fusion alike -- sees identical samples and identical
folds, which is the whole point of the table.

An earlier version also emitted population='all' rows (EHR probes over all 1638
admissions, 10/31/52/95 positives) into the SAME csv. Those are removed: a read
that forgot to filter population=='both' compared numbers across cohorts with
different positive rates, and the column invited exactly that mistake. Run the
EHR-only-with-more-power question as its own analysis if it is wanted.

COMPARABILITY WITH train_probe_concat_fusion.py
-----------------------------------------------
Deliberately identical: DATA_PATH, HORIZONS, the both-subset predicate, the
outer 5-fold StratifiedGroupKFold(shuffle, seed 42) grouped by patient, the
inner 3-fold StratifiedGroupKFold(shuffle, seed 42), C_GRID, DEFAULT_C,
MIN_POS_FOR_INNER_CV, every LogisticRegression argument, the scaler-first
pipeline, the single-class-fold skip, the shuffled-label control, and the
metric field names.

The intended difference is the joint embedding: this file mixes into 128-d with
a selected alpha, that file concatenates into 256-d. One unintended difference
remains and is NOT fixed here: that file averages five outer partitions
(seeds 42-46) and reports auroc_std_repeat, this file evaluates one partition
and reports auroc_std_fold. The two spreads are not interchangeable; only the
seed-42 partition of that file is comparable to these numbers.

Outer CV: 5-fold StratifiedGroupKFold, group=patient_id, seed 42.
Inner CV: 3-fold StratifiedGroupKFold on the training portion, scoring=roc_auc.

Usage (compute node -- ~4700 logistic fits, minutes not seconds):
    python train_probe_regularized.py
    python train_probe_regularized.py --no-shuffle-control   # ~2x faster
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

import os

import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

import joblib

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FitFailedWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*least populated class.*')
warnings.filterwarnings('ignore', message='.*invalid value encountered in cast.*')

DATA_PATH = str(paths.data_root() / "ds1/downstream_features_admission_origbert_pre.joblib")
OUT_DIR = str(paths.data_root() / "ds1")
HORIZONS = ['label_30', 'label_90', 'label_180', 'label_365']
N_FOLDS_OUTER = 5
N_FOLDS_INNER = 3
RANDOM_SEED = 42
# The grid FLOOR is what matters here, not its ceiling. Train and test AUROC
# move monotonically against C over this range -- less shrinkage, more
# memorisation, worse generalisation, all the way down -- and the selected C
# lands near the bottom of the useful region. The grid therefore has to reach
# several decades below that region, or the optimum sits pinned at the boundary
# and the selection is reporting the edge of the grid instead of a choice.
C_GRID = np.logspace(-8, 2, 21)

# Inner CV needs >= 2 positives in every inner fold to fit AND to score, and
# group constraints can push all positives of a fold into one patient. Below
# this many positives in the outer training portion we do not attempt to select
# C at all -- we fall back to a fixed conservative value and say so, because a C
# chosen from 2 positives is noise dressed up as tuning.
MIN_POS_FOR_INNER_CV = 2 * N_FOLDS_INNER      # 6
DEFAULT_C = 1e-5                              # plateau of the table above

# Fusion mixing weight. alpha is a HYPERPARAMETER: it is selected inside the
# inner CV jointly with C, never swept against the outer test set.
#   fused = (v_ehr + alpha*v_ppg) / (1 + alpha)     [both-subset: ppg_mask==1]
# alpha=0 reduces to EHR_post, so it is inside the search space -- the model can
# choose to ignore PPG entirely, and how often it does is itself the result.
# Note 1/(1+alpha) is a global constant that StandardScaler removes downstream,
# so alpha controls mixing DIRECTION only.
# A learnable gate is deliberately NOT used: selecting alpha by inner CV answers
# "does ANY mixing weight generalise", which subsumes "can a gate find it".
ALPHA_GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]


def make_folds(groups, y, n_splits=N_FOLDS_OUTER, seed=RANDOM_SEED):
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros((len(y), 1)), y, groups=groups))


class FusionMixer(BaseEstimator, TransformerMixin):
    """Mix the two aligned 128-d vectors with weight alpha.

    Input X is [v_ehr | v_ppg] concatenated (dim*2 columns). Output is the
    fused dim-column matrix. Being a pipeline step means alpha is an ordinary
    sklearn hyperparameter, so GridSearchCV selects it jointly with C using
    inner-fold validation only -- the outer test fold never influences it.

    It sits AFTER StandardScaler, and that order is what makes this probe an
    exact restriction of the concatenation probe. Writing a for the standardised
    EHR columns and b for the standardised PPG columns,

        z = (a + alpha*b) / (1 + alpha)
        logit = w'z = [w/(1+alpha)]' a  +  [alpha*w/(1+alpha)]' b

    which is concatenation under the constraint w_ppg = alpha * w_ehr. alpha = 0
    therefore reproduces the EHR_post row exactly, and no alpha can beat
    concatenation on the training objective -- only, through the smaller
    hypothesis class, on held-out data.

    (This file previously mixed BEFORE scaling. The old comment claiming the
    1/(1+alpha) factor is "removed downstream by StandardScaler, so alpha
    controls direction only" described that arrangement and no longer applies:
    the constant now survives into the logit, where the L2 penalty and the
    searched C absorb it.)
    """
    def __init__(self, alpha=0.0, dim=128):
        self.alpha = alpha
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ve, vp = X[:, :self.dim], X[:, self.dim:]
        return (ve + self.alpha * vp) / (1.0 + self.alpha)


def build_estimator(fusion=False, dim=128):
    """StandardScaler FIRST, then the optional mixer, then L2 logistic.

    class_weight='balanced' mirrors the pos_weight=n_neg/n_pos used before.
    Scaling is safe because the L2 penalty makes the fit scale-aware.

    The scaler has to come first, and not only so that the single-modality rows
    and the fusion row share a front end. It is what makes the fusion model a
    STRICT SPECIAL CASE of the concatenation model -- see FusionMixer for the
    algebra. With the mixer first (as this file used to have it) the scaler is
    fit on the already-fused columns, so its means and variances depend on
    alpha, the nesting breaks, and "concatenation subsumes every fixed mixing
    ratio" stops being true of the models actually fitted here.

    Cost of the order: the fused columns are no longer unit-variance, since
    standardisation happens before the aggregation. That only introduces a
    global scale, which the L2 penalty absorbs through C -- and C is searched
    in the inner CV anyway.
    """
    steps = [('sc', StandardScaler())]
    if fusion:
        steps.append(('mix', FusionMixer(dim=dim)))
    steps.append(('lr', LogisticRegression(penalty='l2', solver='lbfgs',
                                           class_weight='balanced',
                                           max_iter=5000)))
    return Pipeline(steps)


def run_cv(X, y, groups, tag, fusion=False, dim=128,
           make_est=None, alpha_grid=None, folds=None):
    """Nested CV.

    Inner 3-fold CV over the outer-training patients selects (alpha, C) jointly
    when fusion=True, or C alone otherwise. The winning setting is refit on the
    FULL outer-training set, then scored once on the outer test fold.

    make_est / alpha_grid exist so a sibling script can reuse this function
    verbatim with a different mixer or a frozen alpha, rather than copying the
    body and letting the two drift. Both default to None, in which case the
    behaviour is exactly what this file did before they were added:
    build_estimator(fusion, dim) and the module-level ALPHA_GRID.

    make_est must be a zero-argument callable returning a fresh Pipeline whoses
    mixing step is named 'mix' with an `alpha` parameter, so that
    'mix__alpha' addresses it and the fallback path can force alpha to 0.
    """
    folds = make_folds(groups, y) if folds is None else folds
    tr_auc, te_auc, te_ap, chosen_C, chosen_a, n_pos_te = [], [], [], [], [], []
    n_defaulted = 0

    make = make_est if make_est is not None else (lambda: build_estimator(fusion, dim))

    grid = {'lr__C': C_GRID}
    if fusion:
        grid['mix__alpha'] = alpha_grid if alpha_grid is not None else ALPHA_GRID

    for tr, te in folds:
        n_pos_te.append(int(y[te].sum()))
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue

        best, C_used, a_used = None, None, None
        if int(y[tr].sum()) >= MIN_POS_FOR_INNER_CV:
            inner = StratifiedGroupKFold(n_splits=N_FOLDS_INNER, shuffle=True,
                                         random_state=RANDOM_SEED)
            gs = GridSearchCV(make(), grid,
                              scoring='roc_auc', cv=inner, n_jobs=-1,
                              error_score=np.nan, refit=True)
            try:
                gs.fit(X[tr], y[tr], groups=groups[tr])
                if np.isfinite(gs.best_score_):
                    best = gs.best_estimator_          # refit on full outer-train
                    C_used = gs.best_params_['lr__C']
                    a_used = gs.best_params_.get('mix__alpha')
            except (ValueError, AttributeError):
                best = None

        if best is None:
            # Too few positives to SELECT anything -- fall back to a fixed
            # conservative C.
            #
            # alpha only falls back when it is genuinely being selected, i.e.
            # when the grid offers a choice. A single-valued alpha_grid means a
            # caller froze alpha (the sweep does this): there is nothing to
            # select on that axis and nothing to fall back from, so the frozen
            # value is kept and only C is defaulted. Forcing alpha=0 there would
            # silently run pure sEHR under a row labelled with some other alpha.
            # With the module-level ALPHA_GRID (6 values) this takes the else
            # branch, so train_probe_regularized's own behaviour is unchanged.
            best = make().set_params(lr__C=DEFAULT_C)
            if fusion:
                a_used = (float(alpha_grid[0])
                          if alpha_grid is not None and len(alpha_grid) == 1
                          else 0.0)
                best.set_params(mix__alpha=a_used)
            best.fit(X[tr], y[tr])
            C_used = DEFAULT_C
            n_defaulted += 1

        p_tr = best.predict_proba(X[tr])[:, 1]
        p_te = best.predict_proba(X[te])[:, 1]     # scored exactly once
        tr_auc.append(roc_auc_score(y[tr], p_tr))
        te_auc.append(roc_auc_score(y[te], p_te))
        te_ap.append(average_precision_score(y[te], p_te))
        chosen_C.append(C_used)
        if a_used is not None:
            chosen_a.append(a_used)

    # Field names follow train_probe_concat_fusion.py so the two CSVs can be
    # concatenated on metric name. This file evaluates ONE outer partition, so
    # it reports the fold-to-fold spread (auroc_std_fold) and no _repeat
    # columns; n_repeats / n_folds_total are constants here and exist only to
    # keep the schema a strict subset of that file's.
    return {
        'probe': tag,
        'n_repeats': 1,
        'n_folds_total': N_FOLDS_OUTER,
        'n_features': int(X.shape[1]),
        'train_auroc': np.mean(tr_auc) if tr_auc else np.nan,
        'auroc_mean': np.mean(te_auc) if te_auc else np.nan,
        'auroc_std_fold': np.std(te_auc) if te_auc else np.nan,
        'auprc_mean': np.mean(te_ap) if te_ap else np.nan,
        'auprc_std_fold': np.std(te_ap) if te_ap else np.nan,
        # Per-fold values, retained so two settings evaluated on the SAME folds
        # can be compared PAIRED. The fold-to-fold spread is largely shared
        # across settings (a fold is hard or easy for every probe alike), so
        # the std of the per-fold difference is much smaller than the
        # difference of the two stds. Without these columns an effect smaller
        # than the fold spread is unreadable, and a mean driven by one lucky
        # fold is indistinguishable from a consistent one. Additive
        # columns only -- no existing field changes.
        'auroc_per_fold': ','.join(f'{v:.6f}' for v in te_auc),
        'auprc_per_fold': ','.join(f'{v:.6f}' for v in te_ap),
        'C_median': float(np.median(chosen_C)) if chosen_C else np.nan,
        # NOTE: a defaulted fold force-writes alpha=0.0 into chosen_a, so these
        # three cannot distinguish "the inner CV chose to ignore PPG" from "the
        # inner CV never ran". Subtract n_folds_C_defaulted before reading them.
        'alpha_median': float(np.median(chosen_a)) if chosen_a else np.nan,
        'alpha_per_fold': ','.join(f'{a:g}' for a in chosen_a) if chosen_a else '',
        'n_folds_alpha_zero': int(sum(1 for a in chosen_a if a == 0.0)),
        'n_folds_C_defaulted': n_defaulted,
        'n_valid_folds': len(te_auc),
        'n_pos': int(y.sum()),
        'median_pos_per_fold': int(np.median(n_pos_te)) if n_pos_te else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA_PATH)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--no-shuffle-control', action='store_true')
    args = ap.parse_args()

    data = joblib.load(args.data)
    both = [d for d in data if d['ppg_mask'] == 1 and d['ehr_mask'] == 1]
    print(f"Loaded {len(data)} admissions | both-modality: {len(both)}\n")

    # Single-modality rows, both-modality subset only. The former
    # population='all' rows (EHR_pre / EHR_post over all 1638 admissions) are
    # gone: they sat in the same CSV as the both-subset rows on a different
    # cohort and a different positive rate, so any read that forgot to filter
    # population=='both' silently compared across cohorts. Every row here now
    # sees identical samples and identical folds, as in
    # train_probe_concat_fusion.py.
    FEATS = [('v_ehr_pre_mlp',  'EHR_pre'),
             ('v_ehr_post_mlp', 'EHR_post'),
             ('v_ppg_pre_mlp',  'PPG_pre'),
             ('v_ppg_post_mlp', 'PPG_post')]

    rows = []
    for horizon in HORIZONS:
        print("=" * 92)
        print(horizon)
        print("=" * 92)
        print(f"  {'probe':9s} {'n':>5s} {'n_pos':>5s} "
              f"{'TRAIN':>7s} {'AUROC':>14s} {'AUPRC':>14s} {'C':>9s} "
              f"{'shuf':>6s} {'folds':>6s}")

        y = np.array([d[horizon] for d in both], dtype=np.int64)
        groups = np.array([str(d['patient_id']) for d in both])

        for key, probe in FEATS:
            X = np.stack([d[key] for d in both]).astype(np.float64)
            r = run_cv(X, y, groups, probe)

            shuf = np.nan
            if not args.no_shuffle_control:
                rng = np.random.default_rng(RANDOM_SEED)
                y_shuf = rng.permutation(y)
                shuf = run_cv(X, y_shuf, groups, probe + '_shuffled')['auroc_mean']

            r.update(horizon=horizon, n_samples=len(both), shuffled_auroc=shuf,
                     memorised=bool(r['train_auroc'] > 0.995))
            rows.append(r)

            flag = '  <-- MEMORISED' if r['memorised'] else ''
            if r['n_folds_C_defaulted']:
                flag += (f"  [C defaulted in "
                         f"{r['n_folds_C_defaulted']}/{r['n_valid_folds']} folds]")
            print(f"  {probe:9s} {len(both):>5d} {r['n_pos']:>5d} "
                  f"{r['train_auroc']:>7.3f} "
                  f"{r['auroc_mean']:>7.3f}±{r['auroc_std_fold']:.3f} "
                  f"{r['auprc_mean']:>7.3f}±{r['auprc_std_fold']:.3f} "
                  f"{r['C_median']:>9.4g} "
                  f"{shuf:>6.3f} {r['n_valid_folds']:>4d}/{N_FOLDS_OUTER}"
                  f"{flag}")

        # ── fusion: alpha selected jointly with C inside the inner CV ──────
        y_b = np.array([d[horizon] for d in both], dtype=np.int64)
        g_b = np.array([str(d['patient_id']) for d in both])
        VE = np.stack([d['v_ehr_post_mlp'] for d in both]).astype(np.float64)
        VP = np.stack([d['v_ppg_post_mlp'] for d in both]).astype(np.float64)
        Xf = np.hstack([VE, VP])                       # FusionMixer splits it
        dim = VE.shape[1]

        r = run_cv(Xf, y_b, g_b, 'Fusion', fusion=True, dim=dim)
        shuf = np.nan
        if not args.no_shuffle_control:
            rng = np.random.default_rng(RANDOM_SEED)
            shuf = run_cv(Xf, rng.permutation(y_b), g_b, 'Fusion_shuffled',
                          fusion=True, dim=dim)['auroc_mean']
        r.update(horizon=horizon, n_samples=len(both),
                 shuffled_auroc=shuf,
                 memorised=bool(r['train_auroc'] > 0.995))
        rows.append(r)

        flag = '  <-- MEMORISED' if r['memorised'] else ''
        flag += f"  [alpha per fold: {r['alpha_per_fold']}]"
        if r['n_folds_alpha_zero']:
            flag += (f" -> alpha=0 (PPG ignored) in "
                     f"{r['n_folds_alpha_zero']}/{r['n_valid_folds']}")
        if r['n_folds_C_defaulted']:
            flag += (f"  [defaulted {r['n_folds_C_defaulted']}"
                     f"/{r['n_valid_folds']}]")
        print(f"  {'Fusion':9s} {len(both):>5d} "
              f"{r['n_pos']:>5d} {r['train_auroc']:>7.3f} "
              f"{r['auroc_mean']:>7.3f}±{r['auroc_std_fold']:.3f} "
              f"{r['auprc_mean']:>7.3f}±{r['auprc_std_fold']:.3f} "
              f"{r['C_median']:>9.4g} "
              f"{shuf:>6.3f} {r['n_valid_folds']:>4d}/{N_FOLDS_OUTER}{flag}")
        print()

    df = pd.DataFrame(rows)
    csv = os.path.join(args.out_dir, 'probe_results_regularized.csv')
    df.to_csv(csv, index=False)
    print(f"Saved {csv}")

    print("\n" + "=" * 92)
    print("HOW TO READ")
    print("=" * 92)
    print("  TRAIN ~ 1.000            -> still memorising; C grid needs to go lower")
    print("  shuffled far from 0.500  -> leak in the harness, not a real signal")
    print("  TEST < 0.5 with TRAIN<1  -> genuinely anti-predictive")
    print("  wide C spread per fold   -> too few positives to select C; treat all")
    print("                              numbers on that row as indicative only")

    print("\n" + "=" * 92)
    print("FUSION vs EHR-ONLY  (both-subset, same folds; alpha chosen in inner CV)")
    print("=" * 92)
    for horizon in HORIZONS:
        fu = df[(df.horizon == horizon) & (df.probe == 'Fusion')]
        ep = df[(df.horizon == horizon) & (df.probe == 'EHR_post')]
        if fu.empty or ep.empty:
            continue
        fu, ep = fu.iloc[0], ep.iloc[0]
        print(f"  {horizon:10s} n_pos={int(fu['n_pos']):>3d}  "
              f"EHR_post {ep['auroc_mean']:.3f}  ->  Fusion {fu['auroc_mean']:.3f} "
              f"({fu['auroc_mean'] - ep['auroc_mean']:+.3f})   "
              f"alpha chosen: [{fu['alpha_per_fold']}]")
    print("\n  alpha=0 in every fold -> the fusion row should match EHR_post")
    print("     EXACTLY, not just closely: with StandardScaler first the mixer")
    print("     at alpha=0 is the identity on the EHR block. A discrepancy there")
    print("     means the pipeline order is wrong.")
    print("  alpha>0 chosen but Fusion <= EHR_post -> inner CV overfit its choice")
    print("     of alpha, which with <20 positives is the expected failure.")
    print("  Judge any delta against auroc_std_fold on the row.")
    print("  Fusion is concatenation restricted to w_ppg = alpha*w_ehr, so it")
    print("     cannot beat train_probe_concat_fusion.py's Joint_concat on the")
    print("     training objective -- only on held-out data, via lower variance.")


if __name__ == '__main__':
    main()
