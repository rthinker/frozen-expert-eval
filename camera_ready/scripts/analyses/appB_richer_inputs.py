# C06 and C07: Evaluate richer prediction-space inputs.
"""RUN A - extended prediction-space ladder.

Evaluates calibrated uncertainty, expert disagreement, mutual information
and domain-conditioned weighting as richer prediction-space inputs.

Six new rungs on top of the submitted 10-method ladder. Every one is a
measurable function of the expert predictions, so none can exceed the
prediction-space Bayes optimum; they can only improve a finite-sample
fit. The point of running them is to measure how much of the published
margin that buys.

  PS_11_disagreement  flat logits + all pairwise |margin_i - margin_j|
  PS_12_uncertainty   per-expert temperature-calibrated probabilities
                      (temperatures fitted on SELECT, reusing the
                      submitted PS_A routine) + entropy of the mean
                      + mean of the entropies + their difference (the
                      BALD-style MUTUAL INFORMATION term) + variance,
                      min, max, median of p(fake)
  PS_13_quadratic     flat logits + all pairwise products of margins
                      + squared margins (full quadratic expansion)
  PS_14_rank          per-expert margins mapped through the empirical
                      CDF fitted on FIT (monotone, scale-free, and NOT
                      transductive - the ECDF never sees TEST)
  PS_15_cv_best       whichever of PS_11..PS_14 wins 5-fold stratified
                      CV inside FIT (selection never touches TEST)
  PS_16_oracle_domain NON-DEPLOYABLE upper envelope: within each held-out
                      generator, 5-fold out-of-fold logistic regression
                      on that generator's own labels. Consumes target
                      labels, so it is an upper bound on any
                      domain-conditioned weighting scheme, not a method.

Each rung is fitted with two heads: logistic regression (matching PS_2's
solver/regularisation) and XGBoost. Engineered features are standardised
with a scaler fitted on FIT only; this is a disclosed deviation from the
submitted PS ladder that can only HELP the prediction side, which is the
conservative direction for the paper's claim.

Also reported: PS_17_oracle_pergen_select, the retrospective per-generator
maximum over every rung. Also non-deployable, also an upper envelope.

Outputs -> camera_ready/output/results/analyses/run_a_extended_ladder_<pool>.json
Writes nothing outside camera_ready/output/.

Usage (from the repository root)::

    python camera_ready/scripts/analyses/appB_richer_inputs.py --pool arch6
    python camera_ready/scripts/analyses/appB_richer_inputs.py --pool clip4 --xgb_mode cv
"""

from __future__ import annotations

import _paths  # noqa: F401  # MUST be first

import argparse
import time
from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import _common as common
from src.models.feature_space import FS_1  # noqa: E402
from src.models.prediction_space import PS_A, _softmax  # noqa: E402
from src.utils.config import (  # noqa: E402
    DS_CONFIG, FIT_TAG, SELECT_TAG, get_pool_experts,
)
from src.utils.dataset_helpers import TEST_GENERATORS  # noqa: E402
from src.utils.metrics import compute_auc  # noqa: E402


# --------------------------------------------------------------------------
# Feature maps. Each takes [N, E, 2] logits and returns [N, F].
# Any state they need is fitted on FIT and passed in explicitly.
# --------------------------------------------------------------------------

def _margins(logits3d: np.ndarray) -> np.ndarray:
    """Per-expert decision margin: logit_fake - logit_real. [N, E]."""
    return (logits3d[:, :, 1] - logits3d[:, :, 0]).astype(np.float64)


def _pair_idx(n: int):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def feat_flat(logits3d: np.ndarray) -> np.ndarray:
    return logits3d.reshape(len(logits3d), -1).astype(np.float64)


def feat_disagreement(logits3d: np.ndarray) -> np.ndarray:
    m = _margins(logits3d)
    pairs = _pair_idx(m.shape[1])
    if not pairs:
        return feat_flat(logits3d)
    diffs = np.stack([np.abs(m[:, i] - m[:, j]) for i, j in pairs], axis=1)
    return np.concatenate([feat_flat(logits3d), diffs], axis=1)


def feat_quadratic(logits3d: np.ndarray) -> np.ndarray:
    m = _margins(logits3d)
    pairs = _pair_idx(m.shape[1])
    blocks = [feat_flat(logits3d), m ** 2]
    if pairs:
        blocks.append(np.stack([m[:, i] * m[:, j] for i, j in pairs], axis=1))
    return np.concatenate(blocks, axis=1)


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def make_uncertainty_featmap(select_logits3d: np.ndarray,
                             select_labels: np.ndarray):
    """Fit per-expert temperatures on SELECT, return a feature map.

    Temperatures come from the submitted PS_A routine so the calibration
    is identical to the published calibrated-averaging rung.
    """
    _, diag = PS_A(select_logits3d, select_labels,
                   select_logits3d[:1], return_diagnostics=True)
    temps = np.asarray(diag["temps"], dtype=np.float64)

    def featmap(logits3d: np.ndarray) -> np.ndarray:
        cal = logits3d.astype(np.float64) / temps[None, :, None]
        p = _softmax(cal)[:, :, 1]                      # [N, E] p(fake)
        pbar = p.mean(axis=1, keepdims=True)            # [N, 1]
        h_of_mean = _entropy(pbar[:, 0])                # H(E[p])
        mean_of_h = _entropy(p).mean(axis=1)            # E[H(p)]
        mutual_info = h_of_mean - mean_of_h             # BALD term
        stats = np.stack([
            h_of_mean, mean_of_h, mutual_info,
            p.var(axis=1), p.min(axis=1), p.max(axis=1),
            np.median(p, axis=1), p.max(axis=1) - p.min(axis=1),
        ], axis=1)
        return np.concatenate([p, pbar, stats], axis=1)

    return featmap, temps


def make_rank_featmap(fit_logits3d: np.ndarray):
    """Empirical-CDF map per expert, fitted on FIT only (not transductive)."""
    ref = np.sort(_margins(fit_logits3d), axis=0)       # [N_fit, E]
    n = ref.shape[0]

    def featmap(logits3d: np.ndarray) -> np.ndarray:
        m = _margins(logits3d)
        out = np.empty_like(m)
        for e in range(m.shape[1]):
            out[:, e] = np.searchsorted(ref[:, e], m[:, e], side="right") / n
        return out

    return featmap


# --------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------

def fit_logreg(X_fit, y_fit, X_test):
    scaler = StandardScaler().fit(X_fit)
    lr = LogisticRegression(
        solver=DS_CONFIG.get("logreg_solver", "lbfgs"),
        C=DS_CONFIG.get("logreg_C", 1.0),
        class_weight=DS_CONFIG.get("logreg_class_weight", "balanced"),
        max_iter=max(DS_CONFIG.get("logreg_max_iter", 1000), 2000),
        random_state=DS_CONFIG.get("logreg_random_state", 1024),
    )
    lr.fit(scaler.transform(X_fit), y_fit)
    return lr.predict_proba(scaler.transform(X_test))[:, 1]


def fit_xgb(X_fit, y_fit, X_test, mode: str = "fixed"):
    """XGBoost head.

    mode="fixed": the family-matched configuration the paper uses for its
    XGBoost-on-features control (n_estimators=200, max_depth=3,
    colsample_bytree=0.5, hist, seed 1024). Fast.
    mode="cv": the full submitted PS_4 grid, 5-fold CV inside FIT. Slow
    (108 configs x 5 folds per rung) but is the like-for-like comparator
    against the published PS_4 ceiling. Use for the numbers you quote.
    """
    from xgboost import XGBClassifier
    n_pos = int((y_fit == 1).sum())
    n_neg = int((y_fit == 0).sum())
    spw = n_neg / max(n_pos, 1)
    seed = DS_CONFIG.get("xgb_random_state", 1024)

    def mk(**kw):
        return XGBClassifier(
            scale_pos_weight=spw, random_state=seed,
            eval_metric="logloss", verbosity=0, tree_method="hist", **kw)

    if mode == "fixed":
        clf = mk(n_estimators=200, max_depth=3, colsample_bytree=0.5)
        clf.fit(X_fit, y_fit)
        return clf.predict_proba(X_test)[:, 1]

    grid = [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr,
         "subsample": ss, "colsample_bytree": cs}
        for n in DS_CONFIG.get("xgb_n_estimators", [50, 100, 200])
        for d in DS_CONFIG.get("xgb_max_depth", [2, 3, 4])
        for lr in DS_CONFIG.get("xgb_learning_rate", [0.05, 0.1, 0.3])
        for ss in DS_CONFIG.get("xgb_subsample", [0.8, 1.0])
        for cs in DS_CONFIG.get("xgb_colsample_bytree", [0.8, 1.0])
    ]
    skf = StratifiedKFold(
        n_splits=DS_CONFIG.get("xgb_cv_folds", 5), shuffle=True,
        random_state=seed)
    best_auc, best = -1.0, grid[0]
    for params in grid:
        aucs = []
        for tr, va in skf.split(X_fit, y_fit):
            clf = mk(**params)
            clf.fit(X_fit[tr], y_fit[tr])
            aucs.append(compute_auc(y_fit[va],
                                    clf.predict_proba(X_fit[va])[:, 1]))
        m = float(np.mean(aucs))
        if m > best_auc:
            best_auc, best = m, params
    clf = mk(**best)
    clf.fit(X_fit, y_fit)
    return clf.predict_proba(X_test)[:, 1]


def cv_auc(X_fit, y_fit, seed: int = 1024, n_folds: int = 5) -> float:
    """Mean 5-fold stratified CV AUROC inside FIT, logistic head."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in skf.split(X_fit, y_fit):
        aucs.append(compute_auc(y_fit[va],
                                fit_logreg(X_fit[tr], y_fit[tr], X_fit[va])))
    return float(np.mean(aucs))


# --------------------------------------------------------------------------
# Oracle rung
# --------------------------------------------------------------------------

def oracle_domain_scores(test_logits3d, test_labels, boundaries, generators,
                         seed: int = 1024, n_folds: int = 5) -> np.ndarray:
    """NON-DEPLOYABLE. Per-generator out-of-fold logistic regression.

    For each held-out generator, fit on 4/5 of that generator's own
    labelled samples and score the held-out 1/5. No sample is ever
    predicted by a model that saw it, but the procedure consumes target
    labels, so it upper-bounds any domain-conditioned weighting scheme.
    """
    scores = np.zeros(len(test_labels), dtype=np.float32)
    for g in generators:
        s, e = boundaries[g]
        X = feat_flat(test_logits3d[s:e])
        y = test_labels[s:e]
        if len(np.unique(y)) < 2 or len(y) < 2 * n_folds:
            # Degenerate generator: fall back to probability averaging so the
            # rung is still defined. Flagged in the result JSON.
            scores[s:e] = _softmax(test_logits3d[s:e]).mean(axis=1)[:, 1]
            continue
        k = min(n_folds, int(np.bincount(y.astype(int)).min()))
        if k < 2:
            scores[s:e] = _softmax(test_logits3d[s:e]).mean(axis=1)[:, 1]
            continue
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        oof = np.zeros(len(y), dtype=np.float32)
        for tr, va in skf.split(X, y):
            oof[va] = fit_logreg(X[tr], y[tr], X[va])
        scores[s:e] = oof
    return scores


# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", default="arch6")
    p.add_argument("--xgb_mode", choices=["fixed", "cv"], default="fixed",
                   help="'cv' reproduces the submitted PS_4 grid. Slow. "
                        "Use it for the numbers you quote.")
    p.add_argument("--seeds", type=int, nargs="+", default=common.DEFAULT_SEEDS)
    p.add_argument("--skip_baseline_ladder", action="store_true",
                   help="Skip refitting the submitted 10-method ladder and "
                        "compare against published macro values instead. "
                        "Only for smoke tests.")
    return p.parse_args()


def main():
    args = parse_args()
    log = common.make_logger("run_a_extended_ladder")
    expert_names = get_pool_experts(args.pool)
    cache_root = _paths.cache_root()
    generators = list(TEST_GENERATORS)

    log("RUN A - extended prediction-space ladder")
    log(f"pool={args.pool}  experts={expert_names}  xgb_mode={args.xgb_mode}")
    log(f"writing to {_paths.RESULTS_DIR} (never into codebase/)")

    t0 = time.time()
    fit_feats, fit_logits3d, fit_labels = common.load_split(
        expert_names, FIT_TAG, cache_root)
    _, sel_logits3d, sel_labels = common.load_split(
        expert_names, SELECT_TAG, cache_root)
    X_test_feat, test_logits3d, test_labels, boundaries, gens = \
        common.load_test_generators(expert_names, cache_root, generators)
    log(f"FIT {fit_logits3d.shape}  SELECT {sel_logits3d.shape}  "
        f"TEST {test_logits3d.shape}")

    # ---- submitted ladder (the published ceiling) ------------------------
    if args.skip_baseline_ladder:
        log("skipping the submitted ladder (smoke-test mode)")
        base_scores, base_best, base_macro, base_all = {}, None, None, {}
    else:
        log("running the submitted 10-method ladder ...")
        base_scores = common.run_ps_ladder(
            fit_logits3d, fit_labels, sel_logits3d, sel_labels,
            test_logits3d, seeds=args.seeds, skip_slow=False, log=log)
        base_best, base_macro, base_all = common.best_of(
            base_scores, test_labels, boundaries, gens)
        log(f"published-ladder ceiling: {base_best} = {base_macro:.4f}")

    # ---- new rungs -------------------------------------------------------
    unc_map, temps = make_uncertainty_featmap(sel_logits3d, sel_labels)
    rank_map = make_rank_featmap(fit_logits3d)

    featmaps = {
        "PS_11_disagreement": feat_disagreement,
        "PS_12_uncertainty_mi": unc_map,
        "PS_13_quadratic": feat_quadratic,
        "PS_14_rank_ecdf": rank_map,
    }

    new_scores: Dict[str, np.ndarray] = {}
    cv_scores: Dict[str, float] = {}
    for name, fm in featmaps.items():
        t1 = time.time()
        X_fit = fm(fit_logits3d)
        X_test = fm(test_logits3d)
        log(f"{name}: {X_fit.shape[1]} features")
        new_scores[f"{name}_logreg"] = np.asarray(
            fit_logreg(X_fit, fit_labels, X_test), dtype=np.float32)
        new_scores[f"{name}_xgb"] = np.asarray(
            fit_xgb(X_fit, fit_labels, X_test, mode=args.xgb_mode),
            dtype=np.float32)
        cv_scores[name] = cv_auc(X_fit, fit_labels)
        log(f"  cv(FIT)={cv_scores[name]:.4f}  ({time.time() - t1:.0f}s)")

    # PS_15: CV-selected best rung. Selection uses FIT only.
    best_by_cv = max(cv_scores, key=cv_scores.get)
    log(f"PS_15_cv_best -> {best_by_cv} (cv={cv_scores[best_by_cv]:.4f})")
    new_scores["PS_15_cv_best_logreg"] = new_scores[f"{best_by_cv}_logreg"]
    new_scores["PS_15_cv_best_xgb"] = new_scores[f"{best_by_cv}_xgb"]

    # PS_16: non-deployable domain oracle.
    log("PS_16_oracle_domain (NON-DEPLOYABLE upper envelope) ...")
    new_scores["PS_16_oracle_domain"] = oracle_domain_scores(
        test_logits3d, test_labels, boundaries, gens)

    # ---- FS_1 anchor -----------------------------------------------------
    log("fitting FS_1 (raw-feature LogReg) ...")
    fs1_scores = np.asarray(
        FS_1(np.concatenate(fit_feats, axis=1), fit_labels, X_test_feat),
        dtype=np.float32)
    fs1_macro = common.macro_auc(
        common.per_generator_auc(test_labels, fs1_scores, boundaries, gens),
        gens)
    log(f"FS_1 = {fs1_macro:.4f}")

    # ---- ceilings --------------------------------------------------------
    deployable = {k: v for k, v in {**base_scores, **new_scores}.items()
                  if not k.startswith("PS_16_oracle")}
    dep_best, dep_macro, dep_all = common.best_of(
        deployable, test_labels, boundaries, gens)
    all_incl = {**base_scores, **new_scores}
    orc_best, orc_macro, orc_all = common.best_of(
        all_incl, test_labels, boundaries, gens)

    # PS_17: retrospective per-generator maximum. Also non-deployable.
    pergen_max = {}
    for g in gens:
        pergen_max[g] = max(
            common.per_generator_auc(test_labels, sc, boundaries, [g])[g]
            for sc in deployable.values())
    ps17_macro = float(np.mean([pergen_max[g] for g in gens]))

    log("")
    log(f"deployable extended ceiling: {dep_best} = {dep_macro:.4f}")
    log(f"including oracles:           {orc_best} = {orc_macro:.4f}")
    log(f"PS_17 per-generator oracle selection = {ps17_macro:.4f}")

    cmp_dep = common.compare_methods(
        test_labels, fs1_scores, deployable[dep_best],
        boundaries, gens, "FS_1_raw_feat_logreg", dep_best)
    log(f"DELTA FS_1 - deployable extended ceiling = {cmp_dep['delta']:+.4f}  "
        f"CI {cmp_dep['generator_bootstrap']['ci_95']}  "
        f"wins {cmp_dep['wins']}/{cmp_dep['n_generators']}  "
        f"sign-flip p = {cmp_dep['sign_flip_permutation']['p_value']:.4f}")

    cmp_orc = common.compare_methods(
        test_labels, fs1_scores, all_incl[orc_best],
        boundaries, gens, "FS_1_raw_feat_logreg", orc_best)

    result = {
        "run": "A_extended_prediction_space_ladder",
        "pool": args.pool,
        "experts": list(expert_names),
        "generators": gens,
        "xgb_mode": args.xgb_mode,
        "seeds": list(args.seeds),
        "calibration_temperatures_from_select": temps.tolist(),
        "cv_auc_inside_fit": cv_scores,
        "ps_15_selected_rung": best_by_cv,
        "published_ladder": {
            "best_name": base_best,
            "best_macro": base_macro,
            "all_macro": {k: float(v) for k, v in base_all.items()},
        },
        "extended_rungs_macro": {
            k: float(v) for k, v in
            common.best_of(new_scores, test_labels, boundaries, gens)[2].items()
        },
        "deployable_ceiling": {"name": dep_best, "macro": float(dep_macro)},
        "oracle_inclusive_ceiling": {"name": orc_best, "macro": float(orc_macro)},
        "ps_17_per_generator_oracle_selection": {
            "macro": ps17_macro, "per_generator": pergen_max,
            "note": "NON-DEPLOYABLE: retrospective per-generator maximum",
        },
        "fs1_macro": float(fs1_macro),
        "comparison_vs_deployable_ceiling": cmp_dep,
        "comparison_vs_oracle_ceiling": cmp_orc,
        "non_deployable_rungs": ["PS_16_oracle_domain",
                                 "PS_17_per_generator_oracle_selection"],
        "disclosed_deviations": [
            "engineered rungs use a StandardScaler fitted on FIT; the "
            "submitted ladder does not standardise. This can only help the "
            "prediction side, the conservative direction for the claim.",
            "xgb_mode='fixed' uses the paper's family-matched XGBoost "
            "configuration rather than the full PS_4 CV grid; rerun with "
            "--xgb_mode cv before quoting numbers.",
        ],
        "published_reference": {
            "arch6_delta": 0.123, "arch6_ci": [0.078, 0.174],
            "arch6_best_ps": 0.843, "clip4_best_ps": 0.939,
        },
        "elapsed_sec": float(time.time() - t0),
    }
    out = common.write_json(
        result,
        _paths.RESULTS_DIR / f"run_a_extended_ladder_{args.pool}.json")
    log(f"saved: {out}")


if __name__ == "__main__":
    main()
