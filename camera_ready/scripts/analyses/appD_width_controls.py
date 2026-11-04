# C09: Evaluate width controls on the standard split.
"""RUN C - width-matched feature-space control.

Tests whether interface width explains the difference between features
and predictions by matching the number of input coordinates.

WIDTH-CONTROL DESIGN
----------------------
For each expert i, fit an UNSUPERVISED projection (PCA) on FIT only to d
coordinates, concatenate across experts, and hand the result to the exact
frozen FS_1 anchor (L2 logistic, C = 1.0, no standardisation). Sweep
d in {1, 2, 4, 8}. At d = 2 on arch6 the feature interface receives
6 x 2 = 12 coordinates - exactly the width of the prediction interface
(6 experts x 2 logits). Same splits, same seeds, same combiner family.

Mirror control, for completeness: a SUPERVISED per-expert linear probe to
d coordinates, fitted on FIT only by iterative deflation (fit logistic
direction, project it out, refit on the residual). Bounds what a *learned*
width-matched projection recovers.

Comparator: PS_4 (cross-validated XGBoost on logits), the submitted
ladder's strongest member on every pool (paper Table A12), refitted here
with the frozen implementation on the full FIT logits. Its macro must
reproduce the published ceiling (0.843 arch6 / 0.939 clip4) to 1e-3 or
the run aborts - that is the parity guard.

The comparison object is the paper's: macro AUROC over the 8 held-out
generators, paired generator-level bootstrap 95% CI (10,000 resamples),
exact sign-flip permutation, BCa. TEST is never modified.

Interpretation rules fixed before the run:
- If width-matched FS still exceeds the PS ceiling: width is not the
  mechanism; the prediction interface is narrow in the wrong directions.
- If it does not: we have quantified how much of the published margin is
  attributable to width alone, which is the cleaner statement.
Note the paper's PCA-64 control (FS_B) is a compression check at high d,
NOT this control; do not conflate them.

Outputs -> camera_ready/output/results/analyses/run_c_width_matched_<pool>.json
Writes nothing outside camera_ready/output/.

Usage (from the repository root)::

    python camera_ready/scripts/analyses/appD_width_controls.py --pool arch6
    python camera_ready/scripts/analyses/appD_width_controls.py --pool arch6 --dims 2 --skip_probe
    python camera_ready/scripts/analyses/appD_width_controls.py --pool clip4
"""

from __future__ import annotations

import _paths  # noqa: F401  # MUST be first

import argparse
import time
from typing import Dict, List, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import _common as common
from src.models.feature_space import FS_1  # noqa: E402
from src.models.prediction_space import PS_4  # noqa: E402
from src.utils.config import FIT_TAG, get_pool_experts  # noqa: E402
from src.utils.dataset_helpers import TEST_GENERATORS  # noqa: E402

PUBLISHED_PS4 = {"arch6": 0.843, "clip4": 0.939}
PARITY_TOL = 1e-3


# --------------------------------------------------------------------------
# Projections, fitted on FIT only
# --------------------------------------------------------------------------

def pca_project(fit_feats: List[np.ndarray], test_feats_cat: np.ndarray,
                dims_per_expert: Sequence[int], d: int, log) -> Dict:
    """Per-expert PCA to d coords on FIT; returns projected FIT/TEST."""
    fit_blocks, test_blocks, evr = [], [], []
    off = 0
    for i, xf in enumerate(fit_feats):
        Di = dims_per_expert[i]
        xt = test_feats_cat[:, off:off + Di]
        off += Di
        p = PCA(n_components=d, svd_solver="randomized", random_state=42)
        fit_blocks.append(p.fit_transform(np.asarray(xf, dtype=np.float32)))
        test_blocks.append(p.transform(np.asarray(xt, dtype=np.float32)))
        evr.append(float(p.explained_variance_ratio_.sum()))
    log(f"    PCA d={d}: mean explained variance across experts "
        f"{float(np.mean(evr)):.3f}")
    return {
        "X_fit": np.concatenate(fit_blocks, axis=1),
        "X_test": np.concatenate(test_blocks, axis=1),
        "explained_variance_ratio_sum_per_expert": evr,
    }


def probe_project(fit_feats: List[np.ndarray], fit_labels: np.ndarray,
                  test_feats_cat: np.ndarray,
                  dims_per_expert: Sequence[int], d: int, log) -> Dict:
    """Per-expert supervised linear probe to d coords by deflation.

    Direction k is the L2-logistic (C = 1.0) solution on the residual after
    projecting out directions 1..k-1. Fitted on FIT only; TEST only ever
    transformed. Bounds a *learned* width-matched projection.
    """
    fit_blocks, test_blocks = [], []
    off = 0
    for i, xf in enumerate(fit_feats):
        Di = dims_per_expert[i]
        Xf = np.asarray(xf, dtype=np.float32).copy()
        Xt = np.asarray(test_feats_cat[:, off:off + Di], dtype=np.float32).copy()
        off += Di
        fdirs, tdirs = [], []
        for _ in range(d):
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            clf.fit(Xf, fit_labels)
            w = clf.coef_.ravel().astype(np.float32)
            nw = float(np.linalg.norm(w))
            if nw == 0.0:
                break
            w /= nw
            fdirs.append(Xf @ w)
            tdirs.append(Xt @ w)
            Xf = Xf - np.outer(Xf @ w, w)
            Xt = Xt - np.outer(Xt @ w, w)
        fit_blocks.append(np.stack(fdirs, axis=1))
        test_blocks.append(np.stack(tdirs, axis=1))
        log(f"    probe d={d}: expert {i} done ({len(fdirs)} directions)")
    return {
        "X_fit": np.concatenate(fit_blocks, axis=1),
        "X_test": np.concatenate(test_blocks, axis=1),
    }


# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", default="arch6")
    p.add_argument("--dims", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--skip_probe", action="store_true",
                   help="Skip the supervised-probe mirror control.")
    p.add_argument("--probe_dims", type=int, nargs="+", default=[2],
                   help="d values for the probe mirror (default: 2, the "
                        "width-matched point).")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    log = common.make_logger("run_c_width_matched")
    expert_names = get_pool_experts(args.pool)
    cache_root = _paths.cache_root()
    generators = list(TEST_GENERATORS)

    log("RUN C - width-matched feature-space control")
    log(f"pool={args.pool}  experts={expert_names}")
    log(f"writing to {_paths.RESULTS_DIR} (never into codebase/)")

    fit_feats, fit_logits3d, fit_labels = common.load_split(
        expert_names, FIT_TAG, cache_root)
    X_test_feat, test_logits3d, test_labels, boundaries, gens = \
        common.load_test_generators(expert_names, cache_root, generators)
    dims_per_expert = [f.shape[1] for f in fit_feats]
    n_experts = len(expert_names)
    ps_width = 2 * n_experts
    log(f"expert feature dims {dims_per_expert}; prediction-interface width "
        f"= {ps_width} floats")

    # ---- comparator: the submitted PS_4, with parity guard ---------------
    log("fitting PS_4 (frozen implementation, full FIT logits) ...")
    ps4_scores = np.asarray(
        PS_4(fit_logits3d, fit_labels, test_logits3d), dtype=np.float32)
    ps4_macro = common.macro_auc(
        common.per_generator_auc(test_labels, ps4_scores, boundaries, gens),
        gens)
    ref = PUBLISHED_PS4.get(args.pool)
    log(f"PS_4 macro = {ps4_macro:.4f}  (published {ref})")
    parity_ok = ref is None or abs(ps4_macro - ref) < PARITY_TOL
    if not parity_ok:
        log(f"PARITY FAILURE: |{ps4_macro:.4f} - {ref}| >= {PARITY_TOL}. "
            f"Numbers from this run are NOT comparable to the paper. "
            f"Aborting without writing a quotable result.")
        raise SystemExit(2)

    sweep = {}
    for d in args.dims:
        log(f"  PCA sweep, d={d} ({n_experts * d} coords)")
        proj = pca_project(fit_feats, X_test_feat, dims_per_expert, d, log)
        scores = np.asarray(
            FS_1(proj["X_fit"], fit_labels, proj["X_test"]), dtype=np.float32)
        cmp = common.compare_methods(
            test_labels, scores, ps4_scores, boundaries, gens,
            f"FS_pca{d}_logreg", "PS_4_xgboost")
        log(f"    FS_pca{d} macro={cmp['macro_a']:.4f}  "
            f"delta vs PS_4={cmp['delta']:+.4f}  "
            f"CI {cmp['generator_bootstrap']['ci_95']}")
        sweep[f"d{d}"] = {
            "coords_total": int(n_experts * d),
            "width_matched": bool(n_experts * d == ps_width),
            "explained_variance_ratio_sum_per_expert":
                proj["explained_variance_ratio_sum_per_expert"],
            "fs_macro": cmp["macro_a"],
            "comparison_vs_ps4": cmp,
        }

    probe = {}
    if not args.skip_probe:
        for d in args.probe_dims:
            log(f"  supervised-probe mirror, d={d}")
            pr = probe_project(fit_feats, fit_labels, X_test_feat,
                               dims_per_expert, d, log)
            scores = np.asarray(
                FS_1(pr["X_fit"], fit_labels, pr["X_test"]), dtype=np.float32)
            cmp = common.compare_methods(
                test_labels, scores, ps4_scores, boundaries, gens,
                f"FS_probe{d}_logreg", "PS_4_xgboost")
            log(f"    FS_probe{d} macro={cmp['macro_a']:.4f}  "
                f"delta vs PS_4={cmp['delta']:+.4f}  "
                f"CI {cmp['generator_bootstrap']['ci_95']}")
            probe[f"d{d}"] = {
                "coords_total": int(n_experts * d),
                "fs_macro": cmp["macro_a"],
                "comparison_vs_ps4": cmp,
            }

    res = {
        "run": "C_width_matched_fs_control",
        "pool": args.pool,
        "experts": list(expert_names),
        "generators": gens,
        "prediction_interface_width_floats": ps_width,
        "projection": "per-expert PCA fitted on FIT only (randomized SVD, "
                      "random_state=42), concatenated; combiner is the frozen "
                      "FS_1 anchor (L2 logistic, C=1.0, unstandardised)",
        "comparator": {
            "name": "PS_4_xgboost (frozen implementation, full FIT)",
            "macro": float(ps4_macro),
            "published_reference": ref,
            "parity_ok": bool(parity_ok),
        },
        "pca_sweep": sweep,
        "probe_mirror": probe,
        "note": "The paper's PCA-64 control (FS_B) is a compression check at "
                "high d, not a width-matched comparison; this run is the "
                "width-matched control. TEST never modified.",
        "elapsed_sec": float(time.time() - t0),
    }
    out = common.write_json(
        res, _paths.RESULTS_DIR / f"run_c_width_matched_{args.pool}.json")
    log(f"saved: {out}")


if __name__ == "__main__":
    main()
