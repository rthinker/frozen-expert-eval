"""Synthetic checks for scripts/review/refit_fs1_v3.py: batched scoring equals one-shot scoring up to
floating-point rounding (BLAS summation order differs with the batch size; observed differences ~6e-16)."""
import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "refit_fs1", Path(__file__).resolve().parents[1] / "scripts" / "review" / "refit_fs1_v3.py")
refit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(refit)

CFG = {"raw_feature_logreg_solver": "lbfgs", "raw_feature_logreg_C": 1.0,
       "raw_feature_logreg_class_weight": "balanced", "raw_feature_logreg_max_iter": 2000,
       "raw_feature_logreg_random_state": 1024}


def _data(seed=0, n_fit=400, n_eval=257, d=37):
    rng = np.random.default_rng(seed)
    X_fit = rng.normal(size=(n_fit, d)).astype(np.float32)
    w = rng.normal(size=d)
    y_fit = (X_fit @ w + rng.normal(scale=2.0, size=n_fit) > 0).astype(np.int32)
    X_eval = rng.normal(size=(n_eval, d)).astype(np.float32)
    return X_fit, y_fit, X_eval


def test_batched_scores_equal_one_shot():
    X_fit, y_fit, X_eval = _data()
    one_shot = refit.fs1_fit_score(CFG, X_fit, y_fit, X_eval, batch=len(X_eval))
    for batch in (1, 7, 64, 256, 10_000):
        batched = refit.fs1_fit_score(CFG, X_fit, y_fit, X_eval, batch=batch)
        np.testing.assert_allclose(batched, one_shot, rtol=0, atol=1e-12)
        # ranking (what AUROC uses) is unchanged
        np.testing.assert_array_equal(np.argsort(batched, kind="stable"), np.argsort(one_shot, kind="stable"))


def test_scores_are_probabilities_in_row_order():
    X_fit, y_fit, X_eval = _data(seed=1)
    s = refit.fs1_fit_score(CFG, X_fit, y_fit, X_eval, batch=50)
    assert s.shape == (len(X_eval),) and np.all((s >= 0) & (s <= 1))
    head = refit.fs1_fit_score(CFG, X_fit, y_fit, X_eval[:50], batch=50)
    np.testing.assert_allclose(s[:50], head, rtol=0, atol=1e-12)
