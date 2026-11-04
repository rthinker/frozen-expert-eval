"""The maintained metrics (camera_ready/maintained/metrics_v2.py): the edge cases the reviewed module handles with
placeholders, and agreement with the reviewed module on the paper's two-class data. Nothing is fitted."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

CR = Path(__file__).resolve().parents[1]
ROOT = CR.parent


def _metrics_v2():
    spec = importlib.util.spec_from_file_location("metrics_v2", CR / "maintained" / "metrics_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reviewed():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.utils import metrics
    return metrics


def test_all_positive_labels():
    m2, rv = _metrics_v2(), _reviewed()
    y, s = np.ones(5, dtype=int), np.linspace(0.1, 0.9, 5)
    assert math.isnan(m2.compute_auc(y, s)) and m2.compute_auprc(y, s) == 1.0
    assert rv.compute_auc(y, s) == 0.5 and rv.compute_auprc(y, s) == 0.0       # the reviewed placeholders


def test_all_negative_labels():
    m2, rv = _metrics_v2(), _reviewed()
    y, s = np.zeros(5, dtype=int), np.linspace(0.1, 0.9, 5)
    assert math.isnan(m2.compute_auc(y, s)) and math.isnan(m2.compute_auprc(y, s))
    assert rv.compute_auc(y, s) == 0.5 and rv.compute_auprc(y, s) == 0.0


def test_normal_binary_labels_match_sklearn_and_the_reviewed_module():
    from sklearn.metrics import average_precision_score, roc_auc_score
    m2, rv = _metrics_v2(), _reviewed()
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random(200) + 0.3 * y
    assert m2.compute_auc(y, s) == roc_auc_score(y, s) == rv.compute_auc(y, s)
    assert m2.compute_auprc(y, s) == average_precision_score(y, s) == rv.compute_auprc(y, s)


def test_tied_methods_are_ties_not_degenerate_resamples():
    m2, rv = _metrics_v2(), _reviewed()
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 60)
    s = rng.random(60)
    new, old = m2.bootstrap_paired_auc(s, s, y, n_boot=200), rv.bootstrap_paired_auc(s, s, y, n_boot=200)
    assert new["n_ties"] == 200 and new["n_degenerate"] == 0 and new["delta_mean"] == 0.0
    assert old["n_degenerate"] == 200            # the reviewed counter takes every tie for a degenerate resample


def test_degenerate_resamples_are_counted_and_left_out():
    m2 = _metrics_v2()
    y, a, b = np.array([0, 1, 1]), np.array([0.1, 0.9, 0.8]), np.array([0.5, 0.4, 0.6])
    res = m2.bootstrap_paired_auc(a, b, y, n_boot=500, seed=3)
    rng, single = np.random.default_rng(3), 0
    for _ in range(500):                             # the same resamples, counted directly
        single += len(np.unique(y[rng.integers(0, 3, size=3)])) < 2
    assert res["n_degenerate"] == single > 0 and res["n_valid"] == 500 - single
    assert -1.0 <= res["ci_95_lo"] <= res["ci_95_hi"] <= 1.0
    empty = m2.bootstrap_paired_auc(a, b, np.ones(3, dtype=int), n_boot=20)
    assert empty["n_degenerate"] == 20 and math.isnan(empty["delta_mean"])


def test_the_paper_two_class_results_are_unchanged():
    """On every stored LOGO prediction of arch6 (both classes in each fold), version 2 equals the reviewed module."""
    m2, rv = _metrics_v2(), _reviewed()
    scores = sorted((CR / "output" / "results" / "mitigation" / "content_disjoint" / "scores").glob("arch6_d256_*.npz"))
    assert len(scores) == 8
    for path in scores:
        with np.load(path) as z:
            y = z["labels"]
            for name in ("FS_1_raw_feat_logreg", "PS_4_xgboost"):
                assert m2.compute_auc(y, z[name]) == rv.compute_auc(y, z[name])
                assert m2.compute_auprc(y, z[name]) == rv.compute_auprc(y, z[name])
            new = m2.bootstrap_paired_auc(z["FS_1_raw_feat_logreg"], z["PS_4_xgboost"], y, n_boot=200)
            old = rv.bootstrap_paired_auc(z["FS_1_raw_feat_logreg"], z["PS_4_xgboost"], y, n_boot=200)
            assert new["n_degenerate"] == 0
            assert all(new[k] == old[k] for k in ("delta_mean", "ci_95_lo", "ci_95_hi", "p_a_gt_b")), path.name
