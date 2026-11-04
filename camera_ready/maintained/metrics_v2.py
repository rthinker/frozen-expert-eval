"""Metrics for reuse beyond the paper: version 2 of the reviewed src/utils/metrics.py (kept unchanged).

The reviewed module returns placeholder values in three edge cases. Version 2 treats them explicitly; on test sets
that hold both classes, the case of every result of the paper, it returns exactly what the reviewed functions return
(camera_ready/tests/test_metrics_v2.py checks this on the stored LOGO predictions):

* compute_auc: AUROC is undefined when the labels hold one class. Version 2 returns NaN; the reviewed function
  returns 0.5, which reads as chance.
* compute_auprc: average precision is 1.0 when every label is positive (any ranking finds every positive at full
  precision) and undefined when none is (there is nothing to recall), NaN. The reviewed function returns 0.0 for
  both.
* bootstrap_paired_auc: a resample that holds one class is degenerate. Version 2 leaves it out of the mean, the
  interval and the probability and counts it in n_degenerate; it counts resamples on which both methods have the same
  AUROC in n_ties. The reviewed function scores a degenerate resample as a delta of 0.0 and counts every delta of
  0.0, ties included, as degenerate. The resamples are drawn as in the reviewed function, so with no degenerate
  resample the results are identical.

The paper's results use the reviewed module (the reviewed files keep their bytes); this module is not called by the
pipeline.
"""
from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _classes(labels):
    return np.unique(np.asarray(labels))


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary AUROC; NaN when the labels hold one class (undefined)."""
    if len(_classes(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary average precision (positive class 1); 1.0 when every label is positive, NaN when none is."""
    y = np.asarray(labels)
    classes = _classes(y)
    if len(classes) < 2:
        if len(y) and classes[0] == 1:
            return 1.0
        return float("nan")
    return float(average_precision_score(y, scores))


def bootstrap_paired_auc(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap of the AUROC difference A minus B over resampled test samples.

    Returns delta_mean, ci_95_lo, ci_95_hi and p_a_gt_b over the resamples that hold both classes (NaN when none
    does), and n_boot, n_valid, n_degenerate (resamples with one class) and n_ties (valid resamples on which the two
    AUROCs are equal).
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = len(labels)
    deltas = np.full(n_boot, np.nan)
    ties = 0
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        if len(np.unique(y)) < 2:
            continue
        auc_a = roc_auc_score(y, scores_a[idx])
        auc_b = roc_auc_score(y, scores_b[idx])
        deltas[i] = auc_a - auc_b
        ties += int(auc_a == auc_b)
    valid = deltas[~np.isnan(deltas)]
    if len(valid):
        ci_lo, ci_hi = np.percentile(valid, [2.5, 97.5])
        mean, p_pos = float(np.mean(valid)), float((valid > 0).mean())
    else:
        ci_lo = ci_hi = mean = p_pos = float("nan")
    return {
        "delta_mean": mean,
        "ci_95_lo": float(ci_lo),
        "ci_95_hi": float(ci_hi),
        "p_a_gt_b": p_pos,
        "n_boot": n_boot,
        "n_valid": int(len(valid)),
        "n_degenerate": int(n_boot - len(valid)),
        "n_ties": ties,
    }
