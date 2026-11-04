"""Metrics: AUC computation and bootstrap confidence intervals.

Uses sklearn.metrics.roc_auc_score directly (numerically identical to
DF40's calculate_metrics_for_train: both wrap sklearn roc_curve + auc).
sklearn is preferred because the cached pipeline is numpy-only and the
indirection through DF40 would force an unnecessary adapter import.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary AUC-ROC. Returns 0.5 if only one class present."""
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def compute_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary AUPRC (average precision). Returns 0.0 if only one class."""
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(average_precision_score(labels, scores))


def bootstrap_paired_auc(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap CI on AUC delta (A minus B).

    Resamples individual test samples with replacement,
    computes AUC for both methods on each resample.

    Returns:
        Dict with keys: delta_mean, ci_95_lo, ci_95_hi,
        p_a_gt_b, n_boot, n_degenerate.
    """
    rng = np.random.default_rng(seed)
    N = len(labels)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N, size=N)
        y = labels[idx]
        if len(np.unique(y)) < 2:
            deltas[i] = 0.0
            continue
        auc_a = roc_auc_score(y, scores_a[idx])
        auc_b = roc_auc_score(y, scores_b[idx])
        deltas[i] = auc_a - auc_b
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    p_pos = float((deltas > 0).mean())
    return {
        "delta_mean": float(np.mean(deltas)),
        "ci_95_lo": float(ci_lo),
        "ci_95_hi": float(ci_hi),
        "p_a_gt_b": p_pos,
        "n_boot": n_boot,
        "n_degenerate": int((deltas == 0.0).sum()),
    }
