"""composition_eval.bootstrap: paired generator-level bootstrap CIs + AUC re-exports."""

import numpy as np

from src.utils.metrics import compute_auc, bootstrap_paired_auc


def generator_level_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    gen_boundaries: dict,
    n_boot: int = 10_000,
    seed: int = 42,
):
    """Generator-level paired bootstrap of delta = AUC(a) - AUC(b).

    Computes per-generator AUCs once (using `gen_boundaries` row slices
    {gen: (lo, hi)}), then resamples generators with replacement
    `n_boot` times. Returns dict with `delta_mean`, `ci_95`,
    `per_gen_aucs_a`, `per_gen_aucs_b`, `n_boot`.
    """
    gens = list(gen_boundaries.keys())
    per_gen_a = {}
    per_gen_b = {}
    for g in gens:
        lo, hi = gen_boundaries[g]
        per_gen_a[g] = float(compute_auc(labels[lo:hi], scores_a[lo:hi]))
        per_gen_b[g] = float(compute_auc(labels[lo:hi], scores_b[lo:hi]))

    a_arr = np.array([per_gen_a[g] for g in gens])
    b_arr = np.array([per_gen_b[g] for g in gens])
    n_gens = len(gens)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_gens, size=n_gens)
        deltas[i] = a_arr[idx].mean() - b_arr[idx].mean()

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_mean": float(deltas.mean()),
        "ci_95": (float(lo), float(hi)),
        "per_gen_aucs_a": per_gen_a,
        "per_gen_aucs_b": per_gen_b,
        "n_boot": int(n_boot),
    }


__all__ = [
    "compute_auc",
    "bootstrap_paired_auc",
    "generator_level_bootstrap",
]
