"""Participation ratio and cumulative variance per pool (Appendix D.1)."""

import _bootstrap  # noqa: F401  # MUST be first: binds CUBLAS_WORKSPACE_CONFIG before torch.

import argparse

import numpy as np

from src.utils.io import save_json, load_all_generators
from src.utils.logging_setup import setup_logging
from src.utils.dataset_helpers import TEST_GENERATORS
from src.utils.config import get_pool_experts, CACHE_ROOT, RESULTS_ROOT, FIGURES_ROOT


def participation_ratio(X, n_components=3000):
    """Compute participation ratio = (sum eigenvalues)^2 / sum(eigenvalues^2).

    Higher PR = more effective dimensions used.
    Uses randomized SVD with n_components for consistency across pools.
    Default n_components=3000 covers the full useful spectrum (k99 < 3000
    for all pools tested).
    """
    X_centered = X - X.mean(axis=0)
    n_comp = min(n_components, min(X_centered.shape) - 1)
    if n_comp >= X_centered.shape[1]:
        # Full SVD when n_components covers the full dimensionality
        _, S, _ = np.linalg.svd(X_centered, full_matrices=False)
    else:
        from sklearn.utils.extmath import randomized_svd
        _, S, _ = randomized_svd(X_centered, n_components=n_comp,
                                 random_state=42)
    eigenvalues = S ** 2 / (len(X) - 1)
    pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    return float(pr), eigenvalues


def cumulative_variance(eigenvalues):
    """Fraction of variance explained as a function of components."""
    total = eigenvalues.sum()
    return np.cumsum(eigenvalues) / total


def k_threshold(cumvar, threshold):
    """Number of components needed to reach threshold fraction of variance."""
    idx = np.searchsorted(cumvar, threshold)
    return int(min(idx + 1, len(cumvar)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=str, default="clip4")
    args = p.parse_args()
    log = setup_logging("08_pca")

    expert_names = get_pool_experts(args.pool)
    cache_root = CACHE_ROOT

    # Pool across all 8 test generators
    log.info(f"Pooling across {len(TEST_GENERATORS)} generators...")
    _, X_feat, X_logit_stacked, _, _ = load_all_generators(
        expert_names, cache_root, TEST_GENERATORS)
    X_logit = X_logit_stacked.reshape(len(X_logit_stacked), -1)
    log.info(f"  Features: {X_feat.shape}, Logits: {X_logit.shape}")

    # Participation ratio
    log.info("Computing participation ratios...")
    pr_feat, eig_feat = participation_ratio(X_feat)
    pr_logit, eig_logit = participation_ratio(X_logit)
    pr_gap = pr_feat / pr_logit

    # Cumulative variance
    cv_feat = cumulative_variance(eig_feat)
    cv_logit = cumulative_variance(eig_logit)
    k95_feat = k_threshold(cv_feat, 0.95)
    k99_feat = k_threshold(cv_feat, 0.99)
    k95_logit = k_threshold(cv_logit, 0.95)
    k99_logit = k_threshold(cv_logit, 0.99)

    log.info(f"\n{'='*60}")
    log.info("Effective dimensionality")
    log.info(f"{'='*60}")
    log.info(f"{'Space':<15s} {'PR':>8s} {'k_95':>6s} {'k_99':>6s} {'dim':>6s}")
    log.info("-" * 45)
    log.info(f"{'Features':<15s} {pr_feat:8.1f} {k95_feat:6d} {k99_feat:6d} "
             f"{X_feat.shape[1]:6d}")
    log.info(f"{'Logits':<15s} {pr_logit:8.1f} {k95_logit:6d} {k99_logit:6d} "
             f"{X_logit.shape[1]:6d}")
    log.info(f"{'PR gap':<15s} {pr_gap:8.1f}x")

    # Cumulative variance plot.
    # Per-pool filename (previously a single `figure2_pca_cumvar.*` file was
    # overwritten each time this script ran for a different pool). x-limit
    # is scaled to just above the feature k99 so the 95% horizontal guide
    # actually intersects the feature curve (the old hardcoded limit of 100
    # could not show the 95% crossing since feature k99 is typically >1000).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = FIGURES_ROOT
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(range(1, len(cv_feat) + 1), cv_feat, label="Features", linewidth=2)
        ax.plot(range(1, len(cv_logit) + 1), cv_logit, label="Logits", linewidth=2)
        ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(0.99, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Number of principal components")
        ax.set_ylabel("Cumulative variance explained")
        ax.set_title(f"PCA: Features vs Logits ({args.pool})")
        ax.legend()
        # Show at least enough components to see the feature 99% crossing,
        # padded by ~20% so the curve has visible headroom.
        upper = int(max(k99_feat, k99_logit) * 1.2) + 10
        ax.set_xlim(0, min(upper, len(cv_feat)))
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(
                fig_dir / f"figure2_pca_cumvar_{args.pool}.{ext}", dpi=150)
        plt.close()
        log.info(
            f"  Cumulative variance plot saved: {fig_dir}/figure2_pca_cumvar_{args.pool}.{{png,pdf}}")
    except ImportError:
        log.warning("  matplotlib not available, skipping cumulative variance plot")

    # Save results
    results = {
        "pool": args.pool,
        "n_samples": int(X_feat.shape[0]),
        "features": {
            "dim": int(X_feat.shape[1]), "pr": pr_feat,
            "k95": k95_feat, "k99": k99_feat,
        },
        "logits": {
            "dim": int(X_logit.shape[1]), "pr": pr_logit,
            "k95": k95_logit, "k99": k99_logit,
        },
        "pr_gap": pr_gap,
    }
    out_path = RESULTS_ROOT / f"08_pca_{args.pool}.json"
    save_json(results, out_path)
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
