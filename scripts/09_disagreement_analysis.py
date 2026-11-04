"""PS_1 AUROC by expert-disagreement tercile.

For each test sample: std(expert P(fake)) -> assign to tercile; report
PS_1 AUC per tercile.
"""

import _bootstrap  # noqa: F401  # MUST be first: binds CUBLAS_WORKSPACE_CONFIG before torch.

import argparse

import numpy as np
from scipy.special import softmax

from src.utils.metrics import compute_auc
from src.utils.io import save_json, load_all_generators
from src.utils.logging_setup import setup_logging
from src.utils.dataset_helpers import TEST_GENERATORS
from src.utils.config import get_pool_experts, CACHE_ROOT, RESULTS_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=str, default="clip4")
    args = p.parse_args()
    log = setup_logging("09_disagree")

    expert_names = get_pool_experts(args.pool)
    cache_root = CACHE_ROOT

    # Pool across all test generators
    _, _, logits, labels, _ = load_all_generators(
        expert_names, cache_root, TEST_GENERATORS)
    probs = softmax(logits, axis=-1)[:, :, 1]  # [N, E] P(fake) per expert

    # Compute disagreement = std of expert P(fake)
    disagreement = probs.std(axis=1)  # [N]

    # Tercile boundaries
    t1 = np.percentile(disagreement, 33.33)
    t2 = np.percentile(disagreement, 66.67)
    log.info(f"Disagreement tercile bounds: {t1:.3f}, {t2:.3f}")

    # Probability-average scores
    prob_avg = probs.mean(axis=1)  # [N]

    terciles = {"Low": disagreement <= t1,
                "Mid": (disagreement > t1) & (disagreement <= t2),
                "High": disagreement > t2}

    log.info(f"\n{'='*60}")
    log.info("AUROC by expert-disagreement tercile")
    log.info(f"{'='*60}")
    log.info(f"{'Tercile':<8s} {'N':>6s} {'Disagree':>10s} {'ProbAvg AUC':>12s}")
    log.info("-" * 40)

    results = {"pool": args.pool, "terciles": {}}
    for name, mask in terciles.items():
        n = int(mask.sum())
        mean_d = float(disagreement[mask].mean())
        auc = compute_auc(labels[mask], prob_avg[mask])
        log.info(f"  {name:<6s} {n:6d} {mean_d:10.3f} {auc:12.4f}")
        results["terciles"][name] = {
            "n": n, "mean_disagreement": mean_d, "prob_avg_auc": auc,
        }

    results["tercile_bounds"] = [float(t1), float(t2)]
    out_path = RESULTS_ROOT / f"09_disagreement_{args.pool}.json"
    save_json(results, out_path)
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
