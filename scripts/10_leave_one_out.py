"""Leave-one-expert-out marginal analysis (Appendix E).

Train FS combiner with and without each expert; multi-seed to separate
FS stochasticity from the expert-removal effect.
"""

import _bootstrap  # noqa: F401  # MUST be first: binds CUBLAS_WORKSPACE_CONFIG before torch.

import argparse

import numpy as np
import torch

from src.utils.config import (
    DEFAULT_TRAIN_TAG, SELECT_TAG,
    CACHE_ROOT, RESULTS_ROOT, FS_SEEDS,
    get_pool_experts, get_feature_dims,
)

from src.training.fs_runtime import train_fs_model, predict_fs
from src.utils.io import save_json, preload_pool_views, get_feature_views
from src.utils.logging_setup import setup_logging
from src.utils.dataset_helpers import TEST_GENERATORS
from src.utils.metrics import compute_auc
from src.utils.naming import tag_suffix as _tag_suffix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=str, default="clip4")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--seeds", type=int, nargs="+", default=FS_SEEDS,
                   help="Seeds for FS training (default: 8 seeds from experiments.yaml)")
    p.add_argument("--train_tag", type=str, default=DEFAULT_TRAIN_TAG)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    log = setup_logging("10_loo")
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    expert_names = get_pool_experts(args.pool)
    cache_root = CACHE_ROOT
    preloaded = preload_pool_views(
        expert_names,
        [args.train_tag, SELECT_TAG, *TEST_GENERATORS],
        cache_root,
    )
    generator_views = {g: preloaded[g] for g in TEST_GENERATORS}

    def eval_subset(subset, seed):
        """Train FS_2 for one seed, return per-generator sample-level scores.

        Returns dict {gen: {"scores": [N], "labels": [N]}}. Aggregation across
        seeds happens in `multi_seed_auc` by averaging per-sample scores first,
        then computing AUC (matches the bootstrap protocol in scripts 06/07).
        """
        dims = get_feature_dims(subset)
        train_feats, train_labels = get_feature_views(preloaded[args.train_tag], subset)
        val_feats, val_labels = get_feature_views(preloaded[SELECT_TAG], subset)
        model, _ = train_fs_model(
            train_feats, train_labels, dims, args.proj_dim,
            val_feats=val_feats, val_labels=val_labels,
            seed=seed, device=device,
        )
        per_gen_scores = {}
        for gen in TEST_GENERATORS:
            feats, labels = get_feature_views(generator_views[gen], subset)
            scores = np.asarray(predict_fs(model, feats, device), dtype=np.float32)
            per_gen_scores[gen] = {"scores": scores, "labels": labels}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return per_gen_scores

    def multi_seed_auc(subset):
        """Mean + std of Avg(8) across seeds, plus AUC on seed-averaged predictions.

        Returns (mean_avg8_seedavg, std_avg8_per_seed, per_seed_avg8).
        `mean_avg8_seedavg` computes per-generator AUC on seed-averaged
        sample-level predictions, then macro-averages. This matches the
        bootstrap protocol in scripts 06/07.
        """
        per_seed_sample_scores = {g: [] for g in TEST_GENERATORS}
        per_seed_avg8 = []
        gen_labels = None
        for s in args.seeds:
            seed_gen_scores = eval_subset(subset, s)
            aucs_this_seed = []
            for g in TEST_GENERATORS:
                scores = seed_gen_scores[g]["scores"]
                labels = seed_gen_scores[g]["labels"]
                per_seed_sample_scores[g].append(scores)
                aucs_this_seed.append(float(compute_auc(labels, scores)))
            per_seed_avg8.append(float(np.mean(aucs_this_seed)))
            if gen_labels is None:
                gen_labels = {g: seed_gen_scores[g]["labels"] for g in TEST_GENERATORS}

        gen_aucs = []
        for g in TEST_GENERATORS:
            avg_scores = np.mean(per_seed_sample_scores[g], axis=0)
            gen_aucs.append(float(compute_auc(gen_labels[g], avg_scores)))
        mean_avg8_seedavg = float(np.mean(gen_aucs))
        std_avg8_per_seed = float(np.std(per_seed_avg8))
        return mean_avg8_seedavg, std_avg8_per_seed, per_seed_avg8

    log.info(f"Pool: {args.pool} ({len(expert_names)} experts)")
    log.info(f"Device: {device}")
    log.info(f"Seeds ({len(args.seeds)}): {args.seeds}")

    full_mean, full_std, full_per_seed = multi_seed_auc(expert_names)
    log.info(f"Full pool Avg(8): {full_mean:.4f} +/- {full_std:.4f}")

    results = {
        "pool": args.pool,
        "train_tag": args.train_tag,
        "proj_dim": args.proj_dim,
        "seeds": list(args.seeds),
        "full_auc_mean": full_mean,
        "full_auc_std": full_std,
        "full_auc_per_seed": full_per_seed,
        "loo": [],
    }
    log.info(f"\n{'Expert removed':<20s} {'Avg(8)':>10s} {'Std':>8s} {'Delta':>10s}")
    log.info("-" * 55)
    for name in expert_names:
        subset = [n for n in expert_names if n != name]
        loo_mean, loo_std, loo_per_seed = multi_seed_auc(subset)
        delta = full_mean - loo_mean
        log.info(f"  -{name:<18s} {loo_mean:10.4f} {loo_std:8.4f} {delta:+10.4f}")
        results["loo"].append({
            "removed": name,
            "avg8_mean": loo_mean,
            "avg8_std": loo_std,
            "avg8_per_seed": loo_per_seed,
            "delta_vs_full": delta,
        })

    out = RESULTS_ROOT / f"10_loo_{args.pool}_d{args.proj_dim}{_tag_suffix(args.train_tag)}.json"
    save_json(results, out)
    log.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
