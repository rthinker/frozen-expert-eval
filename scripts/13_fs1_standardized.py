"""FS_1 with StandardScaler for the feature-preprocessing robustness check.

Refits FS_1 after z-scoring FIT features and applying the same scaler to
TEST. Tests whether the FS_1 advantage depends on raw feature scale.
"""
import _bootstrap  # noqa: F401  # MUST be first: binds CUBLAS_WORKSPACE_CONFIG before torch.

import argparse
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.models.feature_space import FS_1
from src.utils.config import (
    DEFAULT_TRAIN_TAG, CACHE_ROOT, RESULTS_ROOT, get_pool_experts,
)
from src.utils.io import save_json, load_pool_data, load_all_generators
from src.utils.metrics import compute_auc
from src.utils.dataset_helpers import TEST_GENERATORS
from src.utils.seed import set_seed
from src.utils.logging_setup import setup_logging


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=str, required=True)
    p.add_argument("--train_tag", type=str, default=DEFAULT_TRAIN_TAG)
    p.add_argument("--seed", type=int, default=1024)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    log = setup_logging("13_fs1_std")

    expert_names = get_pool_experts(args.pool)
    log.info(f"Pool: {args.pool}, train: {args.train_tag}")
    log.info(f"Experts: {expert_names}")

    log.info("Loading FIT features...")
    tr_feats, _, tr_labels = load_pool_data(
        expert_names, args.train_tag, CACHE_ROOT)
    X_train_feat = np.concatenate(tr_feats, axis=1)
    log.info(f"  FIT feat shape: {X_train_feat.shape}")

    log.info("Loading test generators...")
    _, X_test_feat, _, te_labels, gen_boundaries = \
        load_all_generators(expert_names, CACHE_ROOT, TEST_GENERATORS)
    log.info(f"  TEST feat shape: {X_test_feat.shape}")

    log.info("Fitting StandardScaler on FIT, applying to TEST...")
    t0 = time.time()
    scaler = StandardScaler().fit(X_train_feat)
    X_train_std = scaler.transform(X_train_feat).astype(np.float32)
    X_test_std = scaler.transform(X_test_feat).astype(np.float32)
    log.info(f"  scaler fit + transform: {time.time()-t0:.1f}s")
    train_std_mean = float(X_train_std.mean())
    train_std_std = float(X_train_std.std())
    test_std_mean = float(X_test_std.mean())
    test_std_std = float(X_test_std.std())
    log.info(f"  train mean: {train_std_mean:+.3e}   std: {train_std_std:.3f}")
    log.info(f"  test  mean: {test_std_mean:+.3e}   std: {test_std_std:.3f}")

    # Run FS_1 on standardized features.
    log.info("Fitting FS_1 (LogReg on STANDARDIZED features)...")
    t0 = time.time()
    test_scores_std = FS_1(
        X_train_std, tr_labels, X_test_std, logger=log)
    log.info(f"  fit + predict: {time.time()-t0:.1f}s")

    # Also rerun FS_1 on raw features for a paired comparison at this invocation.
    log.info("Fitting FS_1 (LogReg on RAW features) for pairing...")
    t0 = time.time()
    test_scores_raw = FS_1(
        X_train_feat.astype(np.float32), tr_labels,
        X_test_feat.astype(np.float32), logger=log)
    log.info(f"  fit + predict: {time.time()-t0:.1f}s")

    per_gen_std = {}
    per_gen_raw = {}
    for g in TEST_GENERATORS:
        s, e = gen_boundaries[g]
        y_g = te_labels[s:e]
        a_std = compute_auc(y_g, test_scores_std[s:e])
        a_raw = compute_auc(y_g, test_scores_raw[s:e])
        per_gen_std[g] = float(a_std)
        per_gen_raw[g] = float(a_raw)
        log.info(f"  {g:15s}: raw={a_raw:.4f}  std={a_std:.4f}  d={a_std-a_raw:+.4f}")

    avg_std = float(np.mean(list(per_gen_std.values())))
    avg_raw = float(np.mean(list(per_gen_raw.values())))
    log.info("-" * 60)
    log.info(f"  avg(8) RAW FS_1 = {avg_raw:.4f}")
    log.info(f"  avg(8) STD FS_1 = {avg_std:.4f}")
    log.info(f"  delta (STD - RAW) = {avg_std - avg_raw:+.4f}")

    result = {
        "pool": args.pool,
        "method": "fs_1_standardized_vs_raw",
        "avg_auc_8_raw": avg_raw,
        "avg_auc_8_std": avg_std,
        "delta_std_minus_raw": avg_std - avg_raw,
        "per_generator_raw": per_gen_raw,
        "per_generator_std": per_gen_std,
        "seed": args.seed,
        "fit_samples": int(len(tr_labels)),
        "feature_dim": int(X_train_feat.shape[1]),
        # Post-standardization stats so reviewers can audit the scaler's
        # behavior without parsing the log file.
        "scaler_diagnostics": {
            "train_mean_after_scale": train_std_mean,
            "train_std_after_scale": train_std_std,
            "test_mean_after_scale": test_std_mean,
            "test_std_after_scale": test_std_std,
        },
    }
    out = RESULTS_ROOT / f"13_fs1_standardized_{args.pool}.json"
    save_json(result, out)
    log.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
