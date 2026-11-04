"""XGBoost on concatenated raw features (FS_D); family-matched control to PS_4.

Uses the same combiner family as PS_4 on the feature interface.
"""

import _bootstrap  # noqa: F401  # MUST be first: binds CUBLAS_WORKSPACE_CONFIG before torch.

import argparse
import time

import numpy as np
from xgboost import XGBClassifier

from src.utils.config import CACHE_ROOT, FIT_TAG, RESULTS_ROOT, get_pool_experts
from src.utils.io import load_concat_features, save_json
from src.utils.metrics import compute_auc
from src.utils.dataset_helpers import TEST_GENERATORS
from src.utils.seed import set_seed
from src.utils.logging_setup import setup_logging


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=str, default="clip4")
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1024)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    log = setup_logging("12_xgb_feat")

    expert_names = get_pool_experts(args.pool)
    cache_root = CACHE_ROOT

    log.info(f"Pool: {args.pool}")
    log.info("Loading features...")
    fit_feat, fit_y = load_concat_features(expert_names, FIT_TAG, cache_root)
    log.info(f"  {FIT_TAG}: {fit_feat.shape}")

    n_neg = int((fit_y == 0).sum())
    n_pos = int((fit_y == 1).sum())
    spw = n_neg / max(n_pos, 1)

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
    }
    log.info(f"Config: {params}")

    log.info("Fitting XGBoost on features...")
    t0 = time.time()
    clf = XGBClassifier(
        **params,
        scale_pos_weight=spw,
        random_state=args.seed,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        tree_method="hist",
    )
    clf.fit(fit_feat, fit_y)
    fit_time = time.time() - t0
    log.info(f"  Fit done ({fit_time:.0f}s)")

    per_gen = {}
    aucs = []
    for g in TEST_GENERATORS:
        gf, gy = load_concat_features(expert_names, g, cache_root)
        scores = clf.predict_proba(gf)[:, 1]
        auc = compute_auc(gy, scores)
        per_gen[g] = float(auc)
        aucs.append(auc)
        log.info(f"  {g:15s}: {auc:.4f}")

    avg = float(np.mean(aucs))
    total = time.time() - t0
    log.info(f"  Avg(8): {avg:.4f} ({total:.0f}s)")

    result = {
        "pool": args.pool,
        "method": "xgboost_on_features",
        "avg_auc_8": avg,
        "per_generator": per_gen,
        "params": params,
        "seed": args.seed,
        "fit_samples": int(len(fit_y)),
        "feature_dim": int(fit_feat.shape[1]),
        "elapsed_s": round(total, 1),
    }
    out = RESULTS_ROOT / f"12_xgb_features_{args.pool}.json"
    save_json(result, out)
    log.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
