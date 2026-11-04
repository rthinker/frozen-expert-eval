"""PS_4 + FS_1 on merged9 (clip4 + arch6 deduplicated); Appendix G."""

import _bootstrap  # noqa: F401  # MUST be first: binds CUBLAS_WORKSPACE_CONFIG before torch.

import time

import numpy as np

from src.utils.config import CACHE_ROOT, FIT_TAG, RESULTS_ROOT, get_pool_experts
from src.utils.io import load_stacked_logits, load_concat_features, save_json
from src.utils.metrics import compute_auc
from src.utils.dataset_helpers import TEST_GENERATORS
from src.models.prediction_space import PS_4
from src.models.feature_space import FS_1
from src.utils.seed import set_seed
from src.utils.logging_setup import setup_logging


def main():
    set_seed(1024)
    log = setup_logging("14_merged")
    pool = "merged9"
    experts = get_pool_experts(pool)
    cr = CACHE_ROOT

    log.info(f"Pool: {pool} ({len(experts)} experts)")
    log.info(f"Experts: {experts}")

    fit_logits, fit_y = load_stacked_logits(experts, FIT_TAG, cr)
    fit_feat, _ = load_concat_features(experts, FIT_TAG, cr)
    log.info(f"{FIT_TAG} logits: {fit_logits.shape}, features: {fit_feat.shape}")

    # Load all test generators
    gen_bounds = {}
    all_logits_parts, all_labels_parts, all_feat_parts = [], [], []
    offset = 0
    for g in TEST_GENERATORS:
        gl, gy = load_stacked_logits(experts, g, cr)
        gf, _ = load_concat_features(experts, g, cr)
        n = len(gy)
        gen_bounds[g] = (offset, offset + n)
        all_logits_parts.append(gl)
        all_labels_parts.append(gy)
        all_feat_parts.append(gf)
        offset += n
    all_logits = np.concatenate(all_logits_parts)
    all_labels = np.concatenate(all_labels_parts)
    all_feat = np.concatenate(all_feat_parts)

    # PS_4: XGBoost on logits
    log.info("PS_4 (XGBoost on logits)...")
    t0 = time.time()
    ps6_scores = PS_4(fit_logits, fit_y, all_logits)
    ps6_per_gen = {}
    ps6_aucs = []
    for g in TEST_GENERATORS:
        s, e = gen_bounds[g]
        auc = compute_auc(all_labels[s:e], ps6_scores[s:e])
        ps6_per_gen[g] = float(auc)
        ps6_aucs.append(auc)
        log.info(f"  {g:15s}: {auc:.4f}")
    ps6_avg = float(np.mean(ps6_aucs))
    log.info(f"  Avg(8): {ps6_avg:.4f} ({time.time()-t0:.0f}s)")

    # FS_1: raw-feat LogReg on features
    log.info("FS_1 (raw-feat LogReg on features)...")
    t0 = time.time()
    rf_scores = FS_1(fit_feat, fit_y, all_feat, logger=log)
    rf_per_gen = {}
    rf_aucs = []
    for g in TEST_GENERATORS:
        s, e = gen_bounds[g]
        auc = compute_auc(all_labels[s:e], rf_scores[s:e])
        rf_per_gen[g] = float(auc)
        rf_aucs.append(auc)
        log.info(f"  {g:15s}: {auc:.4f}")
    rf_avg = float(np.mean(rf_aucs))
    log.info(f"  Avg(8): {rf_avg:.4f} ({time.time()-t0:.0f}s)")

    delta = rf_avg - ps6_avg
    log.info(f"\nPS_4: {ps6_avg:.4f}, FS_1: {rf_avg:.4f}, "
             f"delta AUROC: {delta:+.4f}")

    result = {
        "pool": pool,
        "expert_names": experts,
        "n_experts": len(experts),
        "logit_dim": int(fit_logits.shape[1] * fit_logits.shape[2]),
        "feature_dim": int(fit_feat.shape[1]),
        "fit_samples": int(len(fit_y)),
        "ps6_xgboost": {
            "avg_auc_8": ps6_avg,
            "per_generator": ps6_per_gen,
        },
        "fs1_raw_feat_logreg": {
            "avg_auc_8": rf_avg,
            "per_generator": rf_per_gen,
        },
        "delta": float(delta),
    }
    out = RESULTS_ROOT / "14_merged9.json"
    save_json(result, out)
    log.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
