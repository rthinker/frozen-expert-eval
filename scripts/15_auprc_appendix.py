"""AUPRC alongside AUROC for arch6, clip4, LOGO arch6; Appendix H.

Reports per-generator class prevalence so AUPRC values are
interpretable. Writes output/results/15_auprc_appendix.json without
modifying any existing result file.
"""

import _bootstrap  # noqa: F401  # MUST be first: binds CUBLAS_WORKSPACE_CONFIG before torch.

import argparse
import time

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.config import (
    DEFAULT_TRAIN_TAG, CACHE_ROOT, RESULTS_ROOT,
    get_pool_experts, get_feature_dims, FS_SEEDS, SELECT_TAG,
)
from src.models.prediction_space import (
    PS_1, PS_4,
    PS_A, PS_B,
    PS_2, PS_C,
    PS_3, PS_F,
    PS_D, PS_E,
)
from src.models.feature_space import FS_1
from src.training.fs_runtime import train_fs_model, predict_fs
from src.utils.metrics import compute_auc, compute_auprc
from src.utils.io import (
    save_json, load_pool_data, load_stacked_logits,
    load_concat_features, load_json,
)
from src.utils.logging_setup import setup_logging
from src.utils.dataset_helpers import TEST_GENERATORS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pools", nargs="+", default=["clip4", "arch6"])
    p.add_argument("--logo_pool", type=str, default="arch6")
    p.add_argument("--logo_proj_dim", type=int, default=256)
    p.add_argument("--seeds", type=int, nargs="+", default=FS_SEEDS)
    p.add_argument("--select_frac", type=float, default=0.15)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def per_generator_metrics(scores, labels, gen_boundaries, generators):
    """Compute AUC and AUPRC per generator, plus prevalence."""
    per_gen = {}
    for g in generators:
        s, e = gen_boundaries[g]
        g_labels = labels[s:e]
        g_scores = scores[s:e]
        n_total = int(e - s)
        n_fake = int(g_labels.sum())
        per_gen[g] = {
            "auc": compute_auc(g_labels, g_scores),
            "auprc": compute_auprc(g_labels, g_scores),
            "n_total": n_total,
            "n_fake": n_fake,
            "prevalence": round(n_fake / n_total, 4) if n_total > 0 else 0.0,
        }
    avg_auc = float(np.mean([per_gen[g]["auc"] for g in generators]))
    avg_auprc = float(np.mean([per_gen[g]["auprc"] for g in generators]))
    return per_gen, avg_auc, avg_auprc


def run_pool(pool, log, seeds, device):
    """Compute AUPRC for PS_1, PS_4, FS_1, FS_2 on a given pool."""
    expert_names = get_pool_experts(pool)
    feature_dims = get_feature_dims(expert_names)
    cache_root = CACHE_ROOT

    log.info(f"\n{'='*60}")
    log.info(f"Pool: {pool}")
    log.info(f"{'='*60}")

    # Load training data
    fit_logits, fit_labels = load_stacked_logits(
        expert_names, DEFAULT_TRAIN_TAG, cache_root)
    sel_logits, sel_labels = load_stacked_logits(
        expert_names, SELECT_TAG, cache_root)
    fit_feat, _ = load_concat_features(
        expert_names, DEFAULT_TRAIN_TAG, cache_root)

    # Load test data
    gen_boundaries = {}
    all_logits_parts, all_feat_parts, all_labels_parts = [], [], []
    offset = 0
    for g in TEST_GENERATORS:
        g_logits, g_labels = load_stacked_logits(expert_names, g, cache_root)
        g_feat, _ = load_concat_features(expert_names, g, cache_root)
        n = len(g_labels)
        gen_boundaries[g] = (offset, offset + n)
        all_logits_parts.append(g_logits)
        all_feat_parts.append(g_feat)
        all_labels_parts.append(g_labels)
        offset += n

    all_logits = np.concatenate(all_logits_parts, axis=0)
    all_feat = np.concatenate(all_feat_parts, axis=0)
    all_labels = np.concatenate(all_labels_parts)

    methods = {}

    # PS_1: Probability averaging
    log.info("  PS_1 (prob avg)...")
    scores = PS_1(all_logits)
    pg, avg_auc, avg_auprc = per_generator_metrics(
        scores, all_labels, gen_boundaries, TEST_GENERATORS)
    methods["PS_1_prob_avg"] = {
        "avg_auc": avg_auc, "avg_auprc": avg_auprc, "per_generator": pg}
    log.info(f"    AUC={avg_auc:.4f}  AUPRC={avg_auprc:.4f}")

    # PS_4: XGBoost
    log.info("  PS_4 (XGBoost)...")
    scores = PS_4(fit_logits, fit_labels, all_logits)
    pg, avg_auc, avg_auprc = per_generator_metrics(
        scores, all_labels, gen_boundaries, TEST_GENERATORS)
    methods["PS_4_xgboost"] = {
        "avg_auc": avg_auc, "avg_auprc": avg_auprc, "per_generator": pg}
    log.info(f"    AUC={avg_auc:.4f}  AUPRC={avg_auprc:.4f}")

    # FS_1: Raw-feature LogReg
    log.info("  FS_1 (raw-feat LogReg)...")
    scores = FS_1(fit_feat, fit_labels, all_feat, logger=log)
    pg, avg_auc, avg_auprc = per_generator_metrics(
        scores, all_labels, gen_boundaries, TEST_GENERATORS)
    methods["FS_1_raw_feat_logreg"] = {
        "avg_auc": avg_auc, "avg_auprc": avg_auprc, "per_generator": pg}
    log.info(f"    AUC={avg_auc:.4f}  AUPRC={avg_auprc:.4f}")

    # FS_2: MLP (retrain with same seeds, take seed-average)
    proj_dim = 32 if pool == "clip4" else 256
    log.info(f"  FS_2 (MLP, d={proj_dim}, {len(seeds)} seeds)...")
    tr_feats, _, tr_labels = load_pool_data(
        expert_names, DEFAULT_TRAIN_TAG, cache_root)
    val_feats, _, val_labels = load_pool_data(
        expert_names, SELECT_TAG, cache_root)
    te_feats_list = []
    for g in TEST_GENERATORS:
        g_feats, _, _ = load_pool_data(expert_names, g, cache_root)
        te_feats_list.append(g_feats)
    # Concatenate per-expert across generators
    n_experts = len(expert_names)
    te_feats_merged = [
        np.concatenate([te_feats_list[gi][ei]
                        for gi in range(len(TEST_GENERATORS))])
        for ei in range(n_experts)
    ]

    seed_scores = []
    for seed in seeds:
        import torch
        model, _ = train_fs_model(
            tr_feats, tr_labels, feature_dims, proj_dim,
            val_feats=val_feats, val_labels=val_labels,
            seed=seed, device=device, classifier_type="mlp",
        )
        s = predict_fs(model, te_feats_merged, device)
        seed_scores.append(s)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_scores = np.mean(seed_scores, axis=0)
    pg, avg_auc, avg_auprc = per_generator_metrics(
        avg_scores, all_labels, gen_boundaries, TEST_GENERATORS)
    methods["FS_2_mlp"] = {
        "avg_auc": avg_auc, "avg_auprc": avg_auprc,
        "proj_dim": proj_dim, "n_seeds": len(seeds),
        "per_generator": pg,
    }
    log.info(f"    AUC={avg_auc:.4f}  AUPRC={avg_auprc:.4f}")

    # Prevalence summary
    prevalence = {}
    for g in TEST_GENERATORS:
        s, e = gen_boundaries[g]
        gl = all_labels[s:e]
        prevalence[g] = {
            "n_total": int(e - s),
            "n_fake": int(gl.sum()),
            "n_real": int((gl == 0).sum()),
            "prevalence_fake": round(float(gl.sum()) / (e - s), 4),
        }

    return {"pool": pool, "methods": methods, "prevalence": prevalence}


def _stack_logits(logits_list):
    return np.stack(logits_list, axis=1)


def _get_ps_method_fn(method_name):
    """Return a callable (fit_logits, fit_labels, sel_logits, sel_labels,
    te_logits, te_labels, seeds) -> scores for a given PS method name."""
    # Map base method names to callables
    METHOD_MAP = {
        "PS_1_prob_avg": lambda fl, fla, sl, sla, tl, tla, seeds:
            PS_1(tl),
        "PS_A_calibrated_avg": lambda fl, fla, sl, sla, tl, tla, seeds:
            PS_A(sl, sla, tl),
        "PS_B_learned_weights": lambda fl, fla, sl, sla, tl, tla, seeds:
            PS_B(sl, sla, tl),
        "PS_2_logreg": lambda fl, fla, sl, sla, tl, tla, seeds:
            PS_2(fl, fla, tl),
        "PS_C_logreg_cv": lambda fl, fla, sl, sla, tl, tla, seeds:
            PS_C(fl, fla, tl),
        "PS_4_xgboost": lambda fl, fla, sl, sla, tl, tla, seeds:
            PS_4(fl, fla, tl),
    }
    # Stochastic methods need seed averaging
    if method_name in ("PS_3_mlp_logits", "PS_D_mlp_probs",
                       "PS_F_mlp_logits_es", "PS_E_mlp_probs_es"):
        def _stochastic_fn(fl, fla, sl, sla, tl, tla, seeds):
            fn_map = {
                "PS_3_mlp_logits": lambda s: PS_3(fl, fla, tl, seed=s),
                "PS_D_mlp_probs": lambda s: PS_D(fl, fla, tl, seed=s),
                "PS_F_mlp_logits_es": lambda s: PS_F(
                    fl, fla, sl, sla, tl, seed=s),
                "PS_E_mlp_probs_es": lambda s: PS_E(
                    fl, fla, sl, sla, tl, seed=s),
            }
            fn = fn_map[method_name]
            all_s = [fn(seed) for seed in seeds]
            return np.mean(all_s, axis=0)
        return _stochastic_fn

    if method_name in METHOD_MAP:
        return METHOD_MAP[method_name]
    raise ValueError(f"Unknown PS method: {method_name}")


def run_logo(pool, proj_dim, seeds, select_frac, log, device):
    """LOGO AUPRC: re-run per-fold best PS and raw-feat LogReg."""
    expert_names = get_pool_experts(pool)
    cache_root = CACHE_ROOT
    results_root = RESULTS_ROOT

    # Read existing LOGO results to get per-fold best PS method names
    logo_path = results_root / f"11_disjoint_{pool}_d{proj_dim}.json"
    if not logo_path.exists():
        log.warning(f"  {logo_path} not found, skipping LOGO AUPRC")
        return None

    logo = load_json(logo_path)

    log.info(f"\n{'='*60}")
    log.info(f"LOGO AUPRC: {pool}, d={proj_dim}")
    log.info(f"{'='*60}")

    # Load all 8 generators
    gen_data = {}
    for g in TEST_GENERATORS:
        gen_data[g] = load_pool_data(expert_names, g, cache_root)
        log.info(f"  {g}: {len(gen_data[g][2])} samples")

    fold_results = []
    for fold_info in logo["per_fold"]:
        held_out = fold_info["held_out"]
        best_ps_name = fold_info["best_ps_name"]
        log.info(f"\n  Fold: held_out={held_out}, best_ps={best_ps_name}")

        # Merge 7 training generators
        train_gens = [g for g in TEST_GENERATORS if g != held_out]
        n_experts = len(expert_names)
        all_feats = [[] for _ in range(n_experts)]
        all_logits = [[] for _ in range(n_experts)]
        all_labels_list = []
        for g in train_gens:
            feats, logits, labels = gen_data[g]
            for i in range(n_experts):
                all_feats[i].append(feats[i])
                all_logits[i].append(logits[i])
            all_labels_list.append(labels)
        merged_feats = [np.concatenate(f) for f in all_feats]
        merged_logits = [np.concatenate(lg) for lg in all_logits]
        merged_labels = np.concatenate(all_labels_list)
        merged_logits3d = _stack_logits(merged_logits)

        # Split FIT/SELECT (same seed=1024 as 11_disjoint_data.py)
        idx = np.arange(len(merged_labels))
        fit_idx, sel_idx = train_test_split(
            idx, test_size=select_frac, random_state=1024,
            stratify=merged_labels)

        fit_feats = [f[fit_idx] for f in merged_feats]
        fit_logits3d = merged_logits3d[fit_idx]
        fit_labels = merged_labels[fit_idx]
        sel_logits3d = merged_logits3d[sel_idx]
        sel_labels = merged_labels[sel_idx]

        # Test data
        te_feats, te_logits, te_labels = gen_data[held_out]
        te_logits3d = _stack_logits(te_logits)
        n_test = len(te_labels)
        n_fake = int(te_labels.sum())
        prevalence = round(n_fake / n_test, 4)

        # Best PS method
        ps_fn = _get_ps_method_fn(best_ps_name)
        ps_scores = ps_fn(fit_logits3d, fit_labels,
                          sel_logits3d, sel_labels,
                          te_logits3d, te_labels, seeds)
        ps_auc = compute_auc(te_labels, ps_scores)
        ps_auprc = compute_auprc(te_labels, ps_scores)

        # Raw-feat LogReg
        rf_scores = FS_1(
            np.concatenate(fit_feats, axis=1),
            fit_labels,
            np.concatenate(te_feats, axis=1),
            logger=log,
        )
        rf_auc = compute_auc(te_labels, rf_scores)
        rf_auprc = compute_auprc(te_labels, rf_scores)

        log.info(f"    PS ({best_ps_name}): AUC={ps_auc:.4f} AUPRC={ps_auprc:.4f}")
        log.info(f"    FS_1 (raw-feat):     AUC={rf_auc:.4f} AUPRC={rf_auprc:.4f}")
        log.info(f"    prevalence(fake): {prevalence}  (n={n_test})")

        fold_results.append({
            "held_out": held_out,
            "n_test": n_test,
            "n_fake": n_fake,
            "prevalence_fake": prevalence,
            "best_ps_name": best_ps_name,
            "best_ps_auc": ps_auc,
            "best_ps_auprc": ps_auprc,
            "raw_feat_auc": rf_auc,
            "raw_feat_auprc": rf_auprc,
        })

    avg_ps_auc = float(np.mean([f["best_ps_auc"] for f in fold_results]))
    avg_ps_auprc = float(np.mean([f["best_ps_auprc"] for f in fold_results]))
    avg_rf_auc = float(np.mean([f["raw_feat_auc"] for f in fold_results]))
    avg_rf_auprc = float(np.mean([f["raw_feat_auprc"] for f in fold_results]))

    log.info("\n  LOGO Summary:")
    log.info(f"    Best PS:    AUC={avg_ps_auc:.4f}  AUPRC={avg_ps_auprc:.4f}")
    log.info(f"    Raw-feat:   AUC={avg_rf_auc:.4f}  AUPRC={avg_rf_auprc:.4f}")
    log.info(f"    Delta AUC:  {avg_rf_auc - avg_ps_auc:+.4f}")
    log.info(f"    Delta AUPRC:{avg_rf_auprc - avg_ps_auprc:+.4f}")

    return {
        "pool": pool,
        "proj_dim": proj_dim,
        "avg_best_ps_auc": avg_ps_auc,
        "avg_best_ps_auprc": avg_ps_auprc,
        "avg_raw_feat_auc": avg_rf_auc,
        "avg_raw_feat_auprc": avg_rf_auprc,
        "delta_auc": float(avg_rf_auc - avg_ps_auc),
        "delta_auprc": float(avg_rf_auprc - avg_ps_auprc),
        "per_fold": fold_results,
    }


def main():
    args = parse_args()
    log = setup_logging("15_auprc")
    import torch
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    log.info("AUPRC Appendix Computation")
    log.info(f"Pools: {args.pools}")
    log.info(f"Device: {device}")

    results = {"pools": {}, "logo": None}

    for pool in args.pools:
        t0 = time.time()
        pool_result = run_pool(pool, log, args.seeds, device)
        results["pools"][pool] = pool_result
        log.info(f"  {pool} done in {time.time()-t0:.0f}s")

    # LOGO
    t0 = time.time()
    logo_result = run_logo(
        args.logo_pool, args.logo_proj_dim,
        args.seeds, args.select_frac, log, device)
    results["logo"] = logo_result
    log.info(f"  LOGO done in {time.time()-t0:.0f}s")

    out = RESULTS_ROOT / "15_auprc_appendix.json"
    save_json(results, out)
    log.info(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
