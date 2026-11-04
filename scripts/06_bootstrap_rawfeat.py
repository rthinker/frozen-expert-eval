"""Paired generator-level bootstrap CI for FS_1 vs best PS; Section 4.2.

Reads `04_ds_{pool}.json::best_ds` plus per-sample scores from the
sibling `04_ds_{pool}_scores.npz` (script 04 must run first). Also
emits a secondary XGBoost-specific delta for continuity.
"""

import _bootstrap  # noqa: F401  # MUST be first: binds CUBLAS_WORKSPACE_CONFIG before torch.

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils.config import (
    DEFAULT_TRAIN_TAG, CACHE_ROOT, RESULTS_ROOT, get_pool_experts,
)

from src.models.feature_space import FS_1
from src.utils.io import save_json, load_pool_data, load_json, load_all_generators
from src.utils.metrics import bootstrap_paired_auc
from src.utils.logging_setup import setup_logging
from src.utils.dataset_helpers import TEST_GENERATORS
from src.utils.naming import tag_suffix


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=str, required=True)
    p.add_argument("--train_tag", type=str, default=DEFAULT_TRAIN_TAG)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_best_ps_scores(pool: str, results_root: Path, train_tag: str, log) -> dict:
    """Return {name: scores_array} for every stored PS method + the best one."""
    ts = tag_suffix(train_tag)
    ds_json = results_root / f"04_ds_{pool}{ts}.json"
    scores_npz = results_root / f"04_ds_{pool}_scores{ts}.npz"
    if not ds_json.exists():
        raise FileNotFoundError(
            f"{ds_json} not found. Run 04_prediction_space.py first."
        )
    if not scores_npz.exists():
        raise FileNotFoundError(
            f"{scores_npz} not found. Run 04_prediction_space.py "
            f"so it produces the scores artifact."
        )

    summary = load_json(ds_json)
    best_ds_name = summary["best_ds"]
    log.info(f"  reading best_ds from {ds_json.name}: {best_ds_name}")
    npz = np.load(scores_npz, allow_pickle=False)

    # A stochastic best_ds (e.g. PS_3_mlp_logits) exists in the npz as one
    # entry per seed: PS_3_mlp_logits_s7, PS_3_mlp_logits_s10, ...
    # We average those for the bootstrap comparator.
    seed_keys = [k for k in npz.keys()
                 if k.startswith(best_ds_name + "_s")]
    if seed_keys:
        log.info(f"  averaging {len(seed_keys)} seeds of {best_ds_name}")
        best_scores = np.mean(
            [npz[k].astype(np.float32) for k in seed_keys], axis=0)
    elif best_ds_name in npz.files:
        best_scores = npz[best_ds_name].astype(np.float32)
    else:
        raise KeyError(
            f"Best DS method {best_ds_name} has no score entries in {scores_npz}."
        )

    test_labels = npz["labels"].astype(np.int32)
    boundaries = npz["gen_boundaries"]
    generators = [str(g) for g in npz["generators"]]
    return {
        "best_name": best_ds_name,
        "best_scores": best_scores,
        "labels": test_labels,
        "boundaries": {g: tuple(boundaries[i]) for i, g in enumerate(generators)},
        "generators": generators,
        "xgb_scores": (
            npz["PS_4_xgboost"].astype(np.float32)
            if "PS_4_xgboost" in npz.files else None
        ),
        "summary": summary,
    }


def main():
    args = parse_args()
    log = setup_logging("06_boot_rf")
    expert_names = get_pool_experts(args.pool)
    results_root = RESULTS_ROOT
    cache_root = CACHE_ROOT

    log.info(f"Pool: {args.pool}, train: {args.train_tag}")
    log.info(f"Experts: {expert_names}")

    log.info("Loading training data...")
    tr_feats, _, tr_labels = load_pool_data(
        expert_names, args.train_tag, cache_root)
    X_train_feat = np.concatenate(tr_feats, axis=1)

    # FS_1 memory-budget gate: same logic as in 04_prediction_space.py.
    # On 16 GiB-RAM hosts the full21 (26591 x 33792) fit OOMs in
    # sklearn lbfgs; write a skip marker instead and exit cleanly so
    # that 07 (which uses FS_2 scores, not FS_1 scores) can still run.
    from src.models.feature_space import (
        fs1_over_budget, fs1_skip_marker, fs1_memory_estimate_gib,
    )
    n_samples, n_features = X_train_feat.shape
    if fs1_over_budget(n_samples, n_features):
        need = fs1_memory_estimate_gib(n_samples, n_features)
        log.info(
            f"FS_1 input matrix would need ~{need:.2f} GiB float64; "
            f"exceeds default 6.0 GiB budget. Writing skip marker."
        )
        log.info(
            "Set ALLOW_FS1_OOM=1 to bypass and run anyway (host "
            "will OOM if it does not have the memory)."
        )
        marker = fs1_skip_marker(
            n_samples, n_features,
            extra={
                "pool": args.pool,
                "train_tag": args.train_tag,
                "comparison": "raw_feat_logreg_vs_best_PS",
            },
        )
        out = results_root / f"06_bootstrap_{args.pool}{tag_suffix(args.train_tag)}.json"
        save_json(marker, out)
        log.info(f"Saved skip marker: {out}")
        return

    log.info("Loading test generators...")
    _, X_test_feat, _, te_labels, gen_boundaries = \
        load_all_generators(expert_names, cache_root, TEST_GENERATORS)
    N_test = len(te_labels)
    for g in TEST_GENERATORS:
        s, e = gen_boundaries[g]
        log.info(f"  {g:15s}: {e - s} samples")
    log.info(f"  Total test: {N_test}")

    log.info("Training raw-feature LogReg...")
    scores_rf = FS_1(X_train_feat, tr_labels, X_test_feat, logger=log)

    ps = _load_best_ps_scores(args.pool, results_root, args.train_tag, log)
    if len(ps["labels"]) != N_test:
        raise RuntimeError(
            f"labels length mismatch: 02 scores has {len(ps['labels'])}, "
            f"live test has {N_test}. Did you regenerate 02 after changing "
            f"the test cache?"
        )
    if not np.array_equal(ps["labels"], te_labels):
        raise RuntimeError("02 labels disagree with freshly loaded labels.")

    log.info("\nPer-generator AUC:")
    per_gen = {}
    for g in TEST_GENERATORS:
        s, e = gen_boundaries[g]
        auc_rf = roc_auc_score(te_labels[s:e], scores_rf[s:e])
        auc_ps = roc_auc_score(te_labels[s:e], ps["best_scores"][s:e])
        per_gen[g] = {
            "raw_feat": float(auc_rf),
            "best_ps": float(auc_ps),
            "n": int(e - s),
        }
        log.info(f"  {g:15s}: RF={auc_rf:.3f}  "
                 f"{ps['best_name']}={auc_ps:.3f}  n={e-s}")

    rf_per_gen = np.array([per_gen[g]["raw_feat"] for g in TEST_GENERATORS])
    ps_per_gen = np.array([per_gen[g]["best_ps"] for g in TEST_GENERATORS])
    avg_rf = float(rf_per_gen.mean())
    avg_ps = float(ps_per_gen.mean())
    log.info(f"\nMacro-avg RF:      {avg_rf:.4f}")
    log.info(f"Macro-avg best PS: {avg_ps:.4f} ({ps['best_name']})")
    log.info(f"Macro delta:       {avg_rf - avg_ps:+.4f}")

    log.info(f"\nSample-level paired bootstrap "
             f"({args.n_boot} resamples, N={N_test})...")
    boot = bootstrap_paired_auc(
        scores_rf, ps["best_scores"], te_labels,
        n_boot=args.n_boot, seed=args.seed)
    log.info(f"  delta mean: {boot['delta_mean']:+.4f}")
    log.info(f"  95% CI: [{boot['ci_95_lo']:+.3f}, "
             f"{boot['ci_95_hi']:+.3f}]")

    rng = np.random.default_rng(args.seed)
    gen_deltas = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, 8, size=8)
        gen_deltas.append(rf_per_gen[idx].mean() - ps_per_gen[idx].mean())
    gen_deltas = np.array(gen_deltas)
    gen_lo, gen_hi = np.percentile(gen_deltas, [2.5, 97.5])
    log.info(f"  generator-level CI: [{gen_lo:+.3f}, {gen_hi:+.3f}]")

    xgb_block = None
    if ps["xgb_scores"] is not None:
        xgb_scores = ps["xgb_scores"]
        per_gen_xgb = []
        for g in TEST_GENERATORS:
            s, e = gen_boundaries[g]
            per_gen_xgb.append(
                roc_auc_score(te_labels[s:e], xgb_scores[s:e]))
        avg_xgb = float(np.mean(per_gen_xgb))
        boot_xgb = bootstrap_paired_auc(
            scores_rf, xgb_scores, te_labels,
            n_boot=args.n_boot, seed=args.seed)
        xgb_block = {
            "macro_avg_xgboost": avg_xgb,
            "macro_delta": float(avg_rf - avg_xgb),
            "sample_level_ci_95": [boot_xgb["ci_95_lo"], boot_xgb["ci_95_hi"]],
        }
        log.info(f"\n[continuity] raw-feat vs XGBoost only: "
                 f"delta={avg_rf - avg_xgb:+.4f}")

    result = {
        "pool": args.pool,
        "train_tag": args.train_tag,
        "comparison": f"raw_feat_logreg_vs_{ps['best_name']}",
        "best_ps_name": ps["best_name"],
        "n_test_samples": N_test,
        "macro_avg_raw_feat": avg_rf,
        "macro_avg_best_ps": avg_ps,
        "macro_delta": float(avg_rf - avg_ps),
        "sample_level_bootstrap": {
            "n_boot": args.n_boot,
            "seed": args.seed,
            "ci_95": [boot["ci_95_lo"], boot["ci_95_hi"]],
            "delta_mean": boot["delta_mean"],
            "p_raw_feat_gt_ps": boot["p_a_gt_b"],
        },
        "generator_level_bootstrap": {
            "ci_95": [float(gen_lo), float(gen_hi)],
        },
        "xgboost_continuity": xgb_block,
        "per_generator": per_gen,
    }
    out = results_root / f"06_bootstrap_{args.pool}{tag_suffix(args.train_tag)}.json"
    save_json(result, out)
    log.info(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
