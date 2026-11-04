"""Paired generator-level bootstrap CI for FS_2 (seed-averaged) vs best PS.

Requires scripts 04 (scores.npz) and 05 (FS checkpoints) to have run.
Also emits a secondary XGBoost-specific delta for continuity.
"""

import _bootstrap  # noqa: F401  # MUST be first: binds CUBLAS_WORKSPACE_CONFIG before torch.

import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.utils.config import (
    DEFAULT_TRAIN_TAG, FS_HIDDEN_DIMS, RESULTS_ROOT,
    CACHE_ROOT, CKPT_ROOT, FS_SEEDS, get_pool_experts, get_feature_dims,
)

from src.models.feature_space import FeatureAssembly
from src.training.fs_runtime import predict_fs
from src.utils.io import save_json, load_all_generators, load_json
from src.utils.metrics import bootstrap_paired_auc
from src.utils.logging_setup import setup_logging
from src.utils.dataset_helpers import TEST_GENERATORS
from src.utils.naming import tag_suffix, checkpoint_name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=str, required=True)
    p.add_argument("--proj_dim", type=int, required=True)
    p.add_argument("--train_tag", type=str, default=DEFAULT_TRAIN_TAG)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def _load_best_ps_scores(pool, results_root, train_tag, log):
    ts = tag_suffix(train_tag)
    ds_json = results_root / f"04_ds_{pool}{ts}.json"
    scores_npz = results_root / f"04_ds_{pool}_scores{ts}.npz"
    if not ds_json.exists() or not scores_npz.exists():
        raise FileNotFoundError(
            "Run 04_prediction_space.py first to produce the summary JSON "
            "and scores.npz artifacts."
        )
    summary = load_json(ds_json)
    best_ds_name = summary["best_ds"]
    npz = np.load(scores_npz, allow_pickle=False)

    seed_keys = [k for k in npz.keys()
                 if k.startswith(best_ds_name + "_s")]
    if seed_keys:
        log.info(f"  averaging {len(seed_keys)} seeds of {best_ds_name}")
        best_scores = np.mean(
            [npz[k].astype(np.float32) for k in seed_keys], axis=0)
    else:
        best_scores = npz[best_ds_name].astype(np.float32)

    return {
        "best_name": best_ds_name,
        "best_scores": best_scores,
        "xgb_scores": (
            npz["PS_4_xgboost"].astype(np.float32)
            if "PS_4_xgboost" in npz.files else None
        ),
        "labels": npz["labels"].astype(np.int32),
    }


def main():
    args = parse_args()
    log = setup_logging("07_boot_mlp")
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    expert_names = get_pool_experts(args.pool)
    feature_dims = get_feature_dims(expert_names)

    log.info(f"Pool: {args.pool}, d={args.proj_dim}")

    _, X_test_feat, _, te_labels, gen_boundaries = \
        load_all_generators(expert_names, CACHE_ROOT, TEST_GENERATORS)
    N_test = len(te_labels)
    feat_list_test = []
    col = 0
    for dim in feature_dims:
        feat_list_test.append(X_test_feat[:, col:col + dim])
        col += dim

    ts = tag_suffix(args.train_tag)
    pattern = f"{args.pool}_mlp_d{args.proj_dim}_seed*_best{ts}.pt"
    checkpoint_seeds = {CKPT_ROOT / checkpoint_name(args.pool, "mlp", args.proj_dim, seed, args.train_tag): seed
                        for seed in FS_SEEDS}
    # Keep the reviewed filename order, including float32 averaging order.
    ckpt_files = sorted(checkpoint_seeds)
    present = set(CKPT_ROOT.glob(pattern))
    expected = set(ckpt_files)
    if present != expected:
        raise FileNotFoundError(f"Incomplete checkpoint set: missing={sorted(p.name for p in expected-present)}, "
                                f"unexpected={sorted(p.name for p in present-expected)}. "
                                "Run the matching 05_feature_space.py step with the configured seeds.")
    log.info(f"Found all {len(ckpt_files)} configured checkpoints")

    all_seed_scores = []
    for ckpt_path in ckpt_files:
        model = FeatureAssembly(
            feature_dims, proj_dim=args.proj_dim,
            classifier_type="mlp",
            expert_dropout=0.0,
            hidden_dims=FS_HIDDEN_DIMS, dropout=0.0,
        ).to(device)
        # Generated checkpoints use tensors and primitive metadata only.
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        expected_seed = checkpoint_seeds[ckpt_path]
        config = state.get("config", {}) if isinstance(state, dict) else {}
        for key, expected_value in {"pool": args.pool, "combiner_type": "mlp", "proj_dim": args.proj_dim,
                                    "seed": expected_seed, "train_tag": args.train_tag}.items():
            if config.get(key) != expected_value:
                raise ValueError(f"Checkpoint {ckpt_path.name}: {key} does not match the requested run")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        scores = predict_fs(model, feat_list_test, device)
        all_seed_scores.append(scores)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scores_mlp = np.mean(all_seed_scores, axis=0)
    log.info(f"Averaged {len(all_seed_scores)} seeds")

    ps = _load_best_ps_scores(args.pool, RESULTS_ROOT, args.train_tag, log)
    if not np.array_equal(ps["labels"], te_labels):
        raise RuntimeError("02 labels disagree with freshly loaded labels.")

    log.info("\nPer-generator AUC:")
    per_gen = {}
    for g in TEST_GENERATORS:
        s, e = gen_boundaries[g]
        a_m = roc_auc_score(te_labels[s:e], scores_mlp[s:e])
        a_p = roc_auc_score(te_labels[s:e], ps["best_scores"][s:e])
        per_gen[g] = {"fs_mlp": float(a_m), "best_ps": float(a_p)}
        log.info(f"  {g:15s}: MLP={a_m:.3f}  "
                 f"{ps['best_name']}={a_p:.3f}")

    mlp_per_gen = np.array([per_gen[g]["fs_mlp"] for g in TEST_GENERATORS])
    ps_per_gen = np.array([per_gen[g]["best_ps"] for g in TEST_GENERATORS])
    avg_mlp = float(mlp_per_gen.mean())
    avg_ps = float(ps_per_gen.mean())
    log.info(f"\nMacro-avg MLP:     {avg_mlp:.4f}")
    log.info(f"Macro-avg best PS: {avg_ps:.4f} ({ps['best_name']})")
    log.info(f"Macro delta:       {avg_mlp - avg_ps:+.4f}")

    log.info(f"\nSample-level bootstrap ({args.n_boot} resamples)...")
    boot = bootstrap_paired_auc(
        scores_mlp, ps["best_scores"], te_labels,
        n_boot=args.n_boot, seed=args.seed)
    log.info(f"  95% CI: [{boot['ci_95_lo']:+.3f}, "
             f"{boot['ci_95_hi']:+.3f}]")

    rng = np.random.default_rng(args.seed)
    gen_deltas = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, 8, size=8)
        gen_deltas.append(mlp_per_gen[idx].mean() - ps_per_gen[idx].mean())
    gen_deltas = np.array(gen_deltas)
    gen_lo, gen_hi = np.percentile(gen_deltas, [2.5, 97.5])
    log.info(f"  generator-level CI: [{gen_lo:+.3f}, {gen_hi:+.3f}]")

    xgb_block = None
    if ps["xgb_scores"] is not None:
        xgb_scores = ps["xgb_scores"]
        per_gen_xgb = [
            roc_auc_score(te_labels[s:e], xgb_scores[s:e])
            for s, e in [gen_boundaries[g] for g in TEST_GENERATORS]
        ]
        avg_xgb = float(np.mean(per_gen_xgb))
        boot_xgb = bootstrap_paired_auc(
            scores_mlp, xgb_scores, te_labels,
            n_boot=args.n_boot, seed=args.seed)
        xgb_block = {
            "macro_avg_xgboost": avg_xgb,
            "macro_delta": float(avg_mlp - avg_xgb),
            "sample_level_ci_95": [boot_xgb["ci_95_lo"], boot_xgb["ci_95_hi"]],
        }

    result = {
        "pool": args.pool,
        "proj_dim": args.proj_dim,
        "train_tag": args.train_tag,
        "comparison": f"fs_mlp_avg_seeds_vs_{ps['best_name']}",
        "best_ps_name": ps["best_name"],
        "n_seeds": len(all_seed_scores),
        "n_test_samples": N_test,
        "macro_avg_mlp": avg_mlp,
        "macro_avg_best_ps": avg_ps,
        "macro_delta": float(avg_mlp - avg_ps),
        "sample_level_bootstrap": {
            "n_boot": args.n_boot,
            "seed": args.seed,
            "ci_95": [boot["ci_95_lo"], boot["ci_95_hi"]],
            "p_mlp_gt_ps": boot["p_a_gt_b"],
        },
        "generator_level_bootstrap": {
            "ci_95": [float(gen_lo), float(gen_hi)],
        },
        "xgboost_continuity": xgb_block,
        "per_generator": per_gen,
    }
    out = RESULTS_ROOT / f"07_bootstrap_mlp_{args.pool}{tag_suffix(args.train_tag)}.json"
    save_json(result, out)
    log.info(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
