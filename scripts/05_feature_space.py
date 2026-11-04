"""Train FeatureAssembly (FS_2) or variants on cached features; reports Avg(8).

`--combiner_type {mlp,linear,pca,random}` selects the FS variant
(Section 4.3 capacity ablation).
"""

import _bootstrap  # noqa: F401  # MUST be first: binds CUBLAS_WORKSPACE_CONFIG before torch.

import argparse
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from src.utils.config import (
    DEFAULT_TRAIN_TAG, EXPERIMENTS_CFG, CACHE_ROOT, SELECT_TAG,
    CKPT_ROOT, RESULTS_ROOT, get_pool_experts, get_feature_dims,
)

from src.models.feature_space import FeatureAssembly, RandomProjectionAssembly
from src.utils.metrics import compute_auc
from src.utils.io import save_json, load_pool_data
from src.utils.logging_setup import setup_logging
from src.utils.dataset_helpers import TEST_GENERATORS
from src.utils.train_subset import balanced_subset_indices
from src.utils.naming import checkpoint_name as make_ckpt_name, tag_suffix as make_tag_suffix
from src.training.fs_runtime import predict_fs, train_fs_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=str, required=True,
                   help="Named pool from experts.yaml")
    p.add_argument("--proj_dim", type=int, default=64)
    p.add_argument("--combiner_type", type=str, default="mlp",
                   choices=["mlp", "linear", "pca", "random"],
                   help="Combiner variant")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=EXPERIMENTS_CFG["feature_assembly"].get(
                       "seeds", [7, 10, 24, 42, 1024, 1973, 3392, 9721]),
                   help="Seeds for multi-seed training (default from experiments.yaml)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs from experiments.yaml")
    p.add_argument("--train_samples", type=int, default=None,
                   help="Balanced FIT subset size. Omit to use config default; "
                        "pass 0 to use all cached FIT samples.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--train_tag", type=str, default=DEFAULT_TRAIN_TAG,
                   help="Cache subdirectory for training data "
                        "(default FIT; override only for experiments)")
    return p.parse_args()


def train_one_seed(
    train_feats, train_labels,
    val_feats, val_labels,
    feature_dims, args, seed, device, log, exp_cfg,
):
    """Train a single combiner for one seed.

    Delegates to `src.training.fs_runtime.train_fs_model` for learned-
    projection variants (mlp, linear). The random-projection variant
    instantiates RandomProjectionAssembly directly and shares the same
    training loop (class-weighted CE, uniform permutation, best-epoch
    selection on SELECT).
    """
    if args.combiner_type == "random":
        from src.utils.seed import set_seed
        import torch.nn as nn
        from src.training.fs_runtime import _to_tensors, _class_weights

        set_seed(seed)
        model = RandomProjectionAssembly(
            feature_dims, proj_dim=args.proj_dim, seed=seed,
        ).to(device)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"  Seed {seed}: {n_trainable:,} trainable params (random proj)")

        train_tensors = _to_tensors(train_feats)
        val_tensors = _to_tensors(val_feats)
        train_labels_t = torch.as_tensor(train_labels, dtype=torch.long)
        n_samples = len(train_labels_t)
        optimiser = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=exp_cfg["lr"], weight_decay=exp_cfg["weight_decay"],
        )
        cw = _class_weights(train_labels)
        if cw is not None:
            cw = cw.to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)

        best_val_auc, best_state = -1.0, None
        batch_size = exp_cfg["batch_size"]
        for epoch in range(exp_cfg["epochs"]):
            model.train()
            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, batch_size):
                idx = indices[start:start + batch_size]
                bf = [t[idx].to(device) for t in train_tensors]
                bl = train_labels_t[idx].to(device)
                optimiser.zero_grad()
                loss = criterion(model(bf), bl)
                loss.backward()
                optimiser.step()
            model.eval()
            val_scores = predict_fs(model, val_tensors, device)
            val_auc = compute_auc(val_labels, val_scores)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        log.info(f"  Seed {seed}: best val AUC = {best_val_auc:.4f}")
        return model, best_val_auc

    # Learned projection: delegate to the canonical fs_runtime trainer.
    model, best_val_auc = train_fs_model(
        train_feats, train_labels, feature_dims, args.proj_dim,
        val_feats=val_feats, val_labels=val_labels,
        seed=seed, device=device,
        classifier_type=args.combiner_type,
        epochs=exp_cfg["epochs"], batch_size=exp_cfg["batch_size"],
        lr=exp_cfg["lr"], weight_decay=exp_cfg["weight_decay"],
        expert_dropout=exp_cfg["expert_dropout"],
        hidden_dims=exp_cfg["hidden_dims"], dropout=exp_cfg["dropout"],
        log=log,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Seed {seed}: {n_trainable:,} trainable params, "
             f"best val AUC = {best_val_auc:.4f}")
    return model, best_val_auc


def run_pca_combiner(train_feats, train_labels, val_feats, val_labels,
                     test_sets, proj_dim, ds_cfg, log,
                     *, pool: str, train_tag: str, select_tag: str):
    """PCA + linear classifier for the width-control analysis. No learned projection.

    The `pool`, `train_tag`, and `select_tag` kwargs are required so the
    output JSON carries the same metadata the manifest scanner expects
    (otherwise tables like `table3_capacity_ablation` silently drop the
    PCA row because its pool defaults to "?").
    """
    X_train = np.concatenate(train_feats, axis=1)
    X_val = np.concatenate(val_feats, axis=1)

    log.info(f"  PCA: input dim={X_train.shape[1]}, target dim={proj_dim}")
    pca_seed = ds_cfg.get(
        "raw_feature_logreg_random_state",
        ds_cfg.get("logreg_random_state", 1024),
    )
    pca = PCA(n_components=proj_dim, random_state=pca_seed)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    lr = LogisticRegression(
        solver=ds_cfg.get(
            "raw_feature_logreg_solver",
            ds_cfg.get("logreg_solver", "lbfgs"),
        ),
        C=ds_cfg.get(
            "raw_feature_logreg_C",
            ds_cfg.get("logreg_C", 1.0),
        ),
        class_weight=ds_cfg.get(
            "raw_feature_logreg_class_weight",
            ds_cfg.get("logreg_class_weight", "balanced"),
        ),
        max_iter=ds_cfg.get("raw_feature_logreg_max_iter", 2000),
        random_state=pca_seed,
    )
    lr.fit(X_train_pca, train_labels)

    val_scores = lr.predict_proba(X_val_pca)[:, 1]
    val_auc = compute_auc(val_labels, val_scores)
    log.info(f"  PCA val AUC: {val_auc:.4f}")

    # Note: key is `combiner_type` (not `combiner`) so the result JSON
    # matches the manifest schema in src/reporting/manifest.py. `pool`,
    # `train_tag`, and `select_tag` are required for manifest classification
    # and for the manifest's train_tag filter to work on non-default tags.
    results = {
        "combiner_type": "pca",
        "pool": pool,
        "train_tag": train_tag,
        "select_tag": select_tag,
        "proj_dim": proj_dim,
        "per_generator": {},
    }
    aucs = []
    for gen_name, (gen_feats, gen_labels) in test_sets.items():
        X_test = np.concatenate(gen_feats, axis=1)
        X_test_pca = pca.transform(X_test)
        scores = lr.predict_proba(X_test_pca)[:, 1]
        auc = compute_auc(gen_labels, scores)
        results["per_generator"][gen_name] = auc
        aucs.append(auc)

    results["avg_auc_8"] = float(np.mean(aucs))
    results["val_auc"] = val_auc
    return results


def main():
    args = parse_args()
    log = setup_logging("05_fs")
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Pool: {args.pool}, combiner: {args.combiner_type}, d={args.proj_dim}")

    exp_cfg = EXPERIMENTS_CFG["feature_assembly"].copy()
    data_cfg = EXPERIMENTS_CFG.get("data", {})
    ds_cfg = EXPERIMENTS_CFG["decision_baselines"]

    if args.epochs is not None:
        exp_cfg["epochs"] = args.epochs

    expert_names = get_pool_experts(args.pool)
    feature_dims = get_feature_dims(expert_names)
    log.info(f"Experts ({len(expert_names)}): {expert_names}")
    log.info(f"Feature dims: {feature_dims}")

    cache_root = CACHE_ROOT
    ckpt_root = CKPT_ROOT
    results_root = RESULTS_ROOT
    tag_suffix = make_tag_suffix(args.train_tag)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    log.info("Loading cached features...")
    train_feats, _, train_labels = load_pool_data(
        expert_names, args.train_tag, cache_root)
    val_feats, _, val_labels = load_pool_data(
        expert_names, SELECT_TAG, cache_root)
    log.info(f"  FIT: {len(train_labels)} samples, "
             f"SELECT: {len(val_labels)} samples")

    effective_train_samples = (
        data_cfg.get("train_subset_size", 0)
        if args.train_samples is None
        else args.train_samples
    )

    if effective_train_samples > 0 and effective_train_samples < len(train_labels):
        keep = balanced_subset_indices(
            train_labels,
            effective_train_samples,
            sample_ids_path=cache_root / args.train_tag / "sample_ids.npy",
        )
        train_feats = [f[keep] for f in train_feats]
        train_labels = train_labels[keep]
        log.info(f"  FIT protocol subset: {len(train_labels)} "
                 f"(r={int((train_labels==0).sum())}, "
                 f"f={int((train_labels==1).sum())}, "
                 f"requested={effective_train_samples})")
    else:
        log.info(f"  FIT protocol subset: using all {len(train_labels)} samples")

    test_sets = {}
    for gen in TEST_GENERATORS:
        gen_feats, _, gen_labels = load_pool_data(expert_names, gen, cache_root)
        test_sets[gen] = (gen_feats, gen_labels)
    log.info(f"  Loaded {len(test_sets)} test generators")

    # PCA combiner (special path: no neural network)
    if args.combiner_type == "pca":
        log.info("\nRunning PCA combiner...")
        results = run_pca_combiner(
            train_feats, train_labels, val_feats, val_labels,
            test_sets, args.proj_dim, ds_cfg, log,
            pool=args.pool,
            train_tag=args.train_tag,
            select_tag=SELECT_TAG,
        )
        log.info(f"  Avg(8) AUC: {results['avg_auc_8']:.4f}")
        for gen, auc in results["per_generator"].items():
            log.info(f"    {gen}: {auc:.4f}")

        out_path = results_root / f"05_fs_{args.pool}_pca_d{args.proj_dim}{tag_suffix}.json"
        save_json(results, out_path)
        log.info(f"  Saved: {out_path}")
        return

    # Trainable combiners (mlp, linear, random)
    all_seed_results = []
    per_seed_scores = {g: [] for g in TEST_GENERATORS}
    for seed in args.seeds:
        log.info(f"\n--- Seed {seed} ---")
        t0 = time.time()

        model, best_val = train_one_seed(
            train_feats, train_labels, val_feats, val_labels,
            feature_dims, args, seed, device, log, exp_cfg,
        )

        seed_results = {"seed": seed, "val_auc": float(best_val), "per_generator": {}}
        aucs = []
        for gen_name, (gen_feats, gen_labels) in test_sets.items():
            scores = predict_fs(model, gen_feats, device)
            per_seed_scores[gen_name].append(scores.astype(np.float32))
            auc = compute_auc(gen_labels, scores)
            seed_results["per_generator"][gen_name] = auc
            aucs.append(auc)

        seed_results["avg_auc_8"] = float(np.mean(aucs))
        elapsed = time.time() - t0

        log.info(f"  Seed {seed}: Avg(8) = {seed_results['avg_auc_8']:.4f} "
                 f"({elapsed:.1f}s)")
        for gen, auc in seed_results["per_generator"].items():
            log.info(f"    {gen}: {auc:.4f}")

        ckpt_path = ckpt_root / make_ckpt_name(
            args.pool, args.combiner_type, args.proj_dim,
            seed, args.train_tag)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "pool": args.pool, "combiner_type": args.combiner_type,
                "proj_dim": args.proj_dim, "seed": seed,
                "train_tag": args.train_tag,
                "feature_dims": feature_dims, "expert_names": expert_names,
            },
            "val_auc": float(best_val),
            "avg_auc_8": float(seed_results["avg_auc_8"]),
        }, str(ckpt_path))

        all_seed_results.append(seed_results)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary across seeds. Compute AUC on seed-averaged per-sample scores
    # (matches the bootstrap protocol in script 07); keep `std_avg8` as the
    # across-seeds std of per-seed macro AUCs for diagnostics.
    avg8_values = [r["avg_auc_8"] for r in all_seed_results]
    std_avg8 = float(np.std(avg8_values))

    per_gen_seed_avg = {}
    per_gen_aucs = []
    for gen in TEST_GENERATORS:
        avg_scores = np.mean(per_seed_scores[gen], axis=0)
        gen_labels = test_sets[gen][1]
        per_gen_seed_avg[gen] = float(compute_auc(gen_labels, avg_scores))
        per_gen_aucs.append(per_gen_seed_avg[gen])
    mean_avg8 = float(np.mean(per_gen_aucs))

    log.info(f"\n{'='*60}")
    log.info("FEATURE-SPACE ASSEMBLY RESULTS (AUC on seed-averaged scores)")
    log.info(f"{'='*60}")
    log.info(f"Pool: {args.pool}, combiner: {args.combiner_type}, d={args.proj_dim}")
    log.info(f"Avg(8): {mean_avg8:.4f} (std across per-seed AUCs: {std_avg8:.4f}, "
             f"seeds: {args.seeds})")

    log.info("Per-generator (seed-averaged scores):")
    for gen in TEST_GENERATORS:
        gen_aucs = [r["per_generator"][gen] for r in all_seed_results]
        log.info(f"  {gen}: {per_gen_seed_avg[gen]:.4f} "
                 f"(per-seed mean {np.mean(gen_aucs):.4f} +/- {np.std(gen_aucs):.4f})")

    # Trainable parameter count. For random-proj, use RandomProjectionAssembly
    # so frozen linear weights are correctly excluded from the count.
    try:
        if args.combiner_type == "random":
            _tmp = RandomProjectionAssembly(
                feature_dims, proj_dim=args.proj_dim, seed=args.seeds[0],
            )
        else:
            _tmp = FeatureAssembly(
                feature_dims, proj_dim=args.proj_dim,
                classifier_type=args.combiner_type,
                expert_dropout=exp_cfg["expert_dropout"],
                hidden_dims=exp_cfg["hidden_dims"],
                dropout=exp_cfg["dropout"],
            )
        n_params = int(sum(p.numel() for p in _tmp.parameters() if p.requires_grad))
        del _tmp
    except Exception:
        n_params = None

    summary = {
        "pool": args.pool,
        "combiner_type": args.combiner_type,
        "proj_dim": args.proj_dim,
        "train_tag": args.train_tag,
        "select_tag": SELECT_TAG,
        "fit_samples": int(len(train_labels)),
        "seeds": args.seeds,
        "mean_avg8": mean_avg8,
        "std_avg8": std_avg8,
        "n_trainable_params": n_params,
        "feature_dims": feature_dims,
        "per_generator_seed_averaged": per_gen_seed_avg,
        "per_seed": all_seed_results,
    }
    out_path = results_root / (
        f"05_fs_{args.pool}_{args.combiner_type}_d{args.proj_dim}{tag_suffix}.json"
    )
    save_json(summary, out_path)
    log.info(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
