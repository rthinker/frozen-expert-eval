"""Run the 10-method prediction-space ladder; Section 4.2.

Stochastic methods (PS_3, PS_F, PS_D, PS_E) average per-sample scores
across 8 seeds before computing AUC, matching scripts 06/07. PS_A
temperatures and PS_B weights are saved as diagnostics. Per-sample test
scores go to a sibling .npz consumed by 06/07. Pass --include_raw_features
to also run FS_1 (raw-feat LogReg).
"""

import _bootstrap  # noqa: F401  # MUST be first: binds CUBLAS_WORKSPACE_CONFIG before torch.

import argparse
import re
import time
from collections import defaultdict

import numpy as np

from src.utils.config import (
    DEFAULT_TRAIN_TAG, EXPERIMENTS_CFG, SELECT_TAG,
    CACHE_ROOT, RESULTS_ROOT, get_pool_experts,
)

from src.models.prediction_space import (
    PS_1, PS_A, PS_B,
    PS_2, PS_C,
    PS_3, PS_F,
    PS_D, PS_E,
    PS_4,
)
from src.models.feature_space import FS_1
from src.utils.seed import set_seed
from src.utils.metrics import compute_auc
from src.utils.io import (
    save_json, load_stacked_logits, load_concat_features,
)
from src.utils.logging_setup import setup_logging
from src.utils.dataset_helpers import TEST_GENERATORS
from src.utils.train_subset import balanced_subset_indices
from src.utils.naming import tag_suffix as _tag_suffix


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=str, required=True)
    p.add_argument("--train_samples", type=int, default=None,
                   help="Balanced FIT subset size. Omit to use config default; "
                        "pass 0 to use all cached FIT samples.")
    p.add_argument("--include_raw_features", action="store_true",
                   help="Also run LogReg on raw concat features")
    p.add_argument("--skip_slow", action="store_true",
                   help="Skip PS_C, PS_F, PS_E, PS_4 (slow CV methods)")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=EXPERIMENTS_CFG["decision_baselines"].get(
                       "mlp_seeds", [7, 10, 24, 42, 1024, 1973, 3392, 9721]),
                   help="Seeds for MLP baselines (PS_3, PS_F, PS_D, PS_E)")
    p.add_argument("--train_tag", type=str, default=DEFAULT_TRAIN_TAG,
                   help="Cache subdirectory for training data "
                        "(default FIT; override only for experiments)")
    return p.parse_args()


def load_logits(expert_names, split_tag, cache_root):
    """Load cached logits for a split. Returns [N, E, 2], labels [N]."""
    return load_stacked_logits(expert_names, split_tag, cache_root)


def load_features(expert_names, split_tag, cache_root):
    """Load cached features for a split. Returns concat [N, sum(D_i)], labels [N]."""
    return load_concat_features(expert_names, split_tag, cache_root)


def evaluate_baseline(name, score_fn, test_sets, log, score_store=None):
    """Run a baseline scoring function on all test generators.

    Args:
        name: Baseline name (e.g., "PS_1_prob_avg").
        score_fn: Callable returning [N_test] scores for one input array.
        test_sets: dict {gen_name: (logits [N,E,2], labels [N])}.
        score_store: Optional dict. If provided, the concatenated per-sample
            test scores are stored under `score_store[name]` (used for the
            bootstrap-vs-actual-best-PS scores.npz artifact).
    """
    t0 = time.time()
    gen_names = list(test_sets.keys())
    gen_sizes = [test_sets[g][0].shape[0] for g in gen_names]
    all_logits = np.concatenate([test_sets[g][0] for g in gen_names], axis=0)
    all_scores = score_fn(all_logits)

    if score_store is not None:
        score_store[name] = np.asarray(all_scores, dtype=np.float32)

    per_gen = {}
    aucs = []
    offset = 0
    for gen_name, n in zip(gen_names, gen_sizes):
        scores = all_scores[offset:offset + n]
        labels = test_sets[gen_name][1]
        auc = compute_auc(labels, scores)
        per_gen[gen_name] = auc
        aucs.append(auc)
        offset += n

    avg8 = float(np.mean(aucs))
    elapsed = time.time() - t0
    log.info(f"  {name}: Avg(8) = {avg8:.4f} ({elapsed:.1f}s)")
    for g, a in per_gen.items():
        log.info(f"    {g}: {a:.4f}")
    return {"name": name, "avg_auc_8": avg8, "per_generator": per_gen}


def _base_method_name(full_name: str) -> str:
    """Strip a trailing _s<digits> seed suffix from a result entry name."""
    return re.sub(r"_s\d+$", "", full_name)


def aggregate_methods(all_results, score_store=None, labels=None, gen_boundaries=None):
    """Aggregate per-seed results, then compute AUC on seed-averaged predictions.

    Stochastic methods (PS_3/5/C/D) emit per-seed entries named with a
    trailing `_s<seed>`; deterministic methods produce a single entry. For
    stochastic methods, we average per-sample test scores across seeds first
    and compute per-generator AUCs on the averaged predictions, matching
    the bootstrap protocol in scripts 06/07. `std_avg8` is the std of the
    per-seed macro AUCs for diagnostics.

    `score_store`, `labels`, and `gen_boundaries` must be supplied together
    to enable seed averaging; otherwise the function falls back to the old
    mean-of-per-seed-AUCs (deterministic methods are unaffected either way).
    """
    grouped = defaultdict(list)
    for r in all_results:
        grouped[_base_method_name(r["name"])].append(r)

    can_seed_average = (
        score_store is not None
        and labels is not None
        and gen_boundaries is not None
    )

    method_summary = {}
    for base, runs in grouped.items():
        avg8s = [r["avg_auc_8"] for r in runs]
        gen_names = list(runs[0]["per_generator"].keys())
        n_runs = len(runs)

        seed_scores = [
            score_store[r["name"]] for r in runs
            if can_seed_average and r["name"] in score_store
        ] if can_seed_average else []

        if n_runs > 1 and len(seed_scores) == n_runs:
            avg_scores = np.mean(seed_scores, axis=0).astype(np.float32)
            per_gen_mean = {}
            per_gen_aucs = []
            for g in gen_names:
                lo, hi = gen_boundaries[g]
                auc = compute_auc(labels[lo:hi], avg_scores[lo:hi])
                per_gen_mean[g] = float(auc)
                per_gen_aucs.append(float(auc))
            mean_avg8 = float(np.mean(per_gen_aucs))
        else:
            mean_avg8 = float(np.mean(avg8s))
            per_gen_mean = {
                g: float(np.mean([r["per_generator"][g] for r in runs]))
                for g in gen_names
            }

        method_summary[base] = {
            "mean_avg8": mean_avg8,
            "std_avg8": float(np.std(avg8s)) if n_runs > 1 else 0.0,
            "n_runs": n_runs,
            "per_generator": per_gen_mean,
        }
    return method_summary


def _log_logreg_cv_diagnostics(log, name, diagnostics):
    """Emit a single polished message for non-fatal LogReg-CV non-convergence."""
    total = diagnostics.get("total_convergence_warning_count", 0)
    if total <= 0:
        return
    cfg = diagnostics["best_config"]
    log.warning(
        "  %s: %d fit(s) hit max_iter=%d during CV/refit; using best config "
        "(penalty=%s, C=%.3g, solver=%s%s). Results are reported, but some "
        "regularized LogReg sub-fits did not fully converge.",
        name,
        total,
        diagnostics["max_iter"],
        cfg["penalty"],
        cfg["C"],
        cfg["solver"],
        "" if cfg["l1_ratio"] is None else f", l1_ratio={cfg['l1_ratio']:.3g}",
    )


def main():
    args = parse_args()
    set_seed(1024)
    log = setup_logging("04_baselines")
    log.info(f"Pool: {args.pool}")

    data_cfg = EXPERIMENTS_CFG.get("data", {})

    expert_names = get_pool_experts(args.pool)
    cache_root = CACHE_ROOT
    results_root = RESULTS_ROOT
    tag_suffix = _tag_suffix(args.train_tag)
    results_root.mkdir(parents=True, exist_ok=True)

    log.info("Loading cached logits...")
    fit_logits, fit_labels = load_logits(
        expert_names, args.train_tag, cache_root)
    select_logits, select_labels = load_logits(
        expert_names, SELECT_TAG, cache_root)
    log.info(f"  FIT: {fit_logits.shape}, SELECT: {select_logits.shape}")

    effective_train_samples = (
        data_cfg.get("train_subset_size", 0)
        if args.train_samples is None
        else args.train_samples
    )
    keep = None
    if effective_train_samples > 0 and effective_train_samples < len(fit_labels):
        keep = balanced_subset_indices(
            fit_labels,
            effective_train_samples,
            sample_ids_path=cache_root / args.train_tag / "sample_ids.npy",
        )
        fit_logits = fit_logits[keep]
        fit_labels = fit_labels[keep]
        log.info(f"  FIT protocol subset: {fit_logits.shape} "
                 f"(requested={effective_train_samples})")
    else:
        log.info(f"  FIT protocol subset: using all {fit_logits.shape[0]} samples")

    test_sets = {}
    for gen in TEST_GENERATORS:
        gen_logits, gen_labels = load_logits(expert_names, gen, cache_root)
        test_sets[gen] = (gen_logits, gen_labels)
    log.info(f"  Loaded {len(test_sets)} test generators")

    all_results = []
    score_store: dict = {}
    diagnostics: dict = {}

    log.info(f"\n{'='*60}")
    log.info("Running prediction-space baselines...")
    log.info(f"{'='*60}")

    # PS_1: Probability Averaging (no fitting)
    r = evaluate_baseline(
        "PS_1_prob_avg", lambda tl: PS_1(tl), test_sets, log,
        score_store=score_store,
    )
    all_results.append(r)

    # PS_A: Calibrated Avg (fitted on SELECT with balanced NLL)
    _, s2_diag = PS_A(
        select_logits, select_labels, select_logits[:1],
        return_diagnostics=True,
    )
    diagnostics["PS_A_temperatures"] = s2_diag["temps"]
    log.info(f"  PS_A fitted temperatures: "
             f"{[round(t, 3) for t in s2_diag['temps']]}")

    def _s2_score(tl):
        return PS_A(select_logits, select_labels, tl)

    r = evaluate_baseline(
        "PS_A_calibrated_avg", _s2_score, test_sets, log, score_store=score_store,
    )
    all_results.append(r)

    # PS_B: Learned Weights (fitted on SELECT by AUC)
    _, s3_diag = PS_B(
        select_logits, select_labels, select_logits[:1],
        return_diagnostics=True,
    )
    diagnostics["PS_B_weights"] = s3_diag["weights"]
    log.info(f"  PS_B fitted weights: {[round(w, 3) for w in s3_diag['weights']]}")

    def _s3_score(tl):
        return PS_B(select_logits, select_labels, tl)

    r = evaluate_baseline(
        "PS_B_learned_weights", _s3_score, test_sets, log, score_store=score_store,
    )
    all_results.append(r)

    # PS_2: LogReg (fitted on FIT)
    r = evaluate_baseline(
        "PS_2_logreg",
        lambda tl: PS_2(fit_logits, fit_labels, tl),
        test_sets, log, score_store=score_store,
    )
    all_results.append(r)

    # PS_3: MLP on logits (fitted on FIT, multiple seeds)
    for seed in args.seeds:
        r = evaluate_baseline(
            f"PS_3_mlp_logits_s{seed}",
            lambda tl, s=seed: PS_3(
                fit_logits, fit_labels, tl, seed=s),
            test_sets, log, score_store=score_store,
        )
        all_results.append(r)

    # PS_D: MLP on probs (fitted on FIT, multiple seeds)
    for seed in args.seeds:
        r = evaluate_baseline(
            f"PS_D_mlp_probs_s{seed}",
            lambda tl, s=seed: PS_D(
                fit_logits, fit_labels, tl, seed=s),
            test_sets, log, score_store=score_store,
        )
        all_results.append(r)

    if not args.skip_slow:
        # PS_C: Regularised LogReg (CV inside FIT)
        ps3_diag = {}

        def _ps3_score(tl):
            scores, diag = PS_C(
                fit_logits, fit_labels, tl, return_diagnostics=True)
            ps3_diag.update(diag)
            return scores

        r = evaluate_baseline(
            "PS_C_logreg_cv",
            _ps3_score,
            test_sets, log, score_store=score_store,
        )
        all_results.append(r)
        diagnostics["PS_C_logreg_cv"] = ps3_diag
        _log_logreg_cv_diagnostics(log, "PS_C_logreg_cv", ps3_diag)

        # PS_F: MLP logits + early stopping (WD on SELECT, refit on FIT)
        for seed in args.seeds:
            r = evaluate_baseline(
                f"PS_F_mlp_logits_es_s{seed}",
                lambda tl, s=seed: PS_F(
                    fit_logits, fit_labels,
                    select_logits, select_labels, tl,
                    seed=s),
                test_sets, log, score_store=score_store,
            )
            all_results.append(r)

        # PS_E: MLP probs + early stopping
        for seed in args.seeds:
            r = evaluate_baseline(
                f"PS_E_mlp_probs_es_s{seed}",
                lambda tl, s=seed: PS_E(
                    fit_logits, fit_labels,
                    select_logits, select_labels, tl,
                    seed=s),
                test_sets, log, score_store=score_store,
            )
            all_results.append(r)

        # PS_4: XGBoost (CV inside FIT)
        r = evaluate_baseline(
            "PS_4_xgboost",
            lambda tl: PS_4(fit_logits, fit_labels, tl),
            test_sets, log, score_store=score_store,
        )
        all_results.append(r)
    else:
        log.info("  Skipping PS_C, PS_F, PS_E, PS_4 (--skip_slow)")

    # ─── Save PS-side artifacts FIRST (before optional FS_1) ───────────
    # Reordered so that an FS_1 OOM (which can happen on the full21
    # pool with ~7 GiB float64 input) does not lose the PS work below.
    # Previously the script saved scores.npz and 04_ds_{pool}.json only
    # after FS_1 succeeded; on memory-tight hosts this left no PS-side
    # output for downstream scripts (06/07).

    # Summary
    log.info(f"\n{'='*60}")
    log.info("DECISION-SPACE BASELINE RESULTS (per run)")
    log.info(f"{'='*60}")
    log.info(f"Pool: {args.pool}")
    log.info(f"{'Method':<30s} {'Avg(8)':>8s}")
    log.info("-" * 40)
    for r in all_results:
        log.info(f"  {r['name']:<28s} {r['avg_auc_8']:8.4f}")

    # Build gen_boundaries + concatenated labels first so stochastic
    # aggregation can compute AUCs on seed-averaged per-sample scores.
    gen_boundaries = {}
    offset = 0
    for gen in TEST_GENERATORS:
        n = test_sets[gen][0].shape[0]
        gen_boundaries[gen] = [int(offset), int(offset + n)]
        offset += n
    all_labels = np.concatenate([test_sets[g][1] for g in TEST_GENERATORS])

    method_summary = aggregate_methods(
        all_results,
        score_store=score_store,
        labels=all_labels,
        gen_boundaries=gen_boundaries,
    )

    log.info(f"\n{'='*60}")
    log.info("SEED-AVERAGED AGGREGATION (AUC on seed-averaged predictions)")
    log.info(f"{'='*60}")
    log.info(f"{'Method':<28s} {'n_runs':>7s} {'mean':>8s} {'std':>8s}")
    log.info("-" * 55)
    for name, m in sorted(method_summary.items(),
                          key=lambda kv: -kv[1]["mean_avg8"]):
        log.info(f"  {name:<26s} {m['n_runs']:7d} "
                 f"{m['mean_avg8']:8.4f} {m['std_avg8']:8.4f}")

    best_ds_name = max(method_summary,
                       key=lambda k: method_summary[k]["mean_avg8"])
    best_ds_mean = method_summary[best_ds_name]["mean_avg8"]
    log.info(f"\nBest DS (seed-averaged): {best_ds_name} = {best_ds_mean:.4f}")

    # Save per-method test scores (for bootstrap vs actual best PS).
    scores_path = results_root / f"04_ds_{args.pool}_scores{tag_suffix}.npz"
    np.savez_compressed(
        scores_path,
        labels=all_labels.astype(np.int32),
        gen_boundaries=np.asarray(
            [gen_boundaries[g] for g in TEST_GENERATORS], dtype=np.int32),
        generators=np.asarray(TEST_GENERATORS),
        **score_store,
    )
    log.info(f"Per-method test scores saved: {scores_path}")

    summary = {
        "pool": args.pool,
        "expert_names": expert_names,
        "train_tag": args.train_tag,
        "select_tag": SELECT_TAG,
        "fit_samples": int(len(fit_labels)),
        "select_samples": int(len(select_labels)),
        "fit_class_balance": {
            "n_real": int((fit_labels == 0).sum()),
            "n_fake": int((fit_labels == 1).sum()),
        },
        "select_class_balance": {
            "n_real": int((select_labels == 0).sum()),
            "n_fake": int((select_labels == 1).sum()),
        },
        "baselines": all_results,
        "method_means": method_summary,
        "diagnostics": diagnostics,
        "best_ds": best_ds_name,
        "best_ds_avg8": best_ds_mean,
        "scores_npz": scores_path.name,
    }
    out_path = results_root / f"04_ds_{args.pool}{tag_suffix}.json"
    save_json(summary, out_path)
    log.info(f"Results saved: {out_path}")

    # ─── Optional FS_1 raw-feat LogReg (gated on memory budget) ────────
    if args.include_raw_features:
        from src.models.feature_space import (
            fs1_over_budget, fs1_skip_marker, fs1_memory_estimate_gib,
        )

        raw_path = results_root / f"04_ds_{args.pool}_rawfeat{tag_suffix}.json"
        # Cheap probe: derive (N, D) from the FIT-side feature shape
        # without loading the full test-side concat, so we can decide
        # before any heavy I/O.
        log.info("\nRunning raw-features LogReg...")
        fit_feat, _ = load_features(expert_names, args.train_tag, cache_root)
        if keep is not None:
            fit_feat = fit_feat[keep]
        n_samples, n_features = fit_feat.shape

        if fs1_over_budget(n_samples, n_features):
            need = fs1_memory_estimate_gib(n_samples, n_features)
            log.info(
                f"  FS_1 input matrix would need ~{need:.2f} GiB float64; "
                f"exceeds default 6.0 GiB budget. Writing skip marker."
            )
            log.info(
                "  Set ALLOW_FS1_OOM=1 to bypass and run anyway "
                "(host will OOM if it does not have the memory)."
            )
            marker = fs1_skip_marker(
                n_samples, n_features,
                extra={"pool": args.pool, "train_tag": args.train_tag},
            )
            save_json(marker, raw_path)
            log.info(f"  Saved skip marker: {raw_path}")
            return

        test_feat_sets = {}
        for gen in TEST_GENERATORS:
            gf, gl = load_features(expert_names, gen, cache_root)
            test_feat_sets[gen] = (gf, gl)

        gen_names = list(test_feat_sets.keys())
        gen_sizes = [test_feat_sets[g][0].shape[0] for g in gen_names]
        all_feat = np.concatenate([test_feat_sets[g][0] for g in gen_names])
        all_scores = FS_1(fit_feat, fit_labels, all_feat, logger=log)

        per_gen = {}
        aucs = []
        t0 = time.time()
        offset = 0
        for gen_name, n in zip(gen_names, gen_sizes):
            scores = all_scores[offset:offset + n]
            labels = test_feat_sets[gen_name][1]
            auc = compute_auc(labels, scores)
            per_gen[gen_name] = auc
            aucs.append(auc)
            offset += n

        avg8 = float(np.mean(aucs))
        log.info(f"  FS_1: Avg(8) = {avg8:.4f} "
                 f"({time.time()-t0:.1f}s)")
        for g, a in per_gen.items():
            log.info(f"    {g}: {a:.4f}")
        raw_feat_result = {
            "name": "fs_1_raw_features_logreg",
            "pool": args.pool,
            "train_tag": args.train_tag,
            "avg_auc_8": avg8,
            "per_generator": per_gen,
        }
        save_json(raw_feat_result, raw_path)
        log.info(f"  Saved raw-features ablation: {raw_path}")


if __name__ == "__main__":
    main()
