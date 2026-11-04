"""Leave-one-generator-out (LOGO) cross-validation; Appendix F.

Train combiners on 7 of the 8 OOD test generators, evaluate on the
8th. Each fold carves a stratified 85/15 FIT/SELECT inside the 7
training generators so PS_A/PS_B/PS_F/PS_E have a held-out SELECT.
Runs the full PS ladder (10 methods) plus FS_1 and FS MLP. Seeds and
score-aggregation match the main bootstrap protocol (06/07): per-sample
average across seeds, then one AUC per method per fold. Saves per-fold
sign-test, Wilcoxon, and bootstrap CI for FS_1 and FS_2 vs best PS.
"""

import _bootstrap  # noqa: F401  # MUST be first: binds CUBLAS_WORKSPACE_CONFIG before torch.

import argparse
import re
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from scipy.stats import binomtest, wilcoxon

from src.models.prediction_space import (
    PS_1, PS_A, PS_B,
    PS_2, PS_C,
    PS_3, PS_F,
    PS_D, PS_E,
    PS_4,
)
from src.models.feature_space import FS_1
from src.utils.metrics import compute_auc
from src.utils.io import save_json, load_pool_data
from src.training.fs_runtime import train_fs_model, predict_fs
from src.utils.logging_setup import setup_logging
from src.utils.dataset_helpers import TEST_GENERATORS
from src.utils.config import (
    FS_SEEDS, CACHE_ROOT, RESULTS_ROOT,
    get_pool_experts, get_feature_dims,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=str, required=True)
    p.add_argument("--proj_dim", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=FS_SEEDS)
    p.add_argument("--select_frac", type=float, default=0.15,
                   help="Stratified FIT/SELECT split inside each fold.")
    p.add_argument("--skip_slow", action="store_true",
                   help="Skip PS_C, PS_F, PS_E, PS_4 for fast iteration.")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def merge_generators(gen_data_list):
    """Merge list of (feats, logits, labels) into single arrays."""
    n_experts = len(gen_data_list[0][0])
    all_feats = [[] for _ in range(n_experts)]
    all_logits = [[] for _ in range(n_experts)]
    all_labels = []
    for feats, logits, labels in gen_data_list:
        for i in range(n_experts):
            all_feats[i].append(feats[i])
            all_logits[i].append(logits[i])
        all_labels.append(labels)
    merged_feats = [np.concatenate(f) for f in all_feats]
    merged_logits = [np.concatenate(lg) for lg in all_logits]
    merged_labels = np.concatenate(all_labels)
    return merged_feats, merged_logits, merged_labels


def _stack_logits(logits_list):
    return np.stack(logits_list, axis=1)


def _split_fit_select(feats, logits3d, labels, select_frac, seed):
    """Stratified split of a merged pool into (fit, select) views."""
    idx = np.arange(len(labels))
    fit_idx, sel_idx = train_test_split(
        idx, test_size=select_frac, random_state=seed, stratify=labels,
    )
    fit_feats = [f[fit_idx] for f in feats]
    sel_feats = [f[sel_idx] for f in feats]
    return (
        fit_feats, logits3d[fit_idx], labels[fit_idx],
        sel_feats, logits3d[sel_idx], labels[sel_idx],
    )


def _run_ps_fold(
    fit_logits3d: np.ndarray,
    fit_labels: np.ndarray,
    sel_logits3d: np.ndarray,
    sel_labels: np.ndarray,
    te_logits3d: np.ndarray,
    te_labels: np.ndarray,
    seeds: List[int],
    skip_slow: bool,
    log,
) -> Dict[str, float]:
    """Run the full PS ladder for one fold. Returns {base_method: AUC}
    where the AUC for stochastic methods is computed on seed-averaged
    per-sample scores.
    """
    per_run_scores: Dict[str, np.ndarray] = {}

    def _log_logreg_cv_diagnostics(name, diagnostics):
        total = diagnostics.get("total_convergence_warning_count", 0)
        if total <= 0:
            return
        cfg = diagnostics["best_config"]
        log.warning(
            "    %s: %d fit(s) hit max_iter=%d during CV/refit; using best "
            "config (penalty=%s, C=%.3g, solver=%s%s).",
            name,
            total,
            diagnostics["max_iter"],
            cfg["penalty"],
            cfg["C"],
            cfg["solver"],
            "" if cfg["l1_ratio"] is None else f", l1_ratio={cfg['l1_ratio']:.3g}",
        )

    def _eval(name, fn):
        scores = np.asarray(fn(), dtype=np.float32)
        per_run_scores[name] = scores
        auc = compute_auc(te_labels, scores)
        log.info(f"    {name:<28s} {auc:.4f}")

    _eval("PS_1_prob_avg", lambda: PS_1(te_logits3d))
    _eval("PS_A_calibrated_avg",
          lambda: PS_A(sel_logits3d, sel_labels, te_logits3d))
    _eval("PS_B_learned_weights",
          lambda: PS_B(sel_logits3d, sel_labels, te_logits3d))
    _eval("PS_2_logreg",
          lambda: PS_2(fit_logits3d, fit_labels, te_logits3d))

    for s in seeds:
        _eval(f"PS_3_mlp_logits_s{s}",
              lambda seed=s: PS_3(
                  fit_logits3d, fit_labels, te_logits3d, seed=seed))
    for s in seeds:
        _eval(f"PS_D_mlp_probs_s{s}",
              lambda seed=s: PS_D(
                  fit_logits3d, fit_labels, te_logits3d, seed=seed))

    if not skip_slow:
        def _ps3_score():
            scores, diag = PS_C(
                fit_logits3d, fit_labels, te_logits3d, return_diagnostics=True)
            _log_logreg_cv_diagnostics("PS_C_logreg_cv", diag)
            return scores

        _eval("PS_C_logreg_cv", _ps3_score)
        for s in seeds:
            _eval(f"PS_F_mlp_logits_es_s{s}",
                  lambda seed=s: PS_F(
                      fit_logits3d, fit_labels,
                      sel_logits3d, sel_labels, te_logits3d,
                      seed=seed))
        for s in seeds:
            _eval(f"PS_E_mlp_probs_es_s{s}",
                  lambda seed=s: PS_E(
                      fit_logits3d, fit_labels,
                      sel_logits3d, sel_labels, te_logits3d,
                      seed=seed))
        _eval("PS_4_xgboost",
              lambda: PS_4(fit_logits3d, fit_labels, te_logits3d))

    # Aggregate: strip _s<seed>, average per-sample scores across seeds,
    # then compute one AUC per base method on the averaged scores.
    grouped = defaultdict(list)
    for name, scores in per_run_scores.items():
        base = re.sub(r"_s\d+$", "", name)
        grouped[base].append(scores)
    aggregated: Dict[str, float] = {}
    for base, score_list in grouped.items():
        avg_scores = np.mean(score_list, axis=0)
        aggregated[base] = float(compute_auc(te_labels, avg_scores))
    return aggregated


def sign_test_p(deltas):
    """Two-sided paired sign test, excluding exact ties; all ties give p=1."""
    d = np.asarray(deltas, dtype=float)
    n_nonzero = int(np.count_nonzero(d))
    return (float(binomtest(int((d > 0).sum()), n_nonzero, 0.5, alternative="two-sided").pvalue)
            if n_nonzero else 1.0)


def main():
    args = parse_args()
    log = setup_logging("11_disjoint")
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    expert_names = get_pool_experts(args.pool)
    feature_dims = get_feature_dims(expert_names)
    cache_root = CACHE_ROOT
    results_root = RESULTS_ROOT

    log.info(f"Pool: {args.pool}, d={args.proj_dim}")
    log.info(f"Experts: {expert_names}")
    log.info(f"Device: {device}")
    log.info("Protocol: leave-one-generator-out overlap-control with full PS ladder")
    log.info(f"Seeds: {args.seeds}")

    log.info("Loading all 8 generators...")
    gen_data = {}
    for g in TEST_GENERATORS:
        gen_data[g] = load_pool_data(expert_names, g, cache_root)
        n = len(gen_data[g][2])
        log.info(f"  {g}: {n} samples")

    fold_results = []
    for held_out in TEST_GENERATORS:
        log.info(f"\n=== Fold: held-out={held_out} ===")
        t0 = time.time()

        train_gens = [g for g in TEST_GENERATORS if g != held_out]
        train_data = [gen_data[g] for g in train_gens]
        tr_feats, tr_logits, tr_labels = merge_generators(train_data)
        te_feats, te_logits, te_labels = gen_data[held_out]
        tr_logits3d = _stack_logits(tr_logits)
        te_logits3d = _stack_logits(te_logits)

        (fit_feats, fit_logits3d, fit_labels,
         sel_feats, sel_logits3d, sel_labels) = _split_fit_select(
            tr_feats, tr_logits3d, tr_labels,
            args.select_frac, seed=1024,
        )

        log.info(f"  FIT: {len(fit_labels)} "
                 f"(real={int((fit_labels==0).sum())}, "
                 f"fake={int((fit_labels==1).sum())}) "
                 f"| SELECT: {len(sel_labels)} "
                 f"(real={int((sel_labels==0).sum())}, "
                 f"fake={int((sel_labels==1).sum())})")
        log.info(f"  TEST: {len(te_labels)} ({held_out})")

        log.info("  PS ladder:")
        ps_means = _run_ps_fold(
            fit_logits3d, fit_labels,
            sel_logits3d, sel_labels,
            te_logits3d, te_labels,
            args.seeds, args.skip_slow, log,
        )
        best_ps_name = max(ps_means, key=ps_means.get)
        best_ps_auc = ps_means[best_ps_name]
        log.info(f"  Best PS: {best_ps_name} = {best_ps_auc:.4f}")

        # Raw-feat LogReg (deterministic)
        rf_scores = FS_1(
            np.concatenate(fit_feats, axis=1),
            fit_labels,
            np.concatenate(te_feats, axis=1),
            logger=log,
        )
        rf_auc = compute_auc(te_labels, rf_scores)
        log.info(f"  Raw-feat LogReg: {rf_auc:.4f}")

        # FS MLP: multi-seed. AUC computed on seed-averaged per-sample
        # scores (matches bootstrap protocol in script 07); fs_std retains
        # the across-seeds std of per-seed AUCs for diagnostics.
        log.info("  FS MLP:")
        fs_aucs = []
        fs_seed_scores = []
        for seed in args.seeds:
            model, _ = train_fs_model(
                fit_feats, fit_labels, feature_dims, args.proj_dim,
                val_feats=sel_feats, val_labels=sel_labels,
                seed=seed, device=device, classifier_type="mlp",
            )
            scores = np.asarray(predict_fs(model, te_feats, device),
                                dtype=np.float32)
            fs_seed_scores.append(scores)
            fs_aucs.append(float(compute_auc(te_labels, scores)))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        fs_avg_scores = np.mean(fs_seed_scores, axis=0)
        fs_mean = float(compute_auc(te_labels, fs_avg_scores))
        fs_std = float(np.std(fs_aucs))
        log.info(f"    FS MLP seed-avg: {fs_mean:.4f} "
                 f"(per-seed std {fs_std:.4f})")

        elapsed = time.time() - t0
        log.info(f"  fold time: {elapsed:.0f}s")

        fold = {
            "held_out": held_out,
            "n_fit": int(len(fit_labels)),
            "n_select": int(len(sel_labels)),
            "n_test": int(len(te_labels)),
            "ps_means": ps_means,
            "best_ps_name": best_ps_name,
            "best_ps_auc": best_ps_auc,
            "raw_feat_auc": float(rf_auc),
            "fs_mlp_mean": fs_mean,
            "fs_mlp_std": fs_std,
            "fs_mlp_per_seed": fs_aucs,
        }
        fold_results.append(fold)

    log.info(f"\n{'='*60}")
    log.info("DISJOINT-DATA RESULTS (LOGO, full PS ladder)")
    log.info(f"{'='*60}")

    best_ps_aucs = np.array([f["best_ps_auc"] for f in fold_results])
    rf_aucs = np.array([f["raw_feat_auc"] for f in fold_results])
    fs_aucs = np.array([f["fs_mlp_mean"] for f in fold_results])

    fs_wins = int((fs_aucs > best_ps_aucs).sum())
    rf_wins = int((rf_aucs > best_ps_aucs).sum())
    n_folds = len(fold_results)

    # Two-sided sign-test p-values (paper convention).
    fs_sign_p = sign_test_p(fs_aucs - best_ps_aucs)
    rf_sign_p = sign_test_p(rf_aucs - best_ps_aucs)

    # Two-sided Wilcoxon signed-rank on fold-deltas.
    fs_deltas = fs_aucs - best_ps_aucs
    rf_deltas = rf_aucs - best_ps_aucs
    fs_wilcoxon_p = (float(wilcoxon(fs_deltas, alternative="two-sided").pvalue) if np.any(fs_deltas != 0) else 1.0)
    rf_wilcoxon_p = (float(wilcoxon(rf_deltas, alternative="two-sided").pvalue) if np.any(rf_deltas != 0) else 1.0)

    # Fold-level bootstrap 95% CI on mean fold-delta (10K resamples,
    # percentile method, fixed seed 1024 for reproducibility).
    rng = np.random.default_rng(1024)
    n_boot = 10_000
    fs_boot_means = np.empty(n_boot)
    rf_boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_folds, size=n_folds)
        fs_boot_means[i] = fs_deltas[idx].mean()
        rf_boot_means[i] = rf_deltas[idx].mean()
    fs_boot_ci = [float(np.percentile(fs_boot_means, 2.5)),
                  float(np.percentile(fs_boot_means, 97.5))]
    rf_boot_ci = [float(np.percentile(rf_boot_means, 2.5)),
                  float(np.percentile(rf_boot_means, 97.5))]

    avg_fs = float(fs_aucs.mean())
    avg_rf = float(rf_aucs.mean())
    avg_best_ps = float(best_ps_aucs.mean())
    delta_fs = avg_fs - avg_best_ps
    delta_rf = avg_rf - avg_best_ps

    hdr = f"{'fold':15s} {'best_ps':>12s} {'ps_name':<25s} {'raw_feat':>10s} {'fs_mlp':>10s}"
    log.info(hdr)
    log.info("-" * len(hdr))
    for f in fold_results:
        log.info(f"{f['held_out']:15s} {f['best_ps_auc']:12.4f} "
                 f"{f['best_ps_name']:<25s} "
                 f"{f['raw_feat_auc']:10.4f} {f['fs_mlp_mean']:10.4f}")
    log.info("-" * len(hdr))
    log.info(f"{'Avg(8)':15s} {avg_best_ps:12.4f} {'':25s} {avg_rf:10.4f} {avg_fs:10.4f}")

    log.info(f"\nFS   vs best PS: {delta_fs:+.4f}  wins={fs_wins}/{n_folds}  "
             f"sign p={fs_sign_p:.4f}  wilcoxon p={fs_wilcoxon_p:.4f}  "
             f"boot 95% CI [{fs_boot_ci[0]:+.4f}, {fs_boot_ci[1]:+.4f}]")
    log.info(f"Raw  vs best PS: {delta_rf:+.4f}  wins={rf_wins}/{n_folds}  "
             f"sign p={rf_sign_p:.4f}  wilcoxon p={rf_wilcoxon_p:.4f}  "
             f"boot 95% CI [{rf_boot_ci[0]:+.4f}, {rf_boot_ci[1]:+.4f}]")

    summary = {
        "pool": args.pool,
        "proj_dim": args.proj_dim,
        "seeds": args.seeds,
        "select_frac": args.select_frac,
        "protocol": "leave_one_generator_out_full_ps_ladder",
        "n_folds": n_folds,
        "avg_fs_mlp": avg_fs,
        "avg_raw_feat_logreg": avg_rf,
        "avg_best_ps": avg_best_ps,
        "delta_fs_vs_ps": delta_fs,
        "delta_rawfeat_vs_ps": delta_rf,
        "fs_wins": fs_wins,
        "rawfeat_wins": rf_wins,
        "fs_sign_test_p": fs_sign_p,                # two-sided
        "rawfeat_sign_test_p": rf_sign_p,           # two-sided
        "fs_wilcoxon_p": fs_wilcoxon_p,             # two-sided
        "rawfeat_wilcoxon_p": rf_wilcoxon_p,        # two-sided
        "fs_fold_bootstrap_ci": fs_boot_ci,         # 95%, 10K resamples, seed 1024
        "rawfeat_fold_bootstrap_ci": rf_boot_ci,    # 95%, 10K resamples, seed 1024
        "sign_test_alternative": "two-sided",
        "wilcoxon_alternative": "two-sided",
        "bootstrap_n_resamples": n_boot,
        "bootstrap_seed": 1024,
        "per_fold": fold_results,
    }
    out = results_root / f"11_disjoint_{args.pool}_d{args.proj_dim}.json"
    save_json(summary, out)
    log.info(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
