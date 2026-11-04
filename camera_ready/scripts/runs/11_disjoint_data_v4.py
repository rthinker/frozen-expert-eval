"""Leave one generator out (LOGO) with optional content-disjoint folds.

Use --exclusion content to remove training images identical to a held-out test image, or --exclusion none for
the reviewed procedure used by gate G0. Each fold uses the reviewed stratified 85/15 FIT/SELECT split
(random_state 1024). AUROC is computed once per method per fold after averaging predictions across seeds.

Each fold records FIT and SELECT membership fingerprints so that the statistics can validate paired runs.
The trusted run-pair mapping in camera_ready/provenance/ covers the supplied records predating fingerprints.

Use --folds for a subset of the eight generators. A full summary is written only when all eight folds are
available. --dry-run prints fold sizes without fitting or writing files. Standalone reruns use a separate
--out-dir. The reproduction harness enables --resume after validating checkpoint provenance and scores.

From the repository root:

    python -B camera_ready/scripts/runs/11_disjoint_data_v4.py --codebase . --pool arch6 --proj_dim 256 \
        --exclusion content --out-dir camera_ready/output/checks/content_disjoint
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cr_paths  # noqa: E402  (must precede torch / src imports)


def _console_logger(name):
    """Version 3: a logger that writes to the console only (a dry run writes no file, not even a log)."""
    import logging
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(h)
    return log


def _fold_problem(folds, generators):
    """Version 3: why `folds` (from --folds) is not a set of held-out generators, each at most once (None if it is)."""
    unknown = sorted(set(folds) - set(generators))
    repeated = sorted({f for f in folds if folds.count(f) > 1})
    why = []
    if unknown:
        why.append(f"not a held-out generator: {', '.join(unknown)} (choose from {', '.join(generators)})")
    if repeated:
        why.append(f"given more than once: {', '.join(repeated)}")
    return "; ".join(why) or None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pool", type=str, required=True)
    p.add_argument("--proj_dim", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="default: FS_SEEDS of the reviewed config")
    p.add_argument("--select_frac", type=float, default=0.15,
                   help="Stratified FIT/SELECT split inside each fold.")
    p.add_argument("--skip_slow", action="store_true",
                   help="Skip PS_C, PS_F, PS_E, PS_4 for fast iteration (never for paper numbers).")
    p.add_argument("--gpu", type=int, default=0)
    # version 2
    p.add_argument("--codebase", help="code base root (default: this repository; MITIGATION_CODEBASE overrides it)")
    p.add_argument("--exclusion", choices=("none", "content"), required=True)
    p.add_argument("--manifest", type=Path,
                   help="exclusion manifest (default: <mitigation>/exclusions/logo_exclusions_<pool>.json)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--folds", nargs="+", help="subset of held-out generators (default: all 8)")
    p.add_argument("--skip-fs2", action="store_true", help="skip the FS MLP (FS2)")
    p.add_argument("--resume", action="store_true", help="reuse fold checkpoints already in --out-dir")
    p.add_argument("--dry-run", action="store_true", help="load data, print fold sizes, fit nothing")
    p.add_argument("--gpu-name", help="run FS2 on the one GPU whose name contains this text (e.g. 5080); "
                                      "exits if it is not visible, never falls back to another GPU or the CPU")
    return p.parse_args()


def _run_ps_fold(fit_logits3d, fit_labels, sel_logits3d, sel_labels, te_logits3d, te_labels,
                 seeds, skip_slow, log):
    """Full PS ladder for one fold, as in the reviewed script. Returns ({base: AUROC}, {base: averaged scores})."""
    import numpy as np
    from src.models.prediction_space import PS_1, PS_A, PS_B, PS_2, PS_C, PS_3, PS_F, PS_D, PS_E, PS_4
    from src.utils.metrics import compute_auc
    per_run_scores: Dict[str, np.ndarray] = {}

    def _log_logreg_cv_diagnostics(name, diagnostics):
        total = diagnostics.get("total_convergence_warning_count", 0)
        if total <= 0:
            return
        cfg = diagnostics["best_config"]
        log.warning("    %s: %d fit(s) hit max_iter=%d during CV/refit; using best config (penalty=%s, C=%.3g, "
                    "solver=%s%s).", name, total, diagnostics["max_iter"], cfg["penalty"], cfg["C"], cfg["solver"],
                    "" if cfg["l1_ratio"] is None else f", l1_ratio={cfg['l1_ratio']:.3g}")

    def _score(name, fn):
        scores = np.asarray(fn(), dtype=np.float32)
        per_run_scores[name] = scores
        auc = compute_auc(te_labels, scores)
        log.info(f"    {name:<28s} {auc:.4f}")

    _score("PS_1_prob_avg", lambda: PS_1(te_logits3d))
    _score("PS_A_calibrated_avg", lambda: PS_A(sel_logits3d, sel_labels, te_logits3d))
    _score("PS_B_learned_weights", lambda: PS_B(sel_logits3d, sel_labels, te_logits3d))
    _score("PS_2_logreg", lambda: PS_2(fit_logits3d, fit_labels, te_logits3d))
    for s in seeds:
        _score(f"PS_3_mlp_logits_s{s}", lambda seed=s: PS_3(fit_logits3d, fit_labels, te_logits3d, seed=seed))
    for s in seeds:
        _score(f"PS_D_mlp_probs_s{s}", lambda seed=s: PS_D(fit_logits3d, fit_labels, te_logits3d, seed=seed))
    if not skip_slow:
        def _ps3_score():
            scores, diag = PS_C(fit_logits3d, fit_labels, te_logits3d, return_diagnostics=True)
            _log_logreg_cv_diagnostics("PS_C_logreg_cv", diag)
            return scores

        _score("PS_C_logreg_cv", _ps3_score)
        for s in seeds:
            _score(f"PS_F_mlp_logits_es_s{s}", lambda seed=s: PS_F(
                fit_logits3d, fit_labels, sel_logits3d, sel_labels, te_logits3d, seed=seed))
        for s in seeds:
            _score(f"PS_E_mlp_probs_es_s{s}", lambda seed=s: PS_E(
                fit_logits3d, fit_labels, sel_logits3d, sel_labels, te_logits3d, seed=seed))
        _score("PS_4_xgboost", lambda: PS_4(fit_logits3d, fit_labels, te_logits3d))

    grouped = defaultdict(list)
    for name, scores in per_run_scores.items():
        grouped[re.sub(r"_s\d+$", "", name)].append(scores)
    aggregated, avg_scores = {}, {}
    for base, score_list in grouped.items():
        avg = np.mean(score_list, axis=0)
        avg_scores[base] = avg.astype(np.float32)
        aggregated[base] = float(compute_auc(te_labels, avg))
    return aggregated, avg_scores


def sign_test_p(deltas):
    """Two-sided paired sign test, excluding exact ties; all ties give p=1."""
    import numpy as np
    from scipy.stats import binomtest

    d = np.asarray(deltas, dtype=float)
    n_nonzero = int(np.count_nonzero(d))
    return (float(binomtest(int((d > 0).sum()), n_nonzero, 0.5, alternative="two-sided").pvalue)
            if n_nonzero else 1.0)


def main():
    args = parse_args()
    _cr_paths.require_validated_resume(args.resume)
    _cr_paths.setup(args.codebase, gpu_name=args.gpu_name)
    import numpy as np
    import torch
    from scipy.stats import wilcoxon
    from logo_common import fold_view, load_content_manifest, load_generators, split_fit_select, write_json
    from ladder_checks import membership_fingerprint                     # version 4
    from src.models.feature_space import FS_1
    from src.training.fs_runtime import predict_fs, train_fs_model
    from src.utils.config import CACHE_ROOT, FS_SEEDS, get_feature_dims, get_pool_experts
    from src.utils.dataset_helpers import TEST_GENERATORS
    from src.utils.metrics import compute_auc

    if args.dry_run:                    # version 3: a dry run writes nothing, not even a log file or a folder
        out_dir = Path(args.out_dir).resolve()
        log = _console_logger(f"11_disjoint_v4_{args.pool}")
    else:
        out_dir = _cr_paths.out_path(args.out_dir / "x").parent
        log = _cr_paths.make_logger(f"11_disjoint_v4_{args.pool}", out_dir)
    seeds = args.seeds or list(FS_SEEDS)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if args.gpu_name:                       # pinned GPU: cuda:0 must be it; no fallback
        if not torch.cuda.is_available() or args.gpu != 0 or \
                args.gpu_name.lower() not in torch.cuda.get_device_name(0).lower():
            raise SystemExit(f"--gpu-name {args.gpu_name}: CUDA device 0 is "
                             f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}")
        device = "cuda:0"
    device_name = torch.cuda.get_device_name(int(device.split(":")[1])) if device.startswith("cuda") else "cpu"
    expert_names = get_pool_experts(args.pool)
    feature_dims = get_feature_dims(expert_names)
    manifest_path = None
    if args.exclusion == "content":
        manifest_path = args.manifest or _cr_paths.EXCLUSION_ROOT / f"logo_exclusions_{args.pool}.json"
        if not Path(manifest_path).exists():
            raise SystemExit(f"no exclusion manifest at {manifest_path}; run logo_exclusions.py first")
    folds = args.folds or list(TEST_GENERATORS)
    why = _fold_problem(folds, list(TEST_GENERATORS))          # version 3
    if why:
        raise SystemExit(f"--folds: {why}")
    env = _cr_paths.environment()

    log.info(f"Pool: {args.pool}, d={args.proj_dim}, exclusion={args.exclusion}, device={device} ({device_name})")
    log.info(f"Experts: {expert_names}")
    log.info(f"Seeds: {seeds}; environment: {env}")
    gen_data = load_generators(expert_names, CACHE_ROOT)
    manifest = load_content_manifest(manifest_path, args.pool, expert_names, gen_data)
    for g in TEST_GENERATORS:
        log.info(f"  {g}: {len(gen_data[g][2])} samples")

    fold_results = []
    for held_out in folds:
        ckpt = out_dir / "folds" / f"{args.pool}_d{args.proj_dim}_{held_out}.json"
        old = json.loads(ckpt.read_text(encoding="utf-8")) if args.resume and ckpt.exists() else None
        if old is not None and all(old.get(k) for k in ("fit_fingerprint", "select_fingerprint")):  # version 4
            fold_results.append(old)
            log.info(f"=== Fold {held_out}: reused checkpoint {ckpt.name}")
            continue
        if old is not None:
            log.info(f"=== Fold {held_out}: checkpoint {ckpt.name} holds no membership fingerprints (an earlier "
                     "version); recomputed")
        log.info(f"\n=== Fold: held-out={held_out} ===")
        t0 = time.time()
        tr_feats, tr_logits3d, tr_labels, te_feats, te_logits3d, te_labels, n_excl = fold_view(
            gen_data, held_out, manifest.get(held_out))
        (fit_feats, fit_logits3d, fit_labels,
         sel_feats, sel_logits3d, sel_labels) = split_fit_select(tr_feats, tr_logits3d, tr_labels,
                                                                 args.select_frac, 1024)
        log.info(f"  excluded training rows: {n_excl}")
        log.info(f"  FIT: {len(fit_labels)} (real={int((fit_labels == 0).sum())}, fake={int((fit_labels == 1).sum())})"
                 f" | SELECT: {len(sel_labels)} (real={int((sel_labels == 0).sum())}, "
                 f"fake={int((sel_labels == 1).sum())})")
        log.info(f"  TEST: {len(te_labels)} ({held_out})")
        if args.dry_run:
            continue

        log.info("  PS ladder:")
        ps_means, ps_scores = _run_ps_fold(fit_logits3d, fit_labels, sel_logits3d, sel_labels,
                                           te_logits3d, te_labels, seeds, args.skip_slow, log)
        best_ps_name = max(ps_means, key=ps_means.get)
        best_ps_auc = ps_means[best_ps_name]
        log.info(f"  Best PS: {best_ps_name} = {best_ps_auc:.4f}")

        rf_scores = FS_1(
            np.concatenate(fit_feats, axis=1), fit_labels, np.concatenate(te_feats, axis=1), logger=log
        )
        rf_auc = compute_auc(te_labels, rf_scores)
        log.info(f"  Raw-feat LogReg: {rf_auc:.4f}")

        fs_mean = fs_std = None
        fs_aucs, fs_avg_scores = [], None
        if not args.skip_fs2:
            log.info("  FS MLP:")
            fs_seed_scores = []
            for seed in seeds:
                model, _ = train_fs_model(fit_feats, fit_labels, feature_dims, args.proj_dim,
                                          val_feats=sel_feats, val_labels=sel_labels,
                                          seed=seed, device=device, classifier_type="mlp")
                scores = np.asarray(predict_fs(model, te_feats, device), dtype=np.float32)
                fs_seed_scores.append(scores)
                fs_aucs.append(float(compute_auc(te_labels, scores)))
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            fs_avg_scores = np.mean(fs_seed_scores, axis=0)
            fs_mean = float(compute_auc(te_labels, fs_avg_scores))
            fs_std = float(np.std(fs_aucs))
            log.info(f"    FS MLP seed-avg: {fs_mean:.4f} (per-seed std {fs_std:.4f})")

        log.info(f"  fold time: {time.time() - t0:.0f}s")
        fold = {"held_out": held_out, "n_fit": int(len(fit_labels)), "n_select": int(len(sel_labels)),
                "n_test": int(len(te_labels)), "n_excluded_training_rows": int(n_excl),
                "fit_fingerprint": membership_fingerprint(fit_feats),           # version 4
                "select_fingerprint": membership_fingerprint(sel_feats),
                "ps_means": ps_means, "best_ps_name": best_ps_name, "best_ps_auc": best_ps_auc,
                "raw_feat_auc": float(rf_auc), "fs_mlp_mean": fs_mean, "fs_mlp_std": fs_std,
                "fs_mlp_per_seed": fs_aucs, "fold_seconds": round(time.time() - t0, 1), "device_name": device_name}
        scores_path = _cr_paths.out_path(out_dir / "scores" / f"{args.pool}_d{args.proj_dim}_{held_out}.npz")
        arrays = {"labels": np.asarray(te_labels, dtype=np.int8),
                  "FS_1_raw_feat_logreg": np.asarray(rf_scores, dtype=np.float32)}
        arrays.update(ps_scores)
        if fs_avg_scores is not None:
            arrays["FS_2_mlp_seed_avg"] = np.asarray(fs_avg_scores, dtype=np.float32)
        np.savez_compressed(scores_path, **arrays)
        fold["scores_file"] = scores_path.relative_to(out_dir).as_posix()
        write_json(fold, ckpt)
        fold_results.append(fold)

    if args.dry_run:
        log.info("dry run: nothing fitted, nothing written")
        return 0
    if sorted(f["held_out"] for f in fold_results) != sorted(TEST_GENERATORS):   # version 3: exactly the eight
        log.info(f"{len({f['held_out'] for f in fold_results})} of {len(TEST_GENERATORS)} folds done; summary needs "
                 "all folds")
        return 0

    order = {g: i for i, g in enumerate(TEST_GENERATORS)}
    fold_results.sort(key=lambda f: order[f["held_out"]])
    best_ps_aucs = np.array([f["best_ps_auc"] for f in fold_results])
    rf_aucs = np.array([f["raw_feat_auc"] for f in fold_results])
    have_fs2 = all(f["fs_mlp_mean"] is not None for f in fold_results)
    fs_aucs = np.array([f["fs_mlp_mean"] for f in fold_results]) if have_fs2 else rf_aucs * np.nan
    n_folds = len(fold_results)
    rf_deltas = rf_aucs - best_ps_aucs
    fs_deltas = fs_aucs - best_ps_aucs
    rf_wins = int((rf_aucs > best_ps_aucs).sum())
    fs_wins = int((fs_aucs > best_ps_aucs).sum()) if have_fs2 else None
    rng = np.random.default_rng(1024)                      # as in the reviewed script: one draw per resample
    fs_boot, rf_boot = np.empty(10_000), np.empty(10_000)
    for i in range(10_000):
        idx = rng.integers(0, n_folds, size=n_folds)
        fs_boot[i] = fs_deltas[idx].mean()
        rf_boot[i] = rf_deltas[idx].mean()
    summary = {
        "pool": args.pool, "proj_dim": args.proj_dim, "seeds": seeds, "select_frac": args.select_frac,
        "protocol": "leave_one_generator_out_full_ps_ladder",
        "version": "11_disjoint_data_v4", "exclusion": args.exclusion,
        "exclusion_manifest": _cr_paths.rel(manifest_path),
        "environment": env, "device": device, "n_folds": n_folds,
        "avg_fs_mlp": float(fs_aucs.mean()) if have_fs2 else None,
        "avg_raw_feat_logreg": float(rf_aucs.mean()), "avg_best_ps": float(best_ps_aucs.mean()),
        "delta_fs_vs_ps": float(fs_aucs.mean() - best_ps_aucs.mean()) if have_fs2 else None,
        "delta_rawfeat_vs_ps": float(rf_aucs.mean() - best_ps_aucs.mean()),
        "fs_wins": fs_wins, "rawfeat_wins": rf_wins,
        "fs_sign_test_p": sign_test_p(fs_deltas) if have_fs2
        else None,
        "rawfeat_sign_test_p": sign_test_p(rf_deltas),
        "fs_wilcoxon_p": (float(wilcoxon(fs_deltas, alternative="two-sided").pvalue) if np.any(fs_deltas != 0) else 1.0) if have_fs2 else None,
        "rawfeat_wilcoxon_p": (float(wilcoxon(rf_deltas, alternative="two-sided").pvalue) if np.any(rf_deltas != 0) else 1.0),
        "fs_fold_bootstrap_ci": [float(np.percentile(fs_boot, 2.5)), float(np.percentile(fs_boot, 97.5))]
        if have_fs2 else None,
        "rawfeat_fold_bootstrap_ci": [float(np.percentile(rf_boot, 2.5)), float(np.percentile(rf_boot, 97.5))],
        "sign_test_alternative": "two-sided", "wilcoxon_alternative": "two-sided",
        "bootstrap_n_resamples": 10_000, "bootstrap_seed": 1024, "per_fold": fold_results,
    }
    out = write_json(summary, out_dir / f"11_disjoint_{args.pool}_d{args.proj_dim}.json")
    log.info(f"FS1 vs best PS (ten methods): {summary['delta_rawfeat_vs_ps']:+.4f}  wins={rf_wins}/{n_folds}  "
             f"CI {summary['rawfeat_fold_bootstrap_ci']}")
    log.info(f"Saved: {_cr_paths.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
