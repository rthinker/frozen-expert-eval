"""Refit the deterministic FS1 raw-feature logistic regression on the standard split and LOGO folds.

The refit reads the released feature cache and the reviewed loaders and
raw_feature_logreg_* settings in configs/experiments.yaml. Scoring runs in
batches to reduce memory. The L-BFGS fit still converts the FIT matrix to
float64 (about 2.0 GiB for 26,591 rows by 9,984 dimensions on standard arch6).

LOGO uses the reviewed folds before the content exclusion. A reference with
excluded training rows is marked as incomparable. The report gives AUROC,
absolute differences and tolerances. The standard-split tolerance of 1e-3
allows for BLAS thread-count effects. PASS requires at least one comparison
and every comparison to pass.

From the repository root:
    python camera_ready/scripts/review/refit_fs1_v3.py --pool arch6 --protocol both

The default report is camera_ready/output/checks/refit_fs1_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def find_codebase(arg):
    """The reviewed code base: the release repository root (three levels above this file) by default."""
    cand = Path(arg).resolve() if arg else Path(__file__).resolve().parents[3]
    if not (cand / "src" / "utils" / "config.py").exists():
        raise SystemExit(f"no reviewed code base at {cand} (expected src/utils/config.py); pass --codebase")
    return cand


def fs1_fit_score(cfg, X_fit, y_fit, X_eval, batch):
    """FS1 exactly as in src/models/feature_space.py::FS_1, with batched predict_proba."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(
        solver=cfg["raw_feature_logreg_solver"],
        C=cfg["raw_feature_logreg_C"],
        class_weight=cfg["raw_feature_logreg_class_weight"],
        max_iter=cfg["raw_feature_logreg_max_iter"],
        random_state=cfg["raw_feature_logreg_random_state"],
    )
    lr.fit(X_fit, y_fit)
    return np.concatenate([lr.predict_proba(X_eval[i:i + batch])[:, 1] for i in range(0, len(X_eval), batch)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--codebase", help="reviewed code base root (default: the release repository root)")
    ap.add_argument("--cache", help="feature cache root (default: <codebase>/output/cached_features)")
    ap.add_argument("--expect-dir", help="stored results of the submitted run (default: <codebase>/output/results)")
    ap.add_argument("--pool", default="arch6")
    ap.add_argument("--protocol", choices=("standard", "logo", "both"), default="both")
    ap.add_argument("--batch", type=int, default=2048, help="rows per scoring batch")
    ap.add_argument("--tol-standard", type=float, default=1e-3)
    ap.add_argument("--tol-logo", type=float, default=1e-6)
    ap.add_argument("--out", type=Path,                         # an ignored folder of camera_ready/
                    default=Path(__file__).resolve().parents[2] / "output" / "checks" / "refit_fs1_report.json")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runs"))
    import _cr_paths
    if args.out:
        args.out = _cr_paths.out_path(args.out)

    codebase = find_codebase(args.codebase)
    sys.path.insert(0, str(codebase))
    from sklearn.model_selection import train_test_split
    from src.utils.config import DS_CONFIG, FIT_TAG, get_pool_experts
    from src.utils.dataset_helpers import TEST_GENERATORS
    from src.utils.io import load_pool_data
    from src.utils.metrics import compute_auc

    cache = Path(args.cache).resolve() if args.cache else codebase / "output" / "cached_features"
    expect = Path(args.expect_dir).resolve() if args.expect_dir else codebase / "output" / "results"
    experts = get_pool_experts(args.pool)

    def shown(p: Path) -> str:
        """Path as recorded in the report: relative to the code base, never a machine-specific prefix."""
        try:
            return "<codebase>/" + p.relative_to(codebase).as_posix()
        except ValueError:
            return p.name

    def stored_result(path):
        """(stored JSON, None), or (None, why it cannot be compared)."""
        if path is None or not path.is_file():
            return None, f"no stored result{(' ' + shown(path)) if path is not None else ''}"
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("skipped") is True:
            return None, f"the stored result {shown(path)} is a skip marker ({data.get('reason', 'unspecified')})"
        return data, None

    report = {"pool": args.pool, "experts": experts, "cache": shown(cache), "batch": args.batch,
              "fs1_settings": {k: DS_CONFIG[k] for k in DS_CONFIG if k.startswith("raw_feature_logreg_")},
              "comparisons": {}}
    ok, compared = True, 0       # PASS needs at least one comparison

    def concat(feats):
        return np.concatenate(feats, axis=1)

    gens = {g: load_pool_data(experts, g, cache) for g in TEST_GENERATORS}

    if args.protocol in ("standard", "both"):
        t0 = time.time()
        fit_f, _, fit_y = load_pool_data(experts, FIT_TAG, cache)
        X_fit = concat(fit_f)
        del fit_f
        # one fit on FIT, scored on every held-out generator (as in the submitted script 04)
        X_all = np.concatenate([concat(gens[g][0]) for g in TEST_GENERATORS], axis=0)
        bounds = np.cumsum([0] + [len(gens[g][2]) for g in TEST_GENERATORS])
        scores = fs1_fit_score(DS_CONFIG, X_fit, fit_y, X_all, args.batch)
        del X_all, X_fit
        per_gen = {g: float(compute_auc(gens[g][2], scores[bounds[i]:bounds[i + 1]]))
                   for i, g in enumerate(TEST_GENERATORS)}
        macro = float(np.mean(list(per_gen.values())))
        stored, why_not = stored_result(expect / f"04_ds_{args.pool}_rawfeat.json")
        rec = {"macro": macro, "per_generator": per_gen, "seconds": round(time.time() - t0, 1)}
        if stored is not None:
            diff = abs(macro - stored["avg_auc_8"])
            rec.update({"stored_macro": stored["avg_auc_8"], "abs_diff": diff, "tolerance": args.tol_standard,
                        "status": "PASS" if diff <= args.tol_standard else "FAIL"})
            ok &= diff <= args.tol_standard
            compared += 1
        else:
            rec["status"] = f"not compared: {why_not}"
        report["comparisons"]["standard"] = rec
        print(f"standard: FS1 macro {macro:.6f}" + (f" (stored {stored['avg_auc_8']:.6f}, {rec['status']})"
                                                      if stored else f" ({rec['status']})"))

    if args.protocol in ("logo", "both"):
        t0 = time.time()
        found = sorted(expect.glob(f"11_disjoint_{args.pool}_d*.json"))   # any projection size
        stored, why_not = stored_result(found[0] if found else None)
        if stored is not None and stored.get("exclusion") not in (None, "none"):
            why_not = (f"the stored run {shown(found[0])} excludes training rows (exclusion "
                       f"{stored.get('exclusion')!r}); this refit reproduces the submitted folds, without exclusion; "
                       "compare with the reviewed run (output/results/)")
            stored = None
        stored_folds = {f["held_out"]: f["raw_feat_auc"] for f in stored["per_fold"]} if stored else {}
        folds = {}
        for held in TEST_GENERATORS:
            train = [gens[g] for g in TEST_GENERATORS if g != held]
            n_exp = len(experts)
            tr_feats = [np.concatenate([t[0][i] for t in train]) for i in range(n_exp)]
            tr_y = np.concatenate([t[2] for t in train])
            idx = np.arange(len(tr_y))
            # same stratified 85/15 FIT/SELECT split as scripts/11_disjoint_data.py (seed 1024)
            fit_idx, _ = train_test_split(idx, test_size=0.15, random_state=1024, stratify=tr_y)
            X_fit = concat([f[fit_idx] for f in tr_feats])
            scores = fs1_fit_score(DS_CONFIG, X_fit, tr_y[fit_idx], concat(gens[held][0]), args.batch)
            folds[held] = float(compute_auc(gens[held][2], scores))
            print(f"  LOGO fold {held}: FS1 {folds[held]:.6f}" +
                  (f" (stored {stored_folds[held]:.6f})" if held in stored_folds else ""))
            del X_fit, tr_feats
        rec = {"mean": float(np.mean(list(folds.values()))), "per_fold": folds, "seconds": round(time.time() - t0, 1)}
        if stored_folds:
            diffs = {g: abs(folds[g] - stored_folds[g]) for g in folds}
            rec.update({"stored_per_fold": stored_folds, "max_abs_diff": max(diffs.values()),
                        "tolerance": args.tol_logo,
                        "status": "PASS" if max(diffs.values()) <= args.tol_logo else "FAIL"})
            ok &= max(diffs.values()) <= args.tol_logo
            compared += 1
        else:
            rec["status"] = f"not compared: {why_not or 'the stored result has no LOGO folds'}"
        report["comparisons"]["logo"] = rec
        print(f"LOGO: FS1 mean {rec['mean']:.6f} ({rec['status']})")

    if not compared:
        print("nothing was compared with a stored result")
    report["status"] = "PASS" if ok and compared else ("NOT COMPARED" if ok else "FAIL")
    _cr_paths.atomic_write_text(args.out, json.dumps(report, indent=2, allow_nan=False))
    print(f"report -> {args.out} ({report['status']})")
    return 0 if ok and compared else 1


if __name__ == "__main__":
    sys.exit(main())
