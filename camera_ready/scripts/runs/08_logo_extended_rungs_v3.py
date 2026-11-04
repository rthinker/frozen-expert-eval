"""Fit richer prediction-space inputs (PS_G to PS_J) in each LOGO fold.

The four input maps use disagreement, uncertainty and mutual information, quadratic terms and rank ECDF.
Their logistic heads use StandardScaler and LogisticRegression (C 1.0, max_iter 2000). The XGBoost heads
use in-fold three-fold CV over capped and widened grids. The SELECT fraction is 15% (random_state 1024).

--exclusion content applies the per-pool exclusion lists before the FIT/SELECT split; --exclusion none uses
the reviewed training rows. Each fold records membership fingerprints for both partitions. Use
09_logo_combined_ceiling_v3.py to combine these runs with the LOGO prediction ladder. Standalone reruns use
a separate --out-dir. The harness enables --resume after validating checkpoint provenance and scores.

From the repository root:

    python -B camera_ready/scripts/runs/08_logo_extended_rungs_v3.py --pool arch6 --exclusion content \
        --out-dir camera_ready/output/checks/richer_inputs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cr_paths  # noqa: E402

SEED = 1024
SELECT_FRAC = 0.15          # matches scripts/11_disjoint_data.py default

# Reduced but honest per-fold grids (verbatim from the reviewed script).
GRID_CAPPED = [
    {"n_estimators": n, "max_depth": d, "learning_rate": 0.1,
     "subsample": 0.8, "colsample_bytree": 0.8}
    for n in (100, 200) for d in (2, 3, 4)
]
GRID_WIDENED = GRID_CAPPED + [
    {"n_estimators": n, "max_depth": d, "learning_rate": 0.05,
     "subsample": 0.8, "colsample_bytree": 0.8}
    for n in (500, 1000) for d in (6, 8, 10)
]


def _fold_problem(folds, generators):
    """Why `folds` (from --folds) is not a set of held-out generators, each at most once (None if it is)."""
    unknown = sorted(set(folds) - set(generators))
    repeated = sorted({f for f in folds if folds.count(f) > 1})
    why = []
    if unknown:
        why.append(f"not a held-out generator: {', '.join(unknown)} (choose from {', '.join(generators)})")
    if repeated:
        why.append(f"given more than once: {', '.join(repeated)}")
    return "; ".join(why) or None


def cv_pick(X, y, grid):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))
    best, best_p = -1.0, grid[0]
    for p in grid:
        aucs = []
        for tr, va in folds:
            c = XGBClassifier(**p, scale_pos_weight=spw, random_state=SEED,
                              eval_metric="logloss", verbosity=0)
            c.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[va], c.predict_proba(X[va])[:, 1]))
        m = float(np.mean(aucs))
        if m > best:
            best, best_p = m, p
    return best_p, best


def fit_xgb(Xf, yf, Xt, params):
    from xgboost import XGBClassifier
    spw = (yf == 0).sum() / max((yf == 1).sum(), 1)
    c = XGBClassifier(**params, scale_pos_weight=spw, random_state=SEED,
                      eval_metric="logloss", verbosity=0)
    c.fit(Xf, yf)
    return c.predict_proba(Xt)[:, 1]


def fit_logreg(Xf, yf, Xt):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xf)
    c = LogisticRegression(max_iter=2000, C=1.0)
    c.fit(sc.transform(Xf), yf)
    return c.predict_proba(sc.transform(Xt))[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", default="arch6")
    ap.add_argument("--regime", choices=["capped", "widened", "both"], default="both")
    ap.add_argument("--codebase", help="code base root (default: this repository; MITIGATION_CODEBASE overrides it)")
    ap.add_argument("--exclusion", choices=("none", "content"), required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--folds", nargs="+")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _cr_paths.require_validated_resume(args.resume)
    _cr_paths.setup(args.codebase)
    import numpy as np
    from sklearn.metrics import roc_auc_score
    import richer_inputs as RA
    from logo_common import fold_view, load_content_manifest, load_generators, split_fit_select, write_json
    from ladder_checks import membership_fingerprint                     # version 3
    from src.utils.config import CACHE_ROOT, get_pool_experts
    from src.utils.dataset_helpers import TEST_GENERATORS

    # version 3: a dry run resolves the folder without creating it (or refusing a stored one); it writes nothing
    out_dir = Path(args.out_dir).resolve() if args.dry_run else _cr_paths.out_path(args.out_dir / "x").parent
    manifest_path = None
    if args.exclusion == "content":
        manifest_path = args.manifest or _cr_paths.EXCLUSION_ROOT / f"logo_exclusions_{args.pool}.json"
        if not Path(manifest_path).exists():
            raise SystemExit(f"no exclusion manifest at {manifest_path}; run logo_exclusions.py first")
    experts = get_pool_experts(args.pool)
    gens = list(TEST_GENERATORS)
    folds = args.folds or gens
    why = _fold_problem(folds, gens)
    if why:
        raise SystemExit(f"--folds: {why}")
    regimes = ["capped", "widened"] if args.regime == "both" else [args.regime]
    print(f"pool={args.pool} experts={len(experts)} regime={args.regime} exclusion={args.exclusion}", flush=True)
    data = load_generators(experts, CACHE_ROOT)
    manifest = load_content_manifest(manifest_path, args.pool, experts, data)

    out = {"pool": args.pool, "seed": SEED, "version": "08_logo_extended_rungs_v3", "exclusion": args.exclusion,
           "exclusion_manifest": _cr_paths.rel(manifest_path),
           "environment": _cr_paths.environment(),
           "grids": {"capped": GRID_CAPPED, "widened": GRID_WIDENED},
           "note": "reduced per-fold grids; see `grids`", "folds": {}}
    per_fold_ceiling = {r: {} for r in regimes}

    for held in folds:
        ckpt = out_dir / "rung_folds" / f"{args.pool}_{held}.json"
        rec = json.loads(ckpt.read_text(encoding="utf-8")) if args.resume and ckpt.exists() else None
        if rec is not None and not all(rec.get(k) for k in ("fit_fingerprint", "select_fingerprint")):  # v3
            print(f"  {held:<14} checkpoint holds no membership fingerprints (an earlier version); recomputed",
                  flush=True)
            rec = None
        if rec is not None:
            print(f"  {held:<14} reused checkpoint", flush=True)
        else:
            t0 = time.time()
            # training data of the fold (other generators; exclusions applied) and the same FIT/SELECT rule as 11
            tr_f, tr_lg, tr_y, _te_f, te_lg, te_y, n_excl = fold_view(data, held, manifest.get(held))
            ff, f_lg, f_y, sf, s_lg, s_y = split_fit_select(tr_f, tr_lg, tr_y, SELECT_FRAC, SEED)
            if args.dry_run:
                print(f"  {held:<14} FIT {len(f_y)}  SELECT {len(s_y)}  TEST {len(te_y)}  excluded {n_excl}")
                continue
            unc, _ = RA.make_uncertainty_featmap(s_lg, s_y)
            rank = RA.make_rank_featmap(f_lg)
            maps = {"disagreement": RA.feat_disagreement, "uncertainty+MI": unc,
                    "quadratic": RA.feat_quadratic, "rank ECDF": rank}
            rec = {"n_fit": int(len(f_y)), "n_test": int(len(te_y)), "n_excluded_training_rows": int(n_excl),
                   "n_select": int(len(s_y)), "fit_fingerprint": membership_fingerprint(ff),     # version 3
                   "select_fingerprint": membership_fingerprint(sf), "rungs": {}}
            scores = {"labels": np.asarray(te_y, dtype=np.int8)}
            for name, fn in maps.items():
                Xf = np.asarray(fn(f_lg), dtype=np.float64)
                Xt = np.asarray(fn(te_lg), dtype=np.float64)
                s_lr = fit_logreg(Xf, f_y, Xt)
                rec["rungs"][f"{name}_logreg"] = float(roc_auc_score(te_y, s_lr))
                scores[f"{name}_logreg"] = np.asarray(s_lr, dtype=np.float32)
                for reg in regimes:
                    grid = GRID_CAPPED if reg == "capped" else GRID_WIDENED
                    p, cv = cv_pick(Xf, f_y, grid)
                    s_x = fit_xgb(Xf, f_y, Xt, p)
                    rec["rungs"][f"{name}_xgb_{reg}"] = float(roc_auc_score(te_y, s_x))
                    scores[f"{name}_xgb_{reg}"] = np.asarray(s_x, dtype=np.float32)
                    rec.setdefault("selected", {})[f"{name}_{reg}"] = {"params": p, "in_fold_cv": cv}
            sp = _cr_paths.out_path(out_dir / "scores" / f"{args.pool}_rungs_{held}.npz")
            np.savez_compressed(sp, **scores)
            rec["scores_file"] = sp.relative_to(out_dir).as_posix()
            rec["fold_seconds"] = round(time.time() - t0, 1)
            write_json(rec, ckpt)
            best_line = ", ".join(f"{k}={v:.4f}" for k, v in sorted(rec["rungs"].items(), key=lambda kv: -kv[1])[:3])
            print(f"  {held:<14} ({time.time() - t0:.0f}s) top: {best_line}", flush=True)
        for reg in regimes:
            vals = [v for k, v in rec["rungs"].items() if k.endswith("_logreg") or k.endswith(f"_xgb_{reg}")]
            per_fold_ceiling[reg][held] = max(vals)
        out["folds"][held] = rec

    if args.dry_run:
        print("dry run: nothing fitted, nothing written")
        return 0
    if len(out["folds"]) != len(gens):
        print(f"{len(out['folds'])} of {len(gens)} folds done; run all folds to write a complete summary")
        return 0
    for reg in regimes:
        out[f"extended_only_ceiling_{reg}"] = float(np.mean([per_fold_ceiling[reg][g] for g in gens]))
        out[f"per_fold_extended_{reg}"] = per_fold_ceiling[reg]
        print(f"  [{reg}] richer inputs alone: mean ceiling {out[f'extended_only_ceiling_{reg}']:.4f}")
    dest = write_json(out, out_dir / f"A3c_logo_extended_rungs_{args.pool}.json")
    print(f"wrote {_cr_paths.rel(dest)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
