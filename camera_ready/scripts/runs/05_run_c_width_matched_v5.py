"""Compare feature-space and prediction-space models at matched widths.

The runner computes per-expert PCA, a supervised deflation probe, FS1 on projected coordinates, PS_4 and FS_D.
PCA uses randomized SVD (random_state 42); the probe uses L2 logistic regression (C 1.0, max_iter 2000).
FS_D uses XGBoost on raw features (200 trees, depth 3, learning rate 0.1, subsample 0.8, colsample 0.5,
hist, scale_pos_weight, seed 1024).

With --protocol standard, gate G4 checks each required method against the packaged references
(--reference-dir overrides camera_ready/inputs). Missing references and failed comparisons fail the gate.
With --protocol logo, each fold uses the same FIT/SELECT split and exclusions as the LOGO ladder, and records
both membership fingerprints. --dry-run writes nothing. The harness alone enables checkpoint reuse after
validation. Use a separate --out-dir for standalone reruns.

From the repository root:

    python -B camera_ready/scripts/runs/05_run_c_width_matched_v5.py --pool arch6 --protocol standard \
        --out-dir camera_ready/output/checks/mechanism --codebase .
    python -B camera_ready/scripts/runs/05_run_c_width_matched_v5.py --pool arch6 --protocol logo \
        --exclusion content --out-dir camera_ready/output/checks/mechanism --codebase .
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cr_paths  # noqa: E402


def _console_logger(name):
    """Version 4: a logger that writes to the console only (a dry run writes no file, not even a log)."""
    import logging
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(h)
    return log


def _fold_problem(folds, generators):
    """Version 4: why `folds` (from --folds) is not a set of held-out generators, each at most once (None if it is)."""
    unknown = sorted(set(folds) - set(generators))
    repeated = sorted({f for f in folds if folds.count(f) > 1})
    why = []
    if unknown:
        why.append(f"not a held-out generator: {', '.join(unknown)} (choose from {', '.join(generators)})")
    if repeated:
        why.append(f"given more than once: {', '.join(repeated)}")
    return "; ".join(why) or None

PARITY_TOL = 1e-3
FSD_PARAMS = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8,
              "colsample_bytree": 0.5}
FSD_SEED = 1024
CR = Path(__file__).resolve().parents[2]           # camera_ready/ in the code repository
REVIEWED_RUN_C = ("arch6",)                        # pools with a reviewed run C (arch6 only)


def default_reference_dir() -> Path:
    """Reviewed run C results: camera_ready/inputs/ in the code repository."""
    return CR / "inputs" if (CR / "inputs").is_dir() else _cr_paths.STORED_REB


def _field(record, *path):
    """record[path[0]][path[1]]... when every level is present and the value is a number, else None."""
    for key in path:
        record = record.get(key) if isinstance(record, dict) else None
    return record if isinstance(record, (int, float)) and not isinstance(record, bool) else None


def g4_gate(pool, macro, dims, probe_dims, skip_probe, skip_fsd, reference_dir, results_root):
    """Gate G4 (version 3): every computed method that has a reviewed value, compared at PARITY_TOL. Returns (gate,
    references, missing): the comparisons, the reference files used, and the expected references that were not found
    (the gate fails when any is missing). For a pool with a reviewed run C, each reviewed field the run computes is
    expected (the PS_4 comparator, the PCA sweep at every --dims width, the probe at every --probe-dims width unless
    --skip-probe), and a method with a reference but no computed value counts as missing too."""
    ref, references, missing = {}, {}, []
    rc = Path(reference_dir) / f"run_c_width_matched_{pool}.json"
    if pool in REVIEWED_RUN_C:
        if rc.is_file():
            r = json.loads(rc.read_text(encoding="utf-8"))
            references["run_c"] = _cr_paths.rel(rc)
            expected = [("PS_4_xgboost", ("comparator", "macro"))]
            expected += [(f"FS_pca{d}_logreg", ("pca_sweep", f"d{d}", "fs_macro")) for d in dims]
            if not skip_probe:
                expected += [(f"FS_probe{d}_logreg", ("probe_mirror", f"d{d}", "fs_macro")) for d in probe_dims]
            for name, path in expected:
                value = _field(r, *path)
                if value is None:
                    missing.append(f"{name} ({'.'.join(path)} not in {rc.name})")
                else:
                    ref[name] = value
        else:
            missing.append(f"reviewed run C ({rc.name} not in {_cr_paths.rel(Path(reference_dir))}; see --reference-dir)")
    for name, fname, needed in (("FS_D_xgb_features", f"12_xgb_features_{pool}.json", not skip_fsd),
                                ("FS_1_raw_feat_logreg", f"04_ds_{pool}_rawfeat.json", True)):
        if not needed:
            continue
        f = Path(results_root) / fname
        value = json.loads(f.read_text(encoding="utf-8")).get("avg_auc_8") if f.is_file() else None
        if value is None:      # absent, or a skip marker without a value
            missing.append(f"{name} ({fname} without avg_auc_8 in the reviewed results)")
        else:
            ref[name] = value
            references[name] = _cr_paths.rel(f)
    missing += [f"{k} (reviewed value present, but this run computed none)" for k in ref if k not in macro]
    gate = {k: {"new": macro[k], "stored": v, "abs_diff": abs(macro[k] - v),
                "status": "PASS" if abs(macro[k] - v) <= PARITY_TOL else "FAIL"}
            for k, v in ref.items() if k in macro}
    return gate, references, missing


# ---------------------------------------------------------------- projections (verbatim from run C)

def pca_project(fit_feats: List, test_feats_cat, dims_per_expert: Sequence[int], d: int, log) -> Dict:
    """Per-expert PCA to d coords on FIT; returns projected FIT/TEST."""
    import numpy as np
    from sklearn.decomposition import PCA
    fit_blocks, test_blocks, evr = [], [], []
    off = 0
    for i, xf in enumerate(fit_feats):
        Di = dims_per_expert[i]
        xt = test_feats_cat[:, off:off + Di]
        off += Di
        p = PCA(n_components=d, svd_solver="randomized", random_state=42)
        fit_blocks.append(p.fit_transform(np.asarray(xf, dtype=np.float32)))
        test_blocks.append(p.transform(np.asarray(xt, dtype=np.float32)))
        evr.append(float(p.explained_variance_ratio_.sum()))
    log(f"    PCA d={d}: mean explained variance across experts {float(np.mean(evr)):.3f}")
    return {"X_fit": np.concatenate(fit_blocks, axis=1), "X_test": np.concatenate(test_blocks, axis=1),
            "explained_variance_ratio_sum_per_expert": evr}


def probe_project(fit_feats: List, fit_labels, test_feats_cat, dims_per_expert: Sequence[int], d: int, log) -> Dict:
    """Per-expert supervised linear probe to d coords by deflation (fitted on FIT only)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    fit_blocks, test_blocks = [], []
    off = 0
    for i, xf in enumerate(fit_feats):
        Di = dims_per_expert[i]
        Xf = np.asarray(xf, dtype=np.float32).copy()
        Xt = np.asarray(test_feats_cat[:, off:off + Di], dtype=np.float32).copy()
        off += Di
        fdirs, tdirs = [], []
        for _ in range(d):
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            clf.fit(Xf, fit_labels)
            w = clf.coef_.ravel().astype(np.float32)
            nw = float(np.linalg.norm(w))
            if nw == 0.0:
                break
            w /= nw
            fdirs.append(Xf @ w)
            tdirs.append(Xt @ w)
            Xf = Xf - np.outer(Xf @ w, w)
            Xt = Xt - np.outer(Xt @ w, w)
        fit_blocks.append(np.stack(fdirs, axis=1))
        test_blocks.append(np.stack(tdirs, axis=1))
        log(f"    probe d={d}: expert {i} done ({len(fdirs)} directions)")
    return {"X_fit": np.concatenate(fit_blocks, axis=1), "X_test": np.concatenate(test_blocks, axis=1)}


def fsd_scores(X_fit, y_fit, X_test):
    """FS_D: XGBoost on raw features, configuration of scripts/12_xgb_features.py."""
    from src.utils.seed import set_seed
    from xgboost import XGBClassifier
    set_seed(FSD_SEED)
    spw = int((y_fit == 0).sum()) / max(int((y_fit == 1).sum()), 1)
    clf = XGBClassifier(**FSD_PARAMS, scale_pos_weight=spw, random_state=FSD_SEED, eval_metric="logloss",
                        verbosity=0, tree_method="hist")
    clf.fit(X_fit, y_fit)
    return clf.predict_proba(X_test)[:, 1]


def methods(fit_feats, fit_logits3d, fit_labels, te_feats_cat, te_logits3d, dims, probe_dims, skip_probe, skip_fsd,
            log):
    """Scores of every compared method on one test set: {name: scores}."""
    import numpy as np
    from src.models.feature_space import FS_1
    from src.models.prediction_space import PS_4
    dims_per_expert = [f.shape[1] for f in fit_feats]
    out = {"PS_4_xgboost": np.asarray(PS_4(fit_logits3d, fit_labels, te_logits3d), dtype=np.float32),
           "FS_1_raw_feat_logreg": np.asarray(FS_1(np.concatenate(fit_feats, axis=1), fit_labels, te_feats_cat),
                                              dtype=np.float32)}
    if not skip_fsd:
        log("    FS_D (XGBoost on features) ...")
        out["FS_D_xgb_features"] = np.asarray(fsd_scores(np.concatenate(fit_feats, axis=1), fit_labels,
                                                         te_feats_cat), dtype=np.float32)
    for d in dims:
        proj = pca_project(fit_feats, te_feats_cat, dims_per_expert, d, log)
        out[f"FS_pca{d}_logreg"] = np.asarray(FS_1(proj["X_fit"], fit_labels, proj["X_test"]), dtype=np.float32)
    if not skip_probe:
        for d in probe_dims:
            pr = probe_project(fit_feats, fit_labels, te_feats_cat, dims_per_expert, d, log)
            out[f"FS_probe{d}_logreg"] = np.asarray(FS_1(pr["X_fit"], fit_labels, pr["X_test"]), dtype=np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", default="arch6")
    ap.add_argument("--protocol", choices=("standard", "logo"), required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--probe_dims", type=int, nargs="+", default=[2])
    ap.add_argument("--skip_probe", action="store_true")
    ap.add_argument("--skip_fsd", action="store_true")
    ap.add_argument("--codebase", help="code base root (default: this repository; MITIGATION_CODEBASE overrides it)")
    ap.add_argument("--exclusion", choices=("none", "content"), default="content", help="LOGO only")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--folds", nargs="+")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reference-dir", type=Path,
                    help="folder with the reviewed run_c_width_matched_<pool>.json (gate G4); default: camera_ready/"
                         "inputs/")
    a = ap.parse_args()
    _cr_paths.require_validated_resume(a.resume)
    if a.reference_dir is not None and not a.reference_dir.is_dir():
        ap.error(f"not a folder of reviewed results: {a.reference_dir}")
    _cr_paths.setup(a.codebase)
    import numpy as np
    from logo_common import fold_view, load_content_manifest, load_generators, split_fit_select, write_json
    from ladder_checks import membership_fingerprint                     # version 5
    from src.utils.config import CACHE_ROOT, FIT_TAG, RESULTS_ROOT, get_pool_experts
    from src.utils.dataset_helpers import TEST_GENERATORS
    from src.utils.io import load_pool_data
    from src.utils.metrics import compute_auc

    if a.dry_run:                       # version 4: a dry run writes nothing, not even a log file or a folder
        out_dir = Path(a.out_dir).resolve()
        logger = _console_logger(f"05_run_c_v5_{a.pool}_{a.protocol}")
    else:
        out_dir = _cr_paths.out_path(a.out_dir / "x").parent
        logger = _cr_paths.make_logger(f"05_run_c_v5_{a.pool}_{a.protocol}", out_dir)
    log = logger.info
    experts = get_pool_experts(a.pool)
    t_start = time.time()

    if a.protocol == "standard":
        fit_feats, fit_logits, fit_labels = load_pool_data(experts, FIT_TAG, CACHE_ROOT)
        data = load_generators(experts, CACHE_ROOT)
        bounds, off = {}, 0
        for g in TEST_GENERATORS:
            n = len(data[g][2])
            bounds[g] = (off, off + n)
            off += n
        te_feats_cat = np.concatenate([np.concatenate(data[g][0], axis=1) for g in TEST_GENERATORS], axis=0)
        te_logits3d = np.concatenate([np.stack(data[g][1], axis=1) for g in TEST_GENERATORS], axis=0)
        te_labels = np.concatenate([data[g][2] for g in TEST_GENERATORS])
        log(f"standard split: FIT {len(fit_labels)}, TEST {len(te_labels)}")
        if a.dry_run:
            return 0
        scores = methods(fit_feats, np.stack(fit_logits, axis=1), fit_labels, te_feats_cat, te_logits3d,
                         a.dims, a.probe_dims, a.skip_probe, a.skip_fsd, log)
        macro = {k: float(np.mean([compute_auc(te_labels[s:e], v[s:e]) for s, e in bounds.values()]))
                 for k, v in scores.items()}
        for k, v in macro.items():
            log(f"  {k:<24} macro {v:.4f}")
        # gate G4: parity with the stored reviewed results; version 3: portable references, all of them required
        gate, references, missing = g4_gate(a.pool, macro, a.dims, a.probe_dims, a.skip_probe, a.skip_fsd,
                                            a.reference_dir or default_reference_dir(), RESULTS_ROOT)
        ok = bool(gate) and not missing and all(g["status"] == "PASS" for g in gate.values())
        res = {"run": "C_v2_standard", "runner": "05_run_c_width_matched_v5", "pool": a.pool, "experts": experts,
               "macro": macro, "gate_G4": gate, "gate_G4_status": "PASS" if ok else "FAIL", "tolerance": PARITY_TOL,
               "gate_G4_references": references, "gate_G4_missing": missing,
               "environment": _cr_paths.environment(), "elapsed_sec": round(time.time() - t_start, 1)}
        out = write_json(res, out_dir / f"run_c_v2_standard_{a.pool}.json")
        log(f"G4 {res['gate_G4_status']}: " + ", ".join(f"{k} {g['abs_diff']:.2e}" for k, g in gate.items()))
        for m in missing:
            log(f"G4 reference missing: {m}")
        log(f"saved {_cr_paths.rel(out)}")
        return 0 if ok else 1

    # ---------------------------------------------------------------- LOGO
    manifest_path = None
    if a.exclusion == "content":
        local = CR / "output" / "results" / "mitigation" / "exclusions"
        manifest_path = a.manifest or (local if local.is_dir() else _cr_paths.EXCLUSION_ROOT) / \
            f"logo_exclusions_{a.pool}.json"
        if not Path(manifest_path).exists():
            raise SystemExit(f"no exclusion manifest at {manifest_path}; run logo_exclusions.py first")
    data = load_generators(experts, CACHE_ROOT)
    folds = a.folds or list(TEST_GENERATORS)
    why = _fold_problem(folds, list(TEST_GENERATORS))          # version 4
    if why:
        raise SystemExit(f"--folds: {why}")
    manifest = load_content_manifest(manifest_path, a.pool, experts, data)
    folds_out = []
    for held in folds:
        ckpt = out_dir / "mechanism_folds" / f"{a.pool}_{held}.json"
        old = json.loads(ckpt.read_text(encoding="utf-8")) if a.resume and ckpt.exists() else None
        if old is not None and all(old.get(k) for k in ("fit_fingerprint", "select_fingerprint")):  # version 5
            folds_out.append(old)
            log(f"fold {held}: reused checkpoint")
            continue
        if old is not None:
            log(f"fold {held}: checkpoint holds no membership fingerprints (an earlier version); recomputed")
        t0 = time.time()
        tr_f, tr_lg, tr_y, te_f, te_lg, te_y, n_excl = fold_view(data, held, manifest.get(held))
        ff, f_lg, f_y, sf, _sl, s_y = split_fit_select(tr_f, tr_lg, tr_y)
        log(f"fold {held}: FIT {len(f_y)}, TEST {len(te_y)}, excluded {n_excl}")
        if a.dry_run:
            continue
        scores = methods(ff, f_lg, f_y, np.concatenate(te_f, axis=1), te_lg, a.dims, a.probe_dims,
                         a.skip_probe, a.skip_fsd, log)
        aucs = {k: float(compute_auc(te_y, v)) for k, v in scores.items()}
        for k, v in aucs.items():
            log(f"  {k:<24} {v:.4f}")
        sp = _cr_paths.out_path(out_dir / "scores" / f"{a.pool}_mechanism_{held}.npz")
        np.savez_compressed(sp, labels=np.asarray(te_y, dtype=np.int8), **scores)
        rec = {"held_out": held, "n_fit": int(len(f_y)), "n_test": int(len(te_y)), "n_select": int(len(s_y)),
               "fit_fingerprint": membership_fingerprint(ff), "select_fingerprint": membership_fingerprint(sf),
               "n_excluded_training_rows": int(n_excl), "auroc": aucs, "scores_file": sp.relative_to(out_dir).as_posix(),
               "fold_seconds": round(time.time() - t0, 1)}
        write_json(rec, ckpt)
        folds_out.append(rec)
    if a.dry_run:
        return 0
    if sorted(r["held_out"] for r in folds_out) != sorted(TEST_GENERATORS):   # version 4: exactly the eight
        log(f"{len({r['held_out'] for r in folds_out})} of {len(TEST_GENERATORS)} folds done; run all folds to "
            "write the summary")
        return 0
    order = {g: i for i, g in enumerate(TEST_GENERATORS)}
    folds_out.sort(key=lambda r: order[r["held_out"]])
    names = list(folds_out[0]["auroc"])
    res = {"run": "C_v2_logo", "runner": "05_run_c_width_matched_v5", "pool": a.pool, "experts": experts,
           "exclusion": a.exclusion,
           "exclusion_manifest": _cr_paths.rel(manifest_path),
           "mean_auroc": {k: float(np.mean([r["auroc"][k] for r in folds_out])) for k in names},
           "per_fold": folds_out, "environment": _cr_paths.environment(),
           "elapsed_sec": round(time.time() - t_start, 1)}
    out = write_json(res, out_dir / f"run_c_v2_logo_{a.pool}.json")
    log("LOGO means: " + ", ".join(f"{k} {v:.4f}" for k, v in res["mean_auroc"].items()))
    log(f"saved {_cr_paths.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
