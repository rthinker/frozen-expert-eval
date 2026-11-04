"""LOGO richer prediction-space inputs with the heads of the standard split, version 2.

Version 2 (the supplied outputs come from version 1): every fold record
also holds the membership fingerprints of FIT and SELECT (``fit_fingerprint``, ``select_fingerprint``,
``scripts/runs/ladder_checks.py``), so that the summary (``summarize_v2.py``) can check that each fold used the rows
of the primary LOGO run it is analysed with; a fold already in --out-dir without them is not reused. Every
computation is that of version 1. The supplied primary runs record no fingerprints, so the summary analyses the
folds of this version with primary LOGO runs rerun by the current runners (the reproducibility harness, block B).

Version 1, as before:

LOGO richer prediction-space inputs with the heads of the standard split, for comparison with the primary LOGO runs.

The primary LOGO richer-input runs (``scripts/runs/08_logo_extended_rungs_v3.py``, described in the paper's Appendix
F) differ from the standard split (Table A3, App. B) in three ways: the logistic head is not class-balanced (the
standard split: "as PS2", which is balanced); the XGBoost heads use 3-fold CV over reduced grids of 6 ("capped") and
12 ("widened") configurations (the standard split: the PS4 grid, 108 configurations, 5-fold CV inside TRAIN); and
PS_K, the CV-selected input, is not run, so the ceiling takes 22 candidates (the ten ladder methods and 4 maps x 3
heads) instead of 20. The paper reports this check in Appendix A.

This script runs the standard split's version in every LOGO fold: the four maps and both heads with the functions of the
standard-split implementation (``scripts/analyses/appB_richer_inputs.py``: feat_disagreement,
make_uncertainty_featmap, feat_quadratic, make_rank_featmap, fit_logreg, fit_xgb(mode="cv"), cv_auc), and PS_K as
there (the map with the best 5-fold CV AUROC of the logistic head inside FIT, under both heads). The folds are those
of the primary content-disjoint runs (``logo_common.fold_view`` with each pool's exclusion list; split_fit_select, 15%
SELECT, seed 1024). ``summarize_v2.py`` then recomputes each pool's ceiling over these 20 methods (the ten ladder
methods of the primary run and PS_G to PS_K under both heads) with ``full_ladder_stats_v4.logo_block``.

The supplied outputs are in camera_ready/output/results/sensitivity/paper_heads/. A rerun writes to --out-dir
(default camera_ready/output/results/sensitivity_rerun/paper_heads/), never into the supplied outputs, the primary
results, the inputs or the feature cache, however the folder is spelled (``_out_guard.py``). Folds run in parallel
processes, each limited to --threads OpenMP threads. Each fold writes its predictions (``<pool>/<pool>_<fold>.npz``)
and then its record (``<pool>/<pool>_<fold>.json``), each through a temporary file renamed into place; the record
holds a fingerprint of everything the fold's result depends on (the pool's logits and the labels in the feature
cache, its exclusion list, the code the fold runs with the reviewed package and its configuration, the package
versions) and the SHA-256 of its predictions.

A fold already in --out-dir is reused only when its record and predictions are complete and consistent (the record
names this fold and its predictions file, which has the recorded SHA-256; the test labels are the held-out
generator's labels in the feature cache; every score array is finite, of the test size, and gives the recorded
AUROC) and its fingerprint equals the current one. Otherwise its files are moved aside to
``<out-dir>/moved_aside/<time>/`` with the reason (nothing is deleted) and the fold is recomputed. ``--plan`` prints
which folds would be reused or recomputed, and why, and changes nothing; ``--no-reuse`` recomputes every fold. Needs
the feature cache. From the repository root:

    python camera_ready/scripts/sensitivity/paper_heads_logo_v2.py --workers 4 --threads 4
    python camera_ready/scripts/sensitivity/paper_heads_logo_v2.py --pools arch6 --folds whichisreal     # one fold
    python camera_ready/scripts/sensitivity/paper_heads_logo_v2.py --plan                           # changes nothing
"""
import argparse
import hashlib
import io
import json
import math
import os
import platform
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _out_guard import output_dir  # noqa: E402

CR = Path(__file__).resolve().parents[2]          # camera_ready/
ROOT = CR.parent
POOLS = ("arch6", "clip4", "strong4", "weak6")
GENS = ["whichisreal", "CollabDiff", "MidJourney", "heygen", "starganv2", "deepfacelab", "stargan", "styleclip"]
RERUN = "camera_ready/output/results/sensitivity_rerun/paper_heads"
MAPS = ("G_disagreement", "H_uncertainty_mi", "I_quadratic", "J_rank_ecdf")
SCORES = tuple(f"{m}_{h}" for m in MAPS for h in ("LR", "XGB"))
AUC_KEYS = SCORES + ("K_cv_selected_LR", "K_cv_selected_XGB")
AUC_TOL = 1e-6                                    # the AUROCs are computed from the float32 arrays that are saved
CACHE = ROOT / "output" / "cached_features"
EXCLUSIONS = CR / "output" / "results" / "mitigation" / "exclusions"
CODE = (Path(__file__).resolve(), CR / "scripts" / "analyses" / "appB_richer_inputs.py",
        CR / "scripts" / "analyses" / "_common.py", CR / "scripts" / "analyses" / "_paths.py",
        CR / "scripts" / "runs" / "logo_common.py", CR / "scripts" / "runs" / "_cr_paths.py",
        CR / "scripts" / "runs" / "ladder_checks.py",                     # version 2
        ROOT / "scripts" / "_bootstrap.py")      # plus every module of src/ and configs/*.yaml (fingerprints)
PACKAGES = ("numpy", "scipy", "scikit-learn", "xgboost")


def write_atomic(path, data):
    """Write bytes through a temporary file in the same folder, then rename it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_fold(out, pool, fold, rec, scores, labels, fingerprint):
    """The predictions first, then the record that names them, with the fingerprint and the predictions' SHA-256."""
    import numpy as np
    buf = io.BytesIO()
    np.savez_compressed(buf, labels=np.asarray(labels, dtype=np.int8), **scores)
    data = buf.getvalue()
    d = Path(out) / pool
    write_atomic(d / f"{pool}_{fold}.npz", data)
    rec = {**rec, "scores_file": f"{pool}/{pool}_{fold}.npz",
           "provenance": {"fingerprint": fingerprint, "scores_sha256": hashlib.sha256(data).hexdigest()}}
    write_atomic(d / f"{pool}_{fold}.json", json.dumps(rec, indent=2).encode("utf-8"))


def fold_task(out_str, pool, fold, threads, fingerprint):
    """One LOGO fold: PS_G to PS_K under both heads, as on the standard split. Runs in a worker process."""
    if threads:
        os.environ["OMP_NUM_THREADS"] = str(threads)
    for p in (CR / "scripts" / "runs", CR / "scripts" / "analyses"):
        sys.path.insert(0, str(p))
    import _cr_paths
    _cr_paths.setup(str(ROOT))
    import numpy as np
    import appB_richer_inputs as B
    from logo_common import fold_view, load_content_manifest, load_generators, split_fit_select
    from ladder_checks import membership_fingerprint                    # version 2
    from src.utils.config import CACHE_ROOT, get_pool_experts
    from src.utils.metrics import compute_auc

    t0 = time.time()
    experts = get_pool_experts(pool)
    data = load_generators(experts, CACHE_ROOT)
    manifest = load_content_manifest(EXCLUSIONS / f"logo_exclusions_{pool}.json", pool, experts, data)
    _tr_f, tr_lg, tr_y, _te_f, te_lg, te_y, n_excl = fold_view(data, fold, manifest.get(fold))
    _ff, f_lg, f_y, _sf, s_lg, s_y = split_fit_select(_tr_f, tr_lg, tr_y, 0.15, 1024)
    unc_map, temps = B.make_uncertainty_featmap(s_lg, s_y)
    maps = {"G_disagreement": B.feat_disagreement, "H_uncertainty_mi": unc_map, "I_quadratic": B.feat_quadratic,
            "J_rank_ecdf": B.make_rank_featmap(f_lg)}
    scores, cv, n_features, seconds = {}, {}, {}, {}
    for name, fm in maps.items():
        t1 = time.time()
        Xf, Xt = fm(f_lg), fm(te_lg)
        n_features[name] = int(Xf.shape[1])
        scores[f"{name}_LR"] = np.asarray(B.fit_logreg(Xf, f_y, Xt), dtype=np.float32)
        scores[f"{name}_XGB"] = np.asarray(B.fit_xgb(Xf, f_y, Xt, mode="cv"), dtype=np.float32)
        cv[name] = B.cv_auc(Xf, f_y)
        seconds[name] = round(time.time() - t1, 1)
    k = max(cv, key=cv.get)
    auc = {n: float(compute_auc(te_y, s)) for n, s in scores.items()}
    auc["K_cv_selected_LR"], auc["K_cv_selected_XGB"] = auc[f"{k}_LR"], auc[f"{k}_XGB"]
    rec = {"pool": pool, "held_out": fold, "n_fit": int(len(f_y)), "n_select": int(len(s_y)), "n_test": int(len(te_y)),
           "n_excluded_training_rows": int(n_excl), "fit_fingerprint": membership_fingerprint(_ff),   # version 2
           "select_fingerprint": membership_fingerprint(_sf), "auc": auc, "cv_auc_logistic_head": cv,
           "k_selected": k,
           "n_features": n_features, "temperatures": [float(x) for x in temps], "seconds_per_map": seconds,
           "fold_seconds": round(time.time() - t0, 1), "omp_threads": threads,
           "implementation": "appB_richer_inputs.py (fit_logreg, fit_xgb mode cv, cv_auc, maps); logo_common folds"}
    write_fold(out_str, pool, fold, rec, scores, te_y, fingerprint)
    return pool, fold, rec["fold_seconds"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pool_experts(pool):
    """The pool's experts, as the folds read them (`src.utils.config`, from configs/experts.yaml)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.utils.config import get_pool_experts
    return list(get_pool_experts(pool))


def fingerprints(pools, folds):
    """{(pool, fold): SHA-256 of what the fold's result depends on}; SystemExit when an input file is missing."""
    code = [*CODE, *sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "configs").glob("*.yaml"))]
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    base = {"code": {p.relative_to(ROOT).as_posix(): _sha256(p) for p in code}, "packages": packages,
            "python": platform.python_version()}
    out = {}
    for pool in pools:
        files = [EXCLUSIONS / f"logo_exclusions_{pool}.json"]
        for g in GENS:                            # every fold trains on the other seven held-out generators
            files += [CACHE / g / "labels.npy"] + [CACHE / g / f"{e}_logits.npy" for e in _pool_experts(pool)]
        missing = [f for f in files if not f.is_file()]
        if missing:
            raise SystemExit(f"needs the feature cache and the exclusion lists: {len(missing)} input file(s) missing, "
                             f"for example {missing[0].relative_to(ROOT).as_posix()}")
        inputs = {f.relative_to(ROOT).as_posix(): _sha256(f) for f in files}
        for fold in folds:
            doc = json.dumps({**base, "inputs": inputs, "pool": pool, "held_out": fold}, sort_keys=True)
            out[pool, fold] = hashlib.sha256(doc.encode("utf-8")).hexdigest()
    return out


def check_fold(out, pool, fold, fingerprint, truth=None):
    """Why the fold stored in `out` cannot be reused (None when it can). `truth`: the held-out generator's labels
    (default: read from the feature cache)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    jp, zp = Path(out) / pool / f"{pool}_{fold}.json", Path(out) / pool / f"{pool}_{fold}.npz"
    try:
        rec = json.loads(jp.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return f"record unreadable ({type(e).__name__})"
    if not isinstance(rec, dict) or rec.get("pool") != pool or rec.get("held_out") != fold:
        return "record of another fold"
    prov = rec.get("provenance") if isinstance(rec.get("provenance"), dict) else {}
    if not prov.get("fingerprint"):
        return "record without a fingerprint (written by an earlier version)"
    if not (rec.get("fit_fingerprint") and rec.get("select_fingerprint")):                    # version 2
        return "record without the membership fingerprints of FIT and SELECT (written by version 1)"
    if prov["fingerprint"] != fingerprint:
        return "inputs, code or packages changed since the fold ran (fingerprint)"
    if rec.get("scores_file") != f"{pool}/{pool}_{fold}.npz":
        return "record names another predictions file"
    if not zp.is_file():
        return "predictions file missing"
    data = zp.read_bytes()
    if hashlib.sha256(data).hexdigest() != prov.get("scores_sha256"):
        return "predictions file is not the one the record was written with"
    auc = rec.get("auc")
    if not isinstance(auc, dict) or set(auc) != set(AUC_KEYS) or not all(
            isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0 for v in auc.values()):
        return "AUROC record incomplete or invalid"
    k = rec.get("k_selected")
    if k not in MAPS or (auc["K_cv_selected_LR"], auc["K_cv_selected_XGB"]) != (auc[f"{k}_LR"], auc[f"{k}_XGB"]):
        return "PS_K record inconsistent with its selected map"
    try:
        with np.load(io.BytesIO(data)) as z:
            arrays = {name: z[name] for name in z.files}
    except Exception as e:  # noqa: BLE001  (a damaged archive can raise several error types)
        return f"predictions file unreadable ({type(e).__name__})"
    if truth is None:
        lp = CACHE / fold / "labels.npy"
        if not lp.is_file():
            return "feature cache missing (the held-out labels cannot be checked)"
        truth = np.load(lp)
    n, y = rec.get("n_test"), arrays.get("labels")
    if y is None or y.ndim != 1 or n != len(y) or not np.array_equal(y, truth):
        return "test labels are not the held-out generator's labels in the feature cache"
    for name in SCORES:
        s = arrays.get(name)
        if s is None or s.ndim != 1 or len(s) != n or not np.all(np.isfinite(s)):
            return f"scores {name} missing, not finite or not of the test size"
        if abs(float(roc_auc_score(y, s)) - auc[name]) > AUC_TOL:
            return f"scores {name} do not give the recorded AUROC"
    return None


def plan(out, pools, folds, fps, reuse=True, apply=True, truth=None):
    """([(pool, fold, reason)] to run, [(pool, fold)] reused). Files of a fold that is not reused, and temporary files
    left by an interrupted write, are moved aside (`apply`), never deleted. `truth`: fold -> held-out labels (tests)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run, reused = [], []

    def move_aside(pool, fold, files, why):
        dest = Path(out) / "moved_aside" / stamp / pool
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.move(str(f), str(dest / f.name))
        with open(dest / f"{pool}_{fold}.reason.txt", "a", encoding="utf-8") as fh:
            fh.write(why + "\n")

    for p in pools:
        for g in folds:
            d = Path(out) / p
            pair = [f for f in (d / f"{p}_{g}.json", d / f"{p}_{g}.npz") if f.exists()]
            leftovers = sorted(d.glob(f"{p}_{g}.*.tmp-*")) if d.is_dir() else []
            if not pair and not leftovers:
                run.append((p, g, "not run yet"))
                continue
            if not (d / f"{p}_{g}.json").exists():
                why = "interrupted (predictions or temporary files without a record)"
            elif not reuse:
                why = "recomputed on request (--no-reuse)"
            else:
                why = check_fold(out, p, g, fps[p, g], None if truth is None else truth(g))
            if why is None:
                reused.append((p, g))
                if leftovers and apply:
                    move_aside(p, g, leftovers, "temporary files of an interrupted write (the record and "
                                                "predictions are complete and were reused)")
                continue
            if apply:
                move_aside(p, g, pair + leftovers, why)
            run.append((p, g, why))
    return run, reused


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pools", nargs="+", default=list(POOLS), choices=list(POOLS))
    ap.add_argument("--folds", nargs="+", default=GENS, choices=GENS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="OpenMP threads per worker (0: library default)")
    ap.add_argument("--out-dir", default=RERUN, help=f"relative to the repository root (default {RERUN})")
    ap.add_argument("--plan", action="store_true",
                    help="print which folds would be reused or recomputed; change nothing")
    ap.add_argument("--no-reuse", action="store_true", help="recompute every fold (its files are moved aside first)")
    a = ap.parse_args()
    out = output_dir(a.out_dir)
    fps = fingerprints(a.pools, a.folds)
    todo, reused = plan(out, a.pools, a.folds, fps, reuse=not a.no_reuse, apply=not a.plan)
    for p, g in reused:
        print(f"  reuse  {p:<8} {g:<12}")
    for p, g, why in todo:
        print(f"  run    {p:<8} {g:<12} {why}")
    print(f"{len(todo)} fold(s) to run, {len(reused)} reused; {a.workers} worker(s), {a.threads or 'default'} "
          "thread(s) each" + (" (plan only: nothing changed)" if a.plan else ""), flush=True)
    if a.plan:
        return 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures = [ex.submit(fold_task, str(out), p, g, a.threads, fps[p, g]) for p, g, _ in todo]
        for f in as_completed(futures):
            pool, fold, sec = f.result()
            why = check_fold(out, pool, fold, fps[pool, fold])
            if why is not None:
                raise SystemExit(f"{pool} {fold}: the fold just written fails its own check: {why}")
            print(f"  {pool:<8} {fold:<12} {sec / 60:6.1f} min   (elapsed {(time.time() - t0) / 60:.1f} min)",
                  flush=True)
    print("summary: python camera_ready/scripts/sensitivity/summarize_v2.py --dir "
          + Path(a.out_dir).parent.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
