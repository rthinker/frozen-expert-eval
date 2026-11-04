"""Compare LOGO result files fold by fold and method by method, with portable reviewed references.

Version 2 keeps the comparisons and tolerances of ``compare_runs.py`` (computationally preserved). It uses the packaged
``camera_ready/inputs/`` references when present; ``--reference-dir`` overrides them. Exclusion manifests default
to this code's ``output/results/mitigation/exclusions/``; ``--exclusion-dir`` overrides that location.

g0  New environment vs the stored reviewed runs (both without exclusion). Deterministic sklearn/scipy methods must
    agree within 1e-6, XGBoost-based ones within 1e-3; the MLP-based methods (PS_3, PS_D, PS_E, PS_F, FS2) are only
    reported (their drift is expected after a hardware change).
g2  Content-disjoint run vs the G0 run (same environment). In the folds whose training data lost no row, every
    method must be identical (1e-9); the folds that lost rows are reported.
g5  Mechanism runner (05_run_c_width_matched_v5.py, LOGO) vs the content-disjoint LOGO run: FS1 and PS_4 per fold
    must be identical (1e-9). Tolerance recorded on 10 October 2026, after arch6 failed at 1e-9 by 7.5e-6 on FS1 only:
    PS_4 stays at 1e-9 and FS1 is judged at --fs1-tol 1e-4; the per-fold FS1 differences are written to the report.
    Precision check, 11 Oct 2026: all 16 saved FS1 score arrays (arch6 and clip4) are identical between runners.
    The ladder scores AUROC before its float32 conversion; the mechanism scores afterward. Scoring the stored
    float32 arrays reproduces the observed metric gap; the earlier process-drift attribution is unsupported.
    See ``../review/check_fs1_score_precision_v2.py`` and ``output/results/mitigation/SCORE_PRECISION_NOTE.md``.
g0log  For pools without a reviewed LOGO result JSON (clip4, strong4, weak6): new run vs the published numerical
    excerpts of the reviewed logs (4 decimals). Deterministic sklearn/scipy methods must equal the printed value
    (|diff| <= 5.1e-5); PS_4 and FS2 are reported only. The historical PS_4 runs used a wider grid (max_depth up to
    10) than the frozen 108-configuration grid. PS_4 did not set any fold's PS ceiling in the reviewed records.
    With --untouched-only, only the folds whose training data lost no row are judged (in a content-disjoint run
    they use exactly the reviewed inputs); the other folds are reported.
p0  Compare the fold checkpoints of a short run (``folds/``, exactly the folds named by --folds, or all eight) with
    the stored reviewed run. Deterministic methods must agree within 1e-6 and PS_4 within 1e-3. The report shows
    whether the MLP-based methods match exactly. Use --quick-pilot for checkpoints made with --skip_slow.
    With --expect-gpu, every checkpoint must name that GPU.

Exit status 1 if the gate fails. A JSON report is written next to the first input unless --out is given.

Usage from the repository root. By hand, --out must name a file outside the stored results, as here; the harness
steps in camera_ready/RELEASE_MANIFEST.md write the stored reports:
    python -B camera_ready/scripts/runs/compare_runs_v2.py g0 --pool arch6 \
        --new-dir camera_ready/output/results/mitigation/g0_reproduce --out camera_ready/output/checks/gate_G0_arch6.json
    python -B camera_ready/scripts/runs/compare_runs_v2.py g2 --pool arch6 \
        --ref-dir camera_ready/output/results/mitigation/g0_reproduce \
        --new-dir camera_ready/output/results/mitigation/content_disjoint --out camera_ready/output/checks/gate_G2_arch6.json
    python -B camera_ready/scripts/runs/compare_runs_v2.py g5 --pool arch6 \
        --logo-dir camera_ready/output/results/mitigation/content_disjoint \
        --mech-dir camera_ready/output/results/mitigation/mechanism --fs1-tol 1e-4 \
        --out camera_ready/output/checks/gate_G5_arch6.json
    python -B camera_ready/scripts/runs/compare_runs_v2.py g0log --pool clip4 --proj_dim 32 \
        --new-dir camera_ready/output/results/mitigation/content_disjoint --untouched-only \
        --log camera_ready/inputs/A4_logo_clip4.txt --out camera_ready/output/checks/gate_G0log_clip4.json
    python -B camera_ready/scripts/runs/compare_runs_v2.py p0 --pool arch6 \
        --new-dir camera_ready/output/results/mitigation/p0_5080 --expect-gpu 5080 --folds starganv2 \
        --quick-pilot --out camera_ready/output/checks/p0_5080_comparison.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cr_paths  # noqa: E402
from ladder_checks import RICHER_CANDIDATES

GENS = ("whichisreal", "CollabDiff", "MidJourney", "heygen", "starganv2", "deepfacelab", "stargan", "styleclip")

EXACT = {"PS_1_prob_avg", "PS_A_calibrated_avg", "PS_B_learned_weights", "PS_2_logreg", "PS_C_logreg_cv",
         "raw_feat_auc"}
XGB = {"PS_4_xgboost"}
STOCHASTIC = {"PS_3_mlp_logits", "PS_D_mlp_probs", "PS_E_mlp_probs_es", "PS_F_mlp_logits_es", "fs_mlp_mean"}

# The excerpts retain the four-decimal values from the reviewed LOGO records.
REVIEWED_EVIDENCE = {"arch6": "A3_logo_arch6.txt", "clip4": "A4_logo_clip4.txt", "weak6": "A4_logo_weak6.txt",
                     "strong4": "A5_logo_strong4.txt"}
LOG_FOLD = re.compile(r"held-out=(\w+)")
LOG_PS = re.compile(r"(?:^|\|\s+INFO\s+\|\s+)(PS_[0-9A-Z]_\w+?)\s+(\d\.\d{4})\s*$")
LOG_FS1 = re.compile(r"Raw-feat LogReg:\s+(\d\.\d{4})")
LOG_FS2 = re.compile(r"FS MLP seed-avg:\s+(\d\.\d{4})")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def read_log(p):
    raw = Path(p).read_bytes()
    return raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")


def log_values(p):
    """{fold: {method: AUROC}} as printed (4 decimals) by a reviewed 11_disjoint_data.py log; per-seed lines skipped."""
    out, fold = {}, None
    for line in read_log(p).splitlines():
        m = LOG_FOLD.search(line)
        if m:
            fold = m.group(1)
            if fold in out:
                raise ValueError(f"duplicate fold in log: {fold}")
            out[fold] = {}
            continue
        if fold is None:
            continue
        m = LOG_PS.search(line)
        if m:
            if not re.search(r"_s\d+$", m.group(1)):
                out[fold][m.group(1)] = float(m.group(2))
            continue
        m = LOG_FS1.search(line)
        if m:
            out[fold]["raw_feat_auc"] = float(m.group(1))
            continue
        m = LOG_FS2.search(line)
        if m:
            out[fold]["fs_mlp_mean"] = float(m.group(1))
    return out


def logo_values(summary):
    """{fold: {method: AUROC}} from an 11_disjoint (reviewed or v2) summary."""
    out = {}
    for f in summary["per_fold"]:
        v = dict(f["ps_means"])
        v["raw_feat_auc"] = f["raw_feat_auc"]
        if f.get("fs_mlp_mean") is not None:
            v["fs_mlp_mean"] = f["fs_mlp_mean"]
        if f["held_out"] in out:
            raise ValueError(f"duplicate fold: {f['held_out']}")
        out[f["held_out"]] = v
    return out


def checkpoint_records(run_dir, pool, proj_dim):
    """Per-fold checkpoints (``folds/<pool>_d<proj>_<fold>.json``) of a run, finished or not."""
    return [load(p) for p in sorted((Path(run_dir) / "folds").glob(f"{pool}_d{proj_dim}_*.json"))]


def rung_values(rungs):
    return {g: dict(r["rungs"]) for g, r in rungs["folds"].items()}


def group(method):
    if method in EXACT or method.endswith("_logreg"):
        return "exact"
    if method in XGB or "_xgb_" in method:
        return "xgboost"
    if method in STOCHASTIC:
        return "stochastic"
    return "other"


ALL_METHODS = EXACT | XGB | STOCHASTIC
LOG_METHODS = EXACT | XGB | {"fs_mlp_mean"}
QUICK_PILOT_METHODS = ALL_METHODS - {"PS_C_logreg_cv", "PS_4_xgboost", "PS_E_mlp_probs_es", "PS_F_mlp_logits_es"}


def require_values(values, folds, methods, label):
    if set(values) != set(folds) or not folds:
        raise ValueError(f"{label}: expected folds {sorted(folds)}, got {sorted(values)}")
    for fold in folds:
        if set(values[fold]) != set(methods) or not methods:
            raise ValueError(f"{label}/{fold}: incomplete or unexpected method set")
        for method, value in values[fold].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{label}/{fold}/{method}: not a finite AUROC: {value!r}")


def checked_exclusions(path):
    records = load(path)["folds"]
    if len(records) != len(GENS) or {r["held_out"] for r in records} != set(GENS):
        raise ValueError("exclusion manifest must contain exactly the eight expected folds")
    for row in records:
        if type(row.get("n_excluded")) is not int or row["n_excluded"] < 0:
            raise ValueError("invalid exclusion count")
    return {r["held_out"] for r in records if r["n_excluded"] > 0}


def diff_table(new, ref, folds=None, methods=None, expected_folds=GENS):
    methods = set(ALL_METHODS if methods is None else methods)
    require_values(new, expected_folds, methods, "new")
    require_values(ref, expected_folds, methods, "reference")
    selected = list(expected_folds if folds is None else folds)
    if not selected or not set(selected) <= set(expected_folds) or len(set(selected)) != len(selected):
        raise ValueError("no valid, nonempty set of folds to compare")
    return [{"fold": g, "method": m, "group": group(m), "new": new[g][m], "ref": ref[g][m],
             "abs_diff": abs(new[g][m] - ref[g][m])} for g in selected for m in sorted(methods)]


def summarise(rows, tol):
    if not rows:
        raise ValueError("no comparisons")
    worst = {}
    for r in rows:
        if any(not math.isfinite(r[k]) for k in ("new", "ref", "abs_diff")):
            raise ValueError("nonfinite gate evidence")
        if not 0 <= r["new"] <= 1 or not 0 <= r["ref"] <= 1:
            raise ValueError("AUROC outside [0, 1]")
        k = r["method"]
        if k not in worst or r["abs_diff"] > worst[k]["abs_diff"]:
            worst[k] = r
    status = {}
    for k, r in sorted(worst.items()):
        t = tol.get(r["group"])
        status[k] = {"group": r["group"], "max_abs_diff": r["abs_diff"], "fold": r["fold"],
                     "checked_folds": sum(x["method"] == k for x in rows),
                     "status": "report" if t is None else ("PASS" if r["abs_diff"] <= t else "FAIL")}
        print(f"  {k:<34} max |diff| {r['abs_diff']:.2e} ({r['fold']}) {status[k]['status']}")
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gate", choices=("g0", "g2", "g5", "g0log", "p0"))
    ap.add_argument("--pool", default="arch6")
    ap.add_argument("--proj_dim", type=int, default=256)
    for option in ("new-dir", "ref-dir", "logo-dir", "mech-dir", "reference-dir", "exclusion-dir", "out"):
        ap.add_argument("--" + option, type=Path)
    ap.add_argument("--log", type=Path, help="g0log: reviewed numerical excerpt or original log")
    ap.add_argument("--untouched-only", action="store_true")
    ap.add_argument("--expect-gpu")
    ap.add_argument("--fs1-tol", type=float, default=1e-9)
    ap.add_argument("--folds", nargs="+", choices=GENS, help="p0: exact expected pilot folds; default all eight")
    ap.add_argument("--quick-pilot", action="store_true", help="p0: compare checkpoints made with --skip_slow")
    a = ap.parse_args()
    if a.folds is not None and (a.gate != "p0" or len(set(a.folds)) != len(a.folds)):
        ap.error("--folds must be distinct and is only for p0")
    if a.quick_pilot and a.gate != "p0":
        ap.error("--quick-pilot is only for p0")
    if not math.isfinite(a.fs1_tol) or not 0 <= a.fs1_tol <= 1:
        ap.error("--fs1-tol must be finite and between 0 and 1")
    code = Path(__file__).resolve().parents[2]
    reference = a.reference_dir or code / "inputs"
    exclusions = a.exclusion_dir or _cr_paths.EXCLUSION_ROOT
    report = {"gate": a.gate, "pool": a.pool, "checked": {"folds": list(GENS)}}
    name = f"11_disjoint_{a.pool}_d{a.proj_dim}.json"
    rname = f"A3c_logo_extended_rungs_{a.pool}.json"
    # Guard destinations before reading inputs or doing work.
    output_dir = a.mech_dir if a.gate == "g5" else a.new_dir
    if output_dir is None:
        ap.error("provide --mech-dir for g5, or --new-dir for the other gates")
    label = {"g0": "G0", "g2": "G2", "g5": "G5", "g0log": "G0log", "p0": "P0"}[a.gate]
    out = _cr_paths.out_path(a.out or output_dir / f"gate_{label}_{a.pool}.json")
    st = {}
    dev_ok = True
    if a.gate in ("g0", "g2"):
        refdir = reference if a.gate == "g0" else a.ref_dir
        if refdir is None:
            ap.error("g2 requires --ref-dir")
        new, ref = logo_values(load(a.new_dir / name)), logo_values(load(refdir / name))
        rn, rr = rung_values(load(a.new_dir / rname)), rung_values(load(refdir / rname))
        touched = checked_exclusions(exclusions / f"logo_exclusions_{a.pool}.json") if a.gate == "g2" else set()
        judged = [g for g in GENS if g not in touched]
        tol = {"exact": 1e-6, "xgboost": 1e-3} if a.gate == "g0" else dict.fromkeys(("exact", "xgboost", "stochastic", "other"), 1e-9)
        st = summarise(diff_table(new, ref, judged), tol)
        st2 = summarise(diff_table(rn, rr, judged, methods=RICHER_CANDIDATES), tol)
        if a.gate == "g0":
            report.update(logo=st, rungs=st2)
        else:
            report["changed_folds"] = summarise(diff_table(new, ref, sorted(touched)), {}) if touched else {}
            report["changed_rungs"] = summarise(diff_table(rn, rr, sorted(touched), methods=RICHER_CANDIDATES), {}) if touched else {}
            report["untouched_folds"] = {**st, **{f"rung:{k}": v for k, v in st2.items()}}
        st = {**st, **{f"rung:{k}": v for k, v in st2.items()}}
        report["checked"].update(logo=96, rungs=96, judged_folds=judged)
    elif a.gate == "g0log":
        log_path = a.log or reference / REVIEWED_EVIDENCE[a.pool]
        if a.log is None and a.reference_dir is not None and not log_path.is_file():
            log_path = log_path.with_suffix(".log")
        new, ref = logo_values(load(a.new_dir / name)), log_values(log_path)
        require_values(new, GENS, ALL_METHODS, "new")
        new = {g: {m: v[m] for m in LOG_METHODS} for g, v in new.items()}
        touched = checked_exclusions(exclusions / f"logo_exclusions_{a.pool}.json") if a.untouched_only else set()
        judged = [g for g in GENS if g not in touched]
        if len(judged) < 6:
            raise ValueError("g0log needs at least six judged folds")
        st = summarise(diff_table(new, ref, judged, methods=LOG_METHODS), {"exact": 5.1e-5})
        if touched:
            report["changed_folds"] = summarise(diff_table(new, ref, sorted(touched), methods=LOG_METHODS), {})
        report.update(log=log_path.name, judged_folds=judged, values=st)
        report["checked"].update(logo=64, judged_folds=judged)
    elif a.gate == "p0":
        recs = checkpoint_records(a.new_dir, a.pool, a.proj_dim)
        new, ref = logo_values({"per_fold": recs}), logo_values(load(reference / name))
        require_values(ref, GENS, ALL_METHODS, "reference")
        folds = a.folds or list(GENS)
        methods = QUICK_PILOT_METHODS if a.quick_pilot else ALL_METHODS
        ref = {g: {method: ref[g][method] for method in methods} for g in folds}
        st = summarise(diff_table(new, ref, methods=methods, expected_folds=folds),
                       {"exact": 1e-6, "xgboost": 1e-3})
        devices = {r["held_out"]: r.get("device_name", "not recorded") for r in recs}
        dev_ok = a.expect_gpu is None or all(a.expect_gpu.lower() in d.lower() for d in devices.values())
        mlp = {k: v["max_abs_diff"] for k, v in st.items() if v["group"] == "stochastic"}
        report.update(folds=folds, values=st, devices=devices, expect_gpu=a.expect_gpu,
                      mlp_max_abs_diff=mlp, mlp_bit_identical=all(x == 0 for x in mlp.values()))
        report["checked"] = {"folds": folds, "logo": len(folds) * len(methods)}
    else:
        logo = logo_values(load(a.logo_dir / name))
        require_values(logo, GENS, ALL_METHODS, "logo")
        records = load(a.mech_dir / f"run_c_v2_logo_{a.pool}.json")["per_fold"]
        mech = {f["held_out"]: f["auroc"] for f in records}
        if len(records) != len(GENS) or set(mech) != set(GENS):
            raise ValueError("mechanism: expected exactly eight distinct folds")
        rows = []
        for g in GENS:
            for method, group_name, mk, lk in (("FS_1", "fs1", "FS_1_raw_feat_logreg", "raw_feat_auc"), ("PS_4", "exact", "PS_4_xgboost", "PS_4_xgboost")):
                n, r = mech[g][mk], logo[g][lk]
                rows.append(dict(fold=g, method=method, group=group_name, new=n, ref=r, abs_diff=abs(n-r)))
        st = summarise(rows, {"exact": 1e-9, "fs1": a.fs1_tol})
        report.update(g5=st, fs1_tolerance=a.fs1_tol,
                      per_fold_fs1_abs_diff={r["fold"]: r["abs_diff"] for r in rows if r["method"] == "FS_1"})
        report["checked"]["mechanism"] = 16
    ok = bool(st) and dev_ok and all(v["status"] in ("PASS", "report") for v in st.values())
    report["status"] = "PASS" if ok else "FAIL"
    _cr_paths.atomic_write_text(out, json.dumps(report, indent=2, allow_nan=False))
    print(f"gate {a.gate.upper()}: {report['status']} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, TypeError) as error:
        raise SystemExit(f"Gate failed: {error}") from error
