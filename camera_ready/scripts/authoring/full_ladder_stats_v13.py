"""Compute the LOGO, width-control and Table 2 statistics from saved results.

The prediction-space ceiling is the strongest tested method on each held-out
generator. For arch6, clip4, strong4 and weak6, the script reports paired
margins, wins, two-sided paired tests, 95% percentile intervals from 10,000
fold resamples (seed 1024), sample-weighted macro results and AUPRC when
per-sample predictions are available. AUPRC uses the AUROC-selected comparator.
The paired sign test excludes ties. Table 2 statistics combine the primary
LOGO results with the standard-split statistics and reviewed results.

The harness uses this script in B31, B32 and C08a. To check the stored
statistics from the repository root:

    python -B camera_ready/scripts/authoring/full_ladder_stats_v13.py --self-test

For standalone analysis, supply --run-dir and --mech-dir for the saved inputs
and --out-dir for a separate output folder. Use --audit-only with --stats to
update the Table 2 statistics in a scratch copy of mitigation_stats.json.
--standard-stats and --reviewed-results select the standard-split inputs.

Full analysis requires complete, compatible LOGO and richer-input runs.
The --allow-ten-methods, --allow-partial-folds and --allow-incomplete-ladder
options permit explicitly labeled partial analyses. See
camera_ready/RELEASE_MANIFEST.md for the harness commands.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runs"))
import ladder_checks  # noqa: E402  version 11: completeness of the ladder, compatibility of paired runs

CODE = Path(__file__).resolve().parents[2]
CAMERA = CODE.parent
CH3 = CAMERA.parent
STORED_LOGO = CH3 / "01.initial_submission" / "codebase" / "output" / "results"
STORED_RUNGS = CH3 / "02.rebuttal" / "runs" / "reprod" / "results"
FULL_LADDER_V1 = CODE / "output" / "results" / "full_ladder_stats.json"
REVIEWED_RESULTS = CODE.parent / "output" / "results"      # version 4: the reviewed results (Block A), for the audit
AUDIT_TOL = 1e-12                                          # version 4: tolerance of the audit's identities
GENS = ["whichisreal", "CollabDiff", "MidJourney", "heygen", "starganv2", "deepfacelab", "stargan", "styleclip"]
BOUNDARY = ("clip4:32", "strong4:256", "weak6:256")      # version 6: the reviewed boundary pools (pool:proj_dim)
MECHANISM = (("mechanism_logo", "arch6", "256"), ("mechanism_logo_clip4", "clip4", "32"))   # version 6: key, pool, dim


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sign_test_p(deltas):
    """Two-sided paired sign test, excluding exact ties; all ties give p=1."""
    d = np.asarray(deltas, dtype=float)
    n_nonzero = int(np.count_nonzero(d))
    return (float(binomtest(int((d > 0).sum()), n_nonzero, 0.5, alternative="two-sided").pvalue)
            if n_nonzero else 1.0)


def sign_flip_p(d):
    d = np.asarray(d, dtype=float)
    obs = abs(d.mean())
    flips = [abs((d * np.array(s)).mean()) for s in itertools.product((1, -1), repeat=len(d))]
    return float(np.mean([f >= obs - 1e-15 for f in flips]))


def fold_stats(named_deltas):
    """Fold-level statistics for several delta vectors sharing one bootstrap stream (default_rng(1024))."""
    names = list(named_deltas)
    D = {k: np.asarray(v, dtype=float) for k, v in named_deltas.items()}
    n = len(next(iter(D.values())))
    rng = np.random.default_rng(1024)
    boot = {k: np.empty(10_000) for k in names}
    for i in range(10_000):
        idx = rng.integers(0, n, size=n)
        for k in names:
            boot[k][i] = D[k][idx].mean()
    out = {}
    for k in names:
        d = D[k]
        wins = int((d > 0).sum())
        out[k] = {"mean_delta": float(d.mean()),
                  "ci95": [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))],
                  "wins": wins, "n": n,
                  "sign_p": sign_test_p(d),
                  "wilcoxon_p": (float(wilcoxon(d, alternative="two-sided").pvalue) if np.any(d != 0) else 1.0),
                  "signflip_p": sign_flip_p(d),
                  "sd_population": float(d.std(ddof=0)), "sd_sample": float(d.std(ddof=1)),
                  "per_fold": [float(x) for x in d]}
    return out


def auprc(labels, scores):
    """Average precision, or None for a fold that holds one class (version 9; version 8 returned 0.0)."""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else None


def logo_block(logo, rungs, run_dir=None):
    """LOGO statistics against the full ladder and against the ten methods only."""
    folds = {f["held_out"]: f for f in logo["per_fold"]}
    order = [g for g in GENS if g in folds]
    fs1 = np.array([folds[g]["raw_feat_auc"] for g in order])
    have_fs2 = all(folds[g].get("fs_mlp_mean") is not None for g in order)
    fs2 = np.array([folds[g]["fs_mlp_mean"] for g in order]) if have_fs2 else None
    ten = np.array([folds[g]["best_ps_auc"] for g in order])
    full, method = [], []
    for g in order:
        best_name, best = folds[g]["best_ps_name"], folds[g]["best_ps_auc"]
        if rungs is not None:
            r = rungs["folds"][g]["rungs"]
            k = max(r, key=r.get)
            if r[k] > best:
                best_name, best = k, r[k]
        full.append(best)
        method.append(best_name)
    full = np.array(full)
    n_test = np.array([folds[g]["n_test"] for g in order], dtype=float)
    w = n_test / n_test.sum()
    res = {"folds": order, "n_test": [int(x) for x in n_test], "fs1": float(fs1.mean()),
           "fs2": float(fs2.mean()) if have_fs2 else None,
           "ceiling_full": float(full.mean()), "ceiling_full_per_fold": [float(x) for x in full],
           "ceiling_method_per_fold": method, "ceiling_ten": float(ten.mean())}
    deltas = {"fs1_full": fs1 - full, "fs1_ten": fs1 - ten}
    if have_fs2:
        deltas.update({"fs2_full": fs2 - full, "fs2_ten": fs2 - ten})
    st = fold_stats(deltas)
    res["stats"] = st
    big = n_test >= 500
    res["fs1_full_delta_n500"] = float((fs1 - full)[big].mean()) if big.any() else None      # version 9: not NaN
    res["weighted"] = {"fs1": float((w * fs1).sum()), "ceiling_full": float((w * full).sum()),
                       "fs2": float((w * fs2).sum()) if have_fs2 else None}
    # AUPRC from the stored per-sample scores (v2 runs only)
    if run_dir is not None:
        ap_fs1, ap_ceil, why = [], [], None
        for g, m in zip(order, method):
            f = folds[g]
            sf = Path(run_dir) / f.get("scores_file", "")
            if not f.get("scores_file") or not sf.exists():
                why = f"no per-sample scores for {g}"
                break
            s = np.load(sf)
            y = s["labels"]
            if m in s.files:
                ceil = s[m]
            else:                                    # version 9: the richer-input scores, checked
                rel = ((rungs or {}).get("folds", {}).get(g) or {}).get("scores_file")
                rsf = Path(run_dir) / rel if rel else None
                if rsf is None or not rsf.exists():
                    why = f"no richer-input scores for {g} (ceiling method {m})"
                    break
                rs = np.load(rsf)
                if m not in rs.files:
                    why = f"the richer-input scores of {g} hold no {m}"
                    break
                if "labels" in rs.files and not np.array_equal(rs["labels"], y):
                    why = f"the richer-input scores of {g} have other labels than the LOGO run's"
                    break
                ceil = rs[m]
            a_fs1, a_ceil = auprc(y, s["FS_1_raw_feat_logreg"]), auprc(y, ceil)
            if a_fs1 is None or a_ceil is None:     # version 9: a one-class fold has no AUPRC
                why = f"{g} holds one class"
                break
            ap_fs1.append(a_fs1)
            ap_ceil.append(a_ceil)
        if why is not None:
            res["auprc_note"] = f"not computed: {why}"
        else:
            res["auprc"] = {"fs1": float(np.mean(ap_fs1)), "ceiling_full": float(np.mean(ap_ceil)),
                            "delta": float(np.mean(ap_fs1) - np.mean(ap_ceil)),
                            "per_fold_fs1": ap_fs1, "per_fold_ceiling": ap_ceil}
    return res


def mechanism_block(mech, logo):
    folds = {f["held_out"]: f["auroc"] for f in mech["per_fold"]}
    order = [g for g in GENS if g in folds]
    lf = {f["held_out"]: f for f in logo["per_fold"]}
    A = {k: np.array([folds[g][k] for g in order]) for k in folds[order[0]]}
    # FS1 per fold comes from the ladder run (one FS1 per fold in the paper); the mechanism run's
    # own FS1 AUROC uses float32 scores (the ladder scores before casting); it is kept only as a check.
    # check_fs1_score_precision.py confirms that the saved score arrays are identical between the runners.
    fs1_mech = A["FS_1_raw_feat_logreg"]
    A["FS_1_raw_feat_logreg"] = np.array([lf[g]["raw_feat_auc"] for g in order])
    fs1_check = {"fs1_source": "ladder run (raw_feat_auc of 11_disjoint_data_v2.py)",
                 "fs1_mechanism_run_max_abs_diff": float(np.max(np.abs(fs1_mech - A["FS_1_raw_feat_logreg"])))}
    comps = {}
    ps4 = A["PS_4_xgboost"]
    fs1 = A["FS_1_raw_feat_logreg"]
    for k in A:
        if k.startswith("FS_pca") or k == "FS_D_xgb_features":
            comps[f"{k}_vs_PS_4"] = A[k] - ps4
        if k.startswith("FS_probe"):
            comps[f"{k}_vs_PS_4"] = A[k] - ps4
            comps[f"{k}_vs_FS_1"] = A[k] - fs1
    if all(lf[g].get("fs_mlp_mean") is not None for g in order):
        comps["FS_2_vs_FS_1"] = np.array([lf[g]["fs_mlp_mean"] for g in order]) - fs1
    return {"mean_auroc": {k: float(v.mean()) for k, v in A.items()}, "comparisons": fold_stats(comps), **fs1_check}


def fmt(x, signed=True):
    if x is None:                                    # version 9: a value that could not be computed
        return "n/a"
    return f"{x:+.3f}" if signed else f"{x:.3f}"


# ---------------------------------------------------------------- version 4: the audit, from its inputs

def _sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rel(p):
    try:
        return Path(p).resolve().relative_to(CODE.parent.resolve()).as_posix()
    except ValueError:
        return Path(p).as_posix()


def audit_inputs(standard_stats, reviewed_dir):
    """The files the audit reads: the standard-split statistics and four reviewed results."""
    reviewed_dir = Path(reviewed_dir)
    return {"standard_stats": Path(standard_stats),
            "ds_arch6": reviewed_dir / "04_ds_arch6.json", "fs1_arch6": reviewed_dir / "04_ds_arch6_rawfeat.json",
            "ds_weak6": reviewed_dir / "04_ds_weak6.json", "fs1_weak6": reviewed_dir / "04_ds_weak6_rawfeat.json"}


def best_of_ps1_to_ps3(ds):
    m = ds["method_means"]
    return max(m[k]["mean_avg8"] for k in ("PS_1_prob_avg", "PS_2_logreg", "PS_3_mlp_logits"))


def audit_block(standard_stats, reviewed_dir, logo, tol=AUDIT_TOL):
    """The audit of Table 2 from its inputs. `logo`: the arch6 LOGO block of logo_block().

    standard-LOGO contrast = standard-split margin over the full ladder - LOGO margin;
    baseline inflation, arch6 = (FS1 - best of PS_1 to PS_3) - standard-split margin over the full ladder;
    baseline inflation, weak6 = (FS1 - best of PS_1 to PS_3) - (FS1 - PS_4).
    (Version 13 uses the paper's names; the keys keep those of versions 4 to 12, overlap_distortion and
    baseline_distortions.)
    Every identity between the inputs is checked to `tol`; "status" is FAIL when one does not hold."""
    files = audit_inputs(standard_stats, reviewed_dir)
    data = {k: load(p) for k, p in files.items()}
    std = data["standard_stats"]["standard"]["arch6"]
    ds_a, ds_w = data["ds_arch6"], data["ds_weak6"]
    fs1_a, fs1_w = data["fs1_arch6"]["avg_auc_8"], data["fs1_weak6"]["avg_auc_8"]
    ps4_a = ds_a["method_means"]["PS_4_xgboost"]["mean_avg8"]
    ps4_w = ds_w["method_means"]["PS_4_xgboost"]["mean_avg8"]
    standard_delta = std["fs1_delta"]
    logo_delta = logo["stats"]["fs1_full"]["mean_delta"]
    checks = []

    def same(name, a, b):
        checks.append({"check": name, "values": [float(a), float(b)], "ok": bool(abs(a - b) <= tol)})
    same("arch6 FS1: standard-split statistics = reviewed result", std["fs1"], fs1_a)
    same("standard-split margin = FS1 - full-ladder ceiling", standard_delta, std["fs1"] - std["ps_ceiling"])
    same("standard-split FS1 - PS_4 = FS1 - reviewed PS_4 (arch6)", std["fs1_minus_ps4"], std["fs1"] - ps4_a)
    same("arch6: the best reviewed method is PS_4", ds_a["best_ds_avg8"], ps4_a)
    same("weak6: the best reviewed method is PS_4", ds_w["best_ds_avg8"], ps4_w)
    same("LOGO margin = mean FS1 - mean ceiling", logo_delta, logo["fs1"] - logo["ceiling_full"])
    same("LOGO margin = mean of its fold margins", logo_delta, float(np.mean(logo["stats"]["fs1_full"]["per_fold"])))
    base = {"arch6": (fs1_a - best_of_ps1_to_ps3(ds_a)) - standard_delta,
            "weak6": (fs1_w - best_of_ps1_to_ps3(ds_w)) - (fs1_w - ps4_w)}
    overlap = standard_delta - logo_delta
    lo, hi = min(overlap, *base.values()), max(overlap, *base.values())
    return {"standard_delta": standard_delta, "logo_delta": logo_delta, "overlap_distortion": overlap,
            "baseline_distortions": base, "range_text": f"{fmt(lo)} to {fmt(hi)}",
            "derivation": {
                "overlap_distortion": "standard.arch6.fs1_delta - logo_arch6.stats.fs1_full.mean_delta",
                "baseline_distortions.arch6": "(04_ds_arch6_rawfeat.avg_auc_8 - best of PS_1 to PS_3 in "
                                              "04_ds_arch6.method_means) - standard.arch6.fs1_delta",
                "baseline_distortions.weak6": "(04_ds_weak6_rawfeat.avg_auc_8 - best of PS_1 to PS_3) - "
                                              "(04_ds_weak6_rawfeat.avg_auc_8 - PS_4_xgboost)"},
            "identity_checks": checks, "identity_tolerance": tol,
            "status": "PASS" if all(c["ok"] for c in checks) else "FAIL",
            "inputs": {k: {"path": _rel(p), "sha256": _sha256(p)} for k, p in files.items()}}


# ---------------------------------------------------------------- version 6: complete inputs, or a named partial run

def check_folds(where, held_out, allow_partial):
    """The held-out generators of a run file, one per fold record (version 6): the eight of GENS, each once, or with
    `allow_partial` at least two of them. Returns the missing ones; stops (SystemExit) otherwise."""
    repeated = sorted({g for g in held_out if held_out.count(g) > 1})
    unknown = sorted({g for g in held_out if g not in GENS})
    if repeated or unknown:
        what = ([f"repeated held-out generators: {', '.join(repeated)}"] if repeated else []) + (
            [f"unexpected held-out generators: {', '.join(unknown)} (expected {', '.join(GENS)})"] if unknown else [])
        raise SystemExit(f"{where}: {'; '.join(what)}. Nothing was written.")
    missing = [g for g in GENS if g not in held_out]
    if missing and not allow_partial:
        raise SystemExit(f"{where}: {len(missing)} of the {len(GENS)} held-out generators missing "
                         f"({', '.join(missing)}). The statistics need all {len(GENS)}; --allow-partial-folds analyses "
                         "the folds present and lists the missing ones. Nothing was written.")
    if len(GENS) - len(missing) < 2:
        raise SystemExit(f"{where}: fewer than two folds, so no statistic can be computed. Nothing was written.")
    return missing


def pairing_problem(logo, rungs):
    """Version 9: why a richer-input run cannot complete a LOGO run (None when it can): another pool or exclusion, or
    a fold whose test or excluded-row counts differ."""
    for k in ("pool", "exclusion"):
        if rungs.get(k) is not None and logo.get(k) is not None and rungs[k] != logo[k]:
            return f"{k} {rungs[k]!r}, not {logo[k]!r}"
    for f in logo["per_fold"]:
        r = (rungs.get("folds") or {}).get(f["held_out"]) or {}
        for k in ("n_test", "n_excluded_training_rows"):
            if r.get(k) is not None and f.get(k) is not None and r[k] != f[k]:
                return f"fold {f['held_out']}: {k} {r[k]}, not {f[k]}"
    return None


def logo_inputs(run_dir, pool, dim, a, notes):
    """The LOGO run of `pool` and its richer-input run, checked (version 6). Returns (LOGO run, richer-input run or
    None, labels for the pool's block); appends a note to `notes` for every gap."""
    p = run_dir / f"11_disjoint_{pool}_d{dim}.json"
    if not p.is_file():
        raise SystemExit(f"{_rel(p)}: not found (the LOGO run of {pool}). Nothing was written.")
    logo = load(p)
    missing = check_folds(_rel(p), [f["held_out"] for f in logo["per_fold"]], a.allow_partial_folds)
    rp = run_dir / f"A3c_logo_extended_rungs_{pool}.json"
    rungs = None
    if rp.is_file():
        rungs = load(rp)
        why = pairing_problem(logo, rungs)           # version 9
        if why:
            raise SystemExit(f"{_rel(rp)} does not belong to {p.name}: {why}. Nothing was written.")
        uncovered = [g for g in GENS if g not in missing and g not in rungs.get("folds", {})]
        if uncovered:
            raise SystemExit(f"{_rel(rp)}: no richer-input runs for {', '.join(uncovered)}, which {p.name} analyses; "
                             "every analysed fold's ceiling is the best of the full ladder. Nothing was written.")
    elif a.allow_ten_methods:
        notes.append(f"{pool}: no richer-input file ({rp.name}), so its ceiling is the best of the ten methods only "
                     "(--allow-ten-methods)")
    else:
        raise SystemExit(f"{_rel(rp)}: not found (the richer-input runs of {pool}). The PS ceiling is the best of the "
                         "full ladder (the ten methods and the richer inputs); --allow-ten-methods analyses the ten "
                         "methods only and says so. Nothing was written.")
    analysed = [g for g in GENS if g not in missing]                     # version 11: the checks below
    if rungs is not None:
        why = ladder_checks.pairing_problem(logo, rungs, p, rp, ("the LOGO run", "its richer-input run"),
                                            folds=analysed)
        if why:
            raise SystemExit(f"{_rel(rp)} cannot complete {p.name}: {why}. Nothing was written.")
    gaps = ladder_checks.ladder_gaps(logo, rungs, folds=analysed)
    if gaps and not a.allow_incomplete_ladder:
        raise SystemExit(f"{pool}: the prediction-space ladder is not complete: {'; '.join(gaps)}. The statistics "
                         "need the ten base methods and the twelve richer-input candidates in every analysed fold; "
                         "--allow-incomplete-ladder analyses what is there and records it as not complete. Nothing "
                         "was written.")
    notes += [f"{pool}: {g} (--allow-incomplete-ladder)" for g in gaps]
    if missing:
        notes.append(f"{pool}: {len(missing)} of {len(GENS)} folds missing ({', '.join(missing)}) "
                     "(--allow-partial-folds)")
    no_fs2 = [f["held_out"] for f in logo["per_fold"] if f.get("fs_mlp_mean") is None]
    if no_fs2:
        notes.append(f"{pool}: no FS2 in {len(no_fs2)} fold(s) ({', '.join(no_fs2)}), so no FS2 statistics")
    ladder = "incomplete" if gaps else ("full" if rungs is not None else "ten methods")     # version 11
    return logo, rungs, {"ladder": ladder, "folds_missing": missing}


def mechanism_inputs(mech_dir, run_dir, a, notes):
    """The mechanism runs under LOGO (Table A8 LOGO columns, App. D.4) and the LOGO runs they take FS1 from, checked
    (version 6). Returns {result key: (mechanism run, LOGO run, folds missing)}."""
    out = {}
    for key, pool, dim in MECHANISM:
        mp, lp = mech_dir / f"run_c_v2_logo_{pool}.json", run_dir / f"11_disjoint_{pool}_d{dim}.json"
        for f, what in ((mp, f"the mechanism run of {pool} under LOGO"),
                        (lp, f"the LOGO run of {pool}, which gives the mechanism its FS1 per fold")):
            if not f.is_file():
                raise SystemExit(f"{_rel(f)}: not found ({what}); without --mech-dir the mechanism is not analysed "
                                 "(and the statistics say so). Nothing was written.")
        mech, logo = load(mp), load(lp)
        missing = check_folds(_rel(mp), [f["held_out"] for f in mech["per_fold"]], a.allow_partial_folds)
        in_logo = [f["held_out"] for f in logo["per_fold"]]
        check_folds(_rel(lp), in_logo, a.allow_partial_folds)
        lacking = [f["held_out"] for f in mech["per_fold"] if f["held_out"] not in in_logo]
        if lacking:
            raise SystemExit(f"{_rel(lp)}: no fold for {', '.join(lacking)}, which {mp.name} analyses (FS1 per fold "
                             "comes from the LOGO run). Nothing was written.")
        why = ladder_checks.pairing_problem(logo, mech, lp, mp, ("the LOGO run", "the mechanism run"),   # v11
                                            folds=[f["held_out"] for f in mech["per_fold"]])
        if why:
            raise SystemExit(f"{_rel(mp)} cannot be analysed with {lp.name}: {why}. Nothing was written.")
        if missing:
            notes.append(f"mechanism {pool}: {len(missing)} of {len(GENS)} folds missing ({', '.join(missing)}) "
                         "(--allow-partial-folds)")
        out[key] = (mech, logo, missing)
    return out


def arch6_basis(block):
    """What the arch6 block behind the audit was computed from, in words (version 6; recorded in the audit block)."""
    words = {"full": "the full ladder", "ten methods": "the ten methods only (--allow-ten-methods)",
             "incomplete": "an incomplete ladder (--allow-incomplete-ladder; input_completeness lists the gaps)",
             None: "a ladder not recorded (statistics written before version 5)"}
    ladder = block.get("ladder")
    missing = [g for g in GENS if g not in block["folds"]]
    return (f"{words.get(ladder, repr(ladder))}; {len(block['folds'])} of {len(GENS)} folds"
            + (f" (missing: {', '.join(missing)})" if missing else ""))


# Standalone runs write only to a separate output path.
# that the reference pins; the harness marks its steps with REPRO_ATTEMPT_ID and backs those results up first.
_REPO = Path(__file__).resolve().parents[3]
_STORED = (_REPO / "camera_ready" / "output" / "results", _REPO / "camera_ready" / "inputs", _REPO / "output" / "results")


def refuse_stored_write(path):
    """Stop before `path` is written among the stored results, unless the harness runs this step."""
    if os.environ.get("REPRO_ATTEMPT_ID"):
        return
    p = os.path.normcase(os.path.realpath(path))
    for folder in _STORED:
        f = os.path.normcase(os.path.realpath(folder))
        if p == f or p.startswith(f.rstrip(os.sep) + os.sep):
            raise SystemExit(f"{Path(path).name} would be written among the stored results that the reference pins "
                             f"({folder}). Run the step through the harness (python camera_ready/scripts/review/"
                             "reproduce_all_v4.py --only <step>), which backs them up first, or write to another "
                             "folder. Nothing was written.")


def write_atomic(path, text):
    """Write `path` through a temporary file, so an interruption cannot leave it half written. Version 10: never
    among the stored results when the script runs by hand."""
    refuse_stored_write(path)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def print_audit(au):
    print(f"audit: standard-LOGO contrast {fmt(au['overlap_distortion'])} (standard {fmt(au['standard_delta'])} - "
          f"LOGO {fmt(au['logo_delta'])}); baseline inflation arch6 {fmt(au['baseline_distortions']['arch6'])}, "
          f"weak6 {fmt(au['baseline_distortions']['weak6'])}; range {au['range_text']}; identities {au['status']}")
    for c in au["identity_checks"]:
        if not c["ok"]:
            print(f"  identity FAILED: {c['check']}: {c['values'][0]!r} vs {c['values'][1]!r}")


# ---------------------------------------------------------------- version 7: where the paper prints each value

NOT_PRINTED = "not printed (kept for checking)"
# Where the camera-ready paper (v8) prints the values of the boundary pools and of the mechanism under LOGO: tables
# formatted for the report and mapped to paper sections. A mechanism value not listed is not printed.
# Version 12: Table A24 and Appendix K of paper v11 are Appendix A in paper v12.
# Version 13: the locations of paper v13, its text checked section by section (the module docstring).
POOL_WHERE = {
    "clip4": (("1", "A8", "A23"), "(Table 1 all five; Table A8 FS1; Table A23 the delta and the wins; Sec. 5.1 the "
                                  "ceiling, FS1, the delta and the wins; Sec. 5.3 the delta and the n>=500 mean; "
                                  "App. A the delta; App. F FS1 and the delta)"),
    "strong4": (("1", "A23"), "(Table 1 all five; Table A23 the delta and the wins; Sec. 5.3 the delta and the "
                              "n>=500 mean; App. A the delta; App. F FS1 and the delta)"),
    "weak6": (("1", "2", "A23"),
              "(Table 1 all five; the delta also in Table 2, as the lowest LOGO margin; Table A23 the ceiling and "
              "FS1; Sec. 5.3 the ceiling and FS1; Sec. 5.4 and App. A, F and G the delta)"),
}
MECH_WHERE = {
    ("arch6", "PS_4_xgboost"): (("A8",), "(LOGO column; also Sec. 5.2 and App. D.4)"),
    ("arch6", "FS_1_raw_feat_logreg"): (("A8",), "(LOGO column)"),
    ("arch6", "FS_D_xgb_features"): (("A8",), "(LOGO column)"),
    ("arch6", "FS_pca2_logreg"): (("A8",), "(LOGO column; also Sec. 5.2 and App. D.4)"),
    ("arch6", "FS_probe2_logreg"): (("A8",), "(LOGO column; also App. D.4)"),
    ("arch6", "FS_pca1_logreg"): ((), "App. D.4"),
    ("arch6", "FS_pca4_logreg"): ((), "App. D.4"),
    ("arch6", "FS_pca8_logreg"): ((), "App. D.4"),
    ("clip4", "PS_4_xgboost"): (("A8",), "(LOGO column)"),
    ("clip4", "FS_1_raw_feat_logreg"): (("A8",), "(LOGO column)"),
    ("clip4", "FS_D_xgb_features"): (("A8",), "(LOGO column)"),
    ("arch6", "FS_D_xgb_features_vs_PS_4"): ((), "Sec. 5.2"),
    ("arch6", "FS_pca2_logreg_vs_PS_4"): ((), "App. A and D.4"),
    ("arch6", "FS_probe2_logreg_vs_PS_4"): ((), "not printed since paper v12 (kept for checking)"),
    ("arch6", "FS_probe2_logreg_vs_FS_1"): ((), "Sec. 5.2, App. A and D.4"),
    ("arch6", "FS_2_vs_FS_1"): ((), "Sec. 5.2"),
    ("clip4", "FS_2_vs_FS_1"): ((), "Sec. 5.2"),
}


def _where(tables, text=""):
    """A paper location: the tables that print the value, then where the text prints it, or why nothing does."""
    if not tables:
        return text
    names = f"Table {tables[0]}" if len(tables) == 1 else f"Tables {', '.join(tables[:-1])} and {tables[-1]}"
    return names + (f" {text}" if text.startswith("(") else f"; {text}" if text else "")


def paper_rows(res):
    """The rows of mitigation_numbers.md (version 7), each with the tables of the camera-ready paper that print its
    value (`tables`), where else the paper prints it or why it does
    not (`text`), the keys of these statistics it comes from (`keys`) and its values (`vals`)."""
    rows = []

    def row(tables, text, quantity, value, keys=(), vals=()):
        rows.append({"tables": tuple(tables), "text": text, "quantity": quantity, "value": value,
                     "keys": tuple(keys), "vals": tuple(vals)})
    lg = res.get("logo_arch6")
    if lg:
        s, k = lg["stats"]["fs1_full"], "logo_arch6.stats.fs1_full"
        row(("1", "2", "A10", "A11", "A13", "A14", "A23"),
            "also the Abstract, Sec. 1, 5.1, 5.4 and 7, App. A, F, G and H.2, and the checklist",
            "LOGO FS1 - PS ceiling (arch6)",
            fmt(s["mean_delta"]), [f"{k}.mean_delta"], [s["mean_delta"]])
        row(("A11", "A14"), "also the Abstract, Sec. 1 and 5.1, App. A and F", "95% CI",
            f"[{fmt(s['ci95'][0])}, {fmt(s['ci95'][1])}]",
            [f"{k}.ci95[0]", f"{k}.ci95[1]"], s["ci95"])
        row(("1", "A14"), "also the Abstract, Sec. 5.1, App. A and F", "folds positive", f"{s['wins']} of {s['n']}",
            [f"{k}.wins", f"{k}.n"], [s["wins"], s["n"]])
        row(("A11",), "also Sec. 5.1 and App. F", "exact sign-flip p", f"{s['signflip_p']:.3f}",
            [f"{k}.signflip_p"], [s["signflip_p"]])
        row(("A11",), "", "sign test p", f"{s['sign_p']:.3f}", [f"{k}.sign_p"], [s["sign_p"]])
        row(("A11",), "", "Wilcoxon p", f"{s['wilcoxon_p']:.3f}", [f"{k}.wilcoxon_p"],
            [s["wilcoxon_p"]])
        # version 9: the margin against the ten methods only (the submitted ladder), printed in App. F
        s10, k10 = lg["stats"]["fs1_ten"], "logo_arch6.stats.fs1_ten"
        row((), "App. F (without the richer inputs, as in the submitted version)",
            "LOGO FS1 - ten-method ceiling (arch6): delta [95% CI], sign-flip p",
            f"{fmt(s10['mean_delta'])} [{fmt(s10['ci95'][0])}, {fmt(s10['ci95'][1])}], {s10['signflip_p']:.3f}",
            [f"{k10}.mean_delta", f"{k10}.ci95[0]", f"{k10}.ci95[1]", f"{k10}.signflip_p"],
            [s10["mean_delta"], *s10["ci95"], s10["signflip_p"]])
        row(("1", "A10", "A13"), "also Sec. 5.1", "PS ceiling (LOGO mean)", fmt(lg["ceiling_full"], False),
            ["logo_arch6.ceiling_full"], [lg["ceiling_full"]])
        row(("1", "A8", "A10", "A13"), "also Sec. 5.1, App. F, H.1 and J.9", "FS1 (LOGO mean)", fmt(lg["fs1"], False),
            ["logo_arch6.fs1"], [lg["fs1"]])
        row(("1",), "App. F", "mean over folds with n >= 500", fmt(lg["fs1_full_delta_n500"]),
            ["logo_arch6.fs1_full_delta_n500"], [lg["fs1_full_delta_n500"]])
        row((), "the reading note of Table A12 (App. F)", "fold-delta SD (population / sample)",
            f"{s['sd_population']:.3f} / {s['sd_sample']:.3f}", [f"{k}.sd_population", f"{k}.sd_sample"])
        row(("1", "A10"), "", "ladder behind the ceiling", lg.get("ladder", "not recorded (version 4)"))
        if "fs2_full" in lg["stats"]:
            s2, k2 = lg["stats"]["fs2_full"], "logo_arch6.stats.fs2_full"
            row(("A10", "A11", "A13"), "", "LOGO FS2 - PS ceiling", fmt(s2["mean_delta"]), [f"{k2}.mean_delta"],
                [s2["mean_delta"]])
            row(("A11",), "", "FS2 CI / sign / Wilcoxon / sign-flip",
                f"[{fmt(s2['ci95'][0])}, {fmt(s2['ci95'][1])}] / {s2['sign_p']:.3f} / {s2['wilcoxon_p']:.3f} / "
                f"{s2['signflip_p']:.3f}",
                [f"{k2}.ci95[0]", f"{k2}.ci95[1]", f"{k2}.sign_p", f"{k2}.wilcoxon_p", f"{k2}.signflip_p"],
                [*s2["ci95"], s2["sign_p"], s2["wilcoxon_p"], s2["signflip_p"]])
            row(("A8", "A10", "A13"), "also App. J.9", "FS2 (LOGO mean)", fmt(lg["fs2"], False), ["logo_arch6.fs2"],
                [lg["fs2"]])
        wt = lg["weighted"]
        row(("A13",), "(LOGO rows)", "weighted ceiling / FS1 delta",
            f"{wt['ceiling_full']:.3f} / {fmt(wt['fs1'] - wt['ceiling_full'])}",
            ["logo_arch6.weighted.ceiling_full", "logo_arch6.weighted.fs1 - weighted.ceiling_full"],
            [wt["ceiling_full"], wt["fs1"] - wt["ceiling_full"]])
        if lg.get("auprc"):
            row((), "App. A and H.2", "LOGO macro AUPRC, FS1 - ceiling method", fmt(lg["auprc"]["delta"]),
                ["logo_arch6.auprc.delta"], [lg["auprc"]["delta"]])
        if res.get("audit"):
            au = res["audit"]
            base = au["baseline_distortions"]
            # version 8: Table A24's first column prints the three distortions too (version 12: the table on
            # training duplicates of Appendix A, which has no number); version 13: the paper's names
            row(("2",), "also Sec. 5.4, App. A and F", "standard-LOGO contrast, arch6 (Table 2's protocol difference)",
                fmt(au["overlap_distortion"]),
                ["audit.overlap_distortion"], [au["overlap_distortion"]])
            row(("2",), "(its three values, also in App. A); the range itself is not printed since paper v12",
                "audit range", au["range_text"], (), [au["overlap_distortion"], *base.values()])
            row(("2",), "also App. A; weak6's also in the Abstract, Sec. 5.4, App. G and the checklist",
                "baseline inflation, arch6 / weak6",
                f"{fmt(base['arch6'])} / {fmt(base['weak6'])}", (), [base["arch6"], base["weak6"]])
            row((), "not printed (a check of Table 2's inputs)", "identities between the audit's inputs",
                au.get("status", "not checked"))
        for i, g in enumerate(lg["folds"]):
            row(("A10",), "", f"best PS per fold ({g})",
                f"{lg['ceiling_full_per_fold'][i]:.3f} ({lg['ceiling_method_per_fold'][i]})",
                [f"logo_arch6.ceiling_full_per_fold[{i}]", f"logo_arch6.ceiling_method_per_fold[{i}]"],
                [lg["ceiling_full_per_fold"][i]])
    shifts = {}                                      # version 9: ten methods to full ladder, per boundary pool
    for pool, b in res.get("logo_boundary", {}).items():
        s, k = b["stats"]["fs1_full"], f"logo_boundary.{pool}"
        if "fs1_ten" in b["stats"]:
            shifts[pool] = abs(s["mean_delta"] - b["stats"]["fs1_ten"]["mean_delta"])
        if pool == "clip4" and b.get("fs2") is not None:     # version 9: printed in Table A8
            row(("A8",), "(LOGO column)", "FS2 (LOGO mean, clip4)", fmt(b["fs2"], False), [f"{k}.fs2"], [b["fs2"]])
        tables, text = POOL_WHERE.get(pool, ((), "not printed (not a pool of the paper)"))
        row(tables, text, f"LOGO {pool}: ceiling / FS1 / delta / wins / n>=500",
            f"{b['ceiling_full']:.3f} / {b['fs1']:.3f} / {fmt(s['mean_delta'])} / {s['wins']}/{s['n']} / "
            f"{fmt(b['fs1_full_delta_n500'])}",
            [f"{k}.ceiling_full", f"{k}.fs1", f"{k}.stats.fs1_full.mean_delta", f"{k}.stats.fs1_full.wins",
             f"{k}.stats.fs1_full.n", f"{k}.fs1_full_delta_n500"],
            [b["ceiling_full"], b["fs1"], s["mean_delta"], s["wins"], s["n"], b["fs1_full_delta_n500"]])
    if shifts:                                       # version 9: "change by at most 0.003" (App. F)
        row((), "App. F", "largest change of a boundary pool's FS1 - ceiling from the ten methods to the full ladder",
            fmt(max(shifts.values()), False),
            ["max over logo_boundary.<pool> of |stats.fs1_full.mean_delta - stats.fs1_ten.mean_delta|"],
            [max(shifts.values())])
    for key, pool in (("mechanism_logo", "arch6"), ("mechanism_logo_clip4", "clip4")):
        mech = res.get(key)
        if not mech:
            continue
        for name, v in mech["mean_auroc"].items():
            tables, text = MECH_WHERE.get((pool, name), ((), NOT_PRINTED))
            row(tables, text, f"{name} (LOGO mean, {pool})", f"{v:.3f}", [f"{key}.mean_auroc.{name}"], [v])
        for name, v in mech["comparisons"].items():
            tables, text = MECH_WHERE.get((pool, name), ((), NOT_PRINTED))
            c = f"{key}.comparisons.{name}"
            row(tables, text, f"{name} ({pool})",
                f"{fmt(v['mean_delta'])} [{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}], {v['wins']}/{v['n']}, "
                f"sign-flip p {v['signflip_p']:.3f}",
                [f"{c}.mean_delta", f"{c}.ci95[0]", f"{c}.ci95[1]", f"{c}.wins", f"{c}.signflip_p"],
                [v["mean_delta"], *v["ci95"]])
    return rows


def numbers_markdown(res):
    """Version 5: the paper's numbers from these statistics, located by the camera-ready table numbering (version 4
    still used the numbering of the submitted paper and compared with its values). Version 6: statistics that are not
    complete (``input_completeness``) open with their notes. Version 7: each row says where the camera-ready paper
    prints its value (paper_rows). Version 8: the header says when sections of the text are named. Version 12:
    Appendix A replaces Appendix K and Table A24 (paper v12), and the header names this version. Version 13: the
    locations and names of paper v13."""
    L = ["# Paper numbers from the LOGO runs", "",
         "Generated by `camera_ready/scripts/authoring/full_ladder_stats_v13.py` from `mitigation_stats.json` in this folder. This report renders selected quantities from the LOGO statistics. The table-building step C24 performs the separate comparison with `camera_ready/provenance/05_number_provenance.csv`. **The \"Paper location\" column uses historical table numbering and text locations from before the appendix was shortened; it is not a current paper index.** Use `camera_ready/provenance/paper_table_locations.json` and `camera_ready/RELEASE_MANIFEST.md` for current locations and additional analyses. For example, the former Table A23 is retained as an additional analysis, not a printed table, and the `n >= 500` subset is reported in Appendix F rather than Table 1. Some retained quantities are no longer printed.",
         ""]
    ic = res.get("input_completeness") or {}
    if ic.get("complete") is False:                                       # version 6
        L += ["**Not complete.** These statistics leave out or replace inputs the paper uses"
              + (f" (options given: {', '.join(ic['options'])})" if ic.get("options") else "") + ":", ""]
        L += [f"- {n}" for n in ic.get("notes", [])] + [""]
    L += ["| Paper location | Quantity | Value |", "|---|---|---|"]
    L += [f"| {_where(r['tables'], r['text'])} | {r['quantity']} | {r['value']} |" for r in paper_rows(res)]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, help="content-disjoint outputs (11, 08 v2)")
    ap.add_argument("--mech-dir", type=Path, help="mechanism outputs (05 v2)")
    ap.add_argument("--boundary", nargs="*", default=list(BOUNDARY),
                    help='boundary pools as pool:proj_dim (default: clip4:32 strong4:256 weak6:256); other selections are recorded as partial')
    ap.add_argument("--out-dir", type=Path, help="default: --run-dir/..")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--reference-dir", type=Path,
                    help="flat directory of reviewed results for --self-test; default: packaged inputs/ "
                         "when present, otherwise the original workspace references")
    ap.add_argument("--standard-stats", type=Path, default=FULL_LADDER_V1,
                    help="standard-split statistics for the audit (default: camera_ready/output/results/"
                         "full_ladder_stats.json, written by step C08)")
    ap.add_argument("--reviewed-results", type=Path, default=REVIEWED_RESULTS,
                    help="reviewed results for the audit (04_ds_arch6/weak6 and their FS1 files; default: the "
                         "repository's output/results/)")
    ap.add_argument("--audit-only", action="store_true",
                    help="recompute the audit block of --stats from the current inputs (step C08a)")
    ap.add_argument("--stats", type=Path,
                    help="with --audit-only: the mitigation_stats.json to update (default: --out-dir/"
                         "mitigation_stats.json)")
    ap.add_argument("--defer-audit", action="store_true",
                    help="version 5: record an audit whose identities do not hold as deferred (exit 0); a later "
                         "--audit-only run (step C08a) recomputes it")
    ap.add_argument("--allow-ten-methods", action="store_true",
                    help="version 6: analyse a pool whose richer-input file is absent over the ten methods (its block "
                         "says ladder: ten methods); without it, such a pool stops the run")
    ap.add_argument("--allow-partial-folds", action="store_true",
                    help="version 6: analyse run files that lack some of the eight held-out generators (at least two "
                         "must remain; every block lists folds_missing); without it, they stop the run")
    ap.add_argument("--allow-incomplete-ladder", action="store_true",
                    help="version 11: analyse folds whose LOGO run lacks base methods or whose richer-input run "
                         "lacks candidates (for example a capped-only run); the block says ladder: incomplete and "
                         "input_completeness lists the gaps; without it, such inputs stop the run")
    a = ap.parse_args()
    names = [spec.partition(":")[0] for spec in a.boundary]                # version 6
    if any(not (n and spec.partition(":")[2].isdigit()) for n, spec in zip(names, a.boundary)):
        ap.error(f"--boundary: expected pool:proj_dim values, got {' '.join(a.boundary)}")
    if len(set(names)) != len(names) or "arch6" in names:
        ap.error("--boundary: name each boundary pool once (arch6 is the primary pool and is always analysed)")

    reference_dir = a.reference_dir
    if reference_dir is None and (CODE / "inputs").is_dir():
        reference_dir = CODE / "inputs"
    if reference_dir is not None and not reference_dir.is_dir():
        ap.error(f"not a reviewed-reference directory: {reference_dir}")
    stored_logo = reference_dir if reference_dir is not None else STORED_LOGO
    stored_rungs = reference_dir if reference_dir is not None else STORED_RUNGS

    if a.self_test:
        logo = load(stored_logo / "11_disjoint_arch6_d256.json")
        rungs = load(stored_rungs / "A3c_logo_extended_rungs_arch6.json")
        problems = ladder_checks.ladder_gaps(logo, rungs)                    # version 11
        why = ladder_checks.pairing_problem(logo, rungs, stored_logo / "11_disjoint_arch6_d256.json",
                                            stored_rungs / "A3c_logo_extended_rungs_arch6.json",
                                            ("the reviewed LOGO run", "the rebuttal's richer-input run"))
        if problems or why:
            raise SystemExit("self-test inputs: " + "; ".join(problems + ([why] if why else [])))
        got = logo_block(logo, rungs)
        # version 8: the stored standard-split statistics (as the audit below), not the working copy C08 rewrites
        audit_dir = reference_dir if reference_dir is not None else a.reviewed_results
        std_path = audit_dir / "full_ladder_stats.json"
        std_path = std_path if std_path.is_file() else a.standard_stats
        ref = load(std_path)["logo_arch6"]
        checks = {
            "ceiling": (got["ceiling_full"], ref["ps_ceiling"]),
            "fs1_delta": (got["stats"]["fs1_full"]["mean_delta"], ref["stats"]["fs1"]["mean_delta"]),
            "fs1_ci_lo": (got["stats"]["fs1_full"]["ci95"][0], ref["stats"]["fs1"]["ci95"][0]),
            "fs1_ci_hi": (got["stats"]["fs1_full"]["ci95"][1], ref["stats"]["fs1"]["ci95"][1]),
            "fs1_signflip": (got["stats"]["fs1_full"]["signflip_p"], ref["stats"]["fs1"]["signflip_p"]),
            "fs1_wilcoxon": (got["stats"]["fs1_full"]["wilcoxon_p"], ref["stats"]["fs1"]["wilcoxon_p"]),
            "fs2_delta": (got["stats"]["fs2_full"]["mean_delta"], ref["stats"]["fs2"]["mean_delta"]),
            "fs2_ci_lo": (got["stats"]["fs2_full"]["ci95"][0], ref["stats"]["fs2"]["ci95"][0]),
            "n500": (got["fs1_full_delta_n500"], ref["fs1_delta_n500"]),
            "weighted_ceiling": (got["weighted"]["ceiling_full"], ref["weighted_ps_ceiling"]),
            "fs1_ten_delta": (got["stats"]["fs1_ten"]["mean_delta"], logo["delta_rawfeat_vs_ps"]),
            "fs1_ten_ci_lo": (got["stats"]["fs1_ten"]["ci95"][0], logo["rawfeat_fold_bootstrap_ci"][0]),
        }
        bad = {k: v for k, v in checks.items() if abs(v[0] - v[1]) > 1e-12}
        for k, (x, y) in checks.items():
            print(f"  {k:<18} {x:.6f} vs {y:.6f}  {'ok' if k not in bad else 'FAIL'}")
        # version 4: the audit derived from the same references, its identities, and the values the paper prints
        au = audit_block(std_path, audit_dir, got)
        for c in au["identity_checks"]:
            print(f"  audit identity: {c['check']}  {'ok' if c['ok'] else 'FAIL'}")
            if not c["ok"]:
                bad[f"audit: {c['check']}"] = c["values"]
        for pool, printed in (("arch6", "+0.074"), ("weak6", "+0.172")):    # Table 2 of the paper
            value = fmt(au["baseline_distortions"][pool])
            print(f"  audit baseline inflation {pool}: {value} (paper {printed})  "
                  f"{'ok' if value == printed else 'FAIL'}")
            if value != printed:
                bad[f"audit_{pool}"] = (value, printed)
        print("self-test", "PASS" if not bad else "FAIL")
        return 0 if not bad else 1

    if a.audit_only:
        stats = a.stats or (a.out_dir / "mitigation_stats.json" if a.out_dir else None)
        if stats is None or not stats.is_file():
            ap.error("--audit-only needs --stats <mitigation_stats.json> (or an --out-dir that holds it)")
        absent = [_rel(p) for p in audit_inputs(a.standard_stats, a.reviewed_results).values() if not p.is_file()]
        if absent:
            raise SystemExit(f"audit inputs missing: {', '.join(absent)}")
        res = load(stats)
        lg = res["logo_arch6"]
        # version 6: the arch6 block behind the audit must be complete, or its gaps allowed by name (version 11: an
        # incomplete ladder needs --allow-incomplete-ladder)
        option = {"ten methods": "--allow-ten-methods", "incomplete": "--allow-incomplete-ladder"}
        allowed = {"ten methods": a.allow_ten_methods, "incomplete": a.allow_incomplete_ladder}
        if lg.get("ladder", "full") != "full" and not allowed.get(lg["ladder"], False):
            raise SystemExit(f"{_rel(stats)}: logo_arch6 was computed over {lg['ladder']!r}, not the full ladder"
                             + (f"; {option[lg['ladder']]} accepts it (the audit then says so)"
                                if lg["ladder"] in option else "") + ". Nothing was written.")
        check_folds(f"{_rel(stats)} (logo_arch6)", list(lg["folds"]), a.allow_partial_folds)
        res["audit"] = audit_block(a.standard_stats, a.reviewed_results, lg)
        res["audit"]["logo_arch6_basis"] = arch6_basis(lg)                  # version 6
        write_atomic(stats, json.dumps(res, indent=2))
        write_atomic(stats.parent / "mitigation_numbers.md", numbers_markdown(res))
        print_audit(res["audit"])
        print(f"written {stats} and {stats.parent / 'mitigation_numbers.md'}")
        return 0 if res["audit"]["status"] == "PASS" else 1

    if a.run_dir is None:
        raise SystemExit("--run-dir is required (or use --self-test)")
    # version 6: every input is checked before anything is computed or written; an incomplete one stops the run
    # unless the option for that partial analysis is given, and every gap is recorded in input_completeness
    notes = []
    pools = [("arch6", "256")] + [tuple(spec.split(":")) for spec in a.boundary]
    checked = {pool: logo_inputs(a.run_dir, pool, dim, a, notes) for pool, dim in pools}
    if sorted(a.boundary) != sorted(BOUNDARY):
        notes.append(f"boundary pools {', '.join(a.boundary) or 'none'} instead of the reviewed "
                     f"{', '.join(BOUNDARY)} (--boundary)")
    if a.mech_dir is not None:
        mech_runs = mechanism_inputs(a.mech_dir, a.run_dir, a, notes)
    else:
        mech_runs = {}
        notes.append("the mechanism under LOGO is not analysed (no --mech-dir)")
    absent = [_rel(p) for p in audit_inputs(a.standard_stats, a.reviewed_results).values() if not p.is_file()]
    if absent:
        raise SystemExit(f"the audit's inputs are missing: {', '.join(absent)} (Table 2; see --standard-stats and "
                         "--reviewed-results). Nothing was written.")
    res = {}
    logo, rungs, labels = checked["arch6"]
    res["logo_arch6"] = logo_block(logo, rungs, a.run_dir)
    res["logo_arch6"]["exclusion"] = logo.get("exclusion")
    res["logo_arch6"].update(labels)                    # version 5: the ladder; version 6: the folds missing
    res["logo_boundary"] = {}
    for pool, _dim in pools[1:]:
        logo_b, rungs_b, labels_b = checked[pool]
        res["logo_boundary"][pool] = logo_block(logo_b, rungs_b, a.run_dir)
        res["logo_boundary"][pool].update(labels_b)
    for pool, block in [("arch6", res["logo_arch6"])] + list(res["logo_boundary"].items()):
        if "auprc" not in block:                                                # version 6
            notes.append(f"{pool}: no AUPRC (the per-sample scores of a fold are missing)")
    # version 4: the audit from its inputs; in a full rerun, step C08a recomputes it after C08 has regenerated the
    # standard-split statistics
    res["audit"] = audit_block(a.standard_stats, a.reviewed_results, res["logo_arch6"])
    res["audit"]["logo_arch6_basis"] = arch6_basis(res["logo_arch6"])         # version 6
    if a.defer_audit and res["audit"]["status"] == "FAIL":        # version 5
        res["audit"]["status"] = "deferred"
        res["audit"]["deferred_reason"] = (
            "the identities between the audit's inputs do not hold: in a full rerun the standard-split "
            "statistics are regenerated after this step (step C08), and step C08a recomputes the audit from them")
    for key, (mech, logo_m, missing) in mech_runs.items():
        res[key] = mechanism_block(mech, logo_m)
        res[key]["folds_missing"] = missing                                     # version 6
    res["input_completeness"] = {                                               # version 6
        "complete": not notes,
        "options": [o for o, on in (("--allow-ten-methods", a.allow_ten_methods),
                                    ("--allow-partial-folds", a.allow_partial_folds),
                                    ("--allow-incomplete-ladder", a.allow_incomplete_ladder)) if on],   # v11
        "notes": notes}
    out_dir = a.out_dir or a.run_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(out_dir / "mitigation_stats.json", json.dumps(res, indent=2))
    write_atomic(out_dir / "mitigation_numbers.md", numbers_markdown(res))
    if res.get("audit"):
        print_audit(res["audit"])
    print(f"written {out_dir / 'mitigation_stats.json'} and {out_dir / 'mitigation_numbers.md'}")
    for n in notes:                                                             # version 6
        print(f"  not complete: {n}")
    return 1 if res.get("audit", {}).get("status") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
