"""Compute fold-level statistics for the Appendix A sensitivity summary.

The sensitivity summarizer imports logo_block and mechanism_block to compare
the primary and alternative runs. These functions report per-fold margins,
macro and weighted means, paired tests and 95% percentile intervals from
10,000 paired fold resamples (seed 1024). For the primary standalone LOGO
statistics and Table 2, use full_ladder_stats_v13.py.
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

CODE = Path(__file__).resolve().parents[2]
CAMERA = CODE.parent
CH3 = CAMERA.parent
STORED_LOGO = CH3 / "01.initial_submission" / "codebase" / "output" / "results"
STORED_RUNGS = CH3 / "02.rebuttal" / "runs" / "reprod" / "results"
FULL_LADDER_V1 = CODE / "output" / "results" / "full_ladder_stats.json"
REVIEWED_RESULTS = CODE.parent / "output" / "results"      # version 4: the reviewed results (Block A), for the audit
AUDIT_TOL = 1e-12                                          # version 4: tolerance of the audit's identities
GENS = ["whichisreal", "CollabDiff", "MidJourney", "heygen", "starganv2", "deepfacelab", "stargan", "styleclip"]


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
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else 0.0


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
    res["fs1_full_delta_n500"] = float((fs1 - full)[big].mean())
    res["weighted"] = {"fs1": float((w * fs1).sum()), "ceiling_full": float((w * full).sum()),
                       "fs2": float((w * fs2).sum()) if have_fs2 else None}
    # AUPRC from the stored per-sample scores (v2 runs only)
    if run_dir is not None:
        ap_fs1, ap_ceil, ok = [], [], True
        for g, m in zip(order, method):
            f = folds[g]
            sf = Path(run_dir) / f.get("scores_file", "")
            if not f.get("scores_file") or not sf.exists():
                ok = False
                break
            s = np.load(sf)
            y = s["labels"]
            ap_fs1.append(auprc(y, s["FS_1_raw_feat_logreg"]))
            if m in s.files:
                ap_ceil.append(auprc(y, s[m]))
            else:
                rs = np.load(Path(run_dir) / rungs["folds"][g]["scores_file"])
                ap_ceil.append(auprc(y, rs[m]))
        if ok:
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

    overlap distortion = standard-split margin over the full ladder - LOGO margin;
    baseline-ladder distortion, arch6 = (FS1 - best of PS_1 to PS_3) - standard-split margin over the full ladder;
    baseline-ladder distortion, weak6 = (FS1 - best of PS_1 to PS_3) - (FS1 - PS_4).
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


def write_atomic(path, text):
    """Write `path` through a temporary file, so an interruption cannot leave it half written."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def print_audit(au):
    print(f"audit: overlap distortion {fmt(au['overlap_distortion'])} (standard {fmt(au['standard_delta'])} - LOGO "
          f"{fmt(au['logo_delta'])}); baseline-ladder distortions arch6 {fmt(au['baseline_distortions']['arch6'])}, "
          f"weak6 {fmt(au['baseline_distortions']['weak6'])}; range {au['range_text']}; identities {au['status']}")
    for c in au["identity_checks"]:
        if not c["ok"]:
            print(f"  identity FAILED: {c['check']}: {c['values'][0]!r} vs {c['values'][1]!r}")


def numbers_markdown(res):
    L = ["# Paper numbers from the mitigation runs", "",
         "Generated by `scripts/authoring/full_ladder_stats_v4.py`. 'v2 now' is the value printed in",
         "`latex/main.tex` before the mitigation; 'new' is from the runs. See",
         "`camera_ready/output/results/mitigation/SCORE_PRECISION_NOTE.md` for the score-precision explanation.", "",
         "| Paper location | Quantity | v2 now | new |", "|---|---|---|---|"]
    lg = res.get("logo_arch6")
    if lg:
        s = lg["stats"]["fs1_full"]
        rows = [
            ("Abstract, Sec. 1, Sec. 5.1, Sec. 7, Table 1, Q1", "LOGO FS1 - PS ceiling (arch6)", "+0.080",
             fmt(s["mean_delta"])),
            ("same", "95% CI", "[+0.024, +0.140]", f"[{fmt(s['ci95'][0])}, {fmt(s['ci95'][1])}]"),
            ("same", "folds positive", "7 of 8", f"{s['wins']} of {s['n']}"),
            ("Sec. 5.1, Table A12", "exact sign-flip p", "0.055", f"{s['signflip_p']:.3f}"),
            ("Sec. 5.1, Table A12", "sign test p", "0.070", f"{s['sign_p']:.3f}"),
            ("Sec. 5.1, Table A12", "Wilcoxon p", "0.055", f"{s['wilcoxon_p']:.3f}"),
            ("Table 1, Sec. 6 step 1, Table A11", "PS ceiling (LOGO mean)", "0.818", fmt(lg["ceiling_full"], False)),
            ("Table 1, Table A11", "FS1 (LOGO mean)", "0.898", fmt(lg["fs1"], False)),
            ("Table 1", "mean over folds with n >= 500", "+0.114", fmt(lg["fs1_full_delta_n500"])),
            ("Table A13 note", "fold-delta SD (population / sample)", "0.085 / 0.091",
             f"{s['sd_population']:.3f} / {s['sd_sample']:.3f}"),
        ]
        if "fs2_full" in lg["stats"]:
            s2 = lg["stats"]["fs2_full"]
            rows += [("Table A11, A12", "LOGO FS2 - PS ceiling", "+0.078", fmt(s2["mean_delta"])),
                     ("Table A12", "FS2 CI / sign / Wilcoxon / sign-flip", "[+0.021, +0.141] / 0.289 / 0.055 / 0.055",
                      f"[{fmt(s2['ci95'][0])}, {fmt(s2['ci95'][1])}] / {s2['sign_p']:.3f} / {s2['wilcoxon_p']:.3f} / "
                      f"{s2['signflip_p']:.3f}"),
                     ("Table A11", "FS2 (LOGO mean)", "0.896", fmt(lg["fs2"], False))]
        wt = lg["weighted"]
        rows += [("Table A14 (LOGO rows)", "weighted ceiling / FS1 delta",
                  "0.827 / +0.095", f"{wt['ceiling_full']:.3f} / {fmt(wt['fs1'] - wt['ceiling_full'])}")]
        if lg.get("auprc"):
            rows += [("App. H.2", "LOGO macro AUPRC, FS1 - ceiling method", "+0.076 (ten methods)",
                      fmt(lg["auprc"]["delta"]))]
        if res.get("audit"):
            au = res["audit"]
            rows += [("Abstract, Sec. 5.4, Table 3, Sec. 6, Sec. 7, Q1", "overlap distortion (standard - LOGO)",
                      "+0.025", fmt(au["overlap_distortion"])),
                     ("same", "audit range", "+0.025 to +0.172", au["range_text"]),
                     ("same", "baseline-ladder distortions, arch6 / weak6", "+0.074 / +0.172",
                      f"{fmt(au['baseline_distortions']['arch6'])} / {fmt(au['baseline_distortions']['weak6'])}"),
                     ("version 4", "identities between the audit's inputs", "-", au.get("status", "not checked"))]
        rows += [("Table A11", f"best PS per fold ({g})", "see Table A11",
                  f"{lg['ceiling_full_per_fold'][i]:.3f} ({lg['ceiling_method_per_fold'][i]})")
                 for i, g in enumerate(lg["folds"])]
        L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    for pool, b in res.get("logo_boundary", {}).items():
        s = b["stats"]["fs1_full"]
        L.append(f"| Table 1, Sec. 5.3 | LOGO {pool}: ceiling / FS1 / delta / wins / n>=500 | see Table 1 | "
                 f"{b['ceiling_full']:.3f} / {b['fs1']:.3f} / {fmt(s['mean_delta'])} / {s['wins']}/{s['n']} / "
                 f"{fmt(b['fs1_full_delta_n500'])} |")
    for key, pool in (("mechanism_logo", "arch6"), ("mechanism_logo_clip4", "clip4")):
        mech = res.get(key)
        if not mech:
            continue
        for k, v in mech["mean_auroc"].items():
            L.append(f"| Table 2 ({pool}, LOGO column), App. D.4 | {k} (LOGO mean, {pool}) | - | {v:.3f} |")
        for k, v in mech["comparisons"].items():
            L.append(f"| Sec. 5.2, App. D.4 | {k} ({pool}) | - | {fmt(v['mean_delta'])} [{fmt(v['ci95'][0])}, "
                     f"{fmt(v['ci95'][1])}], {v['wins']}/{v['n']}, sign-flip p {v['signflip_p']:.3f} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, help="content-disjoint outputs (11, 08 v2)")
    ap.add_argument("--mech-dir", type=Path, help="mechanism outputs (05 v2)")
    ap.add_argument("--boundary", nargs="*", default=["clip4:32", "strong4:256", "weak6:256"],
                    help='boundary pools as pool:proj_dim (default: clip4:32 strong4:256 weak6:256)')
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
    a = ap.parse_args()

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
        got = logo_block(logo, rungs)
        ref = load(FULL_LADDER_V1)["logo_arch6"]
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
        audit_dir = reference_dir if reference_dir is not None else a.reviewed_results
        std_path = audit_dir / "full_ladder_stats.json"
        std_path = std_path if std_path.is_file() else a.standard_stats
        au = audit_block(std_path, audit_dir, got)
        for c in au["identity_checks"]:
            print(f"  audit identity: {c['check']}  {'ok' if c['ok'] else 'FAIL'}")
            if not c["ok"]:
                bad[f"audit: {c['check']}"] = c["values"]
        for pool, printed in (("arch6", "+0.074"), ("weak6", "+0.172")):    # Table 2 of the paper
            value = fmt(au["baseline_distortions"][pool])
            print(f"  audit baseline-ladder distortion {pool}: {value} (paper {printed})  "
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
        res["audit"] = audit_block(a.standard_stats, a.reviewed_results, res["logo_arch6"])
        write_atomic(stats, json.dumps(res, indent=2))
        write_atomic(stats.parent / "mitigation_numbers.md", numbers_markdown(res))
        print_audit(res["audit"])
        print(f"written {stats} and {stats.parent / 'mitigation_numbers.md'}")
        return 0 if res["audit"]["status"] == "PASS" else 1

    if a.run_dir is None:
        raise SystemExit("--run-dir is required (or use --self-test)")
    res = {}
    logo = load(a.run_dir / "11_disjoint_arch6_d256.json")
    rp = a.run_dir / "A3c_logo_extended_rungs_arch6.json"
    res["logo_arch6"] = logo_block(logo, load(rp) if rp.exists() else None, a.run_dir)
    res["logo_arch6"]["exclusion"] = logo.get("exclusion")
    res["logo_boundary"] = {}
    for spec in a.boundary:
        pool, d = spec.split(":")
        p = a.run_dir / f"11_disjoint_{pool}_d{d}.json"
        if p.exists():
            rpp = a.run_dir / f"A3c_logo_extended_rungs_{pool}.json"
            res["logo_boundary"][pool] = logo_block(load(p), load(rpp) if rpp.exists() else None, a.run_dir)
            res["logo_boundary"][pool]["ladder"] = "full" if rpp.exists() else "ten methods"
    # version 4: the audit from its inputs; in a full rerun, step C08a recomputes it after C08 has regenerated the
    # standard-split statistics
    absent = [_rel(p) for p in audit_inputs(a.standard_stats, a.reviewed_results).values() if not p.is_file()]
    if not absent:
        res["audit"] = audit_block(a.standard_stats, a.reviewed_results, res["logo_arch6"])
    else:
        print(f"audit not computed (inputs missing: {', '.join(absent)})")
    if a.mech_dir is not None and (a.mech_dir / "run_c_v2_logo_arch6.json").exists():
        res["mechanism_logo"] = mechanism_block(load(a.mech_dir / "run_c_v2_logo_arch6.json"), logo)
    clip4_logo = a.run_dir / "11_disjoint_clip4_d32.json"       # b2: clip4 column (needs the a4 clip4 run)
    if a.mech_dir is not None and (a.mech_dir / "run_c_v2_logo_clip4.json").exists() and clip4_logo.exists():
        res["mechanism_logo_clip4"] = mechanism_block(load(a.mech_dir / "run_c_v2_logo_clip4.json"), load(clip4_logo))
    out_dir = a.out_dir or a.run_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(out_dir / "mitigation_stats.json", json.dumps(res, indent=2))
    write_atomic(out_dir / "mitigation_numbers.md", numbers_markdown(res))
    if res.get("audit"):
        print_audit(res["audit"])
    print(f"written {out_dir / 'mitigation_stats.json'} and {out_dir / 'mitigation_numbers.md'}")
    return 1 if res.get("audit", {}).get("status") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
