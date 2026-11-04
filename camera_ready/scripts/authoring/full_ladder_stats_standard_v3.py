"""Compute standard-split statistics against the full prediction-space ladder.

For arch6 and clip4, the ceiling is the method with the highest macro AUROC
among the ten base methods and the ten richer-input candidates. Intervals use
10,000 paired generator bootstrap resamples (seed 42). Sample-weighted macro
AUROC weights each generator's AUROC by its number of test rows.

The output also carries the historical arch6 LOGO comparison from before
content exclusions. That comparison uses paired fold bootstrap resamples
(seed 1024), the sign and Wilcoxon tests and the exact sign-flip test. The
current LOGO statistics are in mitigation/mitigation_stats.json.

The harness runs this calculation as C08. For standalone use, pass
--codebase, --run-a-dir, --logo-ceiling and an --out path outside the stored
results. The script reads existing metrics and fits no models.
"""
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

GENS = ["whichisreal", "CollabDiff", "MidJourney", "heygen",
        "starganv2", "deepfacelab", "stargan", "styleclip"]


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


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def gen_ci(a, b, seed=42, n_boot=10_000):
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(a), size=len(a))           # version 2: as many generators as there are
        d[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return [float(lo), float(hi)]


def sign_test_p(deltas):
    """Two-sided paired sign test, excluding exact ties; all ties give p=1."""
    d = np.asarray(deltas, dtype=float)
    n_nonzero = int(np.count_nonzero(d))
    return (float(binomtest(int((d > 0).sum()), n_nonzero, 0.5, alternative="two-sided").pvalue)
            if n_nonzero else 1.0)


def sign_flip_p(deltas):
    obs = abs(deltas.mean())
    flips = [abs((deltas * np.array(s)).mean()) for s in itertools.product((1, -1), repeat=len(deltas))]
    return float(np.mean([f >= obs - 1e-15 for f in flips]))


def logo_stats(fs1, fs2, ps, seed=1024, n_boot=10_000):
    d1, d2 = fs1 - ps, fs2 - ps
    rng = np.random.default_rng(seed)
    b1, b2 = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(ps), size=len(ps))
        b1[i], b2[i] = d1[idx].mean(), d2[idx].mean()
    out = {}
    for name, d, b in (("fs1", d1, b1), ("fs2", d2, b2)):
        wins = int((d > 0).sum())
        out[name] = {
            "mean_delta": float(d.mean()),
            "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "wins": wins,
            "sign_p": sign_test_p(d),
            "wilcoxon_p": (float(wilcoxon(d, alternative="two-sided").pvalue) if np.any(d != 0) else 1.0),
            "signflip_p": sign_flip_p(d),
            "sd_population": float(d.std(ddof=0)),
            "sd_sample": float(d.std(ddof=1)),
            "per_fold": [float(x) for x in d],
        }
    return out


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  check {label}: {got:.6f} vs stored {want:.6f} -> {'ok' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(f"validation failed: {label}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--codebase", type=Path, required=True)
    ap.add_argument("--run-a-dir", type=Path, required=True,
                    help="folder with run_a_extended_ladder_<pool>.json")
    ap.add_argument("--logo-ceiling", type=Path, required=True,
                    help="A3c_logo_combined_ceiling_arch6.json of the July LOGO run")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = a.codebase / "output" / "results"
    out = {"standard": {}, "logo_arch6": {}, "dim_sweep_clip4": {}}

    # ---- standard split, arch6 and clip4
    for pool in ("arch6", "clip4"):
        b06 = load(res / f"06_bootstrap_{pool}.json")
        b07 = load(res / f"07_bootstrap_mlp_{pool}.json")
        ext = load(a.run_a_dir / f"run_a_extended_ladder_{pool}.json")
        fs1 = np.array([b06["per_generator"][g]["raw_feat"] for g in GENS])
        ps4 = np.array([b06["per_generator"][g]["best_ps"] for g in GENS])
        fs2 = np.array([b07["per_generator"][g]["fs_mlp"] for g in GENS])
        n = np.array([b06["per_generator"][g]["n"] for g in GENS], dtype=float)
        ceil_name = ext["deployable_ceiling"]["name"]
        psf = np.array([ext["comparison_vs_deployable_ceiling"]["per_generator_b"][g] for g in GENS])
        # the full-ladder ceiling is the best method of the whole ladder
        ladder_best = max(list(ext["published_ladder"]["all_macro"].items())
                          + [(k, v) for k, v in ext["extended_rungs_macro"].items() if "oracle" not in k],
                          key=lambda kv: kv[1])
        print(f"{pool}: full-ladder ceiling {ladder_best[0]} {ladder_best[1]:.4f}")
        if ladder_best[0] != ceil_name:
            raise SystemExit(f"{pool}: deployable ceiling {ceil_name} is not the ladder best {ladder_best[0]}")
        check(f"{pool} PS ceiling macro", psf.mean(), ext["deployable_ceiling"]["macro"], 1e-9)
        stored = b06["generator_level_bootstrap"]
        lo, hi = gen_ci(fs1, ps4)
        check(f"{pool} FS1-PS4 generator CI lo", lo, stored["ci_95_lo"] if "ci_95_lo" in stored else stored["ci_95"][0], 1e-12)
        check(f"{pool} FS1-PS4 generator CI hi", hi, stored["ci_95_hi"] if "ci_95_hi" in stored else stored["ci_95"][1], 1e-12)
        d1 = fs1 - psf
        d2 = fs2 - psf
        w = n / n.sum()
        out["standard"][pool] = {
            "ps_ceiling_method": ceil_name,
            "ps_ceiling": float(psf.mean()),
            "fs1": float(fs1.mean()),
            "fs2": float(fs2.mean()),
            "fs1_delta": float(d1.mean()),
            "fs1_ci95": gen_ci(fs1, psf),
            "fs1_wins": int((d1 > 0).sum()),
            "fs1_signflip_p": sign_flip_p(d1),
            "fs2_delta": float(d2.mean()),
            "fs2_ci95": gen_ci(fs2, psf),
            "weighted_fs1": float((w * fs1).sum()),
            "weighted_fs2": float((w * fs2).sum()),
            "weighted_ps_ceiling": float((w * psf).sum()),
            "weighted_ps4": float((w * ps4).sum()),
            "fs1_minus_ps4": float((fs1 - ps4).mean()),
        }
        out["standard"][pool]["weighted_fs1_delta"] = out["standard"][pool]["weighted_fs1"] - out["standard"][pool]["weighted_ps_ceiling"]
        out["standard"][pool]["weighted_fs2_delta"] = out["standard"][pool]["weighted_fs2"] - out["standard"][pool]["weighted_ps_ceiling"]

    # ---- LOGO, arch6
    logo = load(res / "11_disjoint_arch6_d256.json")
    comb = load(a.logo_ceiling)
    order = [f["held_out"] for f in logo["per_fold"]]
    fs1 = np.array([f["raw_feat_auc"] for f in logo["per_fold"]])
    fs2 = np.array([f["fs_mlp_mean"] for f in logo["per_fold"]])
    ps10 = np.array([f["best_ps_auc"] for f in logo["per_fold"]])
    byf = {f["fold"]: f for f in comb["per_fold"]}
    psf = np.array([byf[g]["combined"] for g in order])
    ref = logo_stats(fs1, fs2, ps10)
    check("LOGO FS1 CI lo (ten-method)", ref["fs1"]["ci95"][0], logo["rawfeat_fold_bootstrap_ci"][0], 1e-12)
    check("LOGO FS1 CI hi (ten-method)", ref["fs1"]["ci95"][1], logo["rawfeat_fold_bootstrap_ci"][1], 1e-12)
    check("LOGO FS2 CI lo (ten-method)", ref["fs2"]["ci95"][0], logo["fs_fold_bootstrap_ci"][0], 1e-12)
    check("LOGO FS2 CI hi (ten-method)", ref["fs2"]["ci95"][1], logo["fs_fold_bootstrap_ci"][1], 1e-12)
    check("LOGO FS1 Wilcoxon (ten-method)", ref["fs1"]["wilcoxon_p"], logo["rawfeat_wilcoxon_p"], 1e-12)
    check("LOGO FS2 Wilcoxon (ten-method)", ref["fs2"]["wilcoxon_p"], logo["fs_wilcoxon_p"], 1e-12)
    check("LOGO combined ceiling mean", psf.mean(), comb["combined_ceiling"], 1e-12)
    n = np.array([logo["per_fold"][i].get("n_test", 0) for i in range(len(order))], dtype=float)
    if not n.all():
        b06 = load(res / "06_bootstrap_arch6.json")
        n = np.array([b06["per_generator"][g]["n"] for g in order], dtype=float)
    w = n / n.sum()
    full = logo_stats(fs1, fs2, psf)
    out["logo_arch6"] = {
        "folds": order,
        "ps_ceiling": float(psf.mean()),
        "ps_ceiling_per_fold": [float(x) for x in psf],
        "changed_folds": {g: {"ten_method": float(a_), "full": float(b_), "best_richer_input": byf[g].get("best_extended_rung")}
                          for g, a_, b_ in zip(order, ps10, psf) if abs(a_ - b_) > 1e-12},
        "fs1": float(fs1.mean()),
        "fs2": float(fs2.mean()),
        "stats": full,
        "fs1_delta_n500": float(np.mean([d for d, nn in zip(fs1 - psf, n) if nn >= 500])),
        "weighted_fs1": float((w * fs1).sum()),
        "weighted_fs2": float((w * fs2).sum()),
        "weighted_ps_ceiling": float((w * psf).sum()),
        "weighted_ps_ten_method": float((w * ps10).sum()),
    }

    # ---- clip4 FS2 dimension sweep against the full-ladder ceiling
    ceil = out["standard"]["clip4"]["ps_ceiling"]
    for d in (32, 64, 256):
        f = res / f"05_fs_clip4_mlp_d{d}.json"
        if f.exists():
            m = load(f)
            auc = float(m.get("seed_avg_avg8", m.get("mean_avg8")))
            out["dim_sweep_clip4"][str(d)] = {"auroc_source_key": "seed_avg_avg8" if "seed_avg_avg8" in m else "mean_avg8",
                                              "auroc": auc, "delta_vs_full_ceiling": auc - ceil}

    refuse_stored_write(a.out)                                         # version 3
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1)[:6000])
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
