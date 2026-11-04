"""Simulate the minimum detectable effect for the eight-fold LOGO tests in Appendix F.

For each sigma (0.05, 0.08, 0.12), simulate 10,000 sets of eight independent
standard normal draws. Shift and scale the same draws across effects on a
0.005 grid. The MDE is the smallest grid effect with power at least 0.80.

A set detects the effect when the 95% percentile bootstrap interval excludes
zero. Compute that interval over all 6,435 distinct resamples, weighted by
their multinomial probabilities. Also compute power for the two-sided exact
sign, exact Wilcoxon, exact sign-flip (all 256 patterns), and one-sample t
tests at alpha 0.05.

For the bootstrap, refine the estimated power at the initial threshold and
the two preceding grid points using 100,000 new sets per sigma. The reference
MDEs are 0.050, 0.080, and 0.115. The script exits with status 1 on a mismatch.

The harness runs this as C10. The simulation needs no feature cache and
takes about ten minutes on a laptop CPU.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _paths  # noqa: F401  # MUST be first

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import stats

import _common as common

N = 8                                          # LOGO folds
ALPHA, TARGET, STEP = 0.05, 0.80, 0.005
SIGMAS = (0.05, 0.08, 0.12)
PAPER = {0.05: 0.050, 0.08: 0.080, 0.12: 0.115}   # Table A12 as printed
OTHER_TESTS = ("sign_test", "wilcoxon", "signflip", "t_test")


def compositions(n):
    """Every distinct bootstrap resample of n values as a count vector (stars and bars), with its probability."""
    rows = []
    for cut in itertools.combinations(range(2 * n - 1), n - 1):
        prev, c = -1, []
        for k in cut:
            c.append(k - prev - 1)
            prev = k
        c.append(2 * n - 2 - prev)
        rows.append(c)
    counts = np.array(rows, dtype=np.int64)
    probs = np.array([math.factorial(n) / np.prod([math.factorial(k) for k in r]) for r in counts]) / n ** n
    return counts, probs


COUNTS, PROBS = compositions(N)                     # 6,435 resamples for n = 8
SIGNS = np.array(list(itertools.product((1.0, -1.0), repeat=N)))   # the 256 sign patterns
T_CRIT = stats.t.ppf(1 - ALPHA / 2, N - 1)


def _sign_reject():
    k = np.arange(N + 1)
    p = np.minimum(1.0, 2 * np.minimum(stats.binom.cdf(k, N, 0.5), stats.binom.sf(k - 1, N, 0.5)))
    return p < ALPHA


def _wilcoxon_reject():
    """Exact two-sided signed-rank test without ties: reject[w] for w = min(W+, W-)."""
    counts = np.zeros(N * (N + 1) // 2 + 1)
    for signs in itertools.product((0, 1), repeat=N):
        counts[sum(r for r, s in zip(range(1, N + 1), signs) if s)] += 1
    return np.minimum(1.0, 2 * np.cumsum(counts) / counts.sum()) < ALPHA


SIGN_REJECT, WILCOXON_REJECT = _sign_reject(), _wilcoxon_reject()


def detects(x, test, chunk=500):
    """For each row of x (sets x 8 fold-deltas): does the test reject 'mean fold-delta = 0'?"""
    if test == "bootstrap_ci":
        out = np.empty(len(x), dtype=bool)
        for s in range(0, len(x), chunk):
            m = x[s:s + chunk] @ (COUNTS.T / N)          # every resample mean
            order = np.argsort(m, axis=1)
            ms = np.take_along_axis(m, order, axis=1)
            cw = np.cumsum(PROBS[order], axis=1)
            rows = np.arange(len(ms))
            lo = ms[rows, np.argmax(cw >= ALPHA / 2, axis=1)]
            hi = ms[rows, np.argmax(cw >= 1 - ALPHA / 2, axis=1)]
            out[s:s + chunk] = (lo > 0) | (hi < 0)
        return out
    if test == "sign_test":
        return SIGN_REJECT[(x > 0).sum(1)]
    if test == "wilcoxon":
        ranks = np.argsort(np.argsort(np.abs(x), axis=1), axis=1) + 1
        w_neg = (ranks * (x < 0)).sum(1)
        return WILCOXON_REJECT[np.minimum(w_neg, N * (N + 1) // 2 - w_neg)]
    if test == "signflip":
        flipped = np.abs(x) @ SIGNS.T / N               # the mean under every sign pattern
        return (np.abs(flipped) >= np.abs(x.mean(1))[:, None] - 1e-12).mean(1) < ALPHA
    if test == "t_test":
        return np.abs(x.mean(1) / (x.std(1, ddof=1) / math.sqrt(N))) > T_CRIT
    raise ValueError(test)


def self_check(rng, log):
    """The vectorised exact tests against scipy on 2,000 random sets."""
    x = 0.04 + 0.08 * rng.standard_normal((2000, N))
    ref = {"sign_test": [stats.binomtest(int((r > 0).sum()), N, 0.5).pvalue < ALPHA for r in x],
           "wilcoxon": [stats.wilcoxon(r, method="exact").pvalue < ALPHA for r in x],
           "t_test": [stats.ttest_1samp(r, 0).pvalue < ALPHA for r in x]}
    for test, want in ref.items():
        got = detects(x, test)
        if not np.array_equal(got, np.array(want)):
            raise SystemExit(f"self-check failed: {test} disagrees with scipy on {int((got != want).sum())} sets")
    log(f"self-check: sign test, Wilcoxon and t-test agree with scipy on 2,000 sets; {len(COUNTS)} resamples")


def first_reaching(deltas, power):
    return next((d for d, p in zip(deltas, power) if p >= TARGET), None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", type=int, default=10_000)
    ap.add_argument("--precision-sets", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20261022, help="the precision check uses seed + 1")
    ap.add_argument("--out", type=Path, default=_paths.RESULTS_DIR / "appF_power_mde.json")
    a = ap.parse_args()
    out = _paths.assert_no_codebase_writes(a.out)
    log = common.make_logger("appF_power_mde")
    t_start = time.time()
    self_check(np.random.default_rng(1), log)
    rng, rng_precise = np.random.default_rng(a.seed), np.random.default_rng(a.seed + 1)
    zs = {s: rng.standard_normal((a.sets, N)) for s in SIGMAS}
    rows = []
    for sigma in SIGMAS:
        t0 = time.time()
        grid = [round(float(d), 3) for d in np.arange(0, 2.5 * sigma + 1e-9, STEP)]
        z = zs[sigma]
        power = {t: [float(detects(d + sigma * z, t).mean()) for d in grid] for t in OTHER_TESTS}
        boot = []
        for d in grid:                                 # the slow test: stop two grid points after 0.80
            boot.append(float(detects(d + sigma * z, "bootstrap_ci").mean()))
            hit = first_reaching(grid, boot)
            if hit is not None and grid.index(hit) + 2 < len(boot):
                break
        mde = {"bootstrap_ci": first_reaching(grid, boot), **{t: first_reaching(grid, power[t]) for t in OTHER_TESTS}}
        log(f"sigma {sigma}: MDE on {a.sets:,} sets: " + ", ".join(f"{t} {v}" for t, v in mde.items()))
        # precision check of the bootstrap-CI MDE: the grid MDE and the two points below, then upwards if needed
        zp = rng_precise.standard_normal((a.precision_sets, N))
        start = max(0, grid.index(mde["bootstrap_ci"]) - 2)
        precise, table = [], None
        for d in grid[start:]:
            p = float(detects(d + sigma * zp, "bootstrap_ci").mean())
            precise.append({"delta": d, "power": p, "se": math.sqrt(p * (1 - p) / a.precision_sets)})
            log(f"  precision check, delta {d:.3f}: power {p:.4f} (se {precise[-1]['se']:.4f})")
            if p >= TARGET and table is None:
                table = d
            if table is not None and d >= mde["bootstrap_ci"]:
                break
        rows.append({"sigma": sigma, "delta": grid, "power": power, "bootstrap_ci_power": boot,
                     "mde": mde, "precision_check": precise, "table_a12": table, "paper": PAPER[sigma],
                     "matches_paper": table == PAPER[sigma],
                     "larger_mde_than_bootstrap_ci": {t: (mde[t] is None or mde[t] > table) for t in OTHER_TESTS},
                     "elapsed_s": round(time.time() - t0, 1)})
        log(f"sigma {sigma}: Table A12 value {table} (printed {PAPER[sigma]}); "
            f"{rows[-1]['elapsed_s'] / 60:.1f} min")
    res = {"paper_item": "Appendix F, Table A12: MDE at 80% power, alpha 0.05 two-sided, n = 8 fold-deltas",
           "n_folds": N, "alpha": ALPHA, "target_power": TARGET, "grid_step": STEP, "sets": a.sets,
           "precision_sets": a.precision_sets, "seed": a.seed, "precision_seed": a.seed + 1,
           "bootstrap": f"exact percentile 95% CI of the mean over the {len(COUNTS)} distinct resamples",
           "sigmas": rows, "table_a12": {str(r["sigma"]): r["table_a12"] for r in rows},
           "all_match_paper": all(r["matches_paper"] for r in rows),
           "note_holds": all(all(r["larger_mde_than_bootstrap_ci"].values()) for r in rows),
           "elapsed_s": round(time.time() - t_start, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    log(f"Table A12 rebuilt: {res['table_a12']} (printed {PAPER}); matches the paper: {res['all_match_paper']}; "
        f"other tests have larger MDEs: {res['note_holds']}; wrote {out}")
    return 0 if res["all_match_paper"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
