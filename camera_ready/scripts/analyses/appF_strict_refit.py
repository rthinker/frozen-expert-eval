# C02 and C11: Diagnose overlap and fit the strict-refit arms.
"""RUN B - strict path-disjoint evaluation, with a matched-size control.

Measures sensitivity to removing training samples whose paths overlap the test sets.

WHAT THE DIAGNOSTIC FOUND (24 Jul 2026, measured on the shipped cache)
----------------------------------------------------------------------
11,886 of the 17,925 held-out TEST samples (66.3%) appear VERBATIM - same
canonical DF40 path, same label - inside the combiner's training data
(FIT + SELECT). Per generator:

    styleclip    3824/3824  100.0%   (2824 of them fake)
    MidJourney   1882/1882  100.0%   ( 877 fake)
    deepfacelab   149/ 149  100.0%   ( 100 fake)
    CollabDiff   1993/2000   99.7%   (1000 fake)
    starganv2    3000/4000   75.0%   ( 999 fake)
    whichisreal   988/2001   49.4%   (real only)
    heygen         50/ 101   49.5%   (real only)
    stargan          0/3968    0.0%

An earlier version of Appendix A described cross-JSON path reuse as "primarily
through real frames". The measurement shows it is broader than that: for
five of the eight generators, FAKE samples of the held-out generator are
in combiner training. This run produces the estimate with that removed.

The paper's primary LOGO protocol (Appendix F) holds out each test generator
and excludes identified feature-identical held-out content from combiner
training. Other reuse and prior frozen-expert exposure can remain. Run B
instead stress-tests the standard split.

THE MATCHED-SIZE CONTROL
------------------------
Removing the identified rows also removes 89% of the REAL class:
FIT goes from 5,438 real / 21,153 fake to 586 real / 16,484 fake.

The matched control removes the same number of rows per class and partition
at random over several seeds. Random removal retains overlap and does not
match the source coverage of the strict arm. Report the two contrasts as
strict minus matched and matched minus full. Both are descriptive. The
historical JSON keys leakage_effect and sample_size_effect encode these
contrasts for compatibility.

LEVELS
------
  exact    drop FIT/SELECT rows whose full canonical id appears in any
           held-out generator split (same underlying file)
  family   drop FIT/SELECT rows whose leading path component is one of
           the eight held-out generator names - i.e. the combiner sees
           NO sample from a held-out generator by any path. Strictest.
           On the shipped cache this selects the same 9,521 FIT rows as
           `exact`; both are reported because they are different claims.

TEST is never modified, so every level stays directly comparable to the
published numbers.

Outputs -> camera_ready/output/results/analyses/run_b_path_disjoint_<pool>_<level>.json
Writes nothing outside camera_ready/output/.

Usage (from the repository root)::

    python camera_ready/scripts/analyses/appF_strict_refit.py --pool arch6 --diagnose
    python camera_ready/scripts/analyses/appF_strict_refit.py --pool arch6 --levels family --stratify
    python camera_ready/scripts/analyses/appF_strict_refit.py --pool arch6 --levels exact family
    python camera_ready/scripts/analyses/appF_strict_refit.py --pool arch6 --levels family --skip_matched
"""

from __future__ import annotations

import _paths  # noqa: F401  # MUST be first

import argparse
import time
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

import _common as common
from src.models.feature_space import FS_1  # noqa: E402
from src.utils.config import (  # noqa: E402
    FIT_TAG, SELECT_TAG, get_pool_experts,
)
from src.utils.dataset_helpers import TEST_GENERATORS  # noqa: E402
from src.utils.metrics import compute_auc  # noqa: E402

LEVELS = ("exact", "family")


# --------------------------------------------------------------------------
# Contamination masks
# --------------------------------------------------------------------------

def heldout_id_set(cache_root, generators: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for g in generators:
        out.update(common.load_sample_ids(g, cache_root).tolist())
    return out


def contaminated_mask(ids: np.ndarray, level: str, heldout_ids: Set[str],
                      generators: Sequence[str]) -> np.ndarray:
    """True where a training row must be dropped at this strictness level."""
    if level == "exact":
        return np.fromiter((str(s) in heldout_ids for s in ids),
                           dtype=bool, count=len(ids))
    if level == "family":
        fam = set(generators)
        return np.fromiter((str(s).split("/")[0] in fam for s in ids),
                           dtype=bool, count=len(ids))
    raise ValueError(f"unknown level {level!r}")


def matched_random_mask(labels: np.ndarray, drop_mask: np.ndarray,
                        seed: int) -> np.ndarray:
    """A random drop mask with the same per-class drop counts as `drop_mask`."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(labels), dtype=bool)
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        n_drop = int((drop_mask & (labels == cls)).sum())
        n_drop = min(n_drop, len(idx))
        out[rng.choice(idx, size=n_drop, replace=False)] = True
    return out


# --------------------------------------------------------------------------
# Fit + evaluate once
# --------------------------------------------------------------------------

def _fit_and_score(expert_names, cache_root, fit_keep, sel_keep,
                   seeds, skip_slow, generators, log, tag: str) -> Dict:
    fit_feats, fit_logits3d, fit_labels = common.load_split(
        expert_names, FIT_TAG, cache_root)
    sel_feats, sel_logits3d, sel_labels = common.load_split(
        expert_names, SELECT_TAG, cache_root)

    fit_feats = [f[fit_keep] for f in fit_feats]
    fit_logits3d = fit_logits3d[fit_keep]
    fit_labels = fit_labels[fit_keep]
    sel_logits3d = sel_logits3d[sel_keep]
    sel_labels = sel_labels[sel_keep]

    if len(np.unique(fit_labels)) < 2 or len(np.unique(sel_labels)) < 2:
        raise RuntimeError(f"[{tag}] a split lost a whole class; cannot fit.")

    log(f"  [{tag}] FIT {len(fit_labels)} "
        f"(real={int((fit_labels == 0).sum())}, "
        f"fake={int((fit_labels == 1).sum())})   "
        f"SELECT {len(sel_labels)} "
        f"(real={int((sel_labels == 0).sum())}, "
        f"fake={int((sel_labels == 1).sum())})")

    X_test_feat, test_logits3d, test_labels, boundaries, gens = \
        common.load_test_generators(expert_names, cache_root, generators)

    ps_scores = common.run_ps_ladder(
        fit_logits3d, fit_labels, sel_logits3d, sel_labels,
        test_logits3d, seeds=seeds, skip_slow=skip_slow, log=None)
    best_name, best_macro, all_macros = common.best_of(
        ps_scores, test_labels, boundaries, gens)

    fs1_scores = np.asarray(
        FS_1(np.concatenate(fit_feats, axis=1), fit_labels, X_test_feat),
        dtype=np.float32)
    fs1_macro = common.macro_auc(
        common.per_generator_auc(test_labels, fs1_scores, boundaries, gens),
        gens)

    cmp = common.compare_methods(
        test_labels, fs1_scores, ps_scores[best_name],
        boundaries, gens, "FS_1_raw_feat_logreg", best_name)

    log(f"  [{tag}] FS_1={fs1_macro:.4f}  bestPS={best_macro:.4f} "
        f"({best_name})  delta={cmp['delta']:+.4f}  "
        f"CI {cmp['generator_bootstrap']['ci_95']}")

    return {
        "tag": tag,
        "n_fit": int(len(fit_labels)),
        "n_fit_real": int((fit_labels == 0).sum()),
        "n_fit_fake": int((fit_labels == 1).sum()),
        "n_select": int(len(sel_labels)),
        "best_ps_name": best_name,
        "best_ps_macro": float(best_macro),
        "ps_ladder_macro": {k: float(v) for k, v in all_macros.items()},
        "fs1_macro": float(fs1_macro),
        "comparison": cmp,
        "_fs1_scores": fs1_scores,
        "_ps_scores": ps_scores[best_name],
        "_test_labels": test_labels,
        "_boundaries": boundaries,
        "_generators": gens,
    }


def _strip_private(d: Dict) -> Dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


# --------------------------------------------------------------------------
# Diagnose
# --------------------------------------------------------------------------

def diagnose(cache_root, generators, log) -> Dict:
    fit_ids = common.load_sample_ids(FIT_TAG, cache_root)
    sel_ids = common.load_sample_ids(SELECT_TAG, cache_root)
    heldout = heldout_id_set(cache_root, generators)

    _, _, fit_labels = common.load_split(
        get_pool_experts("clip4")[:1], FIT_TAG, cache_root)
    _, _, sel_labels = common.load_split(
        get_pool_experts("clip4")[:1], SELECT_TAG, cache_root)

    report = {}
    for level in LEVELS:
        fm = contaminated_mask(fit_ids, level, heldout, generators)
        sm = contaminated_mask(sel_ids, level, heldout, generators)
        report[level] = {
            "fit_total": int(len(fm)),
            "fit_dropped": int(fm.sum()),
            "fit_dropped_pct": float(100.0 * fm.mean()),
            "fit_kept": int((~fm).sum()),
            "fit_kept_real": int(((fit_labels == 0) & ~fm).sum()),
            "fit_kept_fake": int(((fit_labels == 1) & ~fm).sum()),
            "select_total": int(len(sm)),
            "select_dropped": int(sm.sum()),
            "select_dropped_pct": float(100.0 * sm.mean()),
            "select_kept": int((~sm).sum()),
            "select_kept_real": int(((sel_labels == 0) & ~sm).sum()),
            "select_kept_fake": int(((sel_labels == 1) & ~sm).sum()),
        }
        r = report[level]
        log(f"{level:8s} FIT drop {r['fit_dropped']:6d}/{r['fit_total']} "
            f"({r['fit_dropped_pct']:5.2f}%) -> keep {r['fit_kept']} "
            f"(real {r['fit_kept_real']}, fake {r['fit_kept_fake']})")

    # test-side view
    per_gen = {}
    train_ids = set(fit_ids.tolist()) | set(sel_ids.tolist())
    total = 0
    for g in generators:
        gi = common.load_sample_ids(g, cache_root)
        _, _, gl = common.load_split(get_pool_experts("clip4")[:1], g, cache_root)
        hit = np.fromiter((str(s) in train_ids for s in gi),
                          dtype=bool, count=len(gi))
        per_gen[g] = {
            "n": int(len(gi)),
            "in_combiner_train": int(hit.sum()),
            "pct": float(100.0 * hit.mean()),
            "real_overlapping": int(((gl == 0) & hit).sum()),
            "fake_overlapping": int(((gl == 1) & hit).sum()),
        }
        total += int(hit.sum())
        p = per_gen[g]
        log(f"  {g:14s} {p['in_combiner_train']:5d}/{p['n']:5d} "
            f"({p['pct']:5.1f}%)  real={p['real_overlapping']} "
            f"fake={p['fake_overlapping']}")
    log(f"  TOTAL test samples present in combiner training: {total}")
    return {"by_level": report, "test_side": per_gen,
            "total_test_in_train": total}


# --------------------------------------------------------------------------

def run_level(level, pool, expert_names, cache_root, generators, seeds,
              skip_slow, stratify, matched_seeds, log) -> Dict:
    t0 = time.time()
    log("")
    log("=" * 70)
    log(f"LEVEL {level}  |  pool={pool}")
    log("=" * 70)

    fit_ids = common.load_sample_ids(FIT_TAG, cache_root)
    sel_ids = common.load_sample_ids(SELECT_TAG, cache_root)
    heldout = heldout_id_set(cache_root, generators)
    fit_drop = contaminated_mask(fit_ids, level, heldout, generators)
    sel_drop = contaminated_mask(sel_ids, level, heldout, generators)
    log(f"dropping FIT {int(fit_drop.sum())}/{len(fit_drop)} "
        f"({100 * fit_drop.mean():.2f}%), SELECT {int(sel_drop.sum())}/"
        f"{len(sel_drop)} ({100 * sel_drop.mean():.2f}%)")

    _, _, fit_labels_full = common.load_split(
        expert_names[:1], FIT_TAG, cache_root)
    _, _, sel_labels_full = common.load_split(
        expert_names[:1], SELECT_TAG, cache_root)

    log("")
    log("A. full training data (reproduces the published protocol)")
    full = _fit_and_score(
        expert_names, cache_root,
        np.ones(len(fit_drop), bool), np.ones(len(sel_drop), bool),
        seeds, skip_slow, generators, log, "full")

    log("")
    log("B. strict: contaminated rows removed")
    strict = _fit_and_score(
        expert_names, cache_root, ~fit_drop, ~sel_drop,
        seeds, skip_slow, generators, log, f"strict_{level}")

    matched: List[Dict] = []
    if matched_seeds:
        log("")
        log("C. matched-size control: same row and class counts, dropped at "
            "random, leakage left intact")
        for s in matched_seeds:
            fm = matched_random_mask(fit_labels_full, fit_drop, seed=s)
            sm = matched_random_mask(sel_labels_full, sel_drop, seed=s + 1)
            matched.append(_fit_and_score(
                expert_names, cache_root, ~fm, ~sm,
                seeds, skip_slow, generators, log, f"matched_s{s}"))

    d_full = full["comparison"]["delta"]
    d_strict = strict["comparison"]["delta"]
    d_match = (float(np.mean([m["comparison"]["delta"] for m in matched]))
               if matched else None)

    log("")
    log("-" * 70)
    log(f"delta full     = {d_full:+.4f}   {full['comparison']['generator_bootstrap']['ci_95']}")
    log(f"delta strict   = {d_strict:+.4f}   {strict['comparison']['generator_bootstrap']['ci_95']}")
    if d_match is not None:
        log(f"delta matched  = {d_match:+.4f}   (mean over {len(matched)} seeds)")
        log(f"  coverage-confounded contrast (strict - matched) = {d_strict - d_match:+.4f}")
        log(f"  descriptive contrast (matched - full)        = {d_match - d_full:+.4f}")
    log("-" * 70)

    res = {
        "run": "B_strict_path_disjoint",
        "level": level,
        "level_definition": {
            "exact": "training row dropped if its full canonical id appears "
                     "in any held-out generator split (same file)",
            "family": "training row dropped if its leading path component is "
                      "one of the held-out generator names (no exposure to a "
                      "held-out generator by any path)",
        }[level],
        "pool": pool,
        "experts": list(expert_names),
        "generators": list(generators),
        "seeds": list(seeds),
        "skip_slow": bool(skip_slow),
        "fit_dropped": int(fit_drop.sum()),
        "fit_dropped_pct": float(100.0 * fit_drop.mean()),
        "select_dropped": int(sel_drop.sum()),
        "select_dropped_pct": float(100.0 * sel_drop.mean()),
        "arm_full": _strip_private(full),
        "arm_strict": _strip_private(strict),
        "arm_matched": [_strip_private(m) for m in matched],
        "matched_seeds": list(matched_seeds),
        "summary": {
            "delta_full": d_full,
            "delta_strict": d_strict,
            "delta_matched_mean": d_match,
            "leakage_effect": (d_strict - d_match) if d_match is not None else None,
            "sample_size_effect": (d_match - d_full) if d_match is not None else None,
        },
        "published_reference": {
            "arch6_standard_protocol": 0.123,
            "arch6_standard_ci": [0.078, 0.174],
            "arch6_logo": 0.082,
            "arch6_logo_ci": [0.026, 0.140],
            "note": "Reviewed arch6 LOGO reference: each fold holds out one test generator "
                    "from combiner training. Other forms of reuse and prior "
                    "frozen-expert exposure can remain.",
        },
        "caveat": "Coverage-confounded combiner-stage stress test. The historical summary keys "
                  "leakage_effect and sample_size_effect hold strict minus matched and matched "
                  "minus full, respectively. Random removal retains overlap and leaves source "
                  "coverage unmatched. Frozen-expert training exposure is a separate upstream issue.",
        "elapsed_sec": float(time.time() - t0),
    }

    if stratify:
        res["contamination_stratification"] = _stratify(
            full, fit_ids, sel_ids, cache_root, generators, log)
    return res


def _stratify(full_arm, fit_ids, sel_ids, cache_root, generators, log) -> Dict:
    """Under the PUBLISHED fit, compare the margin on contaminated vs clean
    test samples. If it is similar in both, leakage is not what produces it."""
    log("")
    log("contamination stratification of TEST (under the full/published fit)")
    train_ids = set(fit_ids.tolist()) | set(sel_ids.tolist())
    flags: List[bool] = []
    for g in generators:
        gi = common.load_sample_ids(g, cache_root)
        flags.extend([str(s) in train_ids for s in gi])
    mask = np.asarray(flags, dtype=bool)

    y = full_arm["_test_labels"]
    a = full_arm["_fs1_scores"]
    b = full_arm["_ps_scores"]
    out = {"n_contaminated": int(mask.sum()), "n_clean": int((~mask).sum()),
           "strata": {}, "note": "pooled across generators, not macro-averaged"}
    for tag, m in (("contaminated", mask), ("clean", ~mask)):
        if m.sum() < 50 or len(np.unique(y[m])) < 2:
            out["strata"][tag] = {"n": int(m.sum()), "note": "too small or single-class"}
            continue
        fa, fb = float(compute_auc(y[m], a[m])), float(compute_auc(y[m], b[m]))
        out["strata"][tag] = {"n": int(m.sum()), "fs1_auc": fa,
                              "best_ps_auc": fb, "delta": fa - fb}
        log(f"  {tag:14s} n={int(m.sum()):6d}  FS_1={fa:.4f}  "
            f"PS={fb:.4f}  delta={fa - fb:+.4f}")
    return out


# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", default="arch6")
    p.add_argument("--levels", nargs="+", default=["family"], choices=list(LEVELS))
    p.add_argument("--diagnose", action="store_true")
    p.add_argument("--stratify", action="store_true")
    p.add_argument("--skip_matched", action="store_true",
                   help="Skip the matched-size control. Then the strict delta "
                        "is NOT interpretable on its own - do not quote it.")
    p.add_argument("--matched_seeds", type=int, nargs="+", default=[7, 42, 1024])
    p.add_argument("--skip_slow", action="store_true",
                   help="Skip PS_C/PS_F/PS_E/PS_4. Then 'best PS' is not the "
                        "ceiling. Smoke tests only.")
    p.add_argument("--seeds", type=int, nargs="+", default=common.DEFAULT_SEEDS)
    return p.parse_args()


def main():
    args = parse_args()
    log = common.make_logger("run_b_path_disjoint")
    expert_names = get_pool_experts(args.pool)
    cache_root = _paths.cache_root()
    generators = list(TEST_GENERATORS)

    log("RUN B - strict path-disjoint evaluation")
    log(f"pool={args.pool}  experts={expert_names}")
    log(f"writing to {_paths.RESULTS_DIR} (never into codebase/)")
    log("")

    diag = diagnose(cache_root, generators, log)
    common.write_json({"run": "B_diagnose", "pool": args.pool, **diag},
                      _paths.RESULTS_DIR / f"run_b_diagnose_{args.pool}.json")
    if args.diagnose:
        log("\ndiagnose-only; no refitting.")
        return

    if args.skip_slow:
        log("WARNING: --skip_slow -> best PS is not the ceiling. Do not quote.")
    if args.skip_matched:
        log("WARNING: --skip_matched -> the strict delta is confounded with "
            "training-set size and class balance. Do not quote it alone.")

    matched_seeds = [] if args.skip_matched else args.matched_seeds
    for level in args.levels:
        res = run_level(level, args.pool, expert_names, cache_root, generators,
                        args.seeds, args.skip_slow, args.stratify,
                        matched_seeds, log)
        out = common.write_json(
            res, _paths.RESULTS_DIR /
            f"run_b_path_disjoint_{args.pool}_{level}.json")
        log(f"saved: {out}")


if __name__ == "__main__":
    main()
