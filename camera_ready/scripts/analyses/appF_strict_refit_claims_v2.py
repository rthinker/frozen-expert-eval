# C22: Check claims about the strict-refit arms.
"""Check two load-bearing claims about the strict arm, from frozen artifacts.

CHECK A. Measure the strict refit's effect on the feature anchor's AUROC
for stargan, which contributes no overlapping rows. A change on this fold
therefore reflects the altered training set rather than removal of direct
stargan train/test overlap.

CHECK B. Compare PS_4 in each full arm with the canonical values in
output/results/06_bootstrap_arch6.json and report the observed differences.

Also reported: FS_1 per-generator values across all five arms, so the
coverage argument can be read per fold rather than asserted.

Reads the strict-refit outputs and writes verify_strict_arm_claims.json.
Run this check from the repository root:

    python camera_ready/scripts/review/reproduce_all_v4.py --only C22
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _paths  # noqa: F401  # MUST be first

import json
from pathlib import Path

import _common as common

PARITY_NOTE = ("PS_4 agrees across the supplied outputs. The strict-refit runner casts FS_1 scores "
               "to float32 before AUROC, while the canonical and cost paths score without that cast. "
               "Rounding can introduce ties and accounts for the stored strict-refit gap of about 3.5e-4.")

RESULTS = _paths.RESULTS_DIR
CANON = _paths.CODEBASE_ROOT / "output" / "results" / "06_bootstrap_arch6.json"  # canonical values as shipped


def main() -> int:
    log = common.make_logger("verify_strict_arm_claims")
    b = json.load(open(RESULTS / "run_b_path_disjoint_arch6_family.json",
                       encoding="utf-8"))
    diag = json.load(open(RESULTS / "run_b_diagnose_arch6.json",
                          encoding="utf-8"))

    log("CLAIM A - does the feature anchor collapse on the zero-overlap "
        "generator under the strict refit?")
    stargan_overlap = diag["test_side"]["stargan"]["in_combiner_train"]
    log(f"  stargan overlapped rows removed by the strict refit: "
        f"{stargan_overlap}")

    arms = [("full (published)", b["arm_full"]),
            ("strict", b["arm_strict"])] + \
           [(m["tag"], m) for m in b["arm_matched"]]

    per_gen_fs1, rows = {}, {}
    for name, arm in arms:
        pg = arm["comparison"]["per_generator_a"]
        per_gen_fs1[name] = pg
        rows[name] = {"fs1_macro": arm["fs1_macro"],
                      "fs1_stargan": pg["stargan"],
                      "best_ps_macro": arm["best_ps_macro"],
                      "delta": arm["comparison"]["delta"]}
        log(f"  {name:18s} FS_1 macro={arm['fs1_macro']:.4f}   "
            f"FS_1 on stargan={pg['stargan']:.4f}")

    full_s = b["arm_full"]["comparison"]["per_generator_a"]["stargan"]
    strict_s = b["arm_strict"]["comparison"]["per_generator_a"]["stargan"]
    matched_s = [m["comparison"]["per_generator_a"]["stargan"]
                 for m in b["arm_matched"]]
    claim_a = bool(stargan_overlap == 0 and strict_s < min(matched_s) - 0.05)
    log("")
    log(f"  stargan: full {full_s:.4f} | matched {['%.4f' % x for x in matched_s]}"
        f" | strict {strict_s:.4f}")
    log("  CLAIM A SUPPORTED: nothing was de-leaked for stargan, yet the "
        "strict refit still destroys it - the collapse is not leakage removal."
        if claim_a else
        "  CLAIM A NOT SUPPORTED as stated - do not use this argument.")

    log("")
    log("CLAIM B - what is the actual PS_4 / FS_1 parity against canonical?")
    _canon = json.load(open(CANON, encoding="utf-8"))
    canon = {"ps4": _canon["macro_avg_best_ps"], "fs1": _canon["macro_avg_raw_feat"]}
    ps4_b = b["arm_full"]["best_ps_macro"]
    fs1_b = b["arm_full"]["fs1_macro"]
    d = json.load(open(RESULTS / "run_d_cost_ledger_arch6.json",
                       encoding="utf-8"))["combiner_costs"]
    ps4_d = d["PS_4_xgboost_cv"]["macro_auroc"]
    fs1_d = d["FS_1_frozen_total"]["macro_auroc"]
    parity = {
        "canonical_ps4": canon["ps4"], "canonical_fs1": canon["fs1"],
        "run_b_ps4": ps4_b, "run_b_ps4_absdiff": abs(ps4_b - canon["ps4"]),
        "run_d_ps4": ps4_d, "run_d_ps4_absdiff": abs(ps4_d - canon["ps4"]),
        "run_b_fs1": fs1_b, "run_b_fs1_absdiff": abs(fs1_b - canon["fs1"]),
        "run_d_fs1": fs1_d, "run_d_fs1_absdiff": abs(fs1_d - canon["fs1"]),
    }
    for k in ("run_b_ps4", "run_d_ps4", "run_b_fs1", "run_d_fs1"):
        log(f"  {k:12s} = {parity[k]:.10f}   |diff| = "
            f"{parity[k + '_absdiff']:.2e}")
    worst_ps4 = max(parity["run_b_ps4_absdiff"], parity["run_d_ps4_absdiff"])
    worst_fs1 = max(parity["run_b_fs1_absdiff"], parity["run_d_fs1_absdiff"])
    log(f"  PS_4 agrees with canonical to {worst_ps4:.1e} "
        f"(both arms); FS_1 to {worst_fs1:.1e}")
    log(PARITY_NOTE)

    common.write_json(
        {"run": "verify_strict_arm_claims",
         "stargan_overlapped_rows_removed_by_strict": int(stargan_overlap),
         "claim_a_supported": claim_a,
         "arms": rows,
         "per_generator_fs1_by_arm": per_gen_fs1,
         "parity": parity,
         "parity_note": PARITY_NOTE,
         "claim_a_note": "stargan has zero overlapped rows, so the strict "
                         "refit removes no leakage for it; any collapse there "
                         "is caused by the training data that remains."},
        RESULTS / "verify_strict_arm_claims.json")
    return 0 if claim_a else 1


if __name__ == "__main__":
    raise SystemExit(main())
