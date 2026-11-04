"""Compute the full LOGO prediction-space ceiling for each fold.

The ceiling takes the maximum AUROC across the ten base ladder methods and twelve richer-input candidates.
The result names the winning method and records FS1 and FS2. Inputs must agree on pool, exclusion, row
counts and recorded FIT/SELECT membership. Supplied runs that predate those fingerprints must appear
together in camera_ready/provenance/trusted_run_pairs.json.

An incomplete ladder stops the calculation unless --allow-incomplete-ladder is given, in which case
the report records the gaps under input_completeness.

From the repository root:

    python -B camera_ready/scripts/runs/09_logo_combined_ceiling_v3.py \
        --logo-json camera_ready/output/results/mitigation/content_disjoint/11_disjoint_arch6_d256.json \
        --rungs-json camera_ready/output/results/mitigation/content_disjoint/A3c_logo_extended_rungs_arch6.json \
        --out camera_ready/output/checks/A3c_logo_combined_ceiling_arch6.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cr_paths  # noqa: E402
import ladder_checks  # noqa: E402  version 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logo-json", type=Path, required=True)
    ap.add_argument("--rungs-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="default: next to --rungs-json, A3c_logo_combined_ceiling_<pool>.json")
    ap.add_argument("--allow-incomplete-ladder", action="store_true",
                    help="version 3: compute the ceiling from an incomplete ladder and record the gaps")
    a = ap.parse_args()
    ext = json.loads(a.rungs_json.read_text(encoding="utf-8"))
    pub = json.loads(a.logo_json.read_text(encoding="utf-8"))
    per_fold = {f["held_out"]: f for f in pub["per_fold"]}
    why = ladder_checks.pairing_problem(pub, ext, a.logo_json, a.rungs_json,           # version 3
                                        ("the LOGO run", "the richer-input run"))
    if why:
        raise SystemExit(f"{a.rungs_json} cannot complete {a.logo_json}: {why}. Nothing was written.")
    gaps = ladder_checks.ladder_gaps(pub, ext)
    if gaps and not a.allow_incomplete_ladder:
        raise SystemExit("the prediction-space ladder is not complete: " + "; ".join(gaps) + ". "
                         "--allow-incomplete-ladder computes the ceiling from what is there and records the gaps. "
                         "Nothing was written.")

    rows, combined = [], []
    for f in pub["per_fold"]:
        gen = f["held_out"]
        rec = ext["folds"][gen]
        p = per_fold[gen]["best_ps_auc"]
        x = max(rec["rungs"].values())
        best_x = max(rec["rungs"], key=rec["rungs"].get)
        c = max(p, x)
        combined.append(c)
        rows.append((gen, p, x, best_x, c, p >= x))

    print(f"{'fold':<14}{'ladder':>11}{'richer':>10}{'combined':>10}   best richer input")
    for gen, p, x, best_x, c, pub_wins in rows:
        flag = "" if pub_wins else "   <- richer input sets the ceiling"
        print(f"{gen:<14}{p:>11.4f}{x:>10.4f}{c:>10.4f}   {best_x}{flag}")

    fs1 = pub["avg_raw_feat_logreg"]
    base = pub["avg_best_ps"]
    comb = float(np.mean(combined))
    out = {
        "pool": pub["pool"], "version": "09_logo_combined_ceiling_v3", "exclusion": pub.get("exclusion"),
        "published_ceiling": base, "extended_only_ceiling": float(np.mean([r[2] for r in rows])),
        "combined_ceiling": comb, "fs1": fs1, "fs2": pub.get("avg_fs_mlp"),
        "delta_published_ladder": fs1 - base, "delta_ladder_plus_rungs": fs1 - comb,
        "folds_where_extended_wins": sum(1 for r in rows if not r[5]),
        "per_fold": [
            {"fold": g, "published": p, "extended": x, "best_extended_rung": b, "combined": c,
             "ceiling_method": (per_fold[g]["best_ps_name"] if pw else b),
             "fs1": per_fold[g]["raw_feat_auc"], "fs2": per_fold[g].get("fs_mlp_mean"),
             "n_test": per_fold[g]["n_test"]}
            for g, p, x, b, c, pw in rows
        ],
        "inputs": {"logo_json": _cr_paths.rel(a.logo_json), "rungs_json": _cr_paths.rel(a.rungs_json)},
    }
    if gaps:                                                                  # version 3
        out["input_completeness"] = {"complete": False, "options": ["--allow-incomplete-ladder"], "notes": gaps}
    dest = a.out or a.rungs_json.parent / f"A3c_logo_combined_ceiling_{pub['pool']}.json"
    dest = _cr_paths.out_path(dest)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nLOGO FS1 - full-ladder ceiling: {fs1 - comb:+.4f} (ten methods only: {fs1 - base:+.4f})")
    print(f"wrote {_cr_paths.rel(dest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
