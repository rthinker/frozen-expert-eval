# C13: Compare the published and corrected score conventions.
"""Diff the corrected Appendix H per-expert table against the as-published one.

Reads the as-published table shipped with the reviewed code (output/results/) and the
corrected table written by camera_ready/scripts/authoring/regenerate_appendix_h_v3.py;
writes camera_ready/output/results/analyses/A2_appendix_h_diff.json.

Cause of the change: scripts/16_individual_expert_auc.py ranked by the raw
fake-class logit (logits[:, 1]); every other scorer in the codebase ranks by
the decision margin logits[:, 1] - logits[:, 0]. These are different orderings
whenever the real-class logit varies across samples.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]          # repository root (the reviewed code)
CAMERA_READY = REPO / "camera_ready"
OUT = CAMERA_READY / "output" / "results" / "analyses"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, help="output file; default: the stored camera_ready/output/results/analyses/"
                    "A2_appendix_h_diff.json, which only the reproducibility harness writes")
    a = ap.parse_args()
    if a.out is None and not os.environ.get("REPRO_ATTEMPT_ID"):
        raise SystemExit("run by hand, this script writes only where --out says: its default output is the stored "
                         "camera_ready/output/results/analyses/A2_appendix_h_diff.json, which the harness rewrites "
                         "after backing it up (python camera_ready/scripts/review/reproduce_all_v4.py --only C13).")
    out = a.out or OUT / "A2_appendix_h_diff.json"
    pub = json.loads(
        (REPO / "output/results/16_individual_expert_auc.json")
        .read_text(encoding="utf-8")
    )
    cor = json.loads(
        (CAMERA_READY / "output/results/16_individual_expert_auc_corrected.json")
        .read_text(encoding="utf-8")
    )
    old, new = pub["per_expert"], cor["per_expert"]

    rows = sorted(
        (
            (abs(new[e]["avg_auc_8"] - old[e]["avg_auc_8"]), e,
             old[e]["avg_auc_8"], new[e]["avg_auc_8"])
            for e in old
        ),
        reverse=True,
    )

    print(f"{'expert':<18}{'published':>10}{'corrected':>11}{'delta':>10}")
    for _, e, o, n in rows:
        print(f"{e:<18}{o:>10.4f}{n:>11.4f}{n - o:>+10.4f}")

    d = [r[0] for r in rows]
    ov = [v["avg_auc_8"] for v in old.values()]
    nv = [v["avg_auc_8"] for v in new.values()]

    cells_old = [x for v in old.values() for x in v["per_generator"].values()]
    cells_new = [x for v in new.values() for x in v["per_generator"].values()]
    cell_d = [
        abs(new[e]["per_generator"][g] - old[e]["per_generator"][g])
        for e in old for g in old[e]["per_generator"]
    ]

    rank_old = [e for _, e in sorted(((o, e) for _, e, o, _ in rows), reverse=True)]
    rank_new = [e for _, e in sorted(((n, e) for _, e, _, n in rows), reverse=True)]

    summary = {
        "cause": "score convention: logits[:,1] -> logits[:,1] - logits[:,0]",
        "experts_changed": sum(1 for x in d if x > 1e-9),
        "experts_changed_ge_0.002": sum(1 for x in d if x >= 0.002),
        "mean_abs_delta": sum(d) / len(d),
        "max_abs_delta": max(d),
        "published_range": [min(ov), max(ov)],
        "corrected_range": [min(nv), max(nv)],
        "cells_ge_0.999_published": sum(1 for x in cells_old if x >= 0.999),
        "cells_ge_0.999_corrected": sum(1 for x in cells_new if x >= 0.999),
        "max_abs_cell_delta": max(cell_d),
        "rank_order_preserved": rank_old == rank_new,
        "per_expert": {
            e: {"published": o, "corrected": n, "delta": n - o}
            for _, e, o, n in rows
        },
    }

    print()
    for k, v in summary.items():
        if k != "per_expert":
            print(f"{k:<28} {v}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
