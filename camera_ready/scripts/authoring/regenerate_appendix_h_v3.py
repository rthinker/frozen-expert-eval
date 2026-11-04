"""Recompute the individual-expert AUROCs for Appendix H from cached logits.

The score is the fake-class logit minus the real-class logit, formed in
float32. The script checks all 168 raw fake-logit AUROCs against
output/results/16_individual_expert_auc.json and checks the 21 corrected
expert averages against camera_ready/inputs/A2_appendix_h_diff.json before
writing the corrected table. It fits no models.

The harness runs this calculation as C12. For standalone use, pass --out
with a path outside the stored results. Use --cache to select the cache root.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve()
CODE = _HERE.parents[2]                        # camera_ready/
CH3 = CODE.parent                              # repository root (the reviewed code); name kept
CACHE = CH3 / "output" / "cached_features"
PUBLISHED = CH3 / "output" / "results" / "16_individual_expert_auc.json"
CORRECTED_MEANS = CODE / "inputs" / "A2_appendix_h_diff.json"  # the stored July record
OUT = CODE / "output" / "results" / "16_individual_expert_auc_corrected.json"
TOL = 1e-9

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


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=OUT, help="output file (version 3; default: the stored file)")
    args = ap.parse_args()
    out = args.out.resolve()
    refuse_stored_write(out)                                           # version 3: before any work

    pub = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    a2 = json.loads(CORRECTED_MEANS.read_text(encoding="utf-8"))["per_expert"]
    experts, gens = pub["experts"], pub["generators"]

    raw, margin, failures = {}, {}, []
    for e in experts:
        raw[e], margin[e] = {}, {}
        for g in gens:
            lg = np.load(args.cache / g / f"{e}_logits.npy")          # stored as float32
            y = np.load(args.cache / g / "labels.npy").astype(np.int32)
            raw[e][g] = auc(y, lg[:, 1].astype(np.float64))
            margin[e][g] = auc(y, (lg[:, 1] - lg[:, 0]).astype(np.float64))
            d = abs(raw[e][g] - pub["per_expert"][e]["per_generator"][g])
            if d > TOL:
                failures.append(f"raw cell {e}/{g} differs from the published table by {d:.3g}")
        m = float(np.mean([margin[e][g] for g in gens]))
        d = abs(m - a2[e]["corrected"])
        if d > TOL:
            failures.append(f"margin average {e} differs from the recorded corrected average by {d:.3g}")

    if failures:
        for f in failures:
            print("FAIL", f)
        print(f"{len(failures)} self-check failure(s); nothing written")
        return 1

    rows = {e: {"per_generator": margin[e], "avg_auc_8": float(np.mean([margin[e][g] for g in gens])),
                "published_avg_auc_8_raw_logit": pub["per_expert"][e]["avg_auc_8"]} for e in experts}
    arch6 = ["fs-clip-base", "fs-clip-large", "fs-xception", "fs-srm", "fs-recce", "fs-rfm"]
    best_arch6 = max(arch6, key=lambda e: rows[e]["avg_auc_8"])
    best_all = max(experts, key=lambda e: rows[e]["avg_auc_8"])
    cells_moved = sum(abs(margin[e][g] - raw[e][g]) >= 0.0005 for e in experts for g in gens)
    doc = {
        "description": "Per-expert individual AUROC on the 8 held-out generators under the corrected score "
                       "convention (margin logits[:,1] - logits[:,0], formed in the stored float32 precision); "
                       "sklearn roc_auc_score, no training. Replaces the raw fake-logit convention of the "
                       "submitted Appendix H table.",
        "metric_definitions": {"cells_moving_at_3dp": "Count of cells with absolute AUROC change >= 0.0005; "
                               "not the count of changed three-decimal displays."},
        "experts": experts, "generators": gens, "per_expert": rows,
        "summary": {"best_arch6_expert": best_arch6, "best_arch6_avg": rows[best_arch6]["avg_auc_8"],
                    "best_overall_expert": best_all, "best_overall_avg": rows[best_all]["avg_auc_8"],
                    "cells_moving_at_3dp": int(cells_moved)},
        "self_checks": {"raw_cells_match_published_within": TOL, "margin_averages_match_recorded_within": TOL,
                        "passed": True},
        "inputs": {"published_table_sha256": sha256(PUBLISHED), "recorded_corrected_means_sha256": sha256(CORRECTED_MEANS)},
        "_written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    raw_avg = {e: float(np.mean([raw[e][g] for g in gens])) for e in experts}
    doc["summary"]["rank_order_of_averages_preserved"] = (
        sorted(experts, key=lambda e: -raw_avg[e]) == sorted(experts, key=lambda e: -rows[e]["avg_auc_8"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"self-checks passed: 168 raw cells match the published table; 21 margin averages match the record")
    print(f"best arch6 expert {best_arch6} {rows[best_arch6]['avg_auc_8']:.4f}; best overall {best_all} "
          f"{rows[best_all]['avg_auc_8']:.4f}; cells with |AUROC change| >= 0.0005: {cells_moved}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
