"""Check FS1 score precision from saved predictions.

The ladder records AUROC before casting FS1 scores to float32 for storage.
The mechanism runner casts first, then records AUROC. Casting can introduce
ties. This check compares their saved arrays and recomputes the latter metric
from each, without fitting a model.

From the repository root:
    python -B camera_ready/scripts/review/check_fs1_score_precision_v2.py --out camera_ready/output/checks/score_precision_check.json

An output file is written only if --out is given. Use an output path outside
the stored results when running this check directly. The harness runs it as B30.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def check(mitigation_dir: Path) -> dict:
    rows = []
    for pool, dim in (("arch6", 256), ("clip4", 32)):
        logo_path = mitigation_dir / "content_disjoint" / f"11_disjoint_{pool}_d{dim}.json"
        mech_path = mitigation_dir / "mechanism" / f"run_c_v2_logo_{pool}.json"
        logo = json.loads(logo_path.read_text(encoding="utf-8"))
        mech = json.loads(mech_path.read_text(encoding="utf-8"))
        ladder_folds = {f["held_out"]: f for f in logo["per_fold"]}
        mech_folds = {f["held_out"]: f for f in mech["per_fold"]}
        if set(ladder_folds) != set(mech_folds) or len(ladder_folds) != 8:
            raise ValueError(f"{pool}: expected the same eight folds in both records")
        for gen, fold in ladder_folds.items():
            mf = mech_folds[gen]
            with np.load(mitigation_dir / "content_disjoint" / fold["scores_file"], allow_pickle=False) as ls, \
                    np.load(mitigation_dir / "mechanism" / mf["scores_file"], allow_pickle=False) as ms:
                labels_equal = np.array_equal(ls["labels"], ms["labels"])
                scores_equal = np.array_equal(ls["FS_1_raw_feat_logreg"], ms["FS_1_raw_feat_logreg"])
                dtype_ok = ls["FS_1_raw_feat_logreg"].dtype == np.dtype("float32") and \
                    ms["FS_1_raw_feat_logreg"].dtype == np.dtype("float32")
                auc = float(roc_auc_score(ls["labels"], ls["FS_1_raw_feat_logreg"]))
                mech_auc = float(roc_auc_score(ms["labels"], ms["FS_1_raw_feat_logreg"]))
            stored_ladder = float(fold["raw_feat_auc"])
            stored_mech = float(mf["auroc"]["FS_1_raw_feat_logreg"])
            ok = labels_equal and scores_equal and dtype_ok and abs(auc - stored_mech) <= 1e-12 and \
                abs(mech_auc - stored_mech) <= 1e-12
            rows.append({"pool": pool, "held_out": gen, "labels_equal": labels_equal,
                         "saved_float32_scores_equal": scores_equal, "both_float32": dtype_ok,
                         "stored_ladder_pre_cast_auc": stored_ladder, "stored_mechanism_post_cast_auc": stored_mech,
                         "auc_from_saved_float32_scores": auc,
                         "ladder_minus_saved_score_auc": stored_ladder - auc,
                         "mechanism_minus_saved_score_auc": stored_mech - auc, "status": "PASS" if ok else "FAIL"})
    return {"status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL", "n_folds": len(rows),
            "max_pre_vs_post_cast_auc_difference": max(abs(r["ladder_minus_saved_score_auc"]) for r in rows),
            "max_mechanism_vs_saved_score_auc_difference": max(abs(r["mechanism_minus_saved_score_auc"]) for r in rows),
            "interpretation": "The saved FS1 predictions are identical between runners. Scoring their float32 "
                              "arrays reproduces the mechanism AUROCs and the recorded gap to the ladder's "
                              "pre-cast AUROCs. These records do not support attributing that gap to process drift. "
                              "Original full-precision prediction vectors were not stored.",
            "per_fold": rows}


def stored_problem(path):
    """Why a run started by hand may not write `path` (None when it may)."""
    if os.environ.get("REPRO_ATTEMPT_ID"):
        return None
    cr = Path(__file__).resolve().parents[2]
    stored = (cr / "output" / "results", cr / "inputs", cr / "reference", cr / "provenance",
              cr.parent / "output" / "results", cr.parent / "output" / "cached_features")
    p = os.path.normcase(os.path.realpath(path))
    for f in stored:
        f = os.path.normcase(os.path.realpath(f))
        if p == f or p.startswith(f.rstrip(os.sep) + os.sep):
            return "it lies among the stored results"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mitigation-dir", type=Path,
                    default=Path(__file__).resolve().parents[2] / "output" / "results" / "mitigation")
    ap.add_argument("--out", type=Path, help="optional JSON report; never overwrites run records")
    args = ap.parse_args()
    report = check(args.mitigation_dir)
    if args.out is not None:
        target = args.out.resolve()
        why = stored_problem(target)
        if why:
            raise SystemExit(f"refusing to write {target}: {why}. Run by hand, write the report elsewhere (for "
                             "example camera_ready/output/checks/); the harness rewrites the stored report (step "
                             "B30) after backing it up.")
        if target.is_relative_to(args.mitigation_dir.resolve()) and target.name != "fs1_score_precision_check.json":
            raise SystemExit("inside the run folder, use fs1_score_precision_check.json to preserve original records")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"score precision: {report['status']}; {report['n_folds']} fold pairs; "
          f"largest pre/post-cast AUROC gap {report['max_pre_vs_post_cast_auc_difference']:.12g}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
