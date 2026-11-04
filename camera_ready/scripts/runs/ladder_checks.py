"""Checks shared by the LOGO statistics, the combined ceiling and the sensitivity summary (26 October 2026).

1. ``ladder_gaps``: is the full prediction-space ladder complete in every analysed fold? The LOGO run must hold the
   ten base methods, with its recorded best equal to the best of them, and the richer-input run the twelve
   candidates (four input maps, each with a logistic head and XGBoost heads over the "capped" and the "widened"
   grids), or, for the sensitivity check with the heads of the standard split, its ten candidates. A run of one grid
   only is named as such. An empty list means complete.
2. ``pairing_problem``: can two runs be analysed together (a LOGO run with its richer-input run, mechanism run,
   combined ceiling or paper-heads run)? They must have the same pool and exclusion, and every fold of the first must
   be in the second with the same row counts (test, FIT, SELECT, excluded training rows). Runs that record the
   membership fingerprints of FIT and SELECT (version 4 of the LOGO runner, version 3 of the richer-input runner and
   version 5 of the width-control runner) must have equal fingerprints; when either run lacks them (the stored runs,
   written by earlier versions), the two files must be listed in the same group of the trusted provenance mapping
   (``camera_ready/provenance/trusted_run_pairs.json``). Matching fold names are never enough.
3. ``membership_fingerprint``: the identity of a set of rows, independent of their order: the SHA-256 of the sorted
   SHA-1 digests of the rows' concatenated features (``logo_common.row_hashes``).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE_METHODS = ("PS_1_prob_avg", "PS_A_calibrated_avg", "PS_B_learned_weights", "PS_2_logreg", "PS_3_mlp_logits",
                "PS_D_mlp_probs", "PS_C_logreg_cv", "PS_F_mlp_logits_es", "PS_E_mlp_probs_es", "PS_4_xgboost")
RICHER_MAPS = ("disagreement", "uncertainty+MI", "quadratic", "rank ECDF")
RICHER_CANDIDATES = tuple(f"{m}_{h}" for m in RICHER_MAPS for h in ("logreg", "xgb_capped", "xgb_widened"))
PAPER_HEADS_CANDIDATES = tuple(f"{m}_{h}" for m in ("G_disagreement", "H_uncertainty_mi", "I_quadratic", "J_rank_ecdf",
                                                    "K_cv_selected") for h in ("LR", "XGB"))
COUNTS = ("n_test", "n_fit", "n_select", "n_excluded_training_rows")
FINGERPRINTS = ("fit_fingerprint", "select_fingerprint")
TRUSTED = Path(__file__).resolve().parents[2] / "provenance" / "trusted_run_pairs.json"


def fold_records(run) -> dict:
    """{held-out generator: fold record} of a LOGO, mechanism, richer-input or combined-ceiling run."""
    per = run.get("per_fold")
    if isinstance(per, list):
        return {f.get("held_out", f.get("fold")): f for f in per}
    folds = run.get("folds")
    return dict(folds) if isinstance(folds, dict) else {}


def ladder_gaps(logo, rungs, folds=None, candidates=RICHER_CANDIDATES) -> list:
    """What the full ladder lacks in the analysed folds (`folds`, default every fold of the LOGO run)."""
    gaps = []
    per = fold_records(logo)
    rich = fold_records(rungs) if rungs is not None else {}
    for g in folds or list(per):
        f = per.get(g) or {}
        ps = f.get("ps_means") or {}
        missing = [m for m in BASE_METHODS if m not in ps]
        if missing:
            gaps.append(f"{g}: {len(missing)} of the {len(BASE_METHODS)} base methods missing ({', '.join(missing)})")
        elif "best_ps_auc" in f:
            best = max(BASE_METHODS, key=lambda m: ps[m])
            if abs(ps[best] - f["best_ps_auc"]) > 1e-12 or f.get("best_ps_name", best) != best:
                gaps.append(f"{g}: the recorded best base method ({f.get('best_ps_name')}, {f['best_ps_auc']}) is not "
                            f"the best of the ten ({best}, {ps[best]})")
        if rungs is None:
            continue
        r = rich.get(g) or {}
        have = r.get("rungs", r.get("auc")) or {}
        absent = [c for c in candidates if c not in have]
        if not absent:
            continue
        if candidates == RICHER_CANDIDATES:
            capped = all(f"{m}_xgb_capped" in have for m in RICHER_MAPS)
            widened = all(f"{m}_xgb_widened" in have for m in RICHER_MAPS)
            if capped and not any(f"{m}_xgb_widened" in have for m in RICHER_MAPS):
                gaps.append(f"{g}: a capped-only richer-input run (no XGBoost head over the widened grid)")
                continue
            if widened and not any(f"{m}_xgb_capped" in have for m in RICHER_MAPS):
                gaps.append(f"{g}: a widened-only richer-input run (no XGBoost head over the capped grid)")
                continue
        gaps.append(f"{g}: {len(absent)} of the {len(candidates)} richer-input candidates missing ({', '.join(absent)})")
    return gaps


def content_digest(path) -> str:
    """SHA-256 of a file's content with its line endings made CRLF, as the stored files have them, so that a copy
    with other line endings (a checkout or a rewrite on Linux or macOS) has the same digest."""
    b = Path(path).read_bytes()
    return hashlib.sha256(b.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()


def trusted_groups(path=None) -> list:
    """The groups of the trusted provenance mapping, each a set of content digests."""
    p = Path(path) if path is not None else TRUSTED
    if not p.is_file():
        return []
    m = json.loads(p.read_text(encoding="utf-8"))
    return [set(g["files"].values()) for g in m["groups"]]


def pairing_problem(left, right, left_path=None, right_path=None, names=("first run", "second run"), trusted=None,
                    folds=None):
    """Why `right` cannot be analysed with `left` (None when it can). `folds`: the folds to compare (default every
    fold of `left`)."""
    a, b = names
    for k in ("pool", "exclusion"):
        if left.get(k) is not None and right.get(k) is not None and left[k] != right[k]:
            return f"{b}: {k} {right[k]!r}, {a}: {left[k]!r}"
    lf, rf = fold_records(left), fold_records(right)
    use = folds or list(lf)
    lacking = [g for g in use if g not in rf]
    if lacking:
        return f"{b} has no fold for {', '.join(lacking)}, which {a} has"
    for g in use:
        for k in COUNTS:
            x, y = lf[g].get(k), rf[g].get(k)
            if x is not None and y is not None and x != y:
                return f"fold {g}: {k} {y} in {b}, {x} in {a} (another split or training set)"
    printed = all(lf[g].get(k) and rf[g].get(k) for g in use for k in FINGERPRINTS)
    if printed:
        for g in use:
            for k in FINGERPRINTS:
                if lf[g][k] != rf[g][k]:
                    return f"fold {g}: {k} differs between {a} and {b} (other training rows)"
        return None
    groups = trusted_groups(trusted)
    if left_path is not None and right_path is not None:
        da, db = content_digest(left_path), content_digest(right_path)
        if any(da in g and db in g for g in groups):
            return None
    return (f"{a} and {b} record no membership fingerprints of FIT and SELECT, and the two files are not listed "
            f"together in the trusted provenance mapping ({TRUSTED.name}); their compatibility cannot be checked")


def membership_fingerprint(feats) -> str:
    """Order-free identity of a set of rows (a list of [N, D_i] feature arrays, one per expert)."""
    from logo_common import row_hashes
    return hashlib.sha256("\n".join(sorted(row_hashes(feats))).encode("ascii")).hexdigest()
