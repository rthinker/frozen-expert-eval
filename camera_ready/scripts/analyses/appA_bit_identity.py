# C05: Compare feature and logit arrays for rows reused across splits.
"""Measure bit identity and label agreement for overlapping sample ids.

For each held-out generator and each arch6 expert, compare the cached feature
rows with the corresponding FIT/SELECT rows using exact equality on the stored
float16 values. Repeat the comparison for logits and count label disagreements.

Report the number of overlapping ids, the subsets identical on every expert's
features or logits, and the label-disagreement count. The global verdict tests
whether all overlapping rows have identical features; partial agreement is
reported explicitly. Inputs are read without modification; results are written
to verify_bit_identity.json in the configured results directory.

Public entry point: camera_ready/scripts/analyses/appA_bit_identity.py.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _paths  # noqa: F401  # MUST be first

from pathlib import Path

import numpy as np

import _common as common
from src.utils.config import FIT_TAG, SELECT_TAG, get_pool_experts  # noqa: E402
from src.utils.dataset_helpers import TEST_GENERATORS  # noqa: E402


def main() -> int:
    log = common.make_logger("verify_bit_identity")
    cache_root = Path(_paths.cache_root())
    generators = list(TEST_GENERATORS)
    experts = get_pool_experts("arch6")

    log("BIT-IDENTITY verification of the overlapping rows")
    log(f"experts: {experts}")

    # index the two training splits by sample id -> (split_tag, row)
    train_pos = {}
    for tag in (FIT_TAG, SELECT_TAG):
        ids = common.load_sample_ids(tag, cache_root)
        for i, s in enumerate(ids):
            train_pos.setdefault(str(s), (tag, i))
    train_labels = {tag: np.load(cache_root / tag / "labels.npy")
                    for tag in (FIT_TAG, SELECT_TAG)}
    log(f"indexed {len(train_pos)} unique training sample ids")

    per_gen = {}
    tot_overlap = tot_feat_ok = tot_logit_ok = tot_label_mismatch = 0

    for g in generators:
        gids = common.load_sample_ids(g, cache_root)
        glab = np.load(cache_root / g / "labels.npy")
        pairs = [(i, train_pos[str(s)]) for i, s in enumerate(gids)
                 if str(s) in train_pos]
        n = len(pairs)
        if n == 0:
            per_gen[g] = {"n_overlap": 0, "note": "no overlapping ids"}
            log(f"  {g:14s} no overlap")
            continue

        g_idx = np.array([p[0] for p in pairs])
        # group the training side by split so each file is read once
        by_tag = {}
        for k, (i, (tag, j)) in enumerate(pairs):
            by_tag.setdefault(tag, []).append((k, j))

        feat_ok = np.ones(n, dtype=bool)
        logit_ok = np.ones(n, dtype=bool)

        for e in experts:
            gf = np.load(cache_root / g / f"{e}_features.npy", mmap_mode="r")
            gl = np.load(cache_root / g / f"{e}_logits.npy", mmap_mode="r")
            for tag, lst in by_tag.items():
                ks = np.array([k for k, _ in lst])
                js = np.array([j for _, j in lst])
                tf = np.load(cache_root / tag / f"{e}_features.npy",
                             mmap_mode="r")
                tl = np.load(cache_root / tag / f"{e}_logits.npy",
                             mmap_mode="r")
                a = np.asarray(gf[g_idx[ks]])
                b = np.asarray(tf[js])
                feat_ok[ks] &= np.all(a == b, axis=1)
                a = np.asarray(gl[g_idx[ks]])
                b = np.asarray(tl[js])
                logit_ok[ks] &= np.all(a == b, axis=1)

        lab_mismatch = 0
        for k, (i, (tag, j)) in enumerate(pairs):
            if int(glab[i]) != int(train_labels[tag][j]):
                lab_mismatch += 1

        per_gen[g] = {
            "n_overlap": int(n),
            "features_bit_identical": int(feat_ok.sum()),
            "logits_bit_identical": int(logit_ok.sum()),
            "features_pct": float(100.0 * feat_ok.mean()),
            "label_mismatches": int(lab_mismatch),
        }
        tot_overlap += n
        tot_feat_ok += int(feat_ok.sum())
        tot_logit_ok += int(logit_ok.sum())
        tot_label_mismatch += lab_mismatch
        log(f"  {g:14s} overlap={n:5d}  features bit-identical="
            f"{int(feat_ok.sum()):5d} ({100.0 * feat_ok.mean():6.2f}%)  "
            f"logits={int(logit_ok.sum()):5d}  label mismatches={lab_mismatch}")

    log("")
    log(f"TOTAL overlapping rows          : {tot_overlap}")
    log(f"  bit-identical on ALL features : {tot_feat_ok} "
        f"({100.0 * tot_feat_ok / max(tot_overlap, 1):.2f}%)")
    log(f"  bit-identical on ALL logits   : {tot_logit_ok}")
    log(f"  label disagreements           : {tot_label_mismatch}")

    claim_holds = tot_feat_ok == tot_overlap and tot_overlap > 0
    log("CLAIM SUPPORTED: every overlapping row is the same image."
        if claim_holds else
        "CLAIM NEEDS QUALIFYING: not all overlapping rows are bit-identical - "
        "the measured fraction is reported above.")

    common.write_json(
        {"run": "verify_bit_identity", "pool": "arch6",
         "experts": list(experts),
         "n_overlap_total": tot_overlap,
         "features_bit_identical_total": tot_feat_ok,
         "logits_bit_identical_total": tot_logit_ok,
         "label_mismatches_total": tot_label_mismatch,
         "claim_supported": bool(claim_holds),
         "per_generator": per_gen,
         "definition": "exact equality (==) on the raw stored arrays for every "
                       "arch6 expert; a row counts as bit-identical only if it "
                       "matches on all six experts",
         "why": "checks for bit-identical cached feature vectors, which "
                "distinguishes genuine reuse from a filename collision"},
        _paths.RESULTS_DIR / "verify_bit_identity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
