"""Authenticate the pinned release cache and check its split structure.

The released bytes and notice must match the code-owned trust contract.
Then verifies per split: labels/sample_ids/fingerprint files exist, live
fingerprint matches the stored one, every expert has features+logits,
feature dims match configs/experts.yaml, row counts align across
experts, and FIT/SELECT are disjoint and unique by sample_id.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.utils.cache_ids import load_sample_ids
from src.utils.config import (
    CACHE_ROOT, EXPERTS_CFG, FIT_TAG, SELECT_TAG,
)
from src.utils.dataset_helpers import TEST_GENERATORS
from src.utils.logging_setup import setup_logging


def _fingerprint(labels: np.ndarray, sample_ids: np.ndarray) -> str:
    h_labels = hashlib.sha256(labels.tobytes()).hexdigest()[:16]
    ids_str = "\n".join(str(s) for s in sample_ids)
    h_ids = hashlib.sha256(ids_str.encode("utf-8")).hexdigest()[:16]
    return f"n={len(labels)}_r={int((labels==0).sum())}_f={int((labels==1).sum())}_{h_labels}_{h_ids}"


def _all_expert_names() -> list:
    seen, out = set(), []
    for pool in EXPERTS_CFG.get("pools", {}).values():
        for name in pool:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def verify_split(split_dir: Path, expert_names: list, log) -> list:
    """Verify one cached split. Returns a list of error strings (empty = OK)."""
    errors = []
    if not split_dir.exists():
        errors.append(f"{split_dir.name}: missing directory")
        return errors

    labels_path = split_dir / "labels.npy"
    ids_path = split_dir / "sample_ids.npy"
    fp_path = split_dir / "fingerprint.txt"

    for p in (labels_path, ids_path, fp_path):
        if not p.exists():
            errors.append(f"{split_dir.name}: missing {p.name}")
            return errors

    labels = np.load(labels_path)
    sample_ids = load_sample_ids(ids_path)
    if len(labels) != len(sample_ids):
        errors.append(
            f"{split_dir.name}: len(labels)={len(labels)} vs "
            f"len(sample_ids)={len(sample_ids)}"
        )
        return errors

    # Fingerprint
    live_fp = _fingerprint(labels, sample_ids)
    stored_fp = fp_path.read_text(encoding="utf-8").strip()
    if live_fp != stored_fp:
        errors.append(
            f"{split_dir.name}: fingerprint mismatch\n"
            f"  stored: {stored_fp}\n"
            f"  live:   {live_fp}"
        )

    # Within-split uniqueness (Phase 2 invariant for FIT/SELECT)
    is_fit_or_select = split_dir.name in (FIT_TAG, SELECT_TAG)
    if is_fit_or_select:
        n_unique = len(np.unique(sample_ids))
        if n_unique != len(sample_ids):
            errors.append(
                f"{split_dir.name}: expected zero duplicate sample_ids, "
                f"found {len(sample_ids) - n_unique} duplicates "
                f"({n_unique} unique of {len(sample_ids)})"
            )

    # Expert coverage + row alignment + dim check
    arch_map = EXPERTS_CFG["architectures"]
    expert_map = EXPERTS_CFG["experts"]

    for name in expert_names:
        feat_p = split_dir / f"{name}_features.npy"
        log_p = split_dir / f"{name}_logits.npy"
        if not feat_p.exists() or not log_p.exists():
            errors.append(f"{split_dir.name}/{name}: missing features or logits")
            continue
        try:
            feat = np.load(feat_p, mmap_mode="r")
            logits = np.load(log_p, mmap_mode="r")
        except Exception as exc:
            errors.append(f"{split_dir.name}/{name}: load failed: {exc}")
            continue

        if feat.shape[0] != len(labels):
            errors.append(
                f"{split_dir.name}/{name}: features rows={feat.shape[0]} "
                f"vs labels={len(labels)}"
            )
        if logits.shape[0] != len(labels):
            errors.append(
                f"{split_dir.name}/{name}: logits rows={logits.shape[0]} "
                f"vs labels={len(labels)}"
            )

        expected_dim = arch_map[expert_map[name]["arch"]]["feature_dim"]
        if feat.ndim < 2 or feat.shape[1] != expected_dim:
            errors.append(
                f"{split_dir.name}/{name}: feature dim {feat.shape[1:]} "
                f"!= expected {expected_dim}"
            )
        if logits.ndim < 2 or logits.shape[1] != 2:
            errors.append(
                f"{split_dir.name}/{name}: logit dim {logits.shape[1:]} "
                f"!= expected 2"
            )

    # Balance report (non-fatal)
    n_real = int((labels == 0).sum())
    n_fake = int((labels == 1).sum())
    log.info(
        f"  {split_dir.name:20s}  n={len(labels):6d}  "
        f"real={n_real:5d}  fake={n_fake:5d}  "
        f"(ratio real/fake={n_real / max(n_fake, 1):.3f})"
    )
    return errors


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--require-fit-select",
        action="store_true",
        help="Require both FIT and SELECT when checking split structure.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    # Authenticate raw bytes against code-owned metadata before any cache read.
    from scripts.review.validate_cache_release import check_checksums, check_license_notice, CheckFailure
    try:
        check_checksums(CACHE_ROOT)
        check_license_notice(CACHE_ROOT)
    except CheckFailure as error:
        raise SystemExit(str(error)) from error
    log = setup_logging("03_verify_cache")

    expert_names = _all_expert_names()
    log.info(f"Expert pool: {len(expert_names)} experts")

    # Expected splits: FIT, SELECT, + 8 test generators (TEST)
    expected_splits = [FIT_TAG, SELECT_TAG, *TEST_GENERATORS]
    errors = []

    log.info("\nVerifying splits:")
    log.info(f"  {'split':20s}  {'N':>6s}  {'real':>5s}  {'fake':>5s}")
    log.info("  " + "-" * 60)

    found_splits = []
    for tag in expected_splits:
        split_dir = CACHE_ROOT / tag
        if not split_dir.exists():
            if tag in (FIT_TAG, SELECT_TAG) and not args.require_fit_select:
                log.warning(f"  {tag}: MISSING (pre-Phase-2 is OK)")
                continue
            errors.append(f"{tag}: missing directory")
            continue
        found_splits.append(tag)
        errors.extend(verify_split(split_dir, expert_names, log))

    # Disjointness check for FIT / SELECT
    if FIT_TAG in found_splits and SELECT_TAG in found_splits:
        fit_ids = load_sample_ids(CACHE_ROOT / FIT_TAG / "sample_ids.npy")
        sel_ids = load_sample_ids(CACHE_ROOT / SELECT_TAG / "sample_ids.npy")
        overlap = np.intersect1d(fit_ids, sel_ids)
        if overlap.size > 0:
            errors.append(
                f"FIT and SELECT share {overlap.size} sample_ids "
                f"(expected 0). First few: {overlap[:5].tolist()}"
            )
        else:
            log.info("\n  FIT / SELECT disjointness: OK (0 shared IDs)")

    log.info("")
    if errors:
        log.error("Cache integrity: FAIL")
        for e in errors:
            log.error(f"  - {e}")
        sys.exit(1)
    log.info("Cache integrity: OK")


if __name__ == "__main__":
    main()
