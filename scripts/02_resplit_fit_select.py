"""Deduplicate by sample_id, then stratified 80/20 -> FIT and SELECT.

Reads the raw train cache from `01_extract.py` and writes
`output/cached_features/{FIT,SELECT}/` with sliced per-expert arrays plus
a fingerprint.txt per split. Pure row-level reindex; no GPU, no DF40
dependency. Class balance is intentionally NOT enforced here; trainers
handle imbalance via class-weighted CE / sklearn `balanced`.

Flags: --select_frac, --seed, --dry_run, --source. Defaults are 0.2 / 1024 / off / RAW_TRAIN_TAG.
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.utils.cache_ids import load_sample_ids
from src.utils.config import (
    CACHE_ROOT, EXPERTS_CFG,
    FIT_TAG, SELECT_TAG, RAW_TRAIN_TAG,
)
from src.utils.logging_setup import setup_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_source_dir(override: Path = None) -> Path:
    """Resolve the raw train cache directory to resplit.

    Resolution order:
      1. Explicit --source override.
      2. CACHE_ROOT / RAW_TRAIN_TAG (the canonical default, produced by
         scripts/01_extract.py).
    """
    if override is not None:
        override = Path(override)
        if not override.exists():
            raise FileNotFoundError(f"--source {override} does not exist")
        return override

    raw = CACHE_ROOT / RAW_TRAIN_TAG
    if raw.exists():
        return raw

    raise FileNotFoundError(
        "No raw train cache found. Expected "
        f"{CACHE_ROOT / RAW_TRAIN_TAG}, produced by 01_extract.py. "
        "Did you already run 01c? Check output/cached_features/ for "
        "existing FIT/ and SELECT/ dirs."
    )


def _relative_to_repo(path: Path) -> str:
    """Render a path relative to the repo root when possible.

    Provenance records must be location-independent so that the cache is
    portable across machines and directory renames. Falls back to the
    basename if the path escapes the repo.
    """
    path = Path(path).resolve()
    try:
        return path.relative_to(_REPO).as_posix()
    except ValueError:
        return path.name


def _all_expert_names() -> list:
    """Union of every expert in every pool (same logic as 01 extraction)."""
    pools = EXPERTS_CFG.get("pools", {})
    seen = set()
    out = []
    for pool in pools.values():
        for name in pool:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _fingerprint(labels: np.ndarray, sample_ids: np.ndarray) -> str:
    h_labels = hashlib.sha256(labels.tobytes()).hexdigest()[:16]
    ids_str = "\n".join(str(s) for s in sample_ids)
    h_ids = hashlib.sha256(ids_str.encode("utf-8")).hexdigest()[:16]
    return f"n={len(labels)}_r={int((labels==0).sum())}_f={int((labels==1).sum())}_{h_labels}_{h_ids}"


def _deduplicate(sample_ids: np.ndarray, labels: np.ndarray):
    """Return indices of first-occurrence rows for each unique sample_id.

    Deterministic by construction: ``np.unique`` returns the index of the
    first occurrence for each unique id in the ordering of ``sample_ids``,
    and we then sort those indices to preserve cache row order. The dedup
    is therefore independent of any RNG seed -- given the same raw cache,
    the same unique row set is selected every time.
    """
    _, first_idx = np.unique(sample_ids, return_index=True)
    first_idx = np.sort(first_idx)
    uniq_labels = labels[first_idx]
    stats = {
        "n_rows_raw": int(len(sample_ids)),
        "n_rows_unique": int(len(first_idx)),
        "n_real_unique": int((uniq_labels == 0).sum()),
        "n_fake_unique": int((uniq_labels == 1).sum()),
    }
    return first_idx, stats


def _stratified_split(labels: np.ndarray, select_frac: float, rng):
    """Return (fit_idx, select_idx) -- disjoint, stratified by label."""
    assert 0.0 < select_frac < 1.0
    real_pos = np.where(labels == 0)[0]
    fake_pos = np.where(labels == 1)[0]
    rng.shuffle(real_pos)
    rng.shuffle(fake_pos)
    n_real_select = int(round(len(real_pos) * select_frac))
    n_fake_select = int(round(len(fake_pos) * select_frac))
    select_real = real_pos[:n_real_select]
    fit_real = real_pos[n_real_select:]
    select_fake = fake_pos[:n_fake_select]
    fit_fake = fake_pos[n_fake_select:]
    fit_idx = np.sort(np.concatenate([fit_real, fit_fake]))
    sel_idx = np.sort(np.concatenate([select_real, select_fake]))
    assert np.intersect1d(fit_idx, sel_idx).size == 0
    return fit_idx, sel_idx


def _slice_cache(
    legacy_dir: Path,
    out_dir: Path,
    expert_names: list,
    row_indices: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    provenance: dict,
    log: logging.Logger,
):
    """Materialise a new cache split by slicing every expert's .npy."""
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "labels.npy", labels[row_indices].astype(np.int32))
    np.save(
        out_dir / "sample_ids.npy",
        np.asarray(sample_ids[row_indices], dtype=str),
        allow_pickle=False,
    )

    for name in expert_names:
        feats_path = legacy_dir / f"{name}_features.npy"
        logits_path = legacy_dir / f"{name}_logits.npy"
        if not feats_path.exists() or not logits_path.exists():
            log.warning(
                f"  expert {name}: missing in legacy cache, skipping. "
                f"(You may want to re-extract.)"
            )
            continue

        legacy_feats = np.load(feats_path, mmap_mode="r")
        legacy_logits = np.load(logits_path, mmap_mode="r")

        sliced_feats = np.ascontiguousarray(legacy_feats[row_indices])
        sliced_logits = np.ascontiguousarray(legacy_logits[row_indices])

        np.save(out_dir / f"{name}_features.npy", sliced_feats)
        np.save(out_dir / f"{name}_logits.npy", sliced_logits)

        del legacy_feats, legacy_logits, sliced_feats, sliced_logits

    new_labels = labels[row_indices]
    new_ids = sample_ids[row_indices]
    fp = _fingerprint(new_labels, new_ids)
    (out_dir / "fingerprint.txt").write_text(fp, encoding="utf-8")

    provenance_out = dict(provenance)
    provenance_out.update({
        "n_rows": int(len(row_indices)),
        "n_real": int((new_labels == 0).sum()),
        "n_fake": int((new_labels == 1).sum()),
        "fingerprint": fp,
    })
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance_out, indent=2), encoding="utf-8"
    )

    log.info(
        f"  wrote {out_dir.name}: "
        f"{len(row_indices)} rows "
        f"({int((new_labels==0).sum())} real / {int((new_labels==1).sum())} fake), "
        f"fp={fp[:24]}..."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--select_frac", type=float, default=0.2,
                   help="Fraction of unique IDs to put in SELECT (default 0.2)")
    p.add_argument("--seed", type=int, default=1024,
                   help="RNG seed for the shuffle (default 1024)")
    p.add_argument("--dry_run", action="store_true",
                   help="Report counts and exit without writing anything.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing FIT/SELECT output directories.")
    p.add_argument("--source", type=str, default=None,
                   help="Override auto-detected raw train cache dir "
                        "(default: output/cached_features/<RAW_TRAIN_TAG>).")
    p.add_argument("--legacy_dir", type=str, default=None,
                   help="Deprecated alias for --source.")
    return p.parse_args()


def main():
    args = parse_args()
    log = setup_logging("02_resplit")
    t0 = time.time()

    source_override = args.source or args.legacy_dir
    legacy_dir = _resolve_source_dir(
        Path(source_override) if source_override else None)
    log.info(f"Raw train cache: {legacy_dir}")

    labels_path = legacy_dir / "labels.npy"
    ids_path = legacy_dir / "sample_ids.npy"
    if not labels_path.exists() or not ids_path.exists():
        log.error(
            f"Legacy cache missing labels.npy or sample_ids.npy in {legacy_dir}"
        )
        sys.exit(1)

    labels = np.load(labels_path)
    sample_ids = load_sample_ids(ids_path)
    log.info(f"Loaded {len(labels)} rows, {len(np.unique(sample_ids))} unique IDs")

    uniq_idx, dedup_stats = _deduplicate(sample_ids, labels)
    log.info(
        f"After dedup: {dedup_stats['n_rows_unique']} rows "
        f"({dedup_stats['n_real_unique']} real, "
        f"{dedup_stats['n_fake_unique']} fake)"
    )

    dedup_labels = labels[uniq_idx]

    rng = np.random.default_rng(args.seed)
    fit_local, sel_local = _stratified_split(dedup_labels, args.select_frac, rng)

    fit_rows = uniq_idx[fit_local]
    sel_rows = uniq_idx[sel_local]

    assert np.intersect1d(fit_rows, sel_rows).size == 0, "FIT/SELECT overlap!"
    assert (
        np.intersect1d(sample_ids[fit_rows], sample_ids[sel_rows]).size == 0
    ), "sample_id overlap across FIT/SELECT!"

    log.info("=" * 60)
    log.info("Resplit plan")
    log.info("=" * 60)
    log.info(f"  FIT:    {len(fit_rows):6d} rows  "
             f"({(dedup_labels[fit_local]==0).sum()} real / "
             f"{(dedup_labels[fit_local]==1).sum()} fake)")
    log.info(f"  SELECT: {len(sel_rows):6d} rows  "
             f"({(dedup_labels[sel_local]==0).sum()} real / "
             f"{(dedup_labels[sel_local]==1).sum()} fake)")
    log.info("  Disjoint by sample_id: YES")

    if args.dry_run:
        log.info("Dry run -- no files written.")
        return

    expert_names = _all_expert_names()
    log.info(f"Experts to slice: {len(expert_names)}")

    out_fit = CACHE_ROOT / FIT_TAG
    out_sel = CACHE_ROOT / SELECT_TAG
    for out in (out_fit, out_sel):
        if out.exists():
            if not args.force:
                log.error(
                    f"Output directory {out} already exists. "
                    f"Pass --force to overwrite."
                )
                sys.exit(2)
            shutil.rmtree(out)

    provenance = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_source": _relative_to_repo(legacy_dir),
        "select_frac": args.select_frac,
        "seed": args.seed,
        "policy": "deduped, FIT/SELECT-disjoint, unbalanced",
        **dedup_stats,
    }

    log.info("\nSlicing FIT cache...")
    _slice_cache(
        legacy_dir, out_fit, expert_names, fit_rows,
        labels, sample_ids,
        {**provenance, "split": "FIT"}, log,
    )
    log.info("\nSlicing SELECT cache...")
    _slice_cache(
        legacy_dir, out_sel, expert_names, sel_rows,
        labels, sample_ids,
        {**provenance, "split": "SELECT"}, log,
    )

    elapsed = time.time() - t0
    log.info(f"\nDone in {elapsed:.1f}s.")
    log.info(f"New caches: {out_fit} and {out_sel}")
    log.info(
        "Each new split has its own fingerprint and split provenance. "
        "scripts/03_verify_cache.py authenticates the pinned release cache."
    )


if __name__ == "__main__":
    main()
