"""Read cached identifiers and hash feature rows for content checks.

Identifiers follow cache row order. For each split, the helpers hash the
stored feature bytes from df40-clip-large for comparison across splits.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.utils.cache_ids import load_sample_ids

CONTENT_EXPERT = "df40-clip-large"


def _hash_expert() -> str:
    """Expert whose stored feature rows define content identity."""
    return CONTENT_EXPERT


def _load_ids(path) -> np.ndarray:
    """Sample ids of one split as a 1-D object array of str, in cache row order."""
    ids = load_sample_ids(Path(path))
    return np.asarray([str(x) for x in ids], dtype=object)


def _content_hashes(split_dir, expert: str) -> list:
    """One SHA-1 hex digest per row of the stored feature array (float16 bytes as stored)."""
    feats = np.load(Path(split_dir) / f"{expert}_features.npy", mmap_mode="r")
    return [hashlib.sha1(np.ascontiguousarray(row).tobytes()).hexdigest() for row in feats]
