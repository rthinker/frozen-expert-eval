"""I/O utilities for results and cached data.

Handles JSON results and numpy feature/logit/label arrays.
All paths are resolved relative to the codebase output directory.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

log = logging.getLogger(__name__)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save dictionary as JSON with pretty printing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file to dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _map_if_needed(array: np.ndarray, dtype) -> np.ndarray:
    """Cast only when needed, preserving mmap-backed arrays when possible."""
    return array if array.dtype == dtype else array.astype(dtype)


def load_pool_data(expert_names: List[str], data_tag: str,
                   cache_root: Path, mmap: bool = False):
    """Load per-expert features, logits, and labels for a dataset tag.

    Args:
        expert_names: List of expert names.
        data_tag: Cache subdirectory name (e.g., "whichisreal").
        cache_root: Root of cached_features directory.
        mmap: If True, use memory-mapped loading (read-only, lower RAM).

    Returns:
        (feats, logits, labels) where feats and logits are lists of
        [N, D_i] and [N, C] arrays respectively.
    """
    ds_dir = Path(cache_root) / data_tag
    mm = "r" if mmap else None
    raw_labels = np.load(ds_dir / "labels.npy", mmap_mode=mm)
    labels = _map_if_needed(raw_labels, np.int32)
    feats, logits = [], []
    for n in expert_names:
        raw_f = np.load(ds_dir / f"{n}_features.npy", mmap_mode=mm)
        feats.append(_map_if_needed(raw_f, np.float32))
        raw_l = np.load(ds_dir / f"{n}_logits.npy", mmap_mode=mm)
        logits.append(_map_if_needed(raw_l, np.float32))
    return feats, logits, labels


def load_stacked_logits(expert_names: List[str], data_tag: str,
                        cache_root: Path, mmap: bool = False):
    """Load a split and return logits stacked as [N, E, C]."""
    _, logits, labels = load_pool_data(
        expert_names, data_tag, cache_root, mmap=mmap)
    return np.stack(logits, axis=1), labels


def load_concat_features(expert_names: List[str], data_tag: str,
                         cache_root: Path, mmap: bool = False):
    """Load a split and return features concatenated as [N, sum(D_i)]."""
    feats, _, labels = load_pool_data(
        expert_names, data_tag, cache_root, mmap=mmap)
    return np.concatenate(feats, axis=1), labels


def preload_pool_views(expert_names: List[str], data_tags: List[str],
                       cache_root: Path, mmap: bool = False):
    """Preload split data into a name-indexable structure for repeated slicing."""
    views = {}
    for tag in data_tags:
        feats, logits, labels = load_pool_data(
            expert_names, tag, cache_root, mmap=mmap)
        views[tag] = {
            "labels": labels,
            "features": {name: feat for name, feat in zip(expert_names, feats)},
            "logits": {name: logit for name, logit in zip(expert_names, logits)},
        }
    return views


def get_feature_views(view: Dict[str, Any], expert_names: List[str]):
    """Select a list-of-arrays feature view plus labels from preloaded data."""
    return [view["features"][n] for n in expert_names], view["labels"]


def load_all_generators(expert_names: List[str], cache_root: Path,
                        test_generators: List[str]):
    """Load features, logits, and labels for all test generators.

    Returns:
        dict mapping generator name to (feats, logits, labels) tuple.
        Also returns concatenated arrays for sample-level analysis:
        (gen_data, all_feats_concat, all_logits_3d, all_labels, gen_boundaries)
    """
    gen_data = {}
    all_feats, all_logits, all_labels = [], [], []
    gen_boundaries = {}
    offset = 0

    for g in test_generators:
        feats, logits, labels = load_pool_data(
            expert_names, g, cache_root)
        gen_data[g] = (feats, logits, labels)
        n = len(labels)
        gen_boundaries[g] = (offset, offset + n)
        all_feats.append(np.concatenate(feats, axis=1))
        all_logits.append(np.stack(logits, axis=1))
        all_labels.append(labels)
        offset += n

    X_feat = np.concatenate(all_feats, axis=0)
    X_logits_3d = np.concatenate(all_logits, axis=0)
    y = np.concatenate(all_labels)

    return gen_data, X_feat, X_logits_3d, y, gen_boundaries
