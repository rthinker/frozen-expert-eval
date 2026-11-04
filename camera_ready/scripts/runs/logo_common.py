"""Shared LOGO helpers for the mitigation runners: data loading, exclusion of shared images, fold views.

The fold construction reproduces ``scripts/11_disjoint_data.py`` (reviewed): the training data of fold g are the
other seven held-out generators, merged in ``TEST_GENERATORS`` order; FIT/SELECT is a stratified 85/15 split with
``random_state=1024``. The only addition is the optional exclusion: rows listed in the manifest written by
``logo_exclusions.py`` are removed from a fold's training data *before* the FIT/SELECT split.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SELECT_FRAC = 0.15
SPLIT_SEED = 1024


def load_generators(expert_names: List[str], cache_root: Path) -> Dict[str, tuple]:
    """{generator: (feats list [N, D_i] float32, logits list [N, 2] float32, labels int32)} in canonical order."""
    from src.utils.dataset_helpers import TEST_GENERATORS
    from src.utils.io import load_pool_data
    return {g: load_pool_data(list(expert_names), g, cache_root) for g in TEST_GENERATORS}


def row_hashes(feats: List[np.ndarray]) -> List[str]:
    """SHA-1 of each row's concatenated feature bytes across the pool's experts (float32 as loaded)."""
    block = np.ascontiguousarray(np.concatenate([np.asarray(f, dtype=np.float32) for f in feats], axis=1))
    return [hashlib.sha1(r.tobytes()).hexdigest() for r in block]


def load_manifest(path: Optional[Path]) -> Dict[str, Dict[str, List[int]]]:
    """Read a manifest for the archived runners, preserving their original input handling."""
    if path is None:
        return {}
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    return {f["held_out"]: {g: list(map(int, idx)) for g, idx in f["excluded_rows"].items()} for f in m["folds"]}


def load_content_manifest(path, pool, experts, gen_data) -> Dict[str, Dict[str, List[int]]]:
    """Validate a G1 manifest against the pool and the cached generator row counts."""
    if path is None:
        return {}
    problem = f"Invalid LOGO exclusion manifest {path}"
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"{problem}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("pool") != pool or manifest.get("experts") != list(experts):
        raise SystemExit(f"{problem}: expected pool {pool!r} and experts {list(experts)!r} in that order")
    if manifest.get("gate_G1") != "PASS":
        raise SystemExit(f"{problem}: gate G1 did not pass")
    folds = manifest.get("folds")
    if not isinstance(folds, list):
        raise SystemExit(f"{problem}: expected a list of folds")

    row_counts = {g: len(data[2]) for g, data in gen_data.items()}
    excluded = {}
    total = 0
    for fold in folds:
        held_out = fold.get("held_out") if isinstance(fold, dict) else None
        if not isinstance(held_out, str) or held_out not in row_counts or held_out in excluded:
            raise SystemExit(f"{problem}: unexpected or repeated held-out generator {held_out!r}")
        rows = fold.get("excluded_rows")
        if not isinstance(rows, dict):
            raise SystemExit(f"{problem}: fold {held_out!r} needs an excluded_rows mapping")
        for generator, indices in rows.items():
            if generator == held_out or generator not in row_counts:
                raise SystemExit(f"{problem}: fold {held_out!r} has invalid training generator {generator!r}")
            if not isinstance(indices, list) or any(
                type(index) is not int or not 0 <= index < row_counts[generator] for index in indices
            ):
                raise SystemExit(f"{problem}: fold {held_out!r}, {generator!r} needs zero-based row indices "
                                 f"within its {row_counts[generator]} cached rows")
            if len(set(indices)) != len(indices):
                raise SystemExit(f"{problem}: fold {held_out!r}, {generator!r} repeats row indices")
        count = sum(map(len, rows.values()))
        if type(fold.get("n_excluded")) is not int or fold["n_excluded"] != count:
            raise SystemExit(f"{problem}: fold {held_out!r} exclusion count does not match its row list")
        excluded[held_out] = rows
        total += count

    missing = sorted(row_counts.keys() - excluded.keys())
    if missing:
        raise SystemExit(f"{problem}: missing held-out generators {', '.join(missing)}")
    if type(manifest.get("rows_excluded_total")) is not int or manifest["rows_excluded_total"] != total:
        raise SystemExit(f"{problem}: total exclusion count does not match its folds")
    return excluded


def fold_view(gen_data: Dict[str, tuple], held_out: str, exclude: Optional[Dict[str, List[int]]] = None):
    """Training data of a fold (other generators, merged, exclusions applied) and its test data.

    Returns (tr_feats list, tr_logits3d, tr_labels, te_feats list, te_logits3d, te_labels, n_excluded).
    """
    from src.utils.dataset_helpers import TEST_GENERATORS
    exclude = exclude or {}
    n_exp = len(gen_data[held_out][0])
    feats = [[] for _ in range(n_exp)]
    logits = [[] for _ in range(n_exp)]
    labels = []
    n_excl = 0
    for g in TEST_GENERATORS:
        if g == held_out:
            continue
        f, lg, y = gen_data[g]
        keep = np.ones(len(y), dtype=bool)
        if exclude.get(g):
            keep[np.asarray(exclude[g], dtype=int)] = False
            n_excl += int((~keep).sum())
        for i in range(n_exp):
            feats[i].append(f[i][keep])
            logits[i].append(lg[i][keep])
        labels.append(y[keep])
    tr_feats = [np.concatenate(x) for x in feats]
    tr_logits3d = np.stack([np.concatenate(x) for x in logits], axis=1)
    tr_labels = np.concatenate(labels)
    te_f, te_lg, te_y = gen_data[held_out]
    return tr_feats, tr_logits3d, tr_labels, list(te_f), np.stack(te_lg, axis=1), te_y, n_excl


def split_fit_select(feats, logits3d, labels, select_frac=SELECT_FRAC, seed=SPLIT_SEED):
    """Stratified FIT/SELECT split, identical to scripts/11_disjoint_data.py::_split_fit_select."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(labels))
    fit_idx, sel_idx = train_test_split(idx, test_size=select_frac, random_state=seed, stratify=labels)
    return ([f[fit_idx] for f in feats], logits3d[fit_idx], labels[fit_idx],
            [f[sel_idx] for f in feats], logits3d[sel_idx], labels[sel_idx])


def write_json(obj, path: Path) -> Path:
    from _cr_paths import out_path
    p = out_path(path)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return p
