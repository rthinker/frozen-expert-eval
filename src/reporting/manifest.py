"""Result-JSON classification for the reporting pipeline.

Prefix-based classification: each result JSON is owned by the script whose
number prefixes its filename. This is brittle-to-rename (rename a script and
update the prefix map) but robust to content-shape drift -- a bug in script 07
will raise inside _parse_disjoint() instead of silently misclassifying.

Scripts should NOT import or call this module directly. Only `17_generate_tables.py`
and unit tests use it.

When script numbers or output prefixes change, update `_HANDLERS` below *and*
the fallback prefix strings inside `_parse_ds` / `_parse_pca`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-script parsers. Each parser takes the loaded JSON and returns a
# classified tuple (bucket_name, key, payload). "key" is used to deduplicate
# within a bucket (e.g. pool name for DS, (pool, combiner, dim) for FS).
# Parsers may raise ValueError if the file fails the expected schema.
# ---------------------------------------------------------------------------

_RAWFEAT_TOKEN = "_rawfeat"


def _parse_ds(d: Dict[str, Any], path: Path):
    """Parse a 04_* JSON.

    The 04 script emits two flavours of JSON under the same prefix:
      - 04_ds_{pool}{tag}.json         -- full DS ladder, schema has `baselines` + `best_ds_avg8`
      - 04_ds_{pool}_rawfeat{tag}.json -- single raw-feat LogReg entry,
        schema has `name == 'FS_1'` and no `baselines`.
    We detect the rawfeat flavour by a `_rawfeat` token anywhere in the stem
    (not just the trailing character) so that non-default train tags don't
    break classification.
    """
    stem = path.stem
    if _RAWFEAT_TOKEN in stem:
        pool = d.get("pool")
        if pool is None:
            # Fall back to stripping 04_ds_ prefix and _rawfeat{tag} suffix.
            pool = stem.replace("04_ds_", "", 1).split(_RAWFEAT_TOKEN, 1)[0]
        return "rawfeat", pool, d
    if "baselines" not in d or "best_ds_avg8" not in d:
        raise ValueError(f"{path.name}: missing baselines/best_ds_avg8")
    pool = d.get("pool", stem)
    return "ds", pool, d


def _parse_fs(d: Dict[str, Any], path: Path):
    if "combiner_type" not in d:
        raise ValueError(f"{path.name}: missing combiner_type")
    pool = d.get("pool", "?")
    ctype = d["combiner_type"]
    dim = d.get("proj_dim", 0)
    return "fs", (pool, ctype, dim), d


def _parse_pca(d: Dict[str, Any], path: Path):
    if "features" not in d or "logits" not in d or "pr_gap" not in d:
        raise ValueError(f"{path.name}: missing features/logits/pr_gap")
    pool = d.get("pool", path.stem.replace("08_pca_", ""))
    return "pca", pool, d


def _parse_disjoint(d: Dict[str, Any], path: Path):
    if "per_fold" not in d:
        raise ValueError(f"{path.name}: expected 'per_fold' key")
    pool = d.get("pool", path.stem)
    return "disjoint", pool, d


def _parse_bootstrap_rawfeat(d: Dict[str, Any], path: Path):
    # 06: sample + generator bootstrap for raw-feat LogReg vs PS
    if "sample_level_bootstrap" not in d:
        raise ValueError(f"{path.name}: expected 'sample_level_bootstrap'")
    pool = d.get("pool", path.stem)
    return "sample_bootstrap", pool, d


def _parse_bootstrap_mlp(d: Dict[str, Any], path: Path):
    # 07: sample + generator bootstrap for FS MLP (seed-averaged) vs PS
    if "sample_level_bootstrap" not in d:
        raise ValueError(f"{path.name}: expected 'sample_level_bootstrap'")
    pool = d.get("pool", path.stem)
    return "sample_bootstrap_mlp", pool, d


def _parse_disagreement(d: Dict[str, Any], path: Path):
    if "terciles" not in d:
        raise ValueError(f"{path.name}: expected 'terciles'")
    pool = d.get("pool", path.stem)
    return "disagreement", pool, d


def _parse_loo(d: Dict[str, Any], path: Path):
    if "loo" not in d:
        raise ValueError(f"{path.name}: expected 'loo'")
    pool = d.get("pool", path.stem)
    return "loo", pool, d


def _parse_xgb_features(d: Dict[str, Any], path: Path):
    """Parse a 14_* JSON: XGBoost on concatenated features (FS_D control)."""
    if d.get("method") != "xgboost_on_features":
        raise ValueError(
            f"{path.name}: expected method == 'xgboost_on_features', "
            f"got {d.get('method')!r}"
        )
    if "avg_auc_8" not in d or "per_generator" not in d:
        raise ValueError(
            f"{path.name}: missing avg_auc_8 or per_generator"
        )
    pool = d.get("pool", path.stem.replace("12_xgb_features_", ""))
    return "xgb_features", pool, d


def _parse_fs1_standardized(d: Dict[str, Any], path: Path):
    """Parse a 17_* JSON: FS_1 raw vs standardized features robustness check."""
    if d.get("method") != "fs_1_standardized_vs_raw":
        raise ValueError(
            f"{path.name}: expected method == 'fs_1_standardized_vs_raw', "
            f"got {d.get('method')!r}"
        )
    required = ("avg_auc_8_raw", "avg_auc_8_std", "delta_std_minus_raw")
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"{path.name}: missing keys {missing}")
    pool = d.get("pool", path.stem.replace("13_fs1_standardized_", ""))
    return "fs1_standardized", pool, d


# ---------------------------------------------------------------------------
# Prefix registry.
# ---------------------------------------------------------------------------

_HANDLERS: Dict[str, Callable[[Dict[str, Any], Path], tuple]] = {
    "04": _parse_ds,
    "05": _parse_fs,
    "06": _parse_bootstrap_rawfeat,
    "07": _parse_bootstrap_mlp,
    "08": _parse_pca,
    "09": _parse_disagreement,
    "10": _parse_loo,
    "11": _parse_disjoint,
    "12": _parse_xgb_features,
    "13": _parse_fs1_standardized,
}


def _prefix_of(path: Path) -> Optional[str]:
    stem = path.stem
    # The prefix is the leading two digits up to the first underscore.
    head = stem.split("_", 1)[0]
    if head.isdigit():
        return head
    return None


def scan_results(results_root: Path,
                 train_tag: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Scan all result JSONs under `results_root` and classify by prefix.

    Args:
        results_root: Path to `output/results/`.
        train_tag: Optional training tag filter. If set, files whose
            `train_tag` field does not match are skipped. Pass None to
            include everything.

    Returns:
        Dict with fixed keys:
            ds, fs, pca, disjoint, sample_bootstrap,
            sample_bootstrap_mlp, rawfeat, disagreement, loo,
            xgb_features, fs1_standardized
        Each value is a dict mapping a per-bucket key to the loaded payload.
    """
    manifest: Dict[str, Dict[str, Any]] = {
        "ds": {}, "fs": {}, "pca": {},
        "disjoint": {},
        "sample_bootstrap": {}, "sample_bootstrap_mlp": {},
        "rawfeat": {}, "disagreement": {}, "loo": {},
        "xgb_features": {}, "fs1_standardized": {},
    }

    root = Path(results_root)
    if not root.exists():
        return manifest

    for path in sorted(root.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            log.warning("scan_results: failed to load %s: %s", path.name, exc)
            continue
        if not isinstance(data, dict):
            continue
        if train_tag is not None and data.get("train_tag") not in (None, train_tag):
            continue

        prefix = _prefix_of(path)
        if prefix is None or prefix not in _HANDLERS:
            continue
        try:
            bucket, key, payload = _HANDLERS[prefix](data, path)
        except ValueError as exc:
            log.warning("scan_results: %s", exc)
            continue
        manifest[bucket][key] = payload

    return manifest
