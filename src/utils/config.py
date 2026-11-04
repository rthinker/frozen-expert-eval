"""Centralised configuration loader.

Loads all YAML config files once and exposes key values as module-level
constants. All scripts should import from here rather than hardcoding.

Usage:
    from src.utils.config import DEFAULT_TRAIN_TAG, FS_EPOCHS, POOLS
"""

import warnings
from pathlib import Path
import yaml

# Resolve codebase root (two levels up from src/utils/)
CODEBASE_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIGS = CODEBASE_ROOT / "configs"

# ---------------------------------------------------------------------------
# paths.yaml
#
# `configs/paths.yaml` is user-specific (absolute paths to the DF40
# dataset and weights) and intentionally excluded from git (.gitignore).
# When it is absent,
# fall back to the shipped template `configs/paths.yaml.example` so
# that `import composition_eval`, --verify-only, and combiner-stage
# work from cached features all continue to function on a fresh
# clone.
#
# The fallback is silent for cache-only users: the placeholder paths in
# `paths.yaml.example` (`/path/to/...`) are never read by combiner-stage
# scripts, only by the upstream feature extractor (`01_extract.py`),
# which surfaces a clear FileNotFoundError if those paths are still the
# template values. We only emit a warning when the example file looks
# like the user customised it without copying to `paths.yaml`, so the
# common cache-only reviewer path is quiet.
# ---------------------------------------------------------------------------
_paths_file = _CONFIGS / "paths.yaml"
if not _paths_file.exists():
    _paths_file = _CONFIGS / "paths.yaml.example"
    # If the example file still contains the template `/path/to/`
    # placeholders, the user is intentionally on the cache-only path
    # and the warning would just be noise. Only complain if the
    # placeholders have been edited out.
    try:
        _example_text = _paths_file.read_text(encoding="utf-8")
    except OSError:
        _example_text = "/path/to/"  # treat unreadable as template
    if "/path/to/" not in _example_text:
        warnings.warn(
            "configs/paths.yaml not found and paths.yaml.example "
            "appears to have been customised in place. Copy "
            "paths.yaml.example to paths.yaml so user-specific paths "
            "are not committed.",
            UserWarning, stacklevel=2,
        )

with open(_paths_file, encoding="utf-8") as f:
    _paths = yaml.safe_load(f)

DEFAULT_TRAIN_TAG = _paths["default_train_tag"]
DEFAULT_SELECT_TAG = _paths.get("default_select_tag", "SELECT")
RAW_TRAIN_TAG = _paths.get("raw_train_tag", "_raw_df40_train")
# Explicit, self-documenting aliases for scripts that care about roles, not cache tags.
FIT_TAG = DEFAULT_TRAIN_TAG
SELECT_TAG = DEFAULT_SELECT_TAG
DATASET_ROOT = _paths.get("dataset_root", "")
WEIGHTS_ROOT = _paths.get("weights_root", "")
DATASET_JSON_FOLDER = _paths.get("dataset_json_folder", "")
OUTPUT_DIR = CODEBASE_ROOT / _paths.get("output_dir", "./output")
HF_CACHE_DIR = CODEBASE_ROOT / _paths.get("hf_cache_dir", "./third_party/hf_cache")

# Derived paths
CACHE_ROOT = OUTPUT_DIR / "cached_features"
CKPT_ROOT = OUTPUT_DIR / "checkpoints"
RESULTS_ROOT = OUTPUT_DIR / "results"
FIGURES_ROOT = OUTPUT_DIR / "figures"
TABLES_ROOT = OUTPUT_DIR / "tables"
LOGS_ROOT = OUTPUT_DIR / "logs"

# ---------------------------------------------------------------------------
# experiments.yaml
# ---------------------------------------------------------------------------
with open(_CONFIGS / "experiments.yaml") as f:
    _exp = yaml.safe_load(f)

# Feature-space assembly defaults
_fs = _exp["feature_assembly"]
FS_PROJ_DIM = _fs["proj_dim"]
FS_HIDDEN_DIMS = _fs["hidden_dims"]
FS_DROPOUT = _fs["dropout"]
FS_EXPERT_DROPOUT = _fs["expert_dropout"]
FS_LR = _fs["lr"]
FS_WEIGHT_DECAY = _fs["weight_decay"]
FS_EPOCHS = _fs["epochs"]
FS_BATCH_SIZE = _fs["batch_size"]
FS_SEEDS = _fs["seeds"]

# Decision-space baseline defaults
_ds = _exp["decision_baselines"]
DS_CONFIG = _ds  # Full dict for scripts that need all DS params

# Evaluation
BOOTSTRAP_RESAMPLES = _exp["evaluation"]["bootstrap_resamples"]
CONFIDENCE_LEVEL = _exp["evaluation"]["confidence_level"]

# Reporting
_rep = _exp.get("reporting", {})
REP_HEADLINE_POOL = _rep.get("headline_pool", "arch6")
REP_HEADLINE_DIM = _rep.get("headline_dim", 256)
REP_SECONDARY_POOL = _rep.get("secondary_pool", "clip4")
REP_SECONDARY_DIM = _rep.get("secondary_dim", 32)
REP_SECONDARY_DIM_ALT = _rep.get("secondary_dim_alt", 256)

# ---------------------------------------------------------------------------
# experts.yaml
# ---------------------------------------------------------------------------
with open(_CONFIGS / "experts.yaml") as f:
    _experts = yaml.safe_load(f)

ARCHITECTURES = _experts["architectures"]
EXPERTS = _experts["experts"]
POOLS = _experts["pools"]
EXPERTS_CFG = _experts     # Full dict for scripts needing raw access
EXPERIMENTS_CFG = _exp     # Full dict for scripts needing raw access


def get_pool_experts(pool_name: str):
    """Return list of expert names for a pool."""
    if pool_name not in POOLS:
        raise ValueError(
            f"Unknown pool: {pool_name}. "
            f"Valid: {list(POOLS.keys())}")
    return POOLS[pool_name]


def get_feature_dims(expert_names: list) -> list:
    """Return feature dimensions for a list of expert names."""
    return [
        ARCHITECTURES[EXPERTS[n]["arch"]]["feature_dim"]
        for n in expert_names
    ]
