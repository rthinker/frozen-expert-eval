"""Dataset name mapping and label prefix resolution.

Maps CLI dataset names to JSON filenames and label prefixes used by
DF40's DeepfakeAbstractBaseDataset.
"""

from typing import Dict

# Canonical dataset definitions
# key: CLI name, json: filename in configs/dataset_json/, prefix: label prefix
DATASETS: Dict[str, Dict[str, str]] = {
    "whichisreal": {"json": "whichisreal.json", "prefix": "whichisreal"},
    "CollabDiff": {"json": "CollabDiff.json", "prefix": "CollabDiff"},
    "MidJourney": {"json": "MidJourney.json", "prefix": "MidJourney"},
    "heygen": {"json": "heygen.json", "prefix": "heygen"},
    "starganv2": {"json": "starganv2.json", "prefix": "starganv2"},
    "deepfacelab": {"json": "deepfacelab.json", "prefix": "deepfacelab"},
    "stargan": {"json": "stargan.json", "prefix": "stargan"},
    "styleclip": {"json": "styleclip.json", "prefix": "styleclip"},
    # Per-regime test splits (used to build SELECT; DF40_all.json test is empty)
    "FSAll_ff": {"json": "FSAll_ff.json", "prefix": "FSAll"},
    "FRAll_ff": {"json": "FRAll_ff.json", "prefix": "FRAll"},
    "EFSAll_ff": {"json": "EFSAll_ff.json", "prefix": "EFSAll"},
    # NOTE: DF40_all has many label prefixes (FSAll_Fake, FRAll_Fake, etc.).
    # This prefix is only used by build_label_dict() as a fallback;
    # 01_extract merges the full label_dict from DF40 train_config.yaml.
    "DF40_all": {"json": "DF40_all.json", "prefix": "DF40_train"},
}

# The 8 OOD test generators (TEST split; see README.md "Data splits")
TEST_GENERATORS = [
    "whichisreal", "CollabDiff", "MidJourney", "heygen",
    "starganv2", "deepfacelab", "stargan", "styleclip",
]


def _get_label_prefix(dataset_name: str) -> str:
    """Get the label prefix for a dataset (e.g., 'whichisreal' -> 'whichisreal')."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASETS[dataset_name]["prefix"]


def build_label_dict(dataset_name: str) -> Dict[str, int]:
    """Build the label_dict DF40 expects: {prefix_Real: 0, prefix_Fake: 1}."""
    prefix = _get_label_prefix(dataset_name)
    return {f"{prefix}_Real": 0, f"{prefix}_Fake": 1}
