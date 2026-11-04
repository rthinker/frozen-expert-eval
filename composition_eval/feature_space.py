"""composition_eval.feature_space: feature-space alternatives (FS_1, FS_2, ablations).

FS_1 (deterministic LogReg on raw concatenated features) is the recommended
anchor for the feature-space branch. FS_2 (learned per-expert projection +
MLP) is included as a capacity check.
"""

from src.models.feature_space import (
    FS_1, FeatureAssembly, RandomProjectionAssembly,
)
from src.training.fs_runtime import train_fs_model, predict_fs

__all__ = [
    "FS_1",
    "FeatureAssembly",
    "RandomProjectionAssembly",
    "train_fs_model",
    "predict_fs",
]
