"""Python interface for frozen-expert composition.

After installing the dependencies and downloading the feature cache as
described in README.md, run a standard-split comparison from the repository root:

    >>> import composition_eval as ce
    >>> result = ce.evaluate(pool="arch6", protocol="headline")
    >>> print(result.headline())

For the paper workflow, including the full prediction-space ladder and
leave-one-generator-out evaluation, see camera_ready/REPRODUCTION.md.
"""

from composition_eval.api import (
    evaluate,
    EvaluationResult,
    load_pool,
    register_pool,
)
from composition_eval.prediction_space import (
    PS_1,
    PS_A,
    PS_B,
    PS_2,
    PS_C,
    PS_3,
    PS_F,
    PS_D,
    PS_E,
    PS_4,
)
from composition_eval.feature_space import (
    FS_1,
    FeatureAssembly,
    RandomProjectionAssembly,
    train_fs_model,
    predict_fs,
)
from composition_eval.bootstrap import (
    compute_auc,
    bootstrap_paired_auc,
    generator_level_bootstrap,
)

__version__ = "0.1.0"

__all__ = [
    # High-level API
    "evaluate",
    "EvaluationResult",
    "load_pool",
    "register_pool",
    # PS ladder
    "PS_1",
    "PS_A",
    "PS_B",
    "PS_2",
    "PS_C",
    "PS_3",
    "PS_F",
    "PS_D",
    "PS_E",
    "PS_4",
    # FS
    "FS_1",
    "FeatureAssembly",
    "RandomProjectionAssembly",
    "train_fs_model",
    "predict_fs",
    # Metrics
    "compute_auc",
    "bootstrap_paired_auc",
    "generator_level_bootstrap",
]
