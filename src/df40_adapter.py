"""DF40 adapter: thin bridge between our code and unmodified DF40.

This is the ONLY file that touches sys.path or os.environ for DF40
(except 00_setup.py, which sets HF_HOME + TRANSFORMERS_OFFLINE=0 for downloads).
All other code imports DF40 functionality through this module.

Usage:
    from src.df40_adapter import init_df40, df40_context
    from src.df40_adapter import get_dataset_class, get_detector_registry

Design:
    - DF40's training/ directory is NOT a Python package (no __init__.py)
    - Adding it to sys.path makes its sub-packages importable as top-level
    - DF40 code uses relative paths internally -> must os.chdir() before use
    - HuggingFace cache is redirected to local dir (offline after setup)
"""

import os
import sys
from pathlib import Path
from contextlib import contextmanager

# Resolve paths
CODEBASE_ROOT = Path(__file__).resolve().parent.parent
DF40_REPO = CODEBASE_ROOT / "third_party" / "DeepfakeBench_DF40"
DF40_TRAINING = DF40_REPO / "training"

# HuggingFace cache: redirect BEFORE any transformers import
HF_CACHE = CODEBASE_ROOT / "third_party" / "hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # Models cached by 00_setup.py
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

_initialised = False


def init_df40() -> None:
    """Add DF40/training to sys.path and trigger detector registration.

    Safe to call multiple times; only runs once.
    """
    global _initialised
    if _initialised:
        return

    if not DF40_TRAINING.exists():
        raise FileNotFoundError(
            f"DF40 training directory not found at {DF40_TRAINING}. "
            f"Run: git clone https://github.com/YZY-stack/DF40.git "
            f"third_party/DeepfakeBench_DF40"
        )

    # Add DF40/training to sys.path so its sub-packages are importable
    training_str = str(DF40_TRAINING)
    if training_str not in sys.path:
        sys.path.insert(0, training_str)

    # Trigger detector registration by importing detectors/__init__.py
    # This populates the global DETECTOR registry via @DETECTOR.register_module
    original_cwd = os.getcwd()
    os.chdir(DF40_TRAINING)
    try:
        import detectors  # noqa: F401 -- side-effect import
    finally:
        os.chdir(original_cwd)

    _initialised = True


@contextmanager
def df40_context():
    """Temporarily chdir to DF40 training root.

    DF40 code uses relative paths internally. Any call to DF40's dataset
    class or detector loading must happen inside this context.
    """
    original = os.getcwd()
    os.chdir(DF40_TRAINING)
    try:
        yield
    finally:
        os.chdir(original)


def get_detector_registry():
    """Return the DF40 DETECTOR registry (dict-like)."""
    init_df40()
    from metrics.registry import DETECTOR
    return DETECTOR


def get_dataset_class():
    """Return DF40's DeepfakeAbstractBaseDataset class."""
    init_df40()
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
    return DeepfakeAbstractBaseDataset


