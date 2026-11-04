"""Deterministic seeding for reproducibility.

Sets all random sources and enables PyTorch deterministic mode.
Call set_seed() once at the start of every script.

Determinism requirements
------------------------
PyTorch >= 1.8 is required for ``torch.use_deterministic_algorithms``.
A small number of CUDA kernels (e.g. ``scatter_add`` on some GPUs) lack
deterministic implementations; calling them under strict determinism
will raise a ``RuntimeError``. This codebase does not use such kernels
in its training loops, so strict mode is safe for every reproducible
pipeline under ``scripts/``.

If you are running in an environment where strict determinism cannot be
enforced (e.g. a newer op added in a fork), pass ``strict=False`` and
the call will fall back to ``warn_only`` mode while still logging a
warning. Never silently swallow the failure.
"""

import logging
import os
import random

import numpy as np
import torch


_log = logging.getLogger(__name__)


def set_seed(seed: int = 1024, *, strict: bool = True) -> None:
    """Set all random seeds for full reproducibility.

    Args:
        seed: Random seed. Default 1024 (DF40 standard).
        strict: If True (default), raise ``RuntimeError`` when PyTorch
            cannot enter deterministic mode. If False, fall back to
            ``warn_only=True`` and emit a visible warning instead of
            failing.

    Raises:
        RuntimeError: If ``strict=True`` and PyTorch refuses to enable
            deterministic algorithms.
    """
    # Order matters: CUBLAS_WORKSPACE_CONFIG must be in os.environ BEFORE
    # the first cuBLAS handle is created, otherwise PyTorch falls back to a
    # non-deterministic workspace and ``torch.use_deterministic_algorithms``
    # will refuse. ``scripts/_bootstrap`` already binds it at import time so
    # this ``setdefault`` is a belt-and-suspenders for callers that bypass
    # the bootstrap (tests, ad-hoc REPL).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except (RuntimeError, AttributeError) as exc:
        if strict:
            raise RuntimeError(
                "Failed to enable PyTorch deterministic algorithms: "
                f"{exc!r}. This codebase requires strict determinism "
                "for reproducibility of reported numbers. Either "
                "upgrade PyTorch (>= 1.8) or explicitly opt out by "
                "passing strict=False to set_seed()."
            ) from exc
        _log.warning(
            "PyTorch strict determinism unavailable (%s); falling back "
            "to warn_only=True. Numerical results may drift.", exc,
        )
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, AttributeError, TypeError):
            _log.warning(
                "warn_only determinism also unavailable; results may "
                "not be fully reproducible on this PyTorch version."
            )
