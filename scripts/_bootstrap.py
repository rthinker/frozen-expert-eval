"""Shared bootstrap for direct script execution.

Two responsibilities, both order-sensitive:

1. **Determinism env vars.** ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` is required
   by ``torch.use_deterministic_algorithms(True)`` and must be set BEFORE
   the cuBLAS handle is created (i.e. before the first GPU matmul). We set
   it at module load time here so importing ``_bootstrap`` *first* in a
   script also wins the race against ``import torch``.

2. **sys.path.** Running ``python scripts/<name>.py`` puts the scripts
   directory on sys.path, not the codebase root. Centralize the one
   sys.path adjustment here instead of duplicating it in every script.

To benefit from (1), scripts must list ``import _bootstrap`` BEFORE any
torch / numpy / sklearn imports. ``set_seed()`` re-applies the env var as
a belt-and-suspenders, but the cuBLAS handle may already be cached by
then, so the early-binding here is what actually matters.
"""

import os
import sys
from pathlib import Path

# (1) Determinism env vars: must precede ``import torch`` to take effect.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# (2) sys.path: codebase root.
CODEBASE = Path(__file__).resolve().parent.parent
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))
