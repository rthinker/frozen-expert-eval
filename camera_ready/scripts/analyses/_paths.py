"""Locate the reviewed code base and the output folders for the post-submission analyses.

Repository layout:

    <repository root>/            the reviewed code (src/, scripts/, configs/, output/), never written
        camera_ready/
            scripts/analyses/     this folder: the post-submission analyses (Block C of
                                  camera_ready/scripts/review/reproduce_all_v4.py)
            output/results/analyses/   their results
            output/logs/analyses/      their logs

* The reviewed code base is the repository root. ``CR_CODEBASE`` overrides it.
* Results go to ``camera_ready/output/results/analyses/`` (``CR_RESULTS_DIR`` overrides it); logs
  go to ``camera_ready/output/logs/analyses/``.
* ``assert_no_codebase_writes`` refuses any path inside the repository that is not under
  ``camera_ready/output/`` (or under ``CR_RESULTS_DIR`` when that is set elsewhere), so the
  reviewed files stay unchanged.
* ``df40_json_dir()`` gives the folder of DF40's own split lists (DF40_all.json, FSAll_ff.json,
  ..., one JSON per held-out generator), needed only by the analyses that read DF40's lists rather
  than the feature cache. It comes from ``--df40-json <dir>`` on the command line or the
  ``DF40_JSON_DIR`` environment variable.

Import this module FIRST in every analysis script, before torch or any ``src.*`` import: it puts
the code base on ``sys.path`` and applies the reviewed determinism prelude (``scripts/_bootstrap.py``).

Direct execution and split-list validation:

* A script of this folder run by hand, not by the reproducibility harness (which sets ``REPRO_ATTEMPT_ID`` for
  every step it runs, after backing the stored results up), refuses before any work when its results would go
  among the stored results the reference pins (``camera_ready/output/results/analyses/`` by default): run it
  through the harness (``--only <step>``), or set ``CR_RESULTS_DIR`` to another folder.
* A script without options prints its description for ``-h``/``--help`` instead of running.
* ``df40_json_dir()`` verifies the split-list hashes written by ``prepare_df40_lists_v2.py`` (step C00).
  DF40's published lists name their files in another form than the cache's sample ids.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ANALYSES = Path(__file__).resolve().parent          # camera_ready/scripts/analyses
CAMERA_READY = ANALYSES.parents[1]                  # camera_ready
REPO_ROOT = CAMERA_READY.parent                     # repository root (the reviewed code)
OUTPUT_ROOT = CAMERA_READY / "output"


def _find_codebase() -> Path:
    p = Path(os.environ.get("CR_CODEBASE") or REPO_ROOT).resolve()
    if not (p / "src" / "utils" / "config.py").is_file():
        raise RuntimeError(f"not a code base root (no src/utils/config.py): {p}. Set CR_CODEBASE.")
    return p


CODEBASE_ROOT = _find_codebase()

# All analysis outputs go here; never into the reviewed code.
RESULTS_DIR = Path(os.environ.get("CR_RESULTS_DIR") or OUTPUT_ROOT / "results" / "analyses").resolve()

# Stored inputs and results that direct script execution must protect
STORED_RESULTS = (OUTPUT_ROOT / "results" / "analyses", OUTPUT_ROOT / "results" / "mitigation",
                  OUTPUT_ROOT / "results" / "sensitivity", CAMERA_READY / "inputs", REPO_ROOT / "output" / "results")


def _inside(path, folder) -> bool:
    p, f = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(folder))
    return p == f or p.startswith(f.rstrip(os.sep) + os.sep)


def _script_run_by_hand():
    """The analysis script run as the main program without the harness (None under the harness, a test or an import)."""
    if os.environ.get("REPRO_ATTEMPT_ID"):
        return None
    f = getattr(sys.modules.get("__main__"), "__file__", None)
    return Path(f) if f and _inside(Path(f).parent, ANALYSES) and _inside(ANALYSES, Path(f).parent) else None


_BY_HAND = _script_run_by_hand()
_HELP = any(a in ("-h", "--help") for a in sys.argv[1:])
if _BY_HAND is not None and _HELP and "argparse" not in _BY_HAND.read_text(encoding="utf-8"):
    print((getattr(sys.modules["__main__"], "__doc__", None) or f"{_BY_HAND.name}: no options").strip())
    raise SystemExit(0)
if _BY_HAND is not None and not _HELP and any(_inside(RESULTS_DIR, f) for f in STORED_RESULTS):
    raise SystemExit(f"{_BY_HAND.name} would write its results to {RESULTS_DIR}, among the stored results the "
                     "reference pins. Run it through the harness, which backs them up first (python "
                     "camera_ready/scripts/review/reproduce_all_v4.py --only <step>), or set CR_RESULTS_DIR to another "
                     "folder.")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = OUTPUT_ROOT / "logs" / "analyses"

# Make the reviewed code importable: `src.*` lives at the code base root, `_bootstrap` in scripts/.
for p in (CODEBASE_ROOT, CODEBASE_ROOT / "scripts"):
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

# Reuse the reviewed determinism prelude verbatim (binds CUBLAS_WORKSPACE_CONFIG before torch is used).
import _bootstrap  # noqa: E402,F401  isort:skip


def cache_root():
    """Cache root from the reviewed config (<code base>/output/cached_features)."""
    from src.utils.config import CACHE_ROOT
    return CACHE_ROOT


def assert_no_codebase_writes(path: Path) -> Path:
    """Guard: refuse a path inside the repository unless it is under camera_ready/output/ or RESULTS_DIR."""
    path = Path(path).resolve()
    for allowed in (OUTPUT_ROOT.resolve(), RESULTS_DIR):
        if path.is_relative_to(allowed):
            return path
    for protected in (REPO_ROOT.resolve(), CODEBASE_ROOT):
        if path.is_relative_to(protected):
            raise RuntimeError(f"Refusing to write inside the reviewed code: {path}")
    return path


def df40_json_dir() -> Path:
    """Folder of DF40's split lists, from --df40-json <dir> or DF40_JSON_DIR; exits with a clear message if absent."""
    from prepare_df40_lists_v2 import verify_prepared_lists

    value = None
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--df40-json" and i + 1 < len(argv):
            value = argv[i + 1]
        elif a.startswith("--df40-json="):
            value = a.split("=", 1)[1]
    value = value or os.environ.get("DF40_JSON_DIR")
    if not value:
        raise SystemExit("This analysis reads DF40's own split lists (DF40_all.json, FSAll_ff.json, FRAll_ff.json, "
                         "EFSAll_ff.json and one JSON per held-out generator), rewritten into the cache's path form by "
                         "camera_ready/scripts/analyses/prepare_df40_lists_v2.py. Pass --df40-json <folder it wrote> or "
                         "set DF40_JSON_DIR (see camera_ready/REPRODUCTION.md for where to obtain the lists).")
    p = Path(value).expanduser().resolve()
    if not p.is_dir():
        raise SystemExit(f"--df40-json / DF40_JSON_DIR is not a folder: {p}")
    verify_prepared_lists(p)
    return p
