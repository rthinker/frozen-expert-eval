"""Resolve code and output paths for the mitigation runners.

setup() puts the local repository on sys.path for read-only use and loads scripts/_bootstrap.py before torch
or src modules are imported. An explicit codebase or MITIGATION_CODEBASE overrides the default.

out_path() refuses outputs in frozen folders and protects supplied release results for standalone runs.
The harness backs up supplied results before running steps that write to those paths.

Run the scripts with python -B to avoid writing bytecode into the codebase.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

RUNS = Path(__file__).resolve().parent
CAMERA = RUNS.parents[2]                      # the repository root
CH3 = CAMERA.parent                           # the folder that holds the repository
FROZEN = (CH3 / "01.initial_submission", CH3 / "02.rebuttal")
LEGACY_CODEBASE = CH3 / "01.initial_submission" / "codebase"
# A standalone release contains src/ beside camera_ready/. Resolve it from this file,
# independently of the caller's working directory; preserve the historical source layout.
DEFAULT_CODEBASE = CAMERA if (CAMERA / "src" / "utils" / "config.py").is_file() else LEGACY_CODEBASE
REBUTTAL = CH3 / "02.rebuttal"
OUT_ROOT = RUNS.parents[1] / "output" / "results" / "mitigation"
EXCLUSION_ROOT = OUT_ROOT / "exclusions"

# stored results of the reviewed runs (read-only references for the gates)
STORED_REB = REBUTTAL / "results"                                         # run_c_width_matched_<pool>.json


def select_gpu(name: str) -> str:
    """Make the one GPU whose name contains ``name`` the only CUDA device (cuda:0); exit if it is not visible.

    Must run before torch initialises CUDA (``setup`` calls it first). nvidia-smi lists GPUs in PCI bus order,
    so CUDA_DEVICE_ORDER is set to PCI_BUS_ID to make its index the CUDA index.
    """
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"--gpu-name {name}: nvidia-smi failed ({exc})")
    gpus = [tuple(x.strip() for x in line.split(",", 1)) for line in out.splitlines() if line.strip()]
    hits = [(i, n) for i, n in gpus if name.lower() in n.lower()]
    if len(hits) != 1:
        raise SystemExit(f"--gpu-name {name}: {len(hits)} matching GPU(s); visible: {gpus}. "
                         "Make the requested device visible in nvidia-smi before running.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = hits[0][0]
    return hits[0][1]


def setup(codebase=None, gpu_name=None) -> Path:
    """Make the reviewed code base importable (read-only) and apply its determinism prelude.

    With ``gpu_name`` the named GPU becomes the only CUDA device (``select_gpu``)."""
    if gpu_name:
        select_gpu(gpu_name)
    cb = Path(codebase or os.environ.get("MITIGATION_CODEBASE") or DEFAULT_CODEBASE).resolve()
    if not (cb / "src" / "utils" / "config.py").is_file():
        raise SystemExit(f"not a code base root (no src/utils/config.py): {cb}")
    for p in (cb, cb / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import _bootstrap  # noqa: F401  reviewed prelude (sets env, sys.path); writes nothing
    return cb


def require_validated_resume(enabled):
    """Checkpoint reuse belongs to the harness, which validates data, code and settings."""
    if enabled and (os.environ.get("REPRO_VALIDATED_RESUME") != "1"
                    or not os.environ.get("REPRO_ATTEMPT_ID")
                    or not os.environ.get("REPRO_FINGERPRINT")):
        raise SystemExit("--resume requires checkpoint validation by the reproduction harness. "
                         "Use camera_ready/scripts/review/reproduce_all_v4.py --resume-run <run>; "
                         "start standalone work without --resume in a new --out-dir.")


_CR = RUNS.parents[1]                         # camera_ready/ (release layout)
_ROOT = RUNS.parents[2]                       # repository root
# Standalone runners cannot write to the supplied release results. The harness backs them up first.
STORED = (_CR / "output" / "results" / "mitigation", _CR / "output" / "results" / "analyses",
          _CR / "output" / "results" / "sensitivity", _CR / "inputs", _CR / "reference", _CR / "provenance",
          _ROOT / "output" / "results", _ROOT / "output" / "cached_features")


def _inside(path, folder) -> bool:
    p, f = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(folder))
    return p == f or p.startswith(f.rstrip(os.sep) + os.sep)


def stored_problem(path):
    """Why a run started by hand may not write `path` (None when it may)."""
    if os.environ.get("REPRO_ATTEMPT_ID"):
        return None
    for f in STORED:
        if _inside(path, f):
            return f"it lies in {f.relative_to(_ROOT).as_posix()}/, among the stored results"
    if _inside(Path(path).parent, _CR / "output" / "results") and _inside(_CR / "output" / "results", Path(path).parent):
        return "it lies directly in camera_ready/output/results/, among the stored results"
    return None


def out_path(path) -> Path:
    """Resolve an output path, refusing anything inside the frozen folders and, for a run started by hand, anything
    among the stored results. Creates the parent."""
    p = Path(path).resolve()
    for f in FROZEN:
        try:
            p.relative_to(f.resolve())
        except ValueError:
            continue
        raise SystemExit(f"refusing to write inside a frozen folder: {p}")
    why = stored_problem(p)
    if why:
        raise SystemExit(f"refusing to write {p}: {why}. A run started by hand writes to a folder of its own (--out-dir "
                         "or --out); the harness (camera_ready/scripts/review/reproduce_all_v4.py --only <step>) backs "
                         "the stored results up and reruns them.")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path, text):
    """Publish a validated report without exposing a partially written file."""
    path = out_path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def rel(path) -> str:
    """Path for records and logs: relative to the repository root, else to the folder that holds it (no
    user-specific prefix)."""
    if path is None:
        return None
    p = Path(path).resolve()
    for base in (CAMERA, CH3):
        try:
            return p.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return p.name


def environment() -> dict:
    """Library versions and device, recorded with every result file."""
    import platform

    import numpy
    import scipy
    import sklearn
    env = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
           "sklearn": sklearn.__version__, "platform": platform.platform()}
    try:
        import xgboost
        env["xgboost"] = xgboost.__version__
    except ImportError:
        env["xgboost"] = None
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        env["torch_cuda_build"] = torch.version.cuda
        env["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    except ImportError:
        env["torch"] = None
    return env


def make_logger(name: str, out_dir: Path):
    """Console and file logger; the file goes to the output folder, never into the code base."""
    import logging
    from datetime import datetime
    out_dir = out_path(Path(out_dir) / "logs" / "x").parent
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(out_dir / f"{name}_{datetime.now():%y%m%d_%H%M%S}.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log
