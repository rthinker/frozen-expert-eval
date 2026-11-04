"""Where a rerun of the sensitivity checks may write: never into their supplied outputs, the primary results, the
inputs or the feature cache.

An output folder is refused when it is one of the protected folders, lies inside one, or contains one. Paths are
compared as resolved paths (`..` and links resolved, case folded on Windows) and, where the folders exist, by file
identity, so another spelling of a protected folder (`sensitivity/dedup/../dedup`, `SENSITIVITY/dedup` on a
case-insensitive file system, an absolute path, a link) is refused as well.
"""
import os
from pathlib import Path

CR = Path(__file__).resolve().parents[2]          # camera_ready/
ROOT = CR.parent                                  # repository root
PROTECTED = (
    CR / "output" / "results" / "sensitivity",    # the supplied outputs of these checks
    CR / "output" / "results" / "mitigation",     # the primary LOGO results
    CR / "output" / "results" / "analyses",       # the primary standard-split and other post-submission results
    CR / "inputs", CR / "reference", CR / "provenance",
    ROOT / "output" / "results",                  # the reviewed results
    ROOT / "output" / "cached_features",          # the feature cache
)


def _norm(path):
    return os.path.normcase(os.path.realpath(path))


def _inside(path, folder):
    """Is `path` the folder or inside it, as resolved and normalized paths?"""
    p, f = _norm(path), _norm(folder)
    try:
        return os.path.commonpath([p, f]) == f
    except ValueError:                            # different drives
        return False


def _same(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def overlaps(path, folder):
    """Is `path` the folder, inside it, or a folder that contains it?"""
    path, folder = Path(path), Path(folder)
    if _inside(path, folder) or _inside(folder, path):
        return True
    # file identity of existing folders: a case-insensitive file system or a link the resolved strings do not show
    if any(a.exists() and _same(a, folder) for a in (path, *path.parents)):
        return True
    return path.exists() and any(_same(path, b) for b in (folder, *folder.parents) if b.exists())


def output_dir(arg):
    """The folder named by --out-dir (relative to the repository root, or absolute); SystemExit when it overlaps a
    protected folder."""
    path = Path(arg)
    path = path if path.is_absolute() else ROOT / path
    for folder in PROTECTED:
        if overlaps(path, folder):
            raise SystemExit(f"--out-dir {arg}: overlaps {folder.relative_to(ROOT).as_posix()}/; a rerun never writes "
                             "into the supplied outputs of the sensitivity checks, the primary results, the inputs or "
                             "the feature cache. Choose another folder (the default is "
                             "camera_ready/output/results/sensitivity_rerun/).")
    return path
