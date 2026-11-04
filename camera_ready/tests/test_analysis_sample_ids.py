"""Exercise the analysis wrapper for safe cached sample identifiers."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def analysis_common(tmp_path, monkeypatch):
    analyses = Path(__file__).resolve().parents[1] / "scripts" / "analyses"
    monkeypatch.syspath_prepend(str(analyses))
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path / "analysis_results"))
    spec = importlib.util.spec_from_file_location("analysis_sample_ids_common", analyses / "_common.py")
    common = importlib.util.module_from_spec(spec)
    paths_already_imported = "_paths" in sys.modules
    try:
        spec.loader.exec_module(common)
        yield common
    finally:
        if not paths_already_imported:
            sys.modules.pop("_paths", None)


@pytest.mark.parametrize("as_bytes", [False, True], ids=["unicode", "utf8-bytes"])
def test_analysis_wrapper_reads_string_ids_in_row_order(analysis_common, tmp_path, as_bytes):
    fit = tmp_path / "FIT"
    fit.mkdir()
    expected = ["group/real/sample_02.jpg", "group/fake/sample_é01.jpg"]
    values = [value.encode("utf-8") for value in expected] if as_bytes else expected
    np.save(fit / "sample_ids.npy", np.asarray(values), allow_pickle=False)

    ids = analysis_common.load_sample_ids("FIT", tmp_path)

    assert ids.dtype == object
    assert ids.tolist() == expected


def test_analysis_wrapper_refuses_unverified_object_ids(analysis_common, tmp_path):
    fit = tmp_path / "FIT"
    fit.mkdir()
    np.save(fit / "sample_ids.npy", np.asarray(["unverified_sample"], dtype=object))

    with pytest.raises(ValueError, match="Unverified object identifier array"):
        analysis_common.load_sample_ids("FIT", tmp_path)


def test_analysis_cache_helpers_honor_the_alternate_codebase(tmp_path):
    analyses = Path(__file__).resolve().parents[1] / "scripts" / "analyses"
    release = analyses.parents[2]
    codebase = tmp_path / "alternate"
    files = {
        "scripts/_bootstrap.py": "",
        "src/__init__.py": "",
        "src/utils/__init__.py": "",
        "src/utils/config.py": "CACHE_ROOT = 'alternate cache'\n",
        "src/utils/cache_ids.py": "def load_sample_ids(path): return ['alternate id']\n",
    }
    for name, contents in files.items():
        path = codebase / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    # Reproduce both paths already being present, with the release root first.
    code = f"""
import sys
sys.path[:0] = {list(map(str, (analyses, release, codebase)))!r}
import _paths
import _verify_cache_helpers as helpers
assert _paths.cache_root() == 'alternate cache'
assert helpers._load_ids('ignored.npy').tolist() == ['alternate id']
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=tmp_path,
        env={**os.environ, "CR_CODEBASE": str(codebase), "CR_RESULTS_DIR": str(tmp_path / "analysis_results")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
