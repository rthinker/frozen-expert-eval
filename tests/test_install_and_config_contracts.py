"""Check installation fallback, licence declarations, FIT configuration and stable softmax.
These checks require no GPU or feature arrays.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent


# ---------------------------------------------------------------------------
# 1. composition_eval imports without configs/paths.yaml
# ---------------------------------------------------------------------------

def test_composition_eval_imports_without_paths_yaml(tmp_path: Path):
    """`import composition_eval` works on a fresh checkout where the user
    has not yet copied paths.yaml.example to paths.yaml."""
    # Build a stripped clone of the codebase: source tree minus configs/paths.yaml.
    workdir = tmp_path / "fresh_checkout"
    shutil.copytree(
        _REPO, workdir,
        ignore=shutil.ignore_patterns(
            "__pycache__", ".pytest_cache", ".ruff_cache",
            ".git", ".venv", "output", "third_party",
            "*.egg-info", ".benchmarks",
        ),
    )
    paths_yaml = workdir / "configs" / "paths.yaml"
    if paths_yaml.exists():
        paths_yaml.unlink()
    assert (workdir / "configs" / "paths.yaml.example").exists(), (
        "paths.yaml.example must ship in the repository; the fallback "
        "depends on it."
    )

    proc = subprocess.run(
        [sys.executable, "-c", "import composition_eval; print('ok')"],
        cwd=str(workdir), capture_output=True, text=True, timeout=60,
    )
    # We accept either:
    #   stdout="ok\n" + a stderr fallback warning (the design) OR
    #   stdout="ok\n" with empty stderr (also fine).
    assert proc.returncode == 0, (
        f"composition_eval import failed without paths.yaml.\n"
        f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
    )
    assert "ok" in proc.stdout


# ---------------------------------------------------------------------------
# 2. License consistency: pyproject = LICENSE = CC-BY-NC-SA-4.0
# ---------------------------------------------------------------------------

def test_pyproject_license_matches_license_file():
    """pyproject.toml's `project.license.text` must agree with LICENSE."""
    pyproject = _REPO / "pyproject.toml"
    license_file = _REPO / "LICENSE"
    with open(pyproject, "rb") as f:
        pp = tomllib.load(f)
    license_text = license_file.read_text(encoding="utf-8")

    declared = pp["project"]["license"]["text"]
    assert declared == "CC-BY-NC-SA-4.0", (
        f"pyproject.toml declares license={declared!r}; the LICENSE file "
        "is CC BY-NC-SA 4.0, so pyproject must say 'CC-BY-NC-SA-4.0'. The cache "
        "(separate artifact) is documented in the dataset card / "
        "LICENSE_NOTICE.txt, not in pyproject."
    )
    assert "Attribution-NonCommercial-ShareAlike 4.0 International" in license_text, (
        "LICENSE file must hold the CC BY-NC-SA 4.0 legal code so license-detection "
        "tools (GitHub, PyPI) classify it correctly."
    )


def test_dataset_license_is_cc_by_nc_sa_4():
    """The packaged cache notice declares the project's CC BY-NC-SA terms."""
    notice = _REPO / "output/cached_features/LICENSE_NOTICE.txt"
    text = notice.read_text(encoding="utf-8")
    assert "CC BY-NC-SA 4.0" in text
    assert "https://creativecommons.org/licenses/by-nc-sa/4.0/" in text
    assert "DF40" in text


# ---------------------------------------------------------------------------
# 3. Train tag is sourced from config, not hardcoded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "script_relpath",
    [
        "scripts/12_xgb_features.py",
        "scripts/14_merged_pool.py",
    ],
)
def test_train_tag_uses_constant(script_relpath: str):
    """Scripts under scripts/ should source the train tag from
    src.utils.config (FIT_TAG / DEFAULT_TRAIN_TAG) rather than
    hardcoding the literal "FIT". Catches regressions where someone
    pastes a literal back in."""
    src = (_REPO / script_relpath).read_text(encoding="utf-8")
    # Allow "FIT" inside docstrings / comments / log messages, but flag
    # the specific call patterns that should use the constant.
    bad_patterns = [
        r'load_concat_features\([^)]*"FIT"',
        r'load_stacked_logits\([^)]*"FIT"',
    ]
    for pat in bad_patterns:
        assert re.search(pat, src) is None, (
            f"{script_relpath} hardcodes 'FIT' in {pat!r}; use FIT_TAG "
            "from src.utils.config so the train tag stays a single source "
            "of truth."
        )


# ---------------------------------------------------------------------------
# 4. PS_B uses scipy.special.softmax (not raw np.exp) - regression guard
# ---------------------------------------------------------------------------

def test_ps_b_uses_scipy_softmax():
    """PS_B's weight reparameterization must use scipy_softmax to avoid
    overflow when Nelder-Mead pushes any w_raw above ~700."""
    src = (_REPO / "src" / "models" / "prediction_space.py").read_text(encoding="utf-8")
    ps_b_block = re.search(
        r"def PS_B\(.*?(?=\ndef |\Z)", src, flags=re.DOTALL
    )
    assert ps_b_block is not None, "PS_B function not found in prediction_space.py"
    body = ps_b_block.group(0)

    # Strip comments (anything after # on a line) so explanatory text in
    # comments doesn't trigger the regex; only check actual code lines.
    code_only = "\n".join(
        line.split("#", 1)[0] for line in body.splitlines()
    )

    # Code must not contain the raw np.exp / sum normalization pattern.
    bad = re.compile(r"np\.exp\([^)]+\)\s*/\s*np\.exp\([^)]+\)\.sum\(\)")
    assert bad.search(code_only) is None, (
        "PS_B uses a raw `np.exp(...) / np.exp(...).sum()` softmax which "
        "overflows when Nelder-Mead pushes any input above ~700. Use "
        "scipy_softmax instead."
    )
    assert "scipy_softmax" in code_only, (
        "PS_B should use scipy_softmax for the weight reparameterization."
    )
