"""Validate the current LOGO exclusion manifest contract without fitting models."""

import json
from pathlib import Path

import pytest


CR = Path(__file__).resolve().parents[1]
EXPERTS = ["expert-one", "expert-two"]
GENERATORS = {"CollabDiff": (None, None, [0, 1]), "styleclip": (None, None, [0, 1, 0])}


@pytest.fixture
def loader(monkeypatch):
    monkeypatch.syspath_prepend(str(CR / "scripts" / "runs"))
    from logo_common import load_content_manifest
    return load_content_manifest


@pytest.fixture
def manifest():
    return {
        "pool": "test-pool",
        "experts": EXPERTS.copy(),
        "gate_G1": "PASS",
        "rows_excluded_total": 1,
        "folds": [
            {"held_out": "CollabDiff", "excluded_rows": {"styleclip": [2]}, "n_excluded": 1},
            {"held_out": "styleclip", "excluded_rows": {}, "n_excluded": 0},
        ],
    }


def test_accepts_valid_zero_based_rows_and_no_manifest(loader, manifest, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert loader(path, "test-pool", EXPERTS, GENERATORS) == {"CollabDiff": {"styleclip": [2]}, "styleclip": {}}
    assert loader(None, "test-pool", EXPERTS, GENERATORS) == {}


@pytest.mark.parametrize(("key", "value", "error"), [
    ("pool", "other-pool", "expected pool"),
    ("experts", EXPERTS[::-1], "in that order"),
    ("gate_G1", "FAIL", "gate G1 did not pass"),
    ("rows_excluded_total", 2, "total exclusion count"),
])
def test_rejects_inconsistent_manifest_metadata(loader, manifest, tmp_path, key, value, error):
    manifest[key] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match=error):
        loader(path, "test-pool", EXPERTS, GENERATORS)


@pytest.mark.parametrize(("key", "value", "error"), [
    ("held_out", "unknown", "unexpected or repeated held-out generator"),
    ("excluded_rows", {"CollabDiff": [0]}, "invalid training generator"),
    ("excluded_rows", {"unknown": [0]}, "invalid training generator"),
    ("excluded_rows", {"styleclip": [-1]}, "zero-based row indices"),
    ("excluded_rows", {"styleclip": [3]}, "zero-based row indices"),
    ("excluded_rows", {"styleclip": [True]}, "zero-based row indices"),
    ("excluded_rows", {"styleclip": [1.0]}, "zero-based row indices"),
    ("excluded_rows", {"styleclip": [2, 2]}, "repeats row indices"),
    ("n_excluded", 2, "exclusion count does not match"),
])
def test_rejects_invalid_fold_records(loader, manifest, tmp_path, key, value, error):
    manifest["folds"][0][key] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match=error):
        loader(path, "test-pool", EXPERTS, GENERATORS)


def test_rejects_repeated_fold(loader, manifest, tmp_path):
    manifest["folds"][1]["held_out"] = "CollabDiff"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="repeated held-out generator"):
        loader(path, "test-pool", EXPERTS, GENERATORS)


def test_rejects_missing_fold(loader, manifest, tmp_path):
    manifest["folds"].pop()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="missing held-out generators styleclip"):
        loader(path, "test-pool", EXPERTS, GENERATORS)


@pytest.mark.parametrize("pool", ["arch6", "clip4", "strong4", "weak6"])
def test_accepts_supplied_manifests(loader, pool):
    from src.utils.config import get_pool_experts
    from src.utils.dataset_helpers import TEST_GENERATORS

    data = {
        generator: (None, None, (CR / "output" / "ids" / f"{generator}.txt").read_text(encoding="utf-8").splitlines())
        for generator in TEST_GENERATORS
    }
    path = CR / "output" / "results" / "mitigation" / "exclusions" / f"logo_exclusions_{pool}.json"
    assert set(loader(path, pool, get_pool_experts(pool), data)) == set(TEST_GENERATORS)
