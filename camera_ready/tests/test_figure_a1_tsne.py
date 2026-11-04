"""Checks for Figure A1 sampling, alignment and invalid-cache handling."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/authoring/figure_a1_tsne.py"


@pytest.fixture
def figure():
    spec = importlib.util.spec_from_file_location("figure_a1", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cache(tmp_path, figure, monkeypatch):
    monkeypatch.setattr(figure, "GENERATORS", ("first", "second"))
    monkeypatch.setattr(figure, "EXPECTED_ROWS", (630, 7))
    for generator, count in zip(figure.GENERATORS, figure.EXPECTED_ROWS):
        root = tmp_path / generator
        root.mkdir()
        np.save(root / "labels.npy", np.arange(count) % 2)
        for expert_index, expert in enumerate(figure.EXPERTS):
            for suffix, width in (("features", 1024), ("logits", 2)):
                values = np.repeat(np.arange(count, dtype=np.float32)[:, None], width, axis=1)
                values += expert_index * 1000
                np.save(root / f"{expert}_{suffix}.npy", values)
    return tmp_path


def test_sampling_keeps_interfaces_and_row_indices_aligned(figure, cache):
    features, logits, groups, indices, counts = figure.load_sample(cache)
    assert counts == {"first": 625, "second": 7}
    assert features.shape == (632, 4096) and logits.shape == (632, 8)
    assert len(set(indices[groups == 0])) == 625
    assert set(indices[groups == 1]) == set(range(7))
    for expert in range(4):
        expected = indices + expert * 1000
        np.testing.assert_array_equal(features[:, expert * 1024], expected)
        np.testing.assert_array_equal(logits[:, expert * 2], expected)
    repeated = figure.load_sample(cache)
    for first, second in zip((features, logits, groups, indices), repeated[:4]):
        np.testing.assert_array_equal(first, second)


def test_wrong_expert_shape_is_rejected(figure, cache):
    np.save(cache / "first" / f"{figure.EXPERTS[0]}_logits.npy", np.zeros((629, 2)))
    with pytest.raises(ValueError, match="unexpected shape"):
        figure.load_sample(cache)


def test_nonfinite_sample_is_rejected(figure, cache):
    np.save(cache / "second" / f"{figure.EXPERTS[0]}_features.npy",
            np.full((7, 1024), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        figure.load_sample(cache)


def test_wrong_partition_size_is_rejected(figure, cache):
    np.save(cache / "second" / "labels.npy", np.zeros(6))
    with pytest.raises(ValueError, match="expected 7 labels"):
        figure.load_sample(cache)
