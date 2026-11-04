"""Tests for naming conventions and subset selection determinism."""

import numpy as np

from src.utils.naming import tag_suffix, checkpoint_name, result_name
from src.utils.config import DEFAULT_TRAIN_TAG


def test_tag_suffix_default():
    """Default train_tag produces empty suffix."""
    assert tag_suffix(DEFAULT_TRAIN_TAG) == ""


def test_tag_suffix_custom():
    """Custom train_tag should produce matching suffix."""
    assert tag_suffix("custom_split") == "_custom_split"


def test_checkpoint_name_default():
    """Checkpoint naming with default tag has no tag in filename."""
    name = checkpoint_name("clip4", "mlp", 64, 1024)
    assert name == "clip4_mlp_d64_seed1024_best.pt"


def test_checkpoint_name_custom():
    """Checkpoint naming with custom tag includes that suffix."""
    name = checkpoint_name("clip4", "mlp", 64, 1024, "custom_split")
    assert name == "clip4_mlp_d64_seed1024_best_custom_split.pt"


def test_result_name_default():
    """Result naming with default tag has no tag suffix."""
    name = result_name("05", "fs", "clip4", "mlp", dim=64)
    assert name == "05_fs_clip4_mlp_d64.json"


def test_result_name_custom():
    """Result naming with custom tag."""
    name = result_name("04", "ds", "clip4", train_tag="custom_split")
    assert name == "04_ds_clip4_custom_split.json"


def test_balanced_subset_deterministic():
    """balanced_subset_indices should be deterministic given same inputs."""
    from src.utils.train_subset import balanced_subset_indices

    np.random.seed(42)
    labels = np.array([0] * 500 + [1] * 500, dtype=np.int32)
    ids_path = None  # no sample_ids needed for determinism test

    idx_a = balanced_subset_indices(labels, 200, sample_ids_path=ids_path)
    idx_b = balanced_subset_indices(labels, 200, sample_ids_path=ids_path)
    assert np.array_equal(idx_a, idx_b), "Same inputs should give same indices"
    assert len(idx_a) == 200


def test_balanced_subset_uses_multiple_source_folders(tmp_path):
    from src.utils.train_subset import balanced_subset_indices

    ids = np.array(
        [f"real/{i:03}.png" for i in range(12)]
        + [f"{generator}/ff/{i:03}.png" for generator in ("a", "b", "c") for i in range(12)]
    )
    labels = np.array([0] * 12 + [1] * 36)
    ids_path = tmp_path / "sample_ids.npy"
    np.save(ids_path, ids)

    keep = balanced_subset_indices(labels, 12, sample_ids_path=ids_path)

    assert np.bincount(labels[keep]).tolist() == [6, 6]
    assert {source: sum(item.startswith(f"{source}/") for item in ids[keep]) for source in "abc"} == {
        "a": 2, "b": 2, "c": 2,
    }
    np.save(ids_path, ids[::-1])
    reordered = balanced_subset_indices(labels[::-1], 12, sample_ids_path=ids_path)
    assert set(ids[keep]) == set(ids[::-1][reordered])
