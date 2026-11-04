"""Integration tests for shared FS runtime and reporting config.

These test the seams between modules, not individual functions.
"""

import json
from pathlib import Path

import numpy as np
import pytest


class TestFSRuntime:
    """Test that the shared FS runtime trains and predicts correctly."""

    @pytest.fixture
    def synthetic_pool(self, tmp_path):
        """Create minimal synthetic cached features for 2 experts."""
        rng = np.random.default_rng(42)
        N_train, N_val, N_test = 200, 50, 50

        for tag, n in [("train", N_train), ("val", N_val), ("gen1", N_test)]:
            d = tmp_path / tag
            d.mkdir()
            labels = rng.integers(0, 2, size=n).astype(np.int32)
            np.save(d / "labels.npy", labels)
            sample_ids = np.array([f"{tag}_{i:04d}" for i in range(n)], dtype=str)
            np.save(d / "sample_ids.npy", sample_ids, allow_pickle=False)
            # Expert A: 16-dim features, correlated with label
            feat_a = rng.standard_normal((n, 16)).astype(np.float32)
            feat_a[:, 0] += labels * 2.0  # inject signal
            np.save(d / "expA_features.npy", feat_a)
            np.save(d / "expA_logits.npy",
                    rng.standard_normal((n, 2)).astype(np.float32))
            # Expert B: 8-dim features
            feat_b = rng.standard_normal((n, 8)).astype(np.float32)
            feat_b[:, 0] += labels * 1.5
            np.save(d / "expB_features.npy", feat_b)
            np.save(d / "expB_logits.npy",
                    rng.standard_normal((n, 2)).astype(np.float32))
        return tmp_path, ["expA", "expB"], [16, 8]

    def test_train_and_predict(self, synthetic_pool):
        """End-to-end: train_fs_model + predict_fs produces valid AUCs."""
        from src.training.fs_runtime import train_fs_model, predict_fs
        from src.utils.metrics import compute_auc

        cache_root, expert_names, feature_dims = synthetic_pool
        train_dir = cache_root / "train"
        val_dir = cache_root / "val"

        train_feats = [np.load(train_dir / f"{n}_features.npy")
                       for n in expert_names]
        train_labels = np.load(train_dir / "labels.npy")
        val_feats = [np.load(val_dir / f"{n}_features.npy")
                     for n in expert_names]
        val_labels = np.load(val_dir / "labels.npy")

        model, best_val = train_fs_model(
            train_feats, train_labels, feature_dims, proj_dim=8,
            val_feats=val_feats, val_labels=val_labels,
            seed=42, epochs=5, batch_size=64,
        )

        assert best_val > 0.0, "Val AUC should be positive"
        assert model.training is False, "Model should be in eval mode"

        test_feats = [np.load(cache_root / "gen1" / f"{n}_features.npy")
                      for n in expert_names]
        test_labels = np.load(cache_root / "gen1" / "labels.npy")
        scores = predict_fs(model, test_feats)

        assert scores.shape == (len(test_labels),)
        assert 0.0 <= scores.min() and scores.max() <= 1.0
        auc = compute_auc(test_labels, scores)
        assert auc > 0.5, f"With injected signal, AUC should be > 0.5, got {auc}"

    def test_cached_loader_mmaps(self, synthetic_pool):
        """load_pool_data should load and mmap correctly."""
        from src.utils.io import load_pool_data

        cache_root, expert_names, _ = synthetic_pool
        feats, logits, labels = load_pool_data(
            expert_names, "train", cache_root, mmap=True)

        assert isinstance(feats[0], np.memmap)
        assert isinstance(logits[0], np.memmap)
        assert labels.dtype == np.int32
        assert feats[0].shape[0] == len(labels)

    def test_zero_copy_tensors(self):
        """_to_tensors uses zero-copy for contiguous float32."""
        from src.training.fs_runtime import _to_tensors

        arr = np.ones((10, 4), dtype=np.float32)
        tensors = _to_tensors([arr])
        assert tensors[0].data_ptr() == arr.ctypes.data, \
            "Should share memory (zero-copy)"

        arr_f = np.asfortranarray(arr)
        tensors_f = _to_tensors([arr_f])
        assert tensors_f[0].shape == (10, 4)


class TestReportingConfig:
    """Test that reporting config loads and exposes expected values."""

    def test_reporting_constants_exist(self):
        from src.utils.config import (
            REP_HEADLINE_POOL, REP_HEADLINE_DIM,
            REP_SECONDARY_POOL, REP_SECONDARY_DIM,
        )
        assert isinstance(REP_HEADLINE_POOL, str)
        assert isinstance(REP_HEADLINE_DIM, int)
        assert isinstance(REP_SECONDARY_POOL, str)
        assert isinstance(REP_SECONDARY_DIM, int)

    def test_experts_cfg_available(self):
        from src.utils.config import EXPERTS_CFG, get_pool_experts
        assert "pools" in EXPERTS_CFG
        assert "experts" in EXPERTS_CFG
        pool_name = list(EXPERTS_CFG["pools"].keys())[0]
        experts = get_pool_experts(pool_name)
        assert isinstance(experts, list)
        assert len(experts) > 0

    def test_path_constants_are_paths(self):
        from src.utils.config import (
            CACHE_ROOT, CKPT_ROOT, RESULTS_ROOT,
            FIGURES_ROOT, TABLES_ROOT, LOGS_ROOT,
        )
        for p in (CACHE_ROOT, CKPT_ROOT, RESULTS_ROOT,
                  FIGURES_ROOT, TABLES_ROOT, LOGS_ROOT):
            assert isinstance(p, Path), f"{p} should be a Path"

    def test_report_manifest_scan(self, tmp_path):
        """scan_results should classify DS/FS/PCA artifacts by filename prefix."""
        from src.reporting.manifest import scan_results

        ds_path = tmp_path / "04_ds_clip4.json"
        ds_path.write_text(json.dumps({
            "pool": "clip4",
            "best_ds_avg8": 0.95,
            "baselines": [],
        }), encoding="utf-8")

        fs_path = tmp_path / "05_fs_clip4_mlp_d32.json"
        fs_path.write_text(json.dumps({
            "pool": "clip4",
            "combiner_type": "mlp",
            "proj_dim": 32,
            "mean_avg8": 0.97,
            "per_seed": [],
        }), encoding="utf-8")

        pca_path = tmp_path / "08_pca_clip4.json"
        pca_path.write_text(json.dumps({
            "pool": "clip4",
            "features": {"pr": 12.0},
            "logits": {"pr": 2.0},
            "pr_gap": 6.0,
        }), encoding="utf-8")

        manifest = scan_results(tmp_path)
        assert "clip4" in manifest["ds"]
        assert ("clip4", "mlp", 32) in manifest["fs"]
        assert "clip4" in manifest["pca"]


class TestRobustnessControls:
    """Tests for the 12_xgb_features (FS_D) and 13_fs1_standardized controls."""

    def test_reproduce_all_includes_new_controls(self):
        """EXPERIMENTS must list arch6 FS_D + both 13_fs1_standardized JSONs."""
        import sys
        # reproduce_all.py moved to scripts/review/.
        script_dir = Path(__file__).resolve().parent.parent / "scripts" / "review"
        sys.path.insert(0, str(script_dir))
        try:
            from reproduce_all import EXPERIMENTS, EXPECTED_TABLES
        finally:
            sys.path.remove(str(script_dir))

        outputs = []
        for _prefix, _cmd, expected_outputs, _det in EXPERIMENTS:
            outputs.extend(expected_outputs)

        assert "12_xgb_features_clip4.json" in outputs, \
            "clip4 FS_D control still required (capacity ablation)"
        assert "12_xgb_features_arch6.json" in outputs, \
            "arch6 FS_D control (family-matched for the flagship) missing"
        assert "13_fs1_standardized_arch6.json" in outputs, \
            "13_fs1_standardized arch6 control missing from EXPERIMENTS"
        assert "13_fs1_standardized_clip4.json" in outputs, \
            "13_fs1_standardized clip4 control missing from EXPERIMENTS"

        # The generated robustness table should be tracked.
        assert "table_robustness_controls.tex" in EXPECTED_TABLES

    def test_reference_marks_standardized_fs1_deterministic(self):
        """The shipped reference classifies both standardized FS1 results as deterministic."""
        reference = Path(__file__).resolve().parents[1] / "output/results/reference_results.json"
        results = json.loads(reference.read_text(encoding="utf-8"))["results"]
        for pool in ("arch6", "clip4"):
            assert results[f"13_fs1_standardized_{pool}.json"]["deterministic"] is True

    def test_manifest_parses_xgb_features(self, tmp_path):
        """12_xgb_features_{pool}.json must land in manifest['xgb_features'][pool]."""
        from src.reporting.manifest import scan_results

        p = tmp_path / "12_xgb_features_arch6.json"
        p.write_text(json.dumps({
            "pool": "arch6",
            "method": "xgboost_on_features",
            "avg_auc_8": 0.954,
            "per_generator": {"whichisreal": 0.9},
            "params": {"n_estimators": 200, "max_depth": 3},
        }), encoding="utf-8")
        manifest = scan_results(tmp_path)
        assert "xgb_features" in manifest
        assert "arch6" in manifest["xgb_features"]
        assert manifest["xgb_features"]["arch6"]["avg_auc_8"] == 0.954

    def test_manifest_parses_fs1_standardized(self, tmp_path):
        """13_fs1_standardized_{pool}.json must land in manifest['fs1_standardized']."""
        from src.reporting.manifest import scan_results

        p = tmp_path / "13_fs1_standardized_clip4.json"
        p.write_text(json.dumps({
            "pool": "clip4",
            "method": "fs_1_standardized_vs_raw",
            "avg_auc_8_raw": 0.9654,
            "avg_auc_8_std": 0.9659,
            "delta_std_minus_raw": 0.0005,
            "per_generator_raw": {"whichisreal": 0.9},
            "per_generator_std": {"whichisreal": 0.9},
        }), encoding="utf-8")
        manifest = scan_results(tmp_path)
        assert "fs1_standardized" in manifest
        assert "clip4" in manifest["fs1_standardized"]
        assert manifest["fs1_standardized"]["clip4"]["delta_std_minus_raw"] == 0.0005

    def test_manifest_rejects_wrong_12_method(self, tmp_path):
        """12_* with wrong `method` field should be skipped, not misclassified."""
        from src.reporting.manifest import scan_results

        p = tmp_path / "12_xgb_features_clip4.json"
        p.write_text(json.dumps({
            "pool": "clip4",
            "method": "NOT_xgboost_on_features",
            "avg_auc_8": 0.9,
            "per_generator": {},
        }), encoding="utf-8")
        manifest = scan_results(tmp_path)
        assert manifest["xgb_features"] == {}

    def test_robustness_table_renders(self, tmp_path):
        """table_robustness_controls should render when both inputs are present."""
        from src.reporting.tables import table_robustness_controls

        manifest = {
            "xgb_features": {
                "clip4": {"avg_auc_8": 0.959},
                "arch6": {"avg_auc_8": 0.954},
            },
            "fs1_standardized": {
                "clip4": {"avg_auc_8_raw": 0.9654, "avg_auc_8_std": 0.9659,
                          "delta_std_minus_raw": 0.0005},
                "arch6": {"avg_auc_8_raw": 0.9657, "avg_auc_8_std": 0.9654,
                          "delta_std_minus_raw": -0.0003},
            },
            # PS_4 XGB logit AUROCs sourced from 04_ds_{pool} ladder.
            # The current 04 schema stores each method's entry as a dict with
            # mean_avg8 / std_avg8 / per_generator; _best_ps_xgb_logit_auc
            # must handle both that and the older scalar fallback.
            "ds": {
                "clip4": {"method_means": {"PS_4_xgboost": {"mean_avg8": 0.939}}},
                "arch6": {"method_means": {"PS_4_xgboost": {"mean_avg8": 0.843}}},
            },
        }
        out = table_robustness_controls(manifest)
        assert out is not None
        # Both sub-tables should appear.
        assert "FS\\_D: XGB on features" in out
        assert "FS\\_1 raw" in out
        # Values present.
        assert "0.9590" in out and "0.9540" in out
        assert "0.9654" in out and "0.9659" in out

    def test_robustness_table_degrades_to_none_when_empty(self):
        """Empty manifest => None (not crash)."""
        from src.reporting.tables import table_robustness_controls
        assert table_robustness_controls({}) is None
