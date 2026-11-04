"""Check class weighting, calibration, expert dropout, projection size and tagged result classification.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score


def test_fs1_logs_solver_iteration_cap(monkeypatch, caplog):
    from src.models.feature_space import FS_1, _DS_CFG

    monkeypatch.setitem(_DS_CFG, "raw_feature_logreg_max_iter", 1)
    features = np.array([[-3.0], [-1.0], [1.0], [3.0]])
    with pytest.warns(ConvergenceWarning):
        scores = FS_1(features, np.array([0, 0, 1, 1]), features)

    assert scores.shape == (4,)
    assert "FS1 solver iterations: 1/1" in caplog.text


# ---------------------------------------------------------------------------
# Class-balanced _train_mlp
# ---------------------------------------------------------------------------

def test_train_mlp_class_balanced_on_imbalanced_data():
    """On 1:5 imbalance, class-weighted CE must recover minority class."""
    from src.models.prediction_space import _train_mlp

    rng = np.random.default_rng(42)
    N_real, N_fake = 200, 1000
    X = np.concatenate([
        rng.standard_normal((N_real, 8)) + np.array([1.0, 0, 0, 0, 0, 0, 0, 0]),
        rng.standard_normal((N_fake, 8)) + np.array([-0.3, 0, 0, 0, 0, 0, 0, 0]),
    ]).astype(np.float32)
    y = np.concatenate([np.zeros(N_real), np.ones(N_fake)]).astype(np.int64)

    model, _ = _train_mlp(X, y, epochs=20, seed=1024)
    with torch.no_grad():
        probs = torch.softmax(
            model(torch.as_tensor(X)), dim=1
        )[:, 1].numpy()
    real_recall = (probs[:N_real] < 0.5).mean()
    auc = roc_auc_score(y, probs)
    assert real_recall > 0.5, (
        f"minority recall {real_recall:.3f} too low "
        f"(suggests class-weighted CE is not applied)"
    )
    assert auc > 0.7, f"AUC {auc:.3f} below sanity threshold"


# ---------------------------------------------------------------------------
# PS_A balanced-NLL temperature scaling, numerically stable
# ---------------------------------------------------------------------------

def test_s2_moves_temperature_off_unity_on_overconfident_expert():
    """PS_A must find non-unity T for overconfident experts."""
    from src.models.prediction_space import PS_A

    rng = np.random.default_rng(42)
    N = 2000
    labels = rng.integers(0, 2, N).astype(int)
    noise0 = rng.standard_normal(N) * 1.5
    logits0 = np.stack([
        -labels * 2.0 + 1.0 + noise0,
        labels * 2.0 - 1.0 - noise0,
    ], axis=1)
    logits1 = logits0 * 4.0
    logits = np.stack([logits0, logits1], axis=1)

    _, diag = PS_A(
        logits, labels, logits[:1], return_diagnostics=True)
    t0, t1 = diag["temps"]

    assert t0 > 1.5, f"expert 0 temperature {t0} not above 1"
    assert t1 > 2.0 * t0, (
        f"expert 1 temperature {t1} should be ~4x expert 0 ({t0}); "
        f"observed ratio {t1 / t0:.2f}"
    )


def test_s2_handles_saturated_logits():
    """Numerically stable PS_A should not pin at the 1.0 L-BFGS start."""
    from src.models.prediction_space import PS_A

    rng = np.random.default_rng(1)
    N = 800
    labels = rng.integers(0, 2, N).astype(int)
    logits = np.stack(
        [
            -labels * 15.0 + rng.standard_normal(N),
            labels * 15.0 + rng.standard_normal(N),
        ],
        axis=1,
    )
    logits = logits[:, None, :]  # [N, 1, 2] -- one expert
    _, diag = PS_A(
        logits, labels, logits[:1], return_diagnostics=True)
    t = diag["temps"][0]
    assert abs(t - 1.0) > 0.2, f"PS_A pinned at T={t}, expected movement"


# ---------------------------------------------------------------------------
# Per-sample expert dropout in FeatureAssembly
# ---------------------------------------------------------------------------

def test_feature_assembly_per_sample_expert_dropout_shape():
    """Training forward pass must produce [B, 2] output."""
    from src.models.feature_space import FeatureAssembly

    dims = [64, 96, 48]
    model = FeatureAssembly(
        dims, proj_dim=16, classifier_type="mlp",
        hidden_dims=[32], dropout=0.0,
        expert_dropout=0.5,
    )
    model.train()
    feats = [torch.randn(7, d) for d in dims]
    out = model(feats)
    assert out.shape == (7, 2)
    assert not torch.isnan(out).any()


def test_expert_dropout_helper_guarantees_one_active():
    """apply_expert_dropout must never fully zero any sample."""
    from src.models.feature_space import apply_expert_dropout

    torch.manual_seed(0)
    B, N, d = 64, 4, 8
    projected = [torch.ones(B, d) for _ in range(N)]
    out = apply_expert_dropout(projected, expert_dropout=0.99, training=True)
    stacked = torch.stack(out, dim=0)
    per_sample_sum = stacked.abs().sum(dim=(0, 2))
    assert (per_sample_sum > 0).all(), (
        "some samples have all experts zeroed"
    )


def test_expert_dropout_helper_eval_noop():
    """Eval mode must be a no-op regardless of expert_dropout."""
    from src.models.feature_space import apply_expert_dropout

    B, N, d = 16, 3, 8
    projected = [torch.randn(B, d) for _ in range(N)]
    snapshot = [p.clone() for p in projected]
    out = apply_expert_dropout(projected, expert_dropout=0.5, training=False)
    for a, b in zip(out, snapshot):
        assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# RandomProjectionAssembly
# ---------------------------------------------------------------------------

def test_random_projection_assembly_forward_shape():
    from src.models.feature_space import RandomProjectionAssembly

    dims = [64, 96]
    model = RandomProjectionAssembly(
        dims, proj_dim=16, expert_dropout=0.25, seed=7)
    model.train()
    out = model([torch.randn(5, d) for d in dims])
    assert out.shape == (5, 2)


def test_random_projection_trainable_param_count_is_smaller():
    """Random-proj freezes linear weights; param count must be strictly
    smaller than the learned-projection FeatureAssembly."""
    from src.models.feature_space import (
        FeatureAssembly, RandomProjectionAssembly,
    )

    dims = [1024, 768, 2048]
    proj_dim = 64
    learned = FeatureAssembly(
        dims, proj_dim=proj_dim, classifier_type="mlp",
        hidden_dims=[128, 64], dropout=0.0, expert_dropout=0.0,
    )
    random = RandomProjectionAssembly(
        dims, proj_dim=proj_dim, expert_dropout=0.0, seed=7)

    n_learned = sum(p.numel() for p in learned.parameters() if p.requires_grad)
    n_random = sum(p.numel() for p in random.parameters() if p.requires_grad)
    assert n_random < n_learned, (
        f"random_trainable={n_random} not below learned_trainable={n_learned}"
    )


# ---------------------------------------------------------------------------
# Manifest tagged rawfeat classification
# ---------------------------------------------------------------------------

def test_manifest_classifies_tagged_rawfeat(tmp_path):
    from src.reporting.manifest import scan_results

    default_rawfeat = tmp_path / "04_ds_clip4_rawfeat.json"
    default_rawfeat.write_text(json.dumps({
        "name": "fs_1_raw_features_logreg",
        "pool": "clip4",
        "train_tag": "FIT",
        "avg_auc_8": 0.97,
        "per_generator": {},
    }))
    tagged_rawfeat = tmp_path / "04_ds_arch6_rawfeat_CUSTOM_TAG.json"
    tagged_rawfeat.write_text(json.dumps({
        "name": "fs_1_raw_features_logreg",
        "pool": "arch6",
        "train_tag": "CUSTOM_TAG",
        "avg_auc_8": 0.96,
        "per_generator": {},
    }))
    tagged_ds = tmp_path / "04_ds_arch6_CUSTOM_TAG.json"
    tagged_ds.write_text(json.dumps({
        "pool": "arch6",
        "train_tag": "CUSTOM_TAG",
        "best_ds_avg8": 0.84,
        "best_ds": "PS_4_xgboost",
        "baselines": [],
    }))

    m_default = scan_results(tmp_path, train_tag="FIT")
    assert "clip4" in m_default["rawfeat"]
    assert "arch6" not in m_default["rawfeat"]

    m_custom = scan_results(tmp_path, train_tag="CUSTOM_TAG")
    assert "arch6" in m_custom["rawfeat"]
    assert "clip4" not in m_custom["rawfeat"]
    assert "arch6" in m_custom["ds"]


# ---------------------------------------------------------------------------
# fs_runtime._class_weights + decision_baselines._train_mlp agreement
# ---------------------------------------------------------------------------

def test_fs_runtime_class_weights_match_ds_weights():
    """The FS and DS ladders must share one balancing convention."""
    from src.training.fs_runtime import _class_weights

    y = np.concatenate([np.zeros(300), np.ones(1200)]).astype(np.int64)
    w = _class_weights(y).numpy()
    n_real, n_fake = 300, 1200
    expected = np.array([
        len(y) / (2.0 * n_real),
        len(y) / (2.0 * n_fake),
    ], dtype=np.float32)
    np.testing.assert_allclose(w, expected, rtol=1e-6)


def test_fs_runtime_class_weights_none_on_single_class():
    from src.training.fs_runtime import _class_weights
    assert _class_weights(np.zeros(10, dtype=np.int64)) is None
    assert _class_weights(np.ones(10, dtype=np.int64)) is None
