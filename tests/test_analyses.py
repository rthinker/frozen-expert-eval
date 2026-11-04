"""test_analyses.py: Tests for analytical functions used in the paper.

Covers: PCA/participation ratio, bootstrap CIs, AUC metric, raw-feat
LogReg, and seed determinism. All tests use synthetic data -- no GPU
or cached features required.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score


# ── PCA / Participation Ratio ────────────────────────────────────────

@pytest.fixture
def pca_bottleneck(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("pca_bottleneck", scripts / "08_pca_bottleneck.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pr_identity_matrix(pca_bottleneck):
    """Identity covariance should give PR = dimensionality."""
    rng = np.random.default_rng(42)
    D = 50
    X = rng.standard_normal((500, D))
    pr, _ = pca_bottleneck.participation_ratio(X)
    # For iid Gaussian, PR should be close to D
    assert abs(pr - D) < D * 0.3, f"PR={pr}, expected ~{D}"


def test_pr_one_dominant_direction(pca_bottleneck):
    """One dominant direction should give PR close to 1."""
    rng = np.random.default_rng(42)
    X = np.zeros((500, 20))
    X[:, 0] = rng.standard_normal(500) * 100
    X[:, 1:] = rng.standard_normal((500, 19)) * 0.01
    pr, _ = pca_bottleneck.participation_ratio(X)
    assert pr < 3.0, f"PR={pr}, expected close to 1"


def test_pr_logits_lower_than_features(pca_bottleneck):
    """Features should have higher PR than logits (bottleneck thesis)."""
    rng = np.random.default_rng(42)
    features = rng.standard_normal((500, 200))
    logits = rng.standard_normal((500, 8))
    pr_feat, _ = pca_bottleneck.participation_ratio(features)
    pr_logit, _ = pca_bottleneck.participation_ratio(logits)
    assert pr_feat > pr_logit, (
        f"Features PR={pr_feat} should exceed logits PR={pr_logit}")


# ── Bootstrap CI (uses shared implementation from src/utils/metrics.py) ──

from src.utils.metrics import bootstrap_paired_auc


def test_bootstrap_identical_scores():
    """Identical scores should give CI centred at zero."""
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, size=200)
    scores = rng.random(200)
    boot = bootstrap_paired_auc(scores, scores, labels, n_boot=1000)
    assert abs(boot["delta_mean"]) < 0.01, "Identical scores should give ~0 delta"


def test_bootstrap_perfect_vs_random():
    """Perfect predictor should have CI entirely above zero vs random."""
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, size=500)
    perfect = labels.astype(float)
    random_scores = rng.random(500)
    boot = bootstrap_paired_auc(perfect, random_scores, labels, n_boot=1000)
    assert boot["ci_95_lo"] > 0, f"Perfect vs random CI lower bound should be >0, got {boot['ci_95_lo']}"


def test_bootstrap_deterministic():
    """Same seed should give same bootstrap results."""
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, size=200)
    scores_a = rng.random(200)
    scores_b = rng.random(200)
    d1 = bootstrap_paired_auc(scores_a, scores_b, labels, n_boot=500, seed=123)
    d2 = bootstrap_paired_auc(scores_a, scores_b, labels, n_boot=500, seed=123)
    assert d1 == d2, "Same seed should give identical results"


# ── AUC Metric ───────────────────────────────────────────────────────

def test_compute_auc_import():
    """compute_auc should be importable from utils."""
    from src.utils.metrics import compute_auc
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    auc = compute_auc(labels, scores)
    assert abs(auc - 1.0) < 0.01, f"Perfect ranking should give AUC~1.0, got {auc}"


def test_compute_auc_random():
    """Random scores should give AUC close to 0.5."""
    from src.utils.metrics import compute_auc
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, size=1000)
    scores = rng.random(1000)
    auc = compute_auc(labels, scores)
    assert 0.4 < auc < 0.6, f"Random scores should give AUC~0.5, got {auc}"


# ── Raw-Feature LogReg ───────────────────────────────────────────────

def test_rawfeat_logreg_beats_logit_logreg():
    """The shipped FS1 and PS2 recover the ordering of the planted signals."""
    from src.models.feature_space import FS_1
    from src.models.prediction_space import PS_2
    rng = np.random.default_rng(42)
    N = 500
    labels = rng.integers(0, 2, size=N)
    # Features: label-correlated in 50 dims
    features = rng.standard_normal((N, 50))
    features[:, :10] += labels[:, None] * 2.0
    # Logits: only 2 dims, weaker signal
    logits = rng.standard_normal((N, 1, 2))
    logits[:, 0, 1] += labels * 0.5

    auc_feat = roc_auc_score(labels[400:], FS_1(features[:400], labels[:400], features[400:]))
    auc_logit = roc_auc_score(labels[400:], PS_2(logits[:400], labels[:400], logits[400:]))

    assert auc_feat > auc_logit, (
        f"Features AUC={auc_feat:.3f} should beat logits AUC={auc_logit:.3f}")


# ── Seed Determinism ─────────────────────────────────────────────────

def test_seed_determinism():
    """set_seed should make numpy + torch deterministic."""
    from src.utils.seed import set_seed
    import torch

    set_seed(42)
    a1 = np.random.rand(10)
    t1 = torch.randn(10).numpy()

    set_seed(42)
    a2 = np.random.rand(10)
    t2 = torch.randn(10).numpy()

    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(t1, t2)


# ── Feature Assembly Forward Pass ────────────────────────────────────

def test_feature_assembly_forward_shape():
    """FeatureAssembly should produce [N, 2] logits from heterogeneous inputs."""
    import torch
    from src.models.feature_space import FeatureAssembly

    dims = [768, 1024, 512]  # 3 experts, different dims
    model = FeatureAssembly(dims, proj_dim=32, classifier_type="mlp",
                            hidden_dims=[64, 32], dropout=0.0,
                            expert_dropout=0.0)
    model.eval()
    feats = [torch.randn(8, d) for d in dims]
    with torch.no_grad():
        out = model(feats)
    assert out.shape == (8, 2), f"Expected (8, 2), got {out.shape}"


def test_feature_assembly_expert_dropout():
    """With expert_dropout > 0, training should still produce valid output."""
    import torch
    from src.models.feature_space import FeatureAssembly

    dims = [256, 256, 256, 256]
    model = FeatureAssembly(dims, proj_dim=16, classifier_type="mlp",
                            hidden_dims=[32], dropout=0.1,
                            expert_dropout=0.5)
    model.train()
    feats = [torch.randn(4, d) for d in dims]
    out = model(feats)
    assert out.shape == (4, 2)
    assert not torch.isnan(out).any(), "NaN in output with expert dropout"


