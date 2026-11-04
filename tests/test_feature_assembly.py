"""Smoke test: FeatureAssembly forward pass on random data."""

import torch

from src.models.feature_space import FeatureAssembly, RandomProjectionAssembly


def test_feature_assembly():
    dims = [1024, 1024, 1024, 1024]  # 4 CLIP-large experts
    d = 64
    B = 8

    # MLP variant
    model = FeatureAssembly(dims, proj_dim=d, classifier_type="mlp")
    inputs = [torch.randn(B, dim) for dim in dims]
    out = model(inputs)
    assert out.shape == (B, 2), f"Expected (8, 2), got {out.shape}"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] FeatureAssembly MLP: out={out.shape}, params={n_params:,}")

    # Linear variant
    model_lin = FeatureAssembly(dims, proj_dim=d, classifier_type="linear")
    out_lin = model_lin(inputs)
    assert out_lin.shape == (B, 2)
    n_lin = sum(p.numel() for p in model_lin.parameters() if p.requires_grad)
    print(f"[OK] FeatureAssembly linear: out={out_lin.shape}, params={n_lin:,}")

    # Heterogeneous dims (cross-architecture pool)
    het_dims = [1024, 768, 2048, 2048]
    model_het = FeatureAssembly(het_dims, proj_dim=d)
    het_inputs = [torch.randn(B, dim) for dim in het_dims]
    out_het = model_het(het_inputs)
    assert out_het.shape == (B, 2)
    print(f"[OK] Heterogeneous dims: {het_dims} -> {out_het.shape}")

    # Random projection
    model_rp = RandomProjectionAssembly(dims, proj_dim=d)
    out_rp = model_rp(inputs)
    assert out_rp.shape == (B, 2)
    n_rp_trainable = sum(p.numel() for p in model_rp.parameters() if p.requires_grad)
    n_rp_total = sum(p.numel() for p in model_rp.parameters())
    print(f"[OK] RandomProjection: trainable={n_rp_trainable:,}, total={n_rp_total:,}")


if __name__ == "__main__":
    test_feature_assembly()
    print("\nAll feature assembly tests passed.")
