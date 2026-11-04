"""Smoke test: decision baselines on synthetic data."""

import numpy as np

from src.models.prediction_space import (
    PS_1, PS_A, PS_B,
    PS_2, PS_3, PS_D,
)
from src.models.feature_space import FS_1

np.random.seed(42)
N_FIT, N_TEST, E = 200, 100, 4

# Synthetic logits: [N, E, 2]
fit_logits = np.random.randn(N_FIT, E, 2).astype(np.float32)
test_logits = np.random.randn(N_TEST, E, 2).astype(np.float32)
fit_labels = np.random.randint(0, 2, N_FIT).astype(np.int32)
test_labels = np.random.randint(0, 2, N_TEST).astype(np.int32)


def check(name, scores, n_expected):
    assert scores.shape == (n_expected,), f"{name}: expected ({n_expected},), got {scores.shape}"
    assert np.isfinite(scores).all(), f"{name}: non-finite values"
    print(f"  [OK] {name}: shape={scores.shape}, range=[{scores.min():.3f}, {scores.max():.3f}]")


def test_baselines():
    print("Testing prediction-space baselines on synthetic data...")

    check("PS_1", PS_1(test_logits), N_TEST)
    check("PS_A", PS_A(fit_logits, fit_labels, test_logits), N_TEST)
    check("PS_B", PS_B(fit_logits, fit_labels, test_logits), N_TEST)
    check("PS_2", PS_2(fit_logits, fit_labels, test_logits), N_TEST)
    check("PS_3", PS_3(fit_logits, fit_labels, test_logits, epochs=3), N_TEST)
    check("PS_D", PS_D(fit_logits, fit_labels, test_logits, epochs=3), N_TEST)

    # Raw features LogReg
    fit_feat = np.random.randn(N_FIT, 4096).astype(np.float32)
    test_feat = np.random.randn(N_TEST, 4096).astype(np.float32)
    check("FS_1", FS_1(fit_feat, fit_labels, test_feat), N_TEST)

    print("\nSkipping PS_C, PS_F, PS_E, PS_4 (slow CV) in smoke test.")


if __name__ == "__main__":
    test_baselines()
    print("\nAll baseline tests passed.")
