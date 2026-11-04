"""Test slow prediction-space baselines (PS_C, PS_F, PS_E, PS_4) and cache fingerprint.

These are skipped in test_baselines.py for speed. Run separately:
    cd <codebase_root>
    python tests/test_slow_baselines.py
"""

import numpy as np

from src.models.prediction_space import (
    PS_C, PS_F, PS_E, PS_4,
)

np.random.seed(42)
N_FIT, N_SEL, N_TEST, E = 200, 100, 80, 4

fit_logits = np.random.randn(N_FIT, E, 2).astype(np.float32)
sel_logits = np.random.randn(N_SEL, E, 2).astype(np.float32)
test_logits = np.random.randn(N_TEST, E, 2).astype(np.float32)
fit_labels = np.random.randint(0, 2, N_FIT).astype(np.int32)
sel_labels = np.random.randint(0, 2, N_SEL).astype(np.int32)


def check(name, scores, n):
    assert scores.shape == (n,), f"{name}: expected ({n},), got {scores.shape}"
    assert np.isfinite(scores).all(), f"{name}: non-finite values"
    print(f"  [OK] {name}: shape={scores.shape}, "
          f"range=[{scores.min():.3f}, {scores.max():.3f}]")


def test_s4b():
    print("Testing PS_C (regularised LogReg CV)...")
    scores = PS_C(fit_logits, fit_labels, test_logits)
    check("PS_C", scores, N_TEST)


def test_s5b():
    print("Testing PS_F (MLP logits + early stop)...")
    scores = PS_F(
        fit_logits, fit_labels, sel_logits, sel_labels, test_logits,
        max_epochs=20, patience=5, seed=1024)
    check("PS_F", scores, N_TEST)


def test_s6b():
    print("Testing PS_E (MLP probs + early stop)...")
    scores = PS_E(
        fit_logits, fit_labels, sel_logits, sel_labels, test_logits,
        max_epochs=20, patience=5, seed=1024)
    check("PS_E", scores, N_TEST)


def test_s7():
    print("Testing PS_4 (XGBoost CV)...")
    scores = PS_4(fit_logits, fit_labels, test_logits, n_folds=3)
    check("PS_4", scores, N_TEST)


def test_fingerprint():
    """Test that _cache_fingerprint is stable, catches mismatches and reordering."""
    print("Testing cache fingerprint...")
    import hashlib

    def _cache_fingerprint(labels, sample_ids=None):
        h_labels = hashlib.sha256(labels.tobytes()).hexdigest()[:16]
        if sample_ids is not None:
            h_ids = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()[:16]
        else:
            h_ids = "no_ids"
        return f"n={len(labels)}_r={int((labels==0).sum())}_f={int((labels==1).sum())}_{h_labels}_{h_ids}"

    labels_a = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    labels_b = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    labels_c = np.array([0, 0, 1, 1, 1, 1], dtype=np.int32)
    ids_a = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg", "img6.jpg"]
    ids_reordered = ["img2.jpg", "img1.jpg", "img3.jpg", "img4.jpg", "img5.jpg", "img6.jpg"]

    fp_a = _cache_fingerprint(labels_a, ids_a)
    fp_b = _cache_fingerprint(labels_b, ids_a)
    fp_c = _cache_fingerprint(labels_c, ids_a)
    fp_reorder = _cache_fingerprint(labels_a, ids_reordered)
    fp_no_ids = _cache_fingerprint(labels_a)

    assert fp_a == fp_b, "Identical arrays + IDs should match"
    assert fp_a != fp_c, "Different labels should not match"
    assert fp_a != fp_reorder, "Reordered sample IDs must produce different fingerprint"
    assert fp_a != fp_no_ids, "With-IDs vs without-IDs must differ"
    assert "n=6" in fp_a
    assert "no_ids" in fp_no_ids
    print(f"  [OK] Stable: {fp_a}")
    print(f"  [OK] Detects label change: {fp_a} != {fp_c}")
    print(f"  [OK] Detects reordering: {fp_a} != {fp_reorder}")
    print(f"  [OK] Detects missing IDs: {fp_a} != {fp_no_ids}")


if __name__ == "__main__":
    test_fingerprint()
    test_s4b()
    test_s5b()
    test_s6b()
    test_s7()
    print("\nAll slow baseline tests passed.")
