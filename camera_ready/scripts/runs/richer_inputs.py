"""Feature maps of the richer prediction-space inputs (PS_G to PS_J), copied from the reviewed run.

Each map takes logits [N, E, 2] and
returns features [N, F]; any state is fitted on FIT (or SELECT, for the temperatures) and passed in explicitly.

Paper names: PS_G pairwise disagreement, PS_H uncertainty and mutual information, PS_I quadratic interactions,
PS_J rank (ECDF). PS_K (the CV-selected choice among them) needs no map of its own.
"""
from __future__ import annotations

import numpy as np


def _margins(logits3d: np.ndarray) -> np.ndarray:
    """Per-expert decision margin: logit_fake - logit_real. [N, E]."""
    return (logits3d[:, :, 1] - logits3d[:, :, 0]).astype(np.float64)


def _pair_idx(n: int):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def feat_flat(logits3d: np.ndarray) -> np.ndarray:
    return logits3d.reshape(len(logits3d), -1).astype(np.float64)


def feat_disagreement(logits3d: np.ndarray) -> np.ndarray:
    m = _margins(logits3d)
    pairs = _pair_idx(m.shape[1])
    if not pairs:
        return feat_flat(logits3d)
    diffs = np.stack([np.abs(m[:, i] - m[:, j]) for i, j in pairs], axis=1)
    return np.concatenate([feat_flat(logits3d), diffs], axis=1)


def feat_quadratic(logits3d: np.ndarray) -> np.ndarray:
    m = _margins(logits3d)
    pairs = _pair_idx(m.shape[1])
    blocks = [feat_flat(logits3d), m ** 2]
    if pairs:
        blocks.append(np.stack([m[:, i] * m[:, j] for i, j in pairs], axis=1))
    return np.concatenate(blocks, axis=1)


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def make_uncertainty_featmap(select_logits3d: np.ndarray,
                             select_labels: np.ndarray):
    """Fit per-expert temperatures on SELECT, return a feature map.

    Temperatures come from the submitted PS_A routine so the calibration
    is identical to the published calibrated-averaging rung.
    """
    from src.models.prediction_space import PS_A, _softmax  # reviewed code base (read-only)
    _, diag = PS_A(select_logits3d, select_labels,
                   select_logits3d[:1], return_diagnostics=True)
    temps = np.asarray(diag["temps"], dtype=np.float64)

    def featmap(logits3d: np.ndarray) -> np.ndarray:
        cal = logits3d.astype(np.float64) / temps[None, :, None]
        p = _softmax(cal)[:, :, 1]                      # [N, E] p(fake)
        pbar = p.mean(axis=1, keepdims=True)            # [N, 1]
        h_of_mean = _entropy(pbar[:, 0])                # H(E[p])
        mean_of_h = _entropy(p).mean(axis=1)            # E[H(p)]
        mutual_info = h_of_mean - mean_of_h             # BALD term
        stats = np.stack([
            h_of_mean, mean_of_h, mutual_info,
            p.var(axis=1), p.min(axis=1), p.max(axis=1),
            np.median(p, axis=1), p.max(axis=1) - p.min(axis=1),
        ], axis=1)
        return np.concatenate([p, pbar, stats], axis=1)

    return featmap, temps


def make_rank_featmap(fit_logits3d: np.ndarray):
    """Empirical-CDF map per expert, fitted on FIT only (not transductive)."""
    ref = np.sort(_margins(fit_logits3d), axis=0)       # [N_fit, E]
    n = ref.shape[0]

    def featmap(logits3d: np.ndarray) -> np.ndarray:
        m = _margins(logits3d)
        out = np.empty_like(m)
        for e in range(m.shape[1]):
            out[:, e] = np.searchsorted(ref[:, e], m[:, e], side="right") / n
        return out

    return featmap
