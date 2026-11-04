"""composition_eval.prediction_space: 10-method prediction-space baseline ladder.

Thin re-export of the PS methods from the research codebase. Public API is
stable; see `src.models.prediction_space` for implementation.
"""

from src.models.prediction_space import (
    PS_1,
    PS_A,
    PS_B,
    PS_2,
    PS_C,
    PS_3,
    PS_F,
    PS_D,
    PS_E,
    PS_4,
)

LADDER = [
    ("PS_1_prob_avg", PS_1, "deterministic"),
    ("PS_A_calibrated_avg", PS_A, "deterministic"),
    ("PS_B_learned_weights", PS_B, "deterministic"),
    ("PS_2_logreg", PS_2, "deterministic"),
    ("PS_C_logreg_cv", PS_C, "deterministic"),
    ("PS_3_mlp_logits", PS_3, "stochastic"),
    ("PS_F_mlp_logits_es", PS_F, "stochastic"),
    ("PS_D_mlp_probs", PS_D, "stochastic"),
    ("PS_E_mlp_probs_es", PS_E, "stochastic"),
    ("PS_4_xgboost", PS_4, "deterministic"),
]

__all__ = [
    "PS_1",
    "PS_A",
    "PS_B",
    "PS_2",
    "PS_C",
    "PS_3",
    "PS_F",
    "PS_D",
    "PS_E",
    "PS_4",
    "LADDER",
]
