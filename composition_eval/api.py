"""composition_eval.api: high-level evaluate() entry point.

`evaluate(pool, protocol="headline")` compares FS_1 with PS_1, PS_2 and PS_4
on the standard split. For the paper workflow, see camera_ready/REPRODUCTION.md.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from src.utils.config import CACHE_ROOT, FIT_TAG, get_pool_experts, get_feature_dims
from src.utils.io import load_concat_features, load_stacked_logits
from src.utils.dataset_helpers import TEST_GENERATORS
from src.utils.metrics import compute_auc
from src.models.prediction_space import (
    PS_1, PS_2, PS_4,
)
from src.models.feature_space import FS_1

from composition_eval.bootstrap import generator_level_bootstrap


@dataclass
class EvaluationResult:
    """Result of `evaluate(pool, protocol)`.

    `protocol` is "headline" (PS_1, PS_2, PS_4 and FS_1). `method_aucs`
    are macro AUCs across held-out generators; `per_generator[m][g]`
    is the per-generator AUC; `headline_ci_95` is a paired
    generator-level bootstrap 95% CI on FS_1 - best PS.
    """

    pool: str
    protocol: str
    method_aucs: Dict[str, float]
    per_generator: Dict[str, Dict[str, float]]
    best_ps_name: str
    headline_delta: float
    headline_ci_95: tuple
    extras: Dict[str, Any] = field(default_factory=dict)

    def headline(self) -> str:
        lo, hi = self.headline_ci_95
        return (
            f"{self.pool}: FS_1 = {self.method_aucs['FS_1']:.4f}, "
            f"{self.best_ps_name} = {self.method_aucs[self.best_ps_name]:.4f}, "
            f"delta AUROC = {self.headline_delta:+.3f} "
            f"(95% CI [{lo:+.3f}, {hi:+.3f}])"
        )


def load_pool(name: str):
    """Resolve a named pool to (expert_names, feature_dims)."""
    expert_names = get_pool_experts(name)
    feature_dims = get_feature_dims(expert_names)
    return expert_names, feature_dims


def register_pool(name: str, experts: list) -> None:
    """Register a pool at runtime.

    Note: for reproducibility the preferred path is to add the pool to
    `configs/experts.yaml::pools` and re-import. This function exists for
    interactive / notebook use only.
    """
    from src.utils.config import POOLS
    POOLS[name] = list(experts)


def _build_gen_boundaries(gen_sizes):
    boundaries = {}
    offset = 0
    for gen, n in zip(TEST_GENERATORS, gen_sizes):
        boundaries[gen] = (offset, offset + n)
        offset += n
    return boundaries


def evaluate(
    pool: str,
    protocol: str = "headline",
    train_tag: Optional[str] = None,
    cache_root: Optional[str] = None,
    n_boot: int = 10_000,
    seed: int = 42,
) -> EvaluationResult:
    """Apply the evaluation protocol to a named pool.

    Parameters
    ----------
    pool : str
        Named pool in `configs/experts.yaml::pools`.
    protocol : str
        "headline" (default): PS_1, PS_2, PS_4, FS_1 only. CPU is sufficient.
        For the full ladder, use the harness documented in
        camera_ready/REPRODUCTION.md.
    train_tag : str, optional
        Which cached split to use as FIT. Defaults to FIT_TAG from
        configs/paths.yaml (= "FIT" in the shipped config).
    cache_root : str, optional
        Path to cached features/logits. Defaults to the codebase's `CACHE_ROOT`.
    n_boot : int
        Bootstrap resamples for headline CI (default 10,000).
    seed : int
        RNG seed.

    Returns
    -------
    EvaluationResult
    """
    if protocol not in {"headline", "full_ladder"}:
        raise ValueError(f"protocol must be 'headline' or 'full_ladder', got {protocol!r}")
    if protocol == "full_ladder":
        raise NotImplementedError(
            "`full_ladder` is available through the reproduction harness. Run "
            "`python camera_ready/scripts/review/reproduce_all_v4.py --dry-run` "
            "to see the available steps. See camera_ready/REPRODUCTION.md."
        )

    root = cache_root or CACHE_ROOT
    tag = train_tag if train_tag is not None else FIT_TAG
    expert_names, feature_dims = load_pool(pool)

    fit_feats, fit_labels = load_concat_features(expert_names, tag, root)
    fit_logits = load_stacked_logits(expert_names, tag, root)[0]

    all_test_labels = []
    all_test_feats = []
    all_test_logits = []
    gen_sizes = []
    for gen in TEST_GENERATORS:
        gf, gl = load_concat_features(expert_names, gen, root)
        g_logits = load_stacked_logits(expert_names, gen, root)[0]
        all_test_feats.append(gf)
        all_test_logits.append(g_logits)
        all_test_labels.append(gl)
        gen_sizes.append(gf.shape[0])
    test_feats = np.concatenate(all_test_feats, axis=0)
    test_logits = np.concatenate(all_test_logits, axis=0)
    test_labels = np.concatenate(all_test_labels, axis=0)
    gen_boundaries = _build_gen_boundaries(gen_sizes)

    fs_1_scores = FS_1(fit_feats, fit_labels, test_feats)
    ps_1_scores = PS_1(test_logits)
    ps_2_scores = PS_2(fit_logits, fit_labels, test_logits)
    ps_4_scores = PS_4(fit_logits, fit_labels, test_logits)

    method_scores = {
        "FS_1": fs_1_scores,
        "PS_1_prob_avg": ps_1_scores,
        "PS_2_logreg": ps_2_scores,
        "PS_4_xgboost": ps_4_scores,
    }

    method_aucs = {}
    per_generator = {}
    for name, scores in method_scores.items():
        gen_aucs = []
        per_gen = {}
        for gen in TEST_GENERATORS:
            lo, hi = gen_boundaries[gen]
            auc = float(compute_auc(test_labels[lo:hi], scores[lo:hi]))
            per_gen[gen] = auc
            gen_aucs.append(auc)
        method_aucs[name] = float(np.mean(gen_aucs))
        per_generator[name] = per_gen

    ps_methods = [n for n in method_aucs if n != "FS_1"]
    best_ps_name = max(ps_methods, key=lambda n: method_aucs[n])
    best_ps_scores = method_scores[best_ps_name]

    boot = generator_level_bootstrap(
        fs_1_scores, best_ps_scores, test_labels, gen_boundaries,
        n_boot=n_boot, seed=seed,
    )
    headline_delta = method_aucs["FS_1"] - method_aucs[best_ps_name]

    return EvaluationResult(
        pool=pool,
        protocol=protocol,
        method_aucs=method_aucs,
        per_generator=per_generator,
        best_ps_name=best_ps_name,
        headline_delta=float(headline_delta),
        headline_ci_95=boot["ci_95"],
        extras={
            "expert_names": expert_names,
            "feature_dims": feature_dims,
            "n_test_samples": int(len(test_labels)),
            "n_fit_samples": int(len(fit_labels)),
            "gen_boundaries": {k: list(v) for k, v in gen_boundaries.items()},
            "bootstrap_n": n_boot,
        },
    )
