# Shared inference utilities for the DF40 analyses.
"""Shared inference utilities for the DF40 analyses.

Everything here reproduces the SUBMITTED protocol exactly, so numbers
produced by Run A and Run B are directly comparable to the paper:

  * macro AUROC  = unweighted mean of the 8 per-generator AUROCs
  * primary CI   = paired GENERATOR-level bootstrap, 10,000 resamples,
                   percentile method, resampling the 8 generators with
                   replacement (matches scripts/06_bootstrap_rawfeat.py)
  * sign-flip test = exact two-sided test over the 8 paired generator
                     differences (2^8 = 256 sign patterns).
  * BCa          = bias-corrected accelerated interval on the mean
                   per-generator delta, jackknifed over generators
  * stochastic methods are seed-averaged at the SAMPLE level before a
    single AUROC is computed (matches scripts/04 and 11)

Nothing in this module writes to the frozen codebase.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

import _paths  # noqa: F401  # MUST be first: sets sys.path + determinism prelude

from src.models.prediction_space import (  # noqa: E402
    PS_1, PS_A, PS_B, PS_2, PS_C, PS_3, PS_D, PS_E, PS_F, PS_4,
)
from src.utils.cache_ids import load_sample_ids as _load_cached_sample_ids  # noqa: E402
from src.utils.dataset_helpers import TEST_GENERATORS  # noqa: E402
from src.utils.io import load_pool_data  # noqa: E402
from src.utils.metrics import compute_auc  # noqa: E402

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42          # matches scripts/06_bootstrap_rawfeat.py default
FOLD_BOOTSTRAP_SEED = 1024   # matches scripts/11_disjoint_data.py
DEFAULT_SEEDS = [7, 10, 24, 42, 1024, 1973, 3392, 9721]


# --------------------------------------------------------------------------
# Cache loading
# --------------------------------------------------------------------------

def load_split(expert_names: Sequence[str], tag: str, cache_root: Path):
    """Load one cache split. Returns (feats_list, logits3d, labels)."""
    feats, logits, labels = load_pool_data(list(expert_names), tag, cache_root)
    return feats, np.stack(logits, axis=1), labels


def load_sample_ids(tag: str, cache_root: Path) -> np.ndarray:
    """Canonical relative DF40 sample ids for a split, as a str array."""
    return _load_cached_sample_ids(Path(cache_root) / tag / "sample_ids.npy")


def load_test_generators(expert_names: Sequence[str], cache_root: Path,
                         generators: Sequence[str] = tuple(TEST_GENERATORS)):
    """Load all held-out generators, concatenated in the canonical order.

    Returns (X_feat, logits3d, labels, boundaries, generators).
    `boundaries[g] = (start, end)` indexes the concatenated arrays.
    """
    feats_cat, logits_cat, labels_cat = [], [], []
    boundaries: Dict[str, Tuple[int, int]] = {}
    offset = 0
    for g in generators:
        feats, logits3d, labels = load_split(expert_names, g, cache_root)
        n = len(labels)
        boundaries[g] = (offset, offset + n)
        feats_cat.append(np.concatenate(feats, axis=1))
        logits_cat.append(logits3d)
        labels_cat.append(labels)
        offset += n
    return (
        np.concatenate(feats_cat, axis=0),
        np.concatenate(logits_cat, axis=0),
        np.concatenate(labels_cat, axis=0),
        boundaries,
        list(generators),
    )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def per_generator_auc(labels: np.ndarray, scores: np.ndarray,
                      boundaries: Dict[str, Tuple[int, int]],
                      generators: Sequence[str]) -> Dict[str, float]:
    out = {}
    for g in generators:
        s, e = boundaries[g]
        out[g] = float(compute_auc(labels[s:e], scores[s:e]))
    return out


def macro_auc(per_gen: Dict[str, float], generators: Sequence[str]) -> float:
    return float(np.mean([per_gen[g] for g in generators]))


def generator_bootstrap_ci(deltas_per_gen: np.ndarray,
                           n_boot: int = BOOTSTRAP_RESAMPLES,
                           seed: int = BOOTSTRAP_SEED) -> Dict[str, float]:
    """Paired generator-level percentile bootstrap on the mean delta.

    Mirrors scripts/06_bootstrap_rawfeat.py: resample the G generators
    with replacement, take the mean of their per-generator deltas.
    """
    rng = np.random.default_rng(seed)
    G = len(deltas_per_gen)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, G, size=G)
        means[i] = deltas_per_gen[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "ci_95": [float(lo), float(hi)],
        "delta_mean": float(deltas_per_gen.mean()),
        "boot_mean": float(means.mean()),
        "p_gt_zero": float((means > 0).mean()),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "method": "paired generator-level percentile bootstrap",
    }


def exact_sign_flip_p(deltas_per_gen: np.ndarray,
                      max_exact: int = 20,
                      n_mc: int = 100_000,
                      seed: int = 1024) -> Dict[str, float]:
    """Two-sided exact paired sign-flip permutation test on the mean delta.

    With G generators there are 2^G sign assignments; enumerate them all
    when G <= max_exact, else Monte-Carlo. Reproduces the paper's
    p = 0.008 on arch6 (2 of 256) and p = 0.047 on LOGO (12 of 256).
    """
    d = np.asarray(deltas_per_gen, dtype=np.float64)
    G = len(d)
    obs = abs(d.mean())
    if G <= max_exact:
        n_perm = 1 << G
        signs = np.array(
            [[1.0 if (m >> j) & 1 else -1.0 for j in range(G)]
             for m in range(n_perm)]
        )
        stats = np.abs(signs @ d / G)
        p = float((stats >= obs - 1e-15).sum() / n_perm)
        exact = True
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_mc, G))
        stats = np.abs(signs @ d / G)
        p = float(((stats >= obs - 1e-15).sum() + 1) / (n_mc + 1))
        n_perm = n_mc
        exact = False
    return {
        "p_value": p,
        "exact": exact,
        "n_permutations": int(n_perm),
        "alternative": "two-sided",
        "statistic": float(d.mean()),
    }


def bca_ci(deltas_per_gen: np.ndarray,
           n_boot: int = BOOTSTRAP_RESAMPLES,
           seed: int = BOOTSTRAP_SEED,
           alpha: float = 0.05) -> Dict[str, float]:
    """Bias-corrected accelerated CI on the mean per-generator delta."""
    d = np.asarray(deltas_per_gen, dtype=np.float64)
    G = len(d)
    theta_hat = d.mean()

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = d[rng.integers(0, G, size=G)].mean()

    prop = float((boots < theta_hat).mean())
    prop = min(max(prop, 1.0 / n_boot), 1.0 - 1.0 / n_boot)
    z0 = norm.ppf(prop)

    jack = np.array([np.delete(d, i).mean() for i in range(G)])
    jbar = jack.mean()
    num = float(((jbar - jack) ** 3).sum())
    den = float(6.0 * (((jbar - jack) ** 2).sum() ** 1.5))
    a = num / den if den != 0 else 0.0

    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        zq = norm.ppf(q)
        adj = z0 + (z0 + zq) / (1 - a * (z0 + zq))
        out.append(float(np.percentile(boots, 100 * norm.cdf(adj))))
    return {"bca_95": out, "z0": float(z0), "acceleration": float(a)}


def compare_methods(labels: np.ndarray,
                    scores_a: np.ndarray,
                    scores_b: np.ndarray,
                    boundaries: Dict[str, Tuple[int, int]],
                    generators: Sequence[str],
                    name_a: str,
                    name_b: str) -> Dict:
    """Full inferential summary for `A - B` under the paper's protocol."""
    pg_a = per_generator_auc(labels, scores_a, boundaries, generators)
    pg_b = per_generator_auc(labels, scores_b, boundaries, generators)
    d = np.array([pg_a[g] - pg_b[g] for g in generators])
    return {
        "method_a": name_a,
        "method_b": name_b,
        "macro_a": macro_auc(pg_a, generators),
        "macro_b": macro_auc(pg_b, generators),
        "delta": float(macro_auc(pg_a, generators) - macro_auc(pg_b, generators)),
        "per_generator_a": pg_a,
        "per_generator_b": pg_b,
        "per_generator_delta": {g: float(x) for g, x in zip(generators, d)},
        "wins": int((d > 0).sum()),
        "n_generators": int(len(d)),
        "generator_bootstrap": generator_bootstrap_ci(d),
        "sign_flip_permutation": exact_sign_flip_p(d),
        "bca": bca_ci(d),
    }


# --------------------------------------------------------------------------
# The submitted 10-method PS ladder, returning per-sample scores
# --------------------------------------------------------------------------

def run_ps_ladder(fit_logits3d: np.ndarray,
                  fit_labels: np.ndarray,
                  sel_logits3d: np.ndarray,
                  sel_labels: np.ndarray,
                  test_logits3d: np.ndarray,
                  seeds: Sequence[int] = tuple(DEFAULT_SEEDS),
                  skip_slow: bool = False,
                  log: Optional[Callable[[str], None]] = None
                  ) -> Dict[str, np.ndarray]:
    """Run the submitted ladder; return {base_method: per-sample scores}.

    Stochastic rungs are averaged at the SAMPLE level across seeds before
    a single AUROC is taken downstream, exactly as scripts/04 and 11 do.
    """
    say = log or (lambda m: None)
    raw: Dict[str, np.ndarray] = {}

    def ev(name, fn):
        raw[name] = np.asarray(fn(), dtype=np.float32)
        say(f"    {name}")

    ev("PS_1_prob_avg", lambda: PS_1(test_logits3d))
    ev("PS_A_calibrated_avg",
       lambda: PS_A(sel_logits3d, sel_labels, test_logits3d))
    ev("PS_B_learned_weights",
       lambda: PS_B(sel_logits3d, sel_labels, test_logits3d))
    ev("PS_2_logreg",
       lambda: PS_2(fit_logits3d, fit_labels, test_logits3d))
    for s in seeds:
        ev(f"PS_3_mlp_logits_s{s}",
           lambda sd=s: PS_3(fit_logits3d, fit_labels, test_logits3d, seed=sd))
    for s in seeds:
        ev(f"PS_D_mlp_probs_s{s}",
           lambda sd=s: PS_D(fit_logits3d, fit_labels, test_logits3d, seed=sd))

    if not skip_slow:
        ev("PS_C_logreg_cv",
           lambda: PS_C(fit_logits3d, fit_labels, test_logits3d))
        for s in seeds:
            ev(f"PS_F_mlp_logits_es_s{s}",
               lambda sd=s: PS_F(fit_logits3d, fit_labels,
                                 sel_logits3d, sel_labels,
                                 test_logits3d, seed=sd))
        for s in seeds:
            ev(f"PS_E_mlp_probs_es_s{s}",
               lambda sd=s: PS_E(fit_logits3d, fit_labels,
                                 sel_logits3d, sel_labels,
                                 test_logits3d, seed=sd))
        ev("PS_4_xgboost",
           lambda: PS_4(fit_logits3d, fit_labels, test_logits3d))
    else:
        say("    (skipping PS_C, PS_F, PS_E, PS_4)")

    grouped = defaultdict(list)
    for name, sc in raw.items():
        grouped[re.sub(r"_s\d+$", "", name)].append(sc)
    return {base: np.mean(v, axis=0).astype(np.float32)
            for base, v in grouped.items()}


def best_of(scores: Dict[str, np.ndarray],
            labels: np.ndarray,
            boundaries: Dict[str, Tuple[int, int]],
            generators: Sequence[str]) -> Tuple[str, float, Dict[str, float]]:
    """Return (best method name, its macro AUROC, {method: macro AUROC}).

    'Best' is the retrospective maximum over the ladder on TEST, which is
    the paper's deliberately conservative comparator choice (Sec 3.2).
    """
    macros = {}
    for name, sc in scores.items():
        macros[name] = macro_auc(
            per_generator_auc(labels, sc, boundaries, generators), generators)
    best = max(macros, key=macros.get)
    return best, macros[best], macros


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_json(obj: Dict, path: Path) -> Path:
    path = _paths.assert_no_codebase_writes(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = dict(obj)
    obj.setdefault("_written_utc", datetime.now(timezone.utc).isoformat())
    obj.setdefault("_protocol_note",
                   "macro AUROC over held-out generators; paired "
                   "generator-level bootstrap (10k, percentile); exact "
                   "two-sided sign-flip permutation; BCa on mean "
                   "per-generator delta")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def make_logger(name: str):
    """Minimal stdout logger; also tees to camera_ready/output/logs/analyses/<name>.log."""
    _paths.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _paths.LOGS_DIR / f"{name}.log"

    def log(msg: str = ""):
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    return log
