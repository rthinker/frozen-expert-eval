"""Per-expert, per-generator AUROC diagnostic (see Appendix H for the paper's score convention).

Deterministic sklearn.roc_auc_score on each expert's fake-class logit
across the 8 OOD test generators. No training. Output schema:
output/results/16_individual_expert_auc.json with `per_expert[expert]
.{per_generator, avg_auc_8}`, plus `experts` and `generators` lists.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  # Consistent with all torch-using scripts even though this one is sklearn-only.

import json
import sys

import numpy as np

from src.utils.config import get_pool_experts
from src.utils.dataset_helpers import TEST_GENERATORS
from src.utils.metrics import compute_auc

CODEBASE = _bootstrap.CODEBASE
CACHE_ROOT = CODEBASE / "output" / "cached_features"
RESULT_PATH = CODEBASE / "output" / "results" / "16_individual_expert_auc.json"


def main() -> int:
    # Enumerate every expert from the full21 pool (all 21 experts in the grid).
    experts = get_pool_experts("full21")
    assert len(experts) == 21, f"expected 21 experts, got {len(experts)}"

    per_expert: dict[str, dict] = {}

    for expert in experts:
        per_gen: dict[str, float] = {}
        for gen in TEST_GENERATORS:
            logits_path = CACHE_ROOT / gen / f"{expert}_logits.npy"
            labels_path = CACHE_ROOT / gen / "labels.npy"
            if not logits_path.exists() or not labels_path.exists():
                print(f"[warn] missing {logits_path.name} or labels.npy in {gen}; skipping",
                      file=sys.stderr)
                continue
            logits = np.load(logits_path, mmap_mode="r")
            labels = np.load(labels_path, mmap_mode="r")
            # fake-class score = column 1 (real=0, fake=1 convention)
            scores = np.asarray(logits[:, 1], dtype=np.float64)
            labels_np = np.asarray(labels, dtype=np.int32)
            per_gen[gen] = compute_auc(labels_np, scores)

        avg = float(np.mean([per_gen[g] for g in TEST_GENERATORS if g in per_gen]))
        per_expert[expert] = {
            "per_generator": per_gen,
            "avg_auc_8": avg,
        }

    out = {
        "description": (
            "Per-expert individual AUROC on each of the 8 held-out OOD generators "
            "and averaged across them. Each score uses the expert's fake-class logit. "
            "Deterministic (sklearn roc_auc_score on cached features)."
        ),
        "experts": experts,
        "generators": list(TEST_GENERATORS),
        "per_expert": per_expert,
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(out, indent=2, sort_keys=False), encoding="utf-8")
    print(f"wrote {RESULT_PATH}")
    print()
    # summary print, sorted by avg descending
    rows = sorted(per_expert.items(), key=lambda kv: kv[1]["avg_auc_8"], reverse=True)
    print(f"{'expert':22s} {'avg_auc_8':>10s}")
    for name, d in rows:
        print(f"{name:22s} {d['avg_auc_8']:>10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
