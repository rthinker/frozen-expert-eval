# C19: Construct the inner cross-fitting assignments.
"""Freeze inner cross-fitting assignments for genuine K-class expert heads.

Protocol overview: camera_ready/scripts/analyses/README.md#three-class-protocol.

This script fits no model. It verifies the already frozen K=3 outer
manifests, checks for exact feature matches between outer test and
training partitions, and
freezes deterministic stratified content-group assignments for the
cross-fitted head outputs used to train the PS combiner.

The harness runs these checks as C19 before appI1_native_k_aligned_v2.py.
From the repository root:

    python camera_ready/scripts/review/reproduce_all_v4.py --only C19
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

REPROD = Path(__file__).resolve().parent
RUNS = REPROD  # the analyses folder holds the shared modules
SEED = 1024
MAX_INNER_FOLDS = 5
VIEWS = ("B_arch6", "A_df40")

import _paths  # noqa: E402,F401
RESULTS_K = _paths.RESULTS_DIR / "three_class"
RESULTS_K.mkdir(parents=True, exist_ok=True)

import _verify_cache_helpers as VC  # noqa: E402  shared cache-reading helpers

from src.utils.config import CACHE_ROOT, FIT_TAG, SELECT_TAG  # noqa: E402


def manifest_digest(rows: list[dict]) -> str:
    return hashlib.sha256(
        "\n".join(
            f"{r['split']}|{r['idx']}|{r['fold']}|{r['cls']}" for r in rows
        ).encode()
    ).hexdigest()


def assignment_digest(outer: dict) -> str:
    lines = []
    for fold in sorted(outer, key=int):
        rec = outer[fold]
        for item in rec["assignments"]:
            lines.append(
                f"{fold}|{item['idx']}|{item['inner_fold']}|"
                f"{item['content_group']}"
            )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def choose_inner_split(
    y: np.ndarray, groups: np.ndarray, classes: list[str], outer_fold: int
) -> tuple[int, np.ndarray]:
    """Largest valid split count in [5, 2], with every row predicted once."""
    class_ids = set(range(len(classes)))
    x_dummy = np.zeros((len(y), 1), dtype=np.uint8)
    for n_splits in range(MAX_INNER_FOLDS, 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=SEED + outer_fold,
        )
        assignment = np.full(len(y), -1, dtype=np.int16)
        valid = True
        try:
            for inner, (train_idx, val_idx) in enumerate(
                splitter.split(x_dummy, y, groups)
            ):
                if set(np.unique(y[train_idx])) != class_ids:
                    valid = False
                    break
                if set(np.unique(y[val_idx])) != class_ids:
                    valid = False
                    break
                if set(groups[train_idx]) & set(groups[val_idx]):
                    valid = False
                    break
                if np.any(assignment[val_idx] != -1):
                    valid = False
                    break
                assignment[val_idx] = inner
        except ValueError:
            valid = False
        if valid and np.all(assignment >= 0):
            return n_splits, assignment
    raise RuntimeError(
        "No valid content-grouped inner cross-fit with at least two folds"
    )


def main() -> None:
    print("computing content fingerprints ...", flush=True)
    expert = VC._hash_expert()
    hashes = {
        FIT_TAG: np.asarray(
            VC._content_hashes(CACHE_ROOT / FIT_TAG, expert), dtype=object
        ),
        SELECT_TAG: np.asarray(
            VC._content_hashes(CACHE_ROOT / SELECT_TAG, expert), dtype=object
        ),
    }
    print(
        f"content fingerprint expert: {expert}; "
        f"FIT {len(hashes[FIT_TAG]):,}, SELECT {len(hashes[SELECT_TAG]):,}"
    )

    summary = {}
    for view in VIEWS:
        source = RESULTS_K / f"K3_FROZEN_{view}.json"
        manifest = json.loads(source.read_text(encoding="utf-8"))
        rows = manifest["manifest"]
        digest = manifest_digest(rows)
        if digest != manifest["manifest_sha256"]:
            raise SystemExit(
                f"{view}: manifest digest mismatch\n"
                f"stored {manifest['manifest_sha256']}\ncomputed {digest}"
            )
        classes = manifest["classes"]
        cidx = {name: i for i, name in enumerate(classes)}
        print(
            f"\n=== {view}: digest {digest[:16]}..., "
            f"{len(rows):,} rows, classes {classes} ==="
        )

        outer = {}
        all_valid = True
        for fold in range(manifest["n_folds"]):
            fit_rows = sorted(
                (
                    r
                    for r in rows
                    if r["fold"] != fold and r["split"] == FIT_TAG
                ),
                key=lambda r: r["idx"],
            )
            select_rows = [
                r
                for r in rows
                if r["fold"] != fold and r["split"] == SELECT_TAG
            ]
            test_rows = [r for r in rows if r["fold"] == fold]

            fit_groups = np.asarray(
                [str(hashes[FIT_TAG][r["idx"]]) for r in fit_rows],
                dtype=object,
            )
            y = np.asarray([cidx[r["cls"]] for r in fit_rows], dtype=np.int16)
            labels_by_content = {}
            for content_group, label in zip(fit_groups, y):
                labels_by_content.setdefault(str(content_group), set()).add(
                    int(label)
                )
            conflicting_content_labels = {
                group: sorted(labels)
                for group, labels in labels_by_content.items()
                if len(labels) != 1
            }
            if conflicting_content_labels:
                raise RuntimeError(
                    f"{view} fold {fold}: "
                    f"{len(conflicting_content_labels)} content groups have "
                    "conflicting K-class labels"
                )

            train_side_content = {
                str(hashes[r["split"]][r["idx"]])
                for r in fit_rows + select_rows
            }
            test_content = {
                str(hashes[r["split"]][r["idx"]]) for r in test_rows
            }
            outer_overlap = len(train_side_content & test_content)
            if outer_overlap:
                all_valid = False

            n_inner, assignment = choose_inner_split(
                y, fit_groups, classes, fold
            )
            inner_counts = Counter(int(value) for value in assignment)
            class_by_inner = {
                str(inner): {
                    cls: int(
                        np.sum((assignment == inner) & (y == class_index))
                    )
                    for class_index, cls in enumerate(classes)
                }
                for inner in range(n_inner)
            }
            rec = {
                "n_fit": len(fit_rows),
                "n_select": len(select_rows),
                "n_test": len(test_rows),
                "outer_content_overlap": outer_overlap,
                "n_inner_folds": n_inner,
                "inner_sizes": {
                    str(key): value for key, value in sorted(inner_counts.items())
                },
                "inner_class_counts": class_by_inner,
                "assignments": [
                    {
                        "idx": int(row["idx"]),
                        "inner_fold": int(inner),
                        "content_group": str(content_group),
                    }
                    for row, inner, content_group in zip(
                        fit_rows, assignment, fit_groups
                    )
                ],
            }
            outer[str(fold)] = rec
            print(
                f"fold {fold}: FIT {len(fit_rows):,}, "
                f"SELECT {len(select_rows):,}, TEST {len(test_rows):,}; "
                f"inner={n_inner}, sizes={dict(sorted(inner_counts.items()))}; "
                f"outer content overlap={outer_overlap}"
            )

        a_digest = assignment_digest(outer)
        output = {
            "design": "camera_ready/scripts/analyses/README.md#three-class-protocol",
            "view": view,
            "source_manifest": source.name,
            "manifest_sha256": digest,
            "content_expert": expert,
            "seed": SEED,
            "assignment_sha256": a_digest,
            "classes": classes,
            "pool": manifest["pool"],
            "n_outer_folds": manifest["n_folds"],
            "valid": bool(all_valid),
            "outer_folds": outer,
        }
        destination = (
            RESULTS_K / f"NATIVE_K_PREFLIGHT_{view}.json"
        )
        destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(
            f"wrote {destination.name}; assignment digest "
            f"{a_digest[:16]}...; {'PASS' if all_valid else 'FAIL'}"
        )
        summary[view] = {
            "valid": bool(all_valid),
            "manifest": digest,
            "assignments": a_digest,
        }

    print("\n=== PRE-FLIGHT SUMMARY ===")
    for view, record in summary.items():
        print(
            f"{view:<10} {'READY' if record['valid'] else 'INVALID'}  "
            f"manifest {record['manifest'][:12]}  "
            f"inner {record['assignments'][:12]}"
        )
    if not all(record["valid"] for record in summary.values()):
        raise SystemExit("pre-flight failed")


if __name__ == "__main__":
    main()
