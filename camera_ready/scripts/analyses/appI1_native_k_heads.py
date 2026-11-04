# Fit and cross-fit three-class heads on frozen expert features.
"""Genuine K-class expert heads followed by FS-versus-PS composition.

Protocol overview: camera_ready/scripts/analyses/README.md#three-class-protocol.
Frozen inner assignments: appI1_native_k_preflight.py.

For each outer fold and each expert:

1. fit a multinomial softmax head directly on that expert's frozen
   representation and the K-class target;
2. produce content-grouped cross-fitted logits for COMPOSER FIT;
3. refit the same head on all outer-FIT rows, freeze it, and produce
   SELECT and TEST logits;
4. select the multiclass PS rung exclusively on SELECT;
5. compare it with a standardized full-width feature logistic diagnostic
   on untouched TEST.

Naming note: this is the targeted diagnostic in
camera_ready/scripts/analyses/README.md#three-class-protocol. Its feature method is
`FS-LR(std)-K3`, its PCA control is `FSB(std)-K3`, and its prediction
result is the `PS+-K3 diagnostic ceiling`. It is not paper FS1/FS2 or the
complete PS1-PS4 ladder.

The head is genuinely K-class. The backbone remains the binary-trained
frozen backbone; this is not end-to-end multiclass expert training.

Usage:
    python camera_ready/scripts/analyses/appI1_native_k_heads.py --view B_arch6
    python camera_ready/scripts/analyses/appI1_native_k_heads.py --view A_df40
    python camera_ready/scripts/analyses/appI1_native_k_heads.py --view B_arch6 --smoke
    python camera_ready/scripts/analyses/appI1_native_k_heads.py --view B_arch6 --resume

Smoke mode uses only outer-FIT rows and never scores an outer TEST fold.
Its output is implementation QA and must not be reported as a result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBClassifier

REPROD = Path(__file__).resolve().parent
RUNS = REPROD  # the analyses folder holds the shared modules

SEED = 1024
N_PERM = 20
HEAD_MAX_ITER = 500
HEAD_TOL = 1e-3
XGB_SET = [
    {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    for n_estimators in (50, 100, 200)
    for max_depth in (2, 3, 4)
]

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
        for item in outer[fold]["assignments"]:
            lines.append(
                f"{fold}|{item['idx']}|{item['inner_fold']}|"
                f"{item['content_group']}"
            )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def macro_ovr(
    y: np.ndarray, probabilities: np.ndarray, classes: list[str]
) -> tuple[float, dict[str, float]]:
    """Macro one-vs-rest AUROC and its per-class components."""
    if probabilities.shape != (len(y), len(classes)):
        raise ValueError(
            f"probability shape {probabilities.shape}; "
            f"expected {(len(y), len(classes))}"
        )
    per_class = {}
    for class_index, class_name in enumerate(classes):
        binary = (y == class_index).astype(np.int8)
        if binary.sum() in (0, len(binary)):
            per_class[class_name] = float("nan")
        else:
            per_class[class_name] = float(
                roc_auc_score(binary, probabilities[:, class_index])
            )
    finite = [value for value in per_class.values() if np.isfinite(value)]
    return float(np.mean(finite)), per_class


def validate_probabilities(probabilities: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError(f"{label}: non-finite probability")
    deviation = float(np.max(np.abs(probabilities.sum(axis=-1) - 1.0)))
    if deviation > 1e-5:
        raise RuntimeError(f"{label}: probability sums drift by {deviation}")


def fit_multiclass_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    evaluation_matrices: list[np.ndarray],
    *,
    max_iter: int = 3000,
    tolerance: float = 1e-4,
) -> tuple[list[np.ndarray], StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=max_iter,
        solver="lbfgs",
        tol=tolerance,
        random_state=SEED,
    )
    model.fit(scaler.transform(x_train), y_train)
    predictions = [
        model.predict_proba(scaler.transform(matrix))
        for matrix in evaluation_matrices
    ]
    return predictions, scaler, model


def fit_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    evaluation_matrices: list[np.ndarray],
    parameters: dict,
    n_classes: int,
) -> list[np.ndarray]:
    model = XGBClassifier(
        **parameters,
        objective="multi:softprob",
        num_class=n_classes,
        random_state=SEED,
        eval_metric="mlogloss",
        verbosity=0,
        n_jobs=1,
    )
    model.fit(np.asarray(x_train, np.float32), y_train)
    return [
        model.predict_proba(np.asarray(matrix, np.float32))
        for matrix in evaluation_matrices
    ]


def decision_logits(
    scaler: StandardScaler,
    model: LogisticRegression,
    matrix: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    logits = np.asarray(
        model.decision_function(scaler.transform(matrix)), dtype=np.float64
    )
    if logits.ndim == 1:
        logits = np.column_stack([-logits, logits])
    if logits.shape[1] != n_classes:
        raise RuntimeError(
            f"head emitted {logits.shape[1]} logits for {n_classes} classes"
        )
    return logits


def materialize(
    experts: list[str], rows: list[dict], classes: list[str]
) -> tuple[list[np.ndarray], np.ndarray, list[dict]]:
    """Load expert feature blocks in deterministic row order."""
    class_index = {class_name: i for i, class_name in enumerate(classes)}
    ordered = sorted(
        rows,
        key=lambda row: (
            0 if row["split"] == FIT_TAG else 1,
            int(row["idx"]),
        ),
    )
    blocks = [[] for _ in experts]
    labels = []
    ordered_rows = []
    for split in (FIT_TAG, SELECT_TAG):
        split_rows = [row for row in ordered if row["split"] == split]
        if not split_rows:
            continue
        indices = np.asarray([row["idx"] for row in split_rows], dtype=np.int64)
        for expert_index, expert in enumerate(experts):
            array = np.load(
                CACHE_ROOT / split / f"{expert}_features.npy", mmap_mode="r"
            )
            blocks[expert_index].append(
                np.asarray(array[indices], dtype=np.float32)
            )
        labels.extend(class_index[row["cls"]] for row in split_rows)
        ordered_rows.extend(split_rows)
    return (
        [np.concatenate(parts, axis=0) for parts in blocks],
        np.asarray(labels, dtype=np.int16),
        ordered_rows,
    )


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """One scalar multiclass temperature, fitted on OOF FIT logits."""

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        lp = log_softmax(logits / temperature, axis=1)
        return float(-np.mean(lp[np.arange(len(labels)), labels]))

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(10.0)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    if not result.success:
        raise RuntimeError(f"temperature fit failed: {result.message}")
    return float(np.exp(result.x))


def build_native_logits(
    experts: list[str],
    train_blocks: list[np.ndarray],
    select_blocks: list[np.ndarray],
    test_blocks: list[np.ndarray],
    y_train: np.ndarray,
    y_select: np.ndarray,
    y_test: np.ndarray,
    inner_assignment: np.ndarray,
    classes: list[str],
    log,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, list[float]]:
    """Cross-fitted FIT logits plus final-head SELECT/TEST logits."""
    n_classes = len(classes)
    n_experts = len(experts)
    train_logits = np.full(
        (len(y_train), n_experts, n_classes), np.nan, dtype=np.float64
    )
    select_logits = np.full(
        (len(y_select), n_experts, n_classes), np.nan, dtype=np.float64
    )
    test_logits = np.full(
        (len(y_test), n_experts, n_classes), np.nan, dtype=np.float64
    )
    quality = {}
    temperatures = []
    expected_classes = np.arange(n_classes)
    inner_folds = sorted(int(value) for value in np.unique(inner_assignment))

    for expert_index, expert in enumerate(experts):
        started = time.time()
        x_train = train_blocks[expert_index]
        inner_iterations = []
        for inner in inner_folds:
            validation = inner_assignment == inner
            fitting = ~validation
            if not np.array_equal(np.unique(y_train[fitting]), expected_classes):
                raise RuntimeError(
                    f"{expert} inner {inner}: a class is absent from head FIT"
                )
            _, scaler, model = fit_multiclass_logreg(
                x_train[fitting],
                y_train[fitting],
                [],
                max_iter=HEAD_MAX_ITER,
                tolerance=HEAD_TOL,
            )
            inner_iterations.append(int(model.n_iter_[0]))
            if not np.array_equal(model.classes_, expected_classes):
                raise RuntimeError(
                    f"{expert} inner {inner}: class order {model.classes_}"
                )
            train_logits[validation, expert_index] = decision_logits(
                scaler, model, x_train[validation], n_classes
            )

        if not np.all(np.isfinite(train_logits[:, expert_index])):
            raise RuntimeError(f"{expert}: cross-fitted logits incomplete")

        _, scaler, model = fit_multiclass_logreg(
            x_train,
            y_train,
            [],
            max_iter=HEAD_MAX_ITER,
            tolerance=HEAD_TOL,
        )
        if not np.array_equal(model.classes_, expected_classes):
            raise RuntimeError(f"{expert}: final class order {model.classes_}")
        select_logits[:, expert_index] = decision_logits(
            scaler, model, select_blocks[expert_index], n_classes
        )
        test_logits[:, expert_index] = decision_logits(
            scaler, model, test_blocks[expert_index], n_classes
        )

        train_probability = softmax(train_logits[:, expert_index], axis=1)
        select_probability = softmax(select_logits[:, expert_index], axis=1)
        test_probability = softmax(test_logits[:, expert_index], axis=1)
        validate_probabilities(train_probability, f"{expert} OOF FIT")
        validate_probabilities(select_probability, f"{expert} SELECT")
        validate_probabilities(test_probability, f"{expert} TEST")
        train_macro, _ = macro_ovr(y_train, train_probability, classes)
        select_macro, select_per = macro_ovr(
            y_select, select_probability, classes
        )
        test_macro, test_per = macro_ovr(y_test, test_probability, classes)
        temperature = fit_temperature(
            train_logits[:, expert_index], y_train
        )
        temperatures.append(temperature)
        quality[expert] = {
            "oof_fit_macro": train_macro,
            "select_macro": select_macro,
            "select_per_class": select_per,
            "test_macro": test_macro,
            "test_per_class": test_per,
            "temperature_from_oof_fit": temperature,
            "inner_n_iter": inner_iterations,
            "final_n_iter": int(model.n_iter_[0]),
            "head_max_iter": HEAD_MAX_ITER,
            "head_tolerance": HEAD_TOL,
            "all_fits_converged_before_cap": bool(
                max(inner_iterations + [int(model.n_iter_[0])])
                < HEAD_MAX_ITER
            ),
        }
        log(
            f"    {expert:<16} head OOF {train_macro:.4f}  "
            f"SELECT {select_macro:.4f}  TEST {test_macro:.4f}  "
            f"T={temperature:.3f}  ({time.time() - started:.0f}s)"
        )

    return (
        train_logits,
        select_logits,
        test_logits,
        quality,
        temperatures,
    )


def entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(probabilities * np.log(probabilities), axis=-1)


def centered_directions(logits: np.ndarray) -> np.ndarray:
    """K-1 independent directions per expert, relative to last class."""
    return logits[:, :, :-1] - logits[:, :, -1, None]


def pairwise_js(probabilities: np.ndarray) -> np.ndarray:
    blocks = []
    for first, second in combinations(range(probabilities.shape[1]), 2):
        p = np.clip(probabilities[:, first], 1e-12, 1.0)
        q = np.clip(probabilities[:, second], 1e-12, 1.0)
        midpoint = 0.5 * (p + q)
        divergence = 0.5 * np.sum(p * np.log(p / midpoint), axis=1)
        divergence += 0.5 * np.sum(q * np.log(q / midpoint), axis=1)
        blocks.append(divergence)
    if not blocks:
        return np.empty((len(probabilities), 0), dtype=np.float64)
    return np.stack(blocks, axis=1)


def confidence_stats(probabilities: np.ndarray) -> np.ndarray:
    sorted_probability = np.sort(probabilities, axis=2)
    margins = sorted_probability[:, :, -1] - sorted_probability[:, :, -2]
    maxima = sorted_probability[:, :, -1]
    return np.column_stack(
        [
            maxima.mean(axis=1),
            maxima.std(axis=1),
            margins.mean(axis=1),
            margins.std(axis=1),
        ]
    )


def make_rank_map(reference: np.ndarray):
    sorted_reference = np.sort(reference, axis=0)
    n_rows = len(sorted_reference)

    def transform(matrix: np.ndarray) -> np.ndarray:
        output = np.empty_like(matrix, dtype=np.float64)
        for column in range(matrix.shape[1]):
            output[:, column] = (
                np.searchsorted(
                    sorted_reference[:, column],
                    matrix[:, column],
                    side="right",
                )
                / n_rows
            )
        return output

    return transform


def build_feature_maps(
    train_logits: np.ndarray,
    select_logits: np.ndarray,
    test_logits: np.ndarray,
    temperatures: list[float],
    smoke: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Multiclass prediction-space feature maps."""
    temperatures_array = np.asarray(temperatures, dtype=np.float64)
    calibrated = [
        matrix / temperatures_array[None, :, None]
        for matrix in (train_logits, select_logits, test_logits)
    ]
    uncalibrated_probability = [
        softmax(matrix, axis=2)
        for matrix in (train_logits, select_logits, test_logits)
    ]
    calibrated_probability = [
        softmax(matrix, axis=2) for matrix in calibrated
    ]

    maps = {
        "raw_logits": tuple(
            matrix.reshape(len(matrix), -1)
            for matrix in (train_logits, select_logits, test_logits)
        ),
        "probabilities": tuple(
            matrix.reshape(len(matrix), -1)
            for matrix in uncalibrated_probability
        ),
    }
    if smoke:
        return maps

    disagreement = []
    uncertainty = []
    for probability in calibrated_probability:
        mean_probability = probability.mean(axis=1)
        per_expert_entropy = entropy(probability)
        entropy_of_mean = entropy(mean_probability)
        mean_entropy = per_expert_entropy.mean(axis=1)
        mutual_information = entropy_of_mean - mean_entropy

        disagreement.append(
            np.concatenate(
                [
                    probability.reshape(len(probability), -1),
                    mean_probability,
                    probability.std(axis=1),
                    pairwise_js(probability),
                ],
                axis=1,
            )
        )
        uncertainty.append(
            np.concatenate(
                [
                    probability.reshape(len(probability), -1),
                    mean_probability,
                    probability.var(axis=1),
                    per_expert_entropy,
                    np.column_stack(
                        [entropy_of_mean, mean_entropy, mutual_information]
                    ),
                    confidence_stats(probability),
                ],
                axis=1,
            )
        )
    maps["disagreement"] = tuple(disagreement)
    maps["calibrated_uncertainty_mi"] = tuple(uncertainty)

    directions = [
        centered_directions(matrix).reshape(len(matrix), -1)
        for matrix in (train_logits, select_logits, test_logits)
    ]
    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    maps["quadratic"] = (
        polynomial.fit_transform(directions[0]),
        polynomial.transform(directions[1]),
        polynomial.transform(directions[2]),
    )
    rank = make_rank_map(directions[0])
    maps["rank"] = tuple(rank(matrix) for matrix in directions)
    return maps


def run_ps_ladder(
    feature_maps: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    y_train: np.ndarray,
    y_select: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    mean_probabilities: tuple[np.ndarray, np.ndarray],
    smoke: bool,
) -> tuple[dict, dict, np.ndarray]:
    """Return SELECT-chosen ceiling, all rungs, and chosen TEST proba."""
    select_mean, test_mean = mean_probabilities
    mean_select_score, _ = macro_ovr(y_select, select_mean, classes)
    mean_test_score, mean_test_per = macro_ovr(y_test, test_mean, classes)
    all_scores = {
        "mean_probability": {
            "select": mean_select_score,
            "test": mean_test_score,
            "per_class": mean_test_per,
        }
    }
    best = {
        "select": mean_select_score,
        "test": mean_test_score,
        "per_class": mean_test_per,
        "rung": "mean_probability",
    }
    best_test_probability = test_mean

    xgb_parameters = [XGB_SET[0]] if smoke else XGB_SET
    for map_name, (x_train, x_select, x_test) in feature_maps.items():
        predictions, _, _ = fit_multiclass_logreg(
            np.asarray(x_train, np.float64),
            y_train,
            [
                np.asarray(x_select, np.float64),
                np.asarray(x_test, np.float64),
            ],
        )
        select_probability, test_probability = predictions
        select_score, _ = macro_ovr(
            y_select, select_probability, classes
        )
        test_score, test_per = macro_ovr(
            y_test, test_probability, classes
        )
        key = f"{map_name}|logreg"
        all_scores[key] = {
            "select": select_score,
            "test": test_score,
            "per_class": test_per,
        }
        if select_score > best["select"]:
            best = {
                "select": select_score,
                "test": test_score,
                "per_class": test_per,
                "rung": key,
            }
            best_test_probability = test_probability

        for parameters in xgb_parameters:
            select_probability, test_probability = fit_xgb(
                x_train,
                y_train,
                [x_select, x_test],
                parameters,
                len(classes),
            )
            select_score, _ = macro_ovr(
                y_select, select_probability, classes
            )
            test_score, test_per = macro_ovr(
                y_test, test_probability, classes
            )
            key = (
                f"{map_name}|xgb_d{parameters['max_depth']}"
                f"_n{parameters['n_estimators']}"
            )
            all_scores[key] = {
                "select": select_score,
                "test": test_score,
                "per_class": test_per,
            }
            if select_score > best["select"]:
                best = {
                    "select": select_score,
                    "test": test_score,
                    "per_class": test_per,
                    "rung": key,
                }
                best_test_probability = test_probability

    return best, all_scores, best_test_probability


def grouped_shuffle_labels(
    rows: list[dict],
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Permute labels across generator families, the outer-fold unit."""
    group_label = {}
    for row, label in zip(rows, labels):
        family_group = row["group"]
        integer_label = int(label)
        if (
            family_group in group_label
            and group_label[family_group] != integer_label
        ):
            raise RuntimeError(
                "one generator-family group carries conflicting K-class labels"
            )
        group_label[family_group] = integer_label
    groups = sorted(group_label)
    shuffled = [group_label[group] for group in groups]
    rng.shuffle(shuffled)
    mapping = dict(zip(groups, shuffled))
    output = np.asarray(
        [mapping[row["group"]] for row in rows], dtype=np.int16
    )
    if len(np.unique(output)) < len(np.unique(labels)):
        return None
    return output


def prepare_smoke_split(
    rows: list[dict],
    preflight_fold: dict,
) -> tuple[list[dict], list[dict], list[dict], np.ndarray]:
    """Internal FIT-only train/select/test split; outer TEST is untouched."""
    preflight_by_index = {
        int(item["idx"]): item
        for item in preflight_fold["assignments"]
    }
    fit_rows = sorted(
        (
            {
                **row,
                "content_group": preflight_by_index[row["idx"]][
                    "content_group"
                ],
            }
            for row in rows
            if row["fold"] != 0 and row["split"] == FIT_TAG
        ),
        key=lambda row: row["idx"],
    )
    train = [
        row
        for row in fit_rows
        if preflight_by_index[row["idx"]]["inner_fold"] in (0, 1, 2)
    ]
    select = [
        row
        for row in fit_rows
        if preflight_by_index[row["idx"]]["inner_fold"] == 3
    ]
    test = [
        row
        for row in fit_rows
        if preflight_by_index[row["idx"]]["inner_fold"] == 4
    ]
    inner = np.asarray(
        [preflight_by_index[row["idx"]]["inner_fold"] for row in train],
        dtype=np.int16,
    )
    unique = sorted(np.unique(inner))
    remap = {value: index for index, value in enumerate(unique)}
    inner = np.asarray([remap[int(value)] for value in inner], dtype=np.int16)
    return train, select, test, inner


def run_fold(
    fold: int,
    experts: list[str],
    classes: list[str],
    train_rows: list[dict],
    select_rows: list[dict],
    test_rows: list[dict],
    inner_assignment: np.ndarray,
    previous_fold: dict | None,
    n_permutations: int,
    smoke: bool,
    log,
) -> tuple[dict, dict[str, np.ndarray]]:
    started = time.time()
    train_blocks, y_train, ordered_train = materialize(
        experts, train_rows, classes
    )
    select_blocks, y_select, ordered_select = materialize(
        experts, select_rows, classes
    )
    test_blocks, y_test, ordered_test = materialize(
        experts, test_rows, classes
    )
    if len(inner_assignment) != len(y_train):
        raise RuntimeError(
            f"fold {fold}: inner assignments {len(inner_assignment)} "
            f"!= materialized FIT rows {len(y_train)}"
        )
    if any(row["split"] != FIT_TAG for row in ordered_train):
        raise RuntimeError("composer/head FIT contains a non-FIT row")

    log(
        f"\n=== {'SMOKE' if smoke else 'outer'} fold {fold}: "
        f"FIT {len(y_train):,} | SELECT {len(y_select):,} | "
        f"TEST {len(y_test):,}; inner={len(np.unique(inner_assignment))} ==="
    )
    log("  fitting genuine K-class heads:")
    (
        train_logits,
        select_logits,
        test_logits,
        head_quality,
        temperatures,
    ) = build_native_logits(
        experts,
        train_blocks,
        select_blocks,
        test_blocks,
        y_train,
        y_select,
        y_test,
        inner_assignment,
        classes,
        log,
    )

    maps = build_feature_maps(
        train_logits,
        select_logits,
        test_logits,
        temperatures,
        smoke,
    )
    calibrated_select = softmax(
        select_logits
        / np.asarray(temperatures, dtype=np.float64)[None, :, None],
        axis=2,
    )
    calibrated_test = softmax(
        test_logits
        / np.asarray(temperatures, dtype=np.float64)[None, :, None],
        axis=2,
    )
    best, all_ps, ps_test_probability = run_ps_ladder(
        maps,
        y_train,
        y_select,
        y_test,
        classes,
        (calibrated_select.mean(axis=1), calibrated_test.mean(axis=1)),
        smoke,
    )

    feature_train = np.concatenate(train_blocks, axis=1)
    feature_test = np.concatenate(test_blocks, axis=1)
    (fs_test_probability,), _, _ = fit_multiclass_logreg(
        feature_train, y_train, [feature_test]
    )
    fs_macro, fs_per = macro_ovr(y_test, fs_test_probability, classes)

    ps_width = len(experts) * len(classes)
    pca = PCA(n_components=ps_width, random_state=SEED)
    pca_train = pca.fit_transform(feature_train)
    pca_test = pca.transform(feature_test)
    (pca_test_probability,), _, _ = fit_multiclass_logreg(
        pca_train, y_train, [pca_test]
    )
    pca_macro, pca_per = macro_ovr(
        y_test, pca_test_probability, classes
    )

    null_fs = []
    null_ps = []
    raw_train, _, raw_test = maps["raw_logits"]
    rng = np.random.default_rng(SEED + fold)
    for _ in range(n_permutations):
        shuffled = grouped_shuffle_labels(ordered_train, y_train, rng)
        if shuffled is None:
            continue
        (null_fs_probability,), _, _ = fit_multiclass_logreg(
            feature_train, shuffled, [feature_test]
        )
        (null_ps_probability,), _, _ = fit_multiclass_logreg(
            raw_train, shuffled, [raw_test]
        )
        null_fs.append(
            macro_ovr(y_test, null_fs_probability, classes)[0]
        )
        null_ps.append(
            macro_ovr(y_test, null_ps_probability, classes)[0]
        )

    test_max_name, test_max_record = max(
        all_ps.items(), key=lambda item: item[1]["test"]
    )
    delta = fs_macro - best["test"]
    previous_delta = None
    if previous_fold is not None:
        previous_delta = (
            previous_fold["fs1"]["macro"]
            - previous_fold["ps_ceiling"]["test"]
        )

    log(
        f"  PS ceiling (SELECT): {best['rung']:<36} "
        f"TEST {best['test']:.4f}"
    )
    log(f"  FS-LR(std)-K3 {fs_macro:.4f}  delta {delta:+.4f}")
    log(
        f"  FS PCA->{ps_width} {pca_macro:.4f}  "
        f"delta {pca_macro - best['test']:+.4f}"
    )
    if previous_delta is not None:
        log(
            f"  previous binary-head delta {previous_delta:+.4f}; "
            f"native-minus-binary {delta - previous_delta:+.4f}"
        )
    log(
        f"  TEST-max diagnostic {test_max_record['test']:.4f} "
        f"({test_max_name}); not adopted"
    )
    if null_fs:
        log(
            f"  grouped null ({len(null_fs)}): "
            f"FS {np.mean(null_fs):.4f}, PS {np.mean(null_ps):.4f} "
            f"(chance 0.5)"
        )
    log(f"  fold time {time.time() - started:.0f}s")

    entry = {
        "fold": fold,
        "smoke_internal_fit_only": smoke,
        "n_train": len(y_train),
        "n_select": len(y_select),
        "n_test": len(y_test),
        "n_inner_folds": int(len(np.unique(inner_assignment))),
        "ps_interface": {
            "stored_width": ps_width,
            "independent_directions": len(experts) * (len(classes) - 1),
        },
        "head_quality": head_quality,
        "temperatures": temperatures,
        "ps_ceiling": best,
        "ps_all": all_ps,
        "fs1": {"macro": fs_macro, "per_class": fs_per},
        "fs_pca": {
            "macro": pca_macro,
            "per_class": pca_per,
            "width": ps_width,
        },
        "delta_fs1_minus_ps": delta,
        "previous_binary_head_delta": previous_delta,
        "native_minus_binary_delta": (
            None if previous_delta is None else delta - previous_delta
        ),
        "test_max_diagnostic": {
            "rung": test_max_name,
            "test": test_max_record["test"],
        },
        "group_shuffled_null": {
            "n_permutations": len(null_fs),
            "fs_mean": float(np.mean(null_fs)) if null_fs else None,
            "ps_mean": float(np.mean(null_ps)) if null_ps else None,
            "chance": 0.5,
        },
        "validity": {
            "crossfit_complete": bool(np.all(np.isfinite(train_logits))),
            "probabilities_finite": True,
            "selection_uses_test": False,
        },
        "method_semantics": {
            "fs1_legacy_key": "FS-LR(std)-K3; not paper FS1",
            "fs_pca": "FSB(std)-K3 width control; not paper FS2",
            "ps_ceiling": "PS+-K3 diagnostic ceiling",
            "paper_PS1_to_PS4_complete": False,
            "paper_FS2_evaluated": False,
        },
    }
    arrays = {
        "y": y_test,
        "fs": fs_test_probability,
        "ps": ps_test_probability,
        "pca": pca_test_probability,
        "native_logits": test_logits,
    }
    return entry, arrays


def load_inputs(view: str) -> tuple[dict, dict, dict | None]:
    manifest_path = RESULTS_K / f"K3_FROZEN_{view}.json"
    preflight_path = RESULTS_K / f"NATIVE_K_PREFLIGHT_{view}.json"
    if not manifest_path.exists() or not preflight_path.exists():
        raise SystemExit(
            "missing frozen inputs; run appI1_k3_freeze_v2.py and "
            "appI1_native_k_preflight.py, or copy their stored outputs from "
            "camera_ready/output/results/analyses/three_class/"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    digest = manifest_digest(manifest["manifest"])
    if digest != manifest["manifest_sha256"]:
        raise SystemExit("source manifest digest mismatch")
    if digest != preflight["manifest_sha256"]:
        raise SystemExit("preflight refers to a different manifest")
    computed_assignment = assignment_digest(preflight["outer_folds"])
    if computed_assignment != preflight["assignment_sha256"]:
        raise SystemExit("preflight inner-assignment digest mismatch")
    if not preflight["valid"]:
        raise SystemExit("preflight is not valid")

    previous_path = RESULTS_K / f"K3_RESULT_{view}.json"
    previous = (
        json.loads(previous_path.read_text(encoding="utf-8"))
        if previous_path.exists()
        else None
    )
    return manifest, preflight, previous


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=("B_arch6", "A_df40"),
                        default="B_arch6")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="internal FIT-only QA; never touches an outer TEST fold",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed full fold JSON+NPZ artifacts",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=None,
        help="grouped null permutations; the full default is 20",
    )
    arguments = parser.parse_args()
    if arguments.smoke and arguments.resume:
        raise SystemExit("--smoke and --resume cannot be combined")

    manifest, preflight, previous = load_inputs(arguments.view)
    rows = manifest["manifest"]
    experts = manifest["pool"]
    classes = manifest["classes"]
    print(
        f"view {arguments.view}: genuine K={len(classes)} heads on "
        f"{len(experts)} frozen backbones; manifest "
        f"{manifest['manifest_sha256'][:16]}..., inner "
        f"{preflight['assignment_sha256'][:16]}..."
    )
    print(f"classes: {classes}")
    print(
        "scope: genuine multiclass heads on binary-trained frozen "
        "backbones; not end-to-end multiclass experts"
    )

    if arguments.smoke:
        fold = preflight["outer_folds"]["0"]
        train_rows, select_rows, test_rows, inner = prepare_smoke_split(
            rows, fold
        )
        entry, arrays = run_fold(
            0,
            experts,
            classes,
            train_rows,
            select_rows,
            test_rows,
            inner,
            None,
            arguments.n_perm or 2,
            True,
            print,
        )
        output = {
            "design": "camera_ready/scripts/analyses/README.md#three-class-protocol",
            "view": arguments.view,
            "status": "SMOKE_INTERNAL_FIT_ONLY_DO_NOT_REPORT",
            "fold": entry,
        }
        destination = (
            RESULTS_K / f"NATIVE_K_SMOKE_{arguments.view}.json"
        )
        npz_destination = (
            RESULTS_K / f"NATIVE_K_SMOKE_{arguments.view}.npz"
        )
        destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
        np.savez_compressed(npz_destination, **arrays)
        print(f"wrote {destination.name} and {npz_destination.name}")
        return

    n_permutations = (
        N_PERM if arguments.n_perm is None else arguments.n_perm
    )
    if n_permutations != N_PERM:
        print(
            f"WARNING: n_perm={n_permutations}; the full run "
            f"uses {N_PERM}"
        )

    entries = []
    arrays_by_fold = {}
    for fold in range(manifest["n_folds"]):
        json_path = (
            RESULTS_K
            / f"NATIVE_K_FOLD_{arguments.view}_{fold}.json"
        )
        npz_path = (
            RESULTS_K
            / f"NATIVE_K_FOLD_{arguments.view}_{fold}.npz"
        )
        if arguments.resume and json_path.exists() and npz_path.exists():
            entry = json.loads(json_path.read_text(encoding="utf-8"))
            with np.load(npz_path) as saved:
                arrays = {name: saved[name] for name in saved.files}
            print(
                f"\n=== outer fold {fold}: resumed "
                f"delta {entry['delta_fs1_minus_ps']:+.4f} ==="
            )
        else:
            train_rows = [
                row
                for row in rows
                if row["fold"] != fold and row["split"] == FIT_TAG
            ]
            select_rows = [
                row
                for row in rows
                if row["fold"] != fold and row["split"] == SELECT_TAG
            ]
            test_rows = [row for row in rows if row["fold"] == fold]
            preflight_by_index = {
                int(item["idx"]): item
                for item in preflight["outer_folds"][str(fold)][
                    "assignments"
                ]
            }
            train_rows = [
                {
                    **row,
                    "content_group": preflight_by_index[row["idx"]][
                        "content_group"
                    ],
                }
                for row in train_rows
            ]
            ordered_train = sorted(train_rows, key=lambda row: row["idx"])
            inner = np.asarray(
                [
                    preflight_by_index[row["idx"]]["inner_fold"]
                    for row in ordered_train
                ],
                dtype=np.int16,
            )
            previous_fold = (
                previous["folds"].get(str(fold))
                if previous is not None
                else None
            )
            entry, arrays = run_fold(
                fold,
                experts,
                classes,
                train_rows,
                select_rows,
                test_rows,
                inner,
                previous_fold,
                n_permutations,
                False,
                print,
            )
            json_path.write_text(
                json.dumps(entry, indent=2), encoding="utf-8"
            )
            np.savez_compressed(npz_path, **arrays)
            print(f"  checkpointed {json_path.name} and {npz_path.name}")
        entries.append(entry)
        arrays_by_fold[fold] = arrays

    deltas = np.asarray(
        [entry["delta_fs1_minus_ps"] for entry in entries], dtype=float
    )
    fs_scores = np.asarray(
        [entry["fs1"]["macro"] for entry in entries], dtype=float
    )
    ps_scores = np.asarray(
        [entry["ps_ceiling"]["test"] for entry in entries], dtype=float
    )
    pca_deltas = np.asarray(
        [
            entry["fs_pca"]["macro"] - entry["ps_ceiling"]["test"]
            for entry in entries
        ],
        dtype=float,
    )
    binary_deltas = [
        entry["previous_binary_head_delta"]
        for entry in entries
        if entry["previous_binary_head_delta"] is not None
    ]

    y_oof = np.concatenate([arrays_by_fold[fold]["y"] for fold in arrays_by_fold])
    pooled = {}
    for name in ("fs", "ps", "pca"):
        probability = np.vstack(
            [arrays_by_fold[fold][name] for fold in arrays_by_fold]
        )
        macro, per_class = macro_ovr(y_oof, probability, classes)
        pooled[name] = {"macro": macro, "per_class": per_class}
    pooled["delta_fs_minus_ps"] = (
        pooled["fs"]["macro"] - pooled["ps"]["macro"]
    )
    pooled["n_rows"] = int(len(y_oof))

    fs_reproduction = None
    if previous is not None:
        previous_fs = float(
            np.mean(
                [
                    previous["folds"][str(fold)]["fs1"]["macro"]
                    for fold in range(manifest["n_folds"])
                ]
            )
        )
        fs_reproduction = {
            "current_mean": float(fs_scores.mean()),
            "previous_mean": previous_fs,
            "absolute_difference": abs(float(fs_scores.mean()) - previous_fs),
            "tolerance": 0.01,
            "passed": abs(float(fs_scores.mean()) - previous_fs) <= 0.01,
        }

    result = {
        "design": "camera_ready/scripts/analyses/README.md#three-class-protocol",
        "view": arguments.view,
        "manifest_sha256": manifest["manifest_sha256"],
        "inner_assignment_sha256": preflight["assignment_sha256"],
        "scope": (
            "genuine K-class softmax heads fitted directly on multiclass "
            "labels; submitted binary-trained backbones remain frozen"
        ),
        "classes": classes,
        "experts": experts,
        "n_permutations": n_permutations,
        "folds": entries,
        "primary": {
            "definition": (
                "mean outer-fold FS-LR(std)-K3 macro OvR AUROC minus "
                "SELECT-chosen PS+-K3 diagnostic ceiling macro OvR AUROC"
            ),
            "per_fold": deltas.tolist(),
            "mean": float(deltas.mean()),
            "folds_positive": int(np.sum(deltas > 0)),
            "n_folds": len(deltas),
            "mean_fs1_legacy_key": float(fs_scores.mean()),
            "mean_feature_logreg_standardized": float(fs_scores.mean()),
            "mean_ps": float(ps_scores.mean()),
            "inference": (
                "DESCRIPTIVE; minimum two-sided sign-flip p "
                f"{2 ** (1 - len(deltas)):.4f}"
            ),
        },
        "secondary": {
            "pca_per_fold_delta": pca_deltas.tolist(),
            "pca_mean_delta": float(pca_deltas.mean()),
            "pooled_oof": pooled,
            "binary_head_per_fold_delta": binary_deltas,
            "native_minus_binary_mean_delta": (
                None
                if not binary_deltas
                else float(deltas.mean() - np.mean(binary_deltas))
            ),
        },
        "validity": {
            "manifest": True,
            "inner_assignments": True,
            "outer_content_overlap": 0,
            "fs1_reproduction": fs_reproduction,
            "passed": bool(
                fs_reproduction is None or fs_reproduction["passed"]
            ),
        },
    }
    destination = (
        RESULTS_K / f"NATIVE_K_RESULT_{arguments.view}.json"
    )
    oof_destination = (
        RESULTS_K / f"NATIVE_K_OOF_{arguments.view}.npz"
    )
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        oof_destination,
        y=y_oof,
        fs=np.vstack(
            [arrays_by_fold[fold]["fs"] for fold in arrays_by_fold]
        ),
        ps=np.vstack(
            [arrays_by_fold[fold]["ps"] for fold in arrays_by_fold]
        ),
        pca=np.vstack(
            [arrays_by_fold[fold]["pca"] for fold in arrays_by_fold]
        ),
        native_logits=np.vstack(
            [
                arrays_by_fold[fold]["native_logits"]
                for fold in arrays_by_fold
            ]
        ),
        classes=np.asarray(classes, dtype=object),
    )

    print("\n" + "=" * 72)
    print(
        f"PRIMARY native-head delta {deltas.mean():+.4f}; "
        f"{int(np.sum(deltas > 0))}/{len(deltas)} folds positive"
    )
    print(f"per fold {[round(value, 4) for value in deltas]}")
    print(
        f"mean FS-LR(std)-K3 {fs_scores.mean():.4f}; "
        f"mean PS {ps_scores.mean():.4f}; "
        f"PCA delta {pca_deltas.mean():+.4f}"
    )
    if binary_deltas:
        print(
            f"previous binary-head mean delta {np.mean(binary_deltas):+.4f}; "
            f"native-minus-binary {deltas.mean() - np.mean(binary_deltas):+.4f}"
        )
    if fs_reproduction is not None:
        print(
            "FS reproduction "
            + ("PASS" if fs_reproduction["passed"] else "FAIL")
            + f" (difference {fs_reproduction['absolute_difference']:.6f})"
        )
    print(f"wrote {destination.name} and {oof_destination.name}")
    if not result["validity"]["passed"]:
        raise SystemExit("validity gate failed; do not report result")


if __name__ == "__main__":
    main()
