# C20: Compare combiners using task-aligned three-class expert outputs.
"""Paper-aligned genuine K-class-head FS-versus-PS comparison.

Method naming: camera_ready/scripts/analyses/README.md#three-class-protocol.
Implementation overview: camera_ready/scripts/analyses/README.md#three-class-protocol.
Data/head overview: camera_ready/scripts/analyses/README.md#three-class-protocol.

This runner does not overwrite the targeted diagnostic artifacts emitted
by appI1_native_k_heads.py. It refits the same K-class heads
and evaluates:

  PS1-K3  mean K-class probabilities;
  PS2-K3  unscaled multinomial LogReg on K-class logits;
  PS3-K3  paper shallow MLP, 10 epochs, eight-seed average;
  PS4-K3  multiclass XGBoost, 5-fold FIT-only CV, full paper grid;
  FS1-K3  unscaled multinomial LogReg on raw concatenated features;
  FS2-K3  paper projection MLP, d=256 on arch6, eight-seed average;
  FSB-K3  PCA to E*K followed by the FS1-K3 logistic recipe;
  PS+-K3  the separately labelled extended diagnostic ladder.

Only the output layer and class-balancing arithmetic are extended from
binary to K classes. Smoke mode uses only outer-FIT rows and is never
reportable.

Usage:
    python camera_ready/scripts/analyses/appI1_native_k_aligned_v2.py --view B_arch6 --smoke
    python camera_ready/scripts/analyses/appI1_native_k_aligned_v2.py --view B_arch6 --resume
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from xgboost import XGBClassifier


REPROD = Path(__file__).resolve().parent
RUNS = REPROD  # the analyses folder holds the shared modules
import _paths  # noqa: E402,F401  reviewed code base on sys.path, determinism prelude
RESULTS = _paths.RESULTS_DIR / "three_class"
# what a cache must have been written for to be reused (set in main from the inputs and the device)
CACHE_KEY: dict = {}


def _reusable(array_path, meta_path, log) -> bool:
    """A cache pair is reused only when both files exist and its metadata records this run's key."""
    if not (array_path.exists() and meta_path.exists()):
        return False
    try:
        key = json.loads(meta_path.read_text(encoding="utf-8")).get("cache_key")
    except (OSError, ValueError):
        key = None
    if key != CACHE_KEY:
        log(f"    {meta_path.name}: not reused (written for another manifest, inner assignment, view or device)")
        return False
    return True
RESULTS.mkdir(parents=True, exist_ok=True)
SEED = 1024
SEEDS = (7, 10, 24, 42, 1024, 1973, 3392, 9721)
PS_EPOCHS = 10
FS_EPOCHS = 10
BATCH_SIZE = 256
PS_HIDDEN = 64
PS_DROPOUT = 0.1
FS_HIDDEN = (512, 256)
FS_DROPOUT = 0.1
FS_EXPERT_DROPOUT = 0.25
LEARNING_RATE = 2e-4
PS2_MAX_ITER = 1000
FS1_MAX_ITER = 2000
PROJ_DIM = {"B_arch6": 256, "A_df40": 256}

XGB_GRID = {
    "n_estimators": (50, 100, 200),
    "max_depth": (2, 3, 4),
    "learning_rate": (0.05, 0.1, 0.3),
    "subsample": (0.8, 1.0),
    "colsample_bytree": (0.8, 1.0),
}

_base_spec = importlib.util.spec_from_file_location(
    "native_k_diagnostic", REPROD / "appI1_native_k_heads.py"
)
BASE = importlib.util.module_from_spec(_base_spec)
_base_spec.loader.exec_module(BASE)

from src.models.feature_space import FeatureAssembly  # noqa: E402
from src.utils.config import FIT_TAG, SELECT_TAG  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


class TinyKMLP(nn.Module):
    """Paper PS3 architecture with a K-class output layer."""

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, PS_HIDDEN),
            nn.ReLU(),
            nn.Dropout(PS_DROPOUT),
            nn.Linear(PS_HIDDEN, n_classes),
        )

    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        return self.net(matrix)


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """K-class extension of the paper's inverse-frequency CE weights."""
    counts = np.bincount(labels.astype(int), minlength=n_classes)
    if np.any(counts == 0):
        raise RuntimeError(f"missing class in training labels: {counts}")
    weights = len(labels) / (n_classes * counts.astype(np.float64))
    return torch.as_tensor(weights, dtype=torch.float32)


def sample_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Balanced sample weights for multiclass PS4-K3."""
    weights = class_weights(labels, n_classes).numpy()
    return weights[labels.astype(int)]


def evaluate(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
) -> dict:
    BASE.validate_probabilities(probabilities, "method output")
    macro, per_class = BASE.macro_ovr(labels, probabilities, classes)
    return {"macro": macro, "per_class": per_class}


def unscaled_logreg(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    matrices: list[np.ndarray],
    *,
    max_iter: int,
) -> tuple[list[np.ndarray], dict]:
    """Paper LR recipe: raw inputs, no StandardScaler."""
    estimator = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(np.asarray(x_fit), y_fit)
    output = [
        estimator.predict_proba(np.asarray(matrix)) for matrix in matrices
    ]
    diagnostics = {
        "scaled": False,
        "solver": "lbfgs",
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": max_iter,
        "n_iter": [int(value) for value in estimator.n_iter_],
        "convergence_warnings": sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        ),
        "classes": [int(value) for value in estimator.classes_],
    }
    return output, diagnostics


@torch.no_grad()
def predict_matrix(
    model: nn.Module,
    matrix: np.ndarray,
    device: str,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(matrix, dtype=torch.float32)
    blocks = []
    for start in range(0, len(tensor), batch_size):
        logits = model(tensor[start : start + batch_size].to(device))
        blocks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(blocks)


def train_ps3_seed(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    matrices: list[np.ndarray],
    n_classes: int,
    seed: int,
    device: str,
    epochs: int,
) -> list[np.ndarray]:
    """K-class extension of the submitted fixed-epoch PS3 trainer."""
    set_seed(seed)
    model = TinyKMLP(x_fit.shape[1], n_classes).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(y_fit, n_classes).to(device)
    )
    x_tensor = torch.as_tensor(x_fit, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_fit, dtype=torch.long)
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(y_tensor))
        for start in range(0, len(y_tensor), BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE]
            optimiser.zero_grad()
            logits = model(x_tensor[indices].to(device))
            loss = criterion(logits, y_tensor[indices].to(device))
            loss.backward()
            optimiser.step()
    return [predict_matrix(model, matrix, device) for matrix in matrices]


def seed_cache_paths(
    view: str, fold: int, method: str, seed: int, smoke: bool
) -> tuple[Path, Path]:
    scope = "SMOKE" if smoke else "FULL"
    stem = f"NATIVE_K_ALIGNED_{scope}_{view}_F{fold}_{method}_S{seed}"
    return RESULTS / f"{stem}.npz", RESULTS / f"{stem}.json"


def paper_ps3(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_select: np.ndarray,
    x_test: np.ndarray,
    n_classes: int,
    view: str,
    fold: int,
    device: str,
    smoke: bool,
    resume: bool,
    log,
) -> tuple[np.ndarray, np.ndarray, dict]:
    seeds = (SEED,) if smoke else SEEDS
    epochs = 1 if smoke else PS_EPOCHS
    select_predictions = []
    test_predictions = []
    for seed in seeds:
        array_path, meta_path = seed_cache_paths(
            view, fold, "PS3_K3", seed, smoke
        )
        if resume and _reusable(array_path, meta_path, log):
            with np.load(array_path) as saved:
                select_probability = saved["select"]
                test_probability = saved["test"]
            log(f"    PS3-K3 seed {seed}: resumed")
        else:
            started = time.time()
            select_probability, test_probability = train_ps3_seed(
                x_fit,
                y_fit,
                [x_select, x_test],
                n_classes,
                seed,
                device,
                epochs,
            )
            np.savez_compressed(
                array_path,
                select=select_probability,
                test=test_probability,
            )
            meta_path.write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "epochs": epochs,
                        "device": device,
                        "paper_aligned": not smoke,
                        "cache_key": CACHE_KEY,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            log(
                f"    PS3-K3 seed {seed}: {time.time() - started:.0f}s"
            )
        select_predictions.append(select_probability)
        test_predictions.append(test_probability)
    return (
        np.mean(select_predictions, axis=0),
        np.mean(test_predictions, axis=0),
        {
            "seeds": list(seeds),
            "epochs": epochs,
            "hidden": PS_HIDDEN,
            "dropout": PS_DROPOUT,
            "lr": LEARNING_RATE,
            "averaging": "per-sample class probabilities",
        },
    )


@torch.no_grad()
def predict_feature_model(
    model: FeatureAssembly,
    blocks: list[np.ndarray],
    device: str,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    tensors = [torch.as_tensor(block, dtype=torch.float32) for block in blocks]
    probabilities = []
    for start in range(0, len(tensors[0]), batch_size):
        batch = [
            tensor[start : start + batch_size].to(device)
            for tensor in tensors
        ]
        probabilities.append(
            torch.softmax(model(batch), dim=1).cpu().numpy()
        )
    return np.concatenate(probabilities)


def train_fs2_seed(
    fit_blocks: list[np.ndarray],
    y_fit: np.ndarray,
    select_blocks: list[np.ndarray],
    y_select: np.ndarray,
    test_blocks: list[np.ndarray],
    n_classes: int,
    proj_dim: int,
    seed: int,
    device: str,
    epochs: int,
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Paper FS2 recipe with only its output layer extended to K classes."""
    set_seed(seed)
    model = FeatureAssembly(
        [block.shape[1] for block in fit_blocks],
        proj_dim=proj_dim,
        num_classes=n_classes,
        classifier_type="mlp",
        expert_dropout=FS_EXPERT_DROPOUT,
        hidden_dims=list(FS_HIDDEN),
        dropout=FS_DROPOUT,
    ).to(device)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(y_fit, n_classes).to(device)
    )
    fit_tensors = [
        torch.as_tensor(block, dtype=torch.float32) for block in fit_blocks
    ]
    y_tensor = torch.as_tensor(y_fit, dtype=torch.long)
    best_macro = -np.inf
    best_epoch = 0
    best_state = None
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(y_tensor))
        for start in range(0, len(y_tensor), BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE]
            batch = [tensor[indices].to(device) for tensor in fit_tensors]
            optimiser.zero_grad()
            logits = model(batch)
            loss = criterion(logits, y_tensor[indices].to(device))
            loss.backward()
            optimiser.step()
        select_probability = predict_feature_model(
            model, select_blocks, device
        )
        select_macro = BASE.macro_ovr(
            y_select, select_probability, classes
        )[0]
        if select_macro > best_macro:
            best_macro = select_macro
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("FS2-K3 failed to retain a SELECT checkpoint")
    model.load_state_dict(best_state)
    return (
        predict_feature_model(model, select_blocks, device),
        predict_feature_model(model, test_blocks, device),
        {"best_select_macro": best_macro, "best_epoch": best_epoch},
    )


def paper_fs2(
    fit_blocks: list[np.ndarray],
    y_fit: np.ndarray,
    select_blocks: list[np.ndarray],
    y_select: np.ndarray,
    test_blocks: list[np.ndarray],
    n_classes: int,
    proj_dim: int,
    classes: list[str],
    view: str,
    fold: int,
    device: str,
    smoke: bool,
    resume: bool,
    log,
) -> tuple[np.ndarray, np.ndarray, dict]:
    seeds = (SEED,) if smoke else SEEDS
    epochs = 1 if smoke else FS_EPOCHS
    select_predictions = []
    test_predictions = []
    seed_records = {}
    for seed in seeds:
        array_path, meta_path = seed_cache_paths(
            view, fold, "FS2_K3", seed, smoke
        )
        if resume and _reusable(array_path, meta_path, log):
            with np.load(array_path) as saved:
                select_probability = saved["select"]
                test_probability = saved["test"]
            record = json.loads(meta_path.read_text(encoding="utf-8"))
            record.pop("cache_key", None)                           # not part of the result
            log(f"    FS2-K3 seed {seed}: resumed")
        else:
            started = time.time()
            select_probability, test_probability, record = train_fs2_seed(
                fit_blocks,
                y_fit,
                select_blocks,
                y_select,
                test_blocks,
                n_classes,
                proj_dim,
                seed,
                device,
                epochs,
                classes,
            )
            record.update(
                {
                    "seed": seed,
                    "epochs": epochs,
                    "device": device,
                    "paper_aligned": not smoke,
                }
            )
            np.savez_compressed(
                array_path,
                select=select_probability,
                test=test_probability,
            )
            meta_path.write_text(
                json.dumps({**record, "cache_key": CACHE_KEY}, indent=2), encoding="utf-8"
            )
            log(
                f"    FS2-K3 seed {seed}: "
                f"best epoch {record['best_epoch']}, "
                f"{time.time() - started:.0f}s"
            )
        select_predictions.append(select_probability)
        test_predictions.append(test_probability)
        seed_records[str(seed)] = record
    return (
        np.mean(select_predictions, axis=0),
        np.mean(test_predictions, axis=0),
        {
            "seeds": list(seeds),
            "epochs": epochs,
            "proj_dim": proj_dim,
            "hidden": list(FS_HIDDEN),
            "dropout": FS_DROPOUT,
            "expert_dropout": FS_EXPERT_DROPOUT,
            "lr": LEARNING_RATE,
            "selection": "best SELECT macro OvR checkpoint per seed",
            "per_seed": seed_records,
        },
    )


def paper_ps4(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_select: np.ndarray,
    x_test: np.ndarray,
    classes: list[str],
    smoke: bool,
    log,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Full paper XGBoost grid with the necessary multiclass metric."""
    n_classes = len(classes)
    grid = (
        [
            {
                "n_estimators": 50,
                "max_depth": 2,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
        ]
        if smoke
        else list(ParameterGrid(XGB_GRID))
    )
    n_splits = 2 if smoke else 5
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=SEED
    )
    best_score = -np.inf
    best_parameters = None
    started = time.time()
    balanced = sample_weights(y_fit, n_classes)
    for index, parameters in enumerate(grid):
        scores = []
        for train_index, validation_index in splitter.split(x_fit, y_fit):
            model = XGBClassifier(
                **parameters,
                objective="multi:softprob",
                num_class=n_classes,
                random_state=SEED,
                eval_metric="mlogloss",
                verbosity=0,
                n_jobs=1,
            )
            model.fit(
                np.asarray(x_fit[train_index], dtype=np.float32),
                y_fit[train_index],
                sample_weight=balanced[train_index],
            )
            probability = model.predict_proba(
                np.asarray(x_fit[validation_index], dtype=np.float32)
            )
            scores.append(
                BASE.macro_ovr(
                    y_fit[validation_index], probability, classes
                )[0]
            )
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_parameters = dict(parameters)
        if (index + 1) % 12 == 0 or index + 1 == len(grid):
            log(
                f"    PS4-K3 CV {index + 1}/{len(grid)}; "
                f"best {best_score:.4f}; "
                f"{time.time() - started:.0f}s"
            )
    if best_parameters is None:
        raise RuntimeError("PS4-K3 grid search produced no winner")
    final = XGBClassifier(
        **best_parameters,
        objective="multi:softprob",
        num_class=n_classes,
        random_state=SEED,
        eval_metric="mlogloss",
        verbosity=0,
        n_jobs=1,
    )
    final.fit(
        np.asarray(x_fit, dtype=np.float32),
        y_fit,
        sample_weight=balanced,
    )
    return (
        final.predict_proba(np.asarray(x_select, dtype=np.float32)),
        final.predict_proba(np.asarray(x_test, dtype=np.float32)),
        {
            "best_parameters": best_parameters,
            "in_fit_cv_macro_ovr": best_score,
            "grid_size": len(grid),
            "cv_folds": n_splits,
            "selection_split": "FIT only",
            "multiclass_extension": (
                "balanced sample weights and macro one-vs-rest AUROC"
            ),
        },
    )


def method_cache_paths(
    view: str, fold: int, method: str, smoke: bool
) -> tuple[Path, Path]:
    scope = "SMOKE" if smoke else "FULL"
    stem = f"NATIVE_K_ALIGNED_{scope}_{view}_F{fold}_{method}"
    return RESULTS / f"{stem}.npz", RESULTS / f"{stem}.json"


def cached_method(
    view: str,
    fold: int,
    method: str,
    smoke: bool,
    resume: bool,
    producer,
    log,
) -> tuple[np.ndarray, np.ndarray, dict]:
    array_path, meta_path = method_cache_paths(
        view, fold, method, smoke
    )
    if resume and _reusable(array_path, meta_path, log):
        with np.load(array_path) as saved:
            select_probability = saved["select"]
            test_probability = saved["test"]
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata.pop("cache_key", None)                             # not part of the result
        log(f"  {method}: resumed")
        return select_probability, test_probability, metadata
    started = time.time()
    select_probability, test_probability, metadata = producer()
    np.savez_compressed(
        array_path, select=select_probability, test=test_probability
    )
    metadata = {**metadata, "elapsed_seconds": time.time() - started}
    meta_path.write_text(
        json.dumps({**metadata, "cache_key": CACHE_KEY}, indent=2), encoding="utf-8"
    )
    log(f"  {method}: completed in {metadata['elapsed_seconds']:.0f}s")
    return select_probability, test_probability, metadata


def head_cache_paths(
    view: str, fold: int, smoke: bool
) -> tuple[Path, Path]:
    scope = "SMOKE" if smoke else "FULL"
    stem = f"NATIVE_K_ALIGNED_{scope}_{view}_F{fold}_HEADS"
    return RESULTS / f"{stem}.npz", RESULTS / f"{stem}.json"


def get_heads(
    view: str,
    fold: int,
    experts: list[str],
    classes: list[str],
    fit_blocks: list[np.ndarray],
    select_blocks: list[np.ndarray],
    test_blocks: list[np.ndarray],
    y_fit: np.ndarray,
    y_select: np.ndarray,
    y_test: np.ndarray,
    inner_assignment: np.ndarray,
    smoke: bool,
    resume: bool,
    log,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, list[float]]:
    array_path, meta_path = head_cache_paths(view, fold, smoke)
    if resume and _reusable(array_path, meta_path, log):
        with np.load(array_path) as saved:
            fit_logits = saved["fit_logits"]
            select_logits = saved["select_logits"]
            test_logits = saved["test_logits"]
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        log("  genuine K-class heads: resumed")
        return (
            fit_logits,
            select_logits,
            test_logits,
            metadata["quality"],
            metadata["temperatures"],
        )
    started = time.time()
    output = BASE.build_native_logits(
        experts,
        fit_blocks,
        select_blocks,
        test_blocks,
        y_fit,
        y_select,
        y_test,
        inner_assignment,
        classes,
        log,
    )
    fit_logits, select_logits, test_logits, quality, temperatures = output
    np.savez_compressed(
        array_path,
        fit_logits=fit_logits,
        select_logits=select_logits,
        test_logits=test_logits,
    )
    meta_path.write_text(
        json.dumps(
            {
                "quality": quality,
                "temperatures": temperatures,
                "elapsed_seconds": time.time() - started,
                "head_specification": (
                    "standardized multinomial heads; "
                    "not a paper composer label"
                ),
                "cache_key": CACHE_KEY,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def rows_for_fold(
    rows: list[dict],
    preflight: dict,
    fold: int,
    smoke: bool,
) -> tuple[list[dict], list[dict], list[dict], np.ndarray]:
    if smoke:
        return BASE.prepare_smoke_split(
            rows, preflight["outer_folds"]["0"]
        )
    fit_rows = [
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
    lookup = {
        int(item["idx"]): item
        for item in preflight["outer_folds"][str(fold)]["assignments"]
    }
    fit_rows = [
        {
            **row,
            "content_group": lookup[row["idx"]]["content_group"],
        }
        for row in fit_rows
    ]
    ordered = sorted(fit_rows, key=lambda row: row["idx"])
    inner = np.asarray(
        [lookup[row["idx"]]["inner_fold"] for row in ordered],
        dtype=np.int16,
    )
    return fit_rows, select_rows, test_rows, inner


def record_method(
    name: str,
    y_select: np.ndarray,
    y_test: np.ndarray,
    select_probability: np.ndarray,
    test_probability: np.ndarray,
    classes: list[str],
    metadata: dict,
) -> dict:
    return {
        "name": name,
        "select": evaluate(y_select, select_probability, classes),
        "test": evaluate(y_test, test_probability, classes),
        "metadata": metadata,
    }


def run_fold(
    view: str,
    fold: int,
    experts: list[str],
    classes: list[str],
    fit_rows: list[dict],
    select_rows: list[dict],
    test_rows: list[dict],
    inner_assignment: np.ndarray,
    device: str,
    smoke: bool,
    resume: bool,
    log,
) -> tuple[dict, dict[str, np.ndarray]]:
    fit_blocks, y_fit, _ = BASE.materialize(experts, fit_rows, classes)
    select_blocks, y_select, _ = BASE.materialize(
        experts, select_rows, classes
    )
    test_blocks, y_test, _ = BASE.materialize(experts, test_rows, classes)
    if len(inner_assignment) != len(y_fit):
        raise RuntimeError("inner assignment count does not match FIT")
    log(
        f"\n=== {'SMOKE' if smoke else 'outer'} fold {fold}: "
        f"FIT {len(y_fit):,} | SELECT {len(y_select):,} | "
        f"TEST {len(y_test):,} ==="
    )

    (
        fit_logits,
        select_logits,
        test_logits,
        head_quality,
        temperatures,
    ) = get_heads(
        view,
        fold,
        experts,
        classes,
        fit_blocks,
        select_blocks,
        test_blocks,
        y_fit,
        y_select,
        y_test,
        inner_assignment,
        smoke,
        resume,
        log,
    )
    x_fit_logits = fit_logits.reshape(len(fit_logits), -1)
    x_select_logits = select_logits.reshape(len(select_logits), -1)
    x_test_logits = test_logits.reshape(len(test_logits), -1)
    feature_fit = np.concatenate(fit_blocks, axis=1)
    feature_select = np.concatenate(select_blocks, axis=1)
    feature_test = np.concatenate(test_blocks, axis=1)
    methods = {}
    probabilities = {}

    select_probability = torch.softmax(
        torch.as_tensor(select_logits), dim=2
    ).numpy().mean(axis=1)
    test_probability = torch.softmax(
        torch.as_tensor(test_logits), dim=2
    ).numpy().mean(axis=1)
    methods["PS1-K3"] = record_method(
        "PS1-K3",
        y_select,
        y_test,
        select_probability,
        test_probability,
        classes,
        {"paper_recipe": True, "selection": "none"},
    )
    probabilities["PS1-K3"] = test_probability

    def produce_ps2():
        predictions, diagnostics = unscaled_logreg(
            x_fit_logits,
            y_fit,
            [x_select_logits, x_test_logits],
            max_iter=PS2_MAX_ITER,
        )
        return predictions[0], predictions[1], diagnostics

    select_probability, test_probability, metadata = cached_method(
        view,
        fold,
        "PS2_K3",
        smoke,
        resume,
        produce_ps2,
        log,
    )
    methods["PS2-K3"] = record_method(
        "PS2-K3",
        y_select,
        y_test,
        select_probability,
        test_probability,
        classes,
        metadata,
    )
    probabilities["PS2-K3"] = test_probability

    select_probability, test_probability, metadata = cached_method(
        view,
        fold,
        "PS3_K3",
        smoke,
        resume,
        lambda: paper_ps3(
            x_fit_logits,
            y_fit,
            x_select_logits,
            x_test_logits,
            len(classes),
            view,
            fold,
            device,
            smoke,
            resume,
            log,
        ),
        log,
    )
    methods["PS3-K3"] = record_method(
        "PS3-K3",
        y_select,
        y_test,
        select_probability,
        test_probability,
        classes,
        metadata,
    )
    probabilities["PS3-K3"] = test_probability

    select_probability, test_probability, metadata = cached_method(
        view,
        fold,
        "PS4_K3",
        smoke,
        resume,
        lambda: paper_ps4(
            x_fit_logits,
            y_fit,
            x_select_logits,
            x_test_logits,
            classes,
            smoke,
            log,
        ),
        log,
    )
    methods["PS4-K3"] = record_method(
        "PS4-K3",
        y_select,
        y_test,
        select_probability,
        test_probability,
        classes,
        metadata,
    )
    probabilities["PS4-K3"] = test_probability

    def produce_fs1():
        predictions, diagnostics = unscaled_logreg(
            feature_fit,
            y_fit,
            [feature_select, feature_test],
            max_iter=FS1_MAX_ITER,
        )
        return predictions[0], predictions[1], diagnostics

    select_probability, test_probability, metadata = cached_method(
        view,
        fold,
        "FS1_K3",
        smoke,
        resume,
        produce_fs1,
        log,
    )
    methods["FS1-K3"] = record_method(
        "FS1-K3",
        y_select,
        y_test,
        select_probability,
        test_probability,
        classes,
        metadata,
    )
    probabilities["FS1-K3"] = test_probability

    select_probability, test_probability, metadata = cached_method(
        view,
        fold,
        "FS2_K3",
        smoke,
        resume,
        lambda: paper_fs2(
            fit_blocks,
            y_fit,
            select_blocks,
            y_select,
            test_blocks,
            len(classes),
            PROJ_DIM[view],
            classes,
            view,
            fold,
            device,
            smoke,
            resume,
            log,
        ),
        log,
    )
    methods["FS2-K3"] = record_method(
        "FS2-K3",
        y_select,
        y_test,
        select_probability,
        test_probability,
        classes,
        metadata,
    )
    probabilities["FS2-K3"] = test_probability

    def produce_fsb():
        width = len(experts) * len(classes)
        pca = PCA(n_components=width, random_state=SEED)
        pca_fit = pca.fit_transform(feature_fit)
        predictions, diagnostics = unscaled_logreg(
            pca_fit,
            y_fit,
            [pca.transform(feature_select), pca.transform(feature_test)],
            max_iter=FS1_MAX_ITER,
        )
        diagnostics.update(
            {
                "pca_width": width,
                "pca_fit_split": "outer FIT only",
                "explained_variance_ratio": float(
                    pca.explained_variance_ratio_.sum()
                ),
            }
        )
        return predictions[0], predictions[1], diagnostics

    select_probability, test_probability, metadata = cached_method(
        view,
        fold,
        f"FSB_K3_{len(experts) * len(classes)}",
        smoke,
        resume,
        produce_fsb,
        log,
    )
    methods[f"FSB-K3-{len(experts) * len(classes)}"] = record_method(
        f"FSB-K3-{len(experts) * len(classes)}",
        y_select,
        y_test,
        select_probability,
        test_probability,
        classes,
        metadata,
    )
    probabilities[f"FSB-K3-{len(experts) * len(classes)}"] = (
        test_probability
    )

    def produce_ps_plus():
        maps = BASE.build_feature_maps(
            fit_logits,
            select_logits,
            test_logits,
            temperatures,
            smoke,
        )
        calibrated_select = torch.softmax(
            torch.as_tensor(
                select_logits
                / np.asarray(temperatures)[None, :, None]
            ),
            dim=2,
        ).numpy()
        calibrated_test = torch.softmax(
            torch.as_tensor(
                test_logits / np.asarray(temperatures)[None, :, None]
            ),
            dim=2,
        ).numpy()
        best, all_scores, best_test = BASE.run_ps_ladder(
            maps,
            y_fit,
            y_select,
            y_test,
            classes,
            (
                calibrated_select.mean(axis=1),
                calibrated_test.mean(axis=1),
            ),
            smoke,
        )
        # The cache contract requires SELECT probabilities. Refit/load the
        # selected rung would duplicate the ladder; its SELECT score is
        # already persisted in metadata, and a NaN placeholder would violate
        # validation. Use PS1-K3 SELECT probabilities only as a cache carrier;
        # the method record below is built directly from `best`.
        return (
            calibrated_select.mean(axis=1),
            best_test,
            {
                "selected": best,
                "all_scores": all_scores,
                "ladder": "extended diagnostic, not PS1-PS4",
                "selection_split": "SELECT",
                "select_probability_carrier": (
                    "calibrated mean only; selected SELECT score is metadata"
                ),
            },
        )

    plus_select_probability, plus_test_probability, plus_metadata = (
        cached_method(
            view,
            fold,
            "PSPLUS_K3",
            smoke,
            resume,
            produce_ps_plus,
            log,
        )
    )
    selected_plus = plus_metadata["selected"]
    plus_test_eval = evaluate(
        y_test, plus_test_probability, classes
    )
    methods["PS+-K3"] = {
        "name": "PS+-K3",
        "select": {
            "macro": selected_plus["select"],
            "per_class": None,
        },
        "test": plus_test_eval,
        "metadata": plus_metadata,
    }
    probabilities["PS+-K3"] = plus_test_probability

    core_names = ("PS1-K3", "PS2-K3", "PS3-K3", "PS4-K3")
    select_core = max(
        core_names, key=lambda name: methods[name]["select"]["macro"]
    )
    test_core = max(
        core_names, key=lambda name: methods[name]["test"]["macro"]
    )
    fs1_score = methods["FS1-K3"]["test"]["macro"]
    fs2_score = methods["FS2-K3"]["test"]["macro"]
    entry = {
        "fold": fold,
        "smoke_internal_fit_only": smoke,
        "n_fit": len(y_fit),
        "n_select": len(y_select),
        "n_test": len(y_test),
        "methods": methods,
        "head_quality": head_quality,
        "core_ps_select_chosen": {
            "method": select_core,
            "test": methods[select_core]["test"]["macro"],
        },
        "core_ps_test_max_diagnostic": {
            "method": test_core,
            "test": methods[test_core]["test"]["macro"],
            "non_deployable": True,
        },
        "contrasts": {
            "FS1_minus_select_chosen_core_PS": (
                fs1_score - methods[select_core]["test"]["macro"]
            ),
            "FS2_minus_select_chosen_core_PS": (
                fs2_score - methods[select_core]["test"]["macro"]
            ),
            "FS1_minus_PS_plus": (
                fs1_score - methods["PS+-K3"]["test"]["macro"]
            ),
            "FS2_minus_PS_plus": (
                fs2_score - methods["PS+-K3"]["test"]["macro"]
            ),
        },
        "validity": {
            "paper_labels_follow_registry": True,
            "composer_inputs_unscaled_for_PS2_FS1_FSB": True,
            "PS4_grid_size": methods["PS4-K3"]["metadata"]["grid_size"],
            "PS3_seed_count": len(
                methods["PS3-K3"]["metadata"]["seeds"]
            ),
            "FS2_seed_count": len(
                methods["FS2-K3"]["metadata"]["seeds"]
            ),
            "selection_uses_test": False,
        },
    }
    arrays = {"y": y_test, **probabilities}
    return entry, arrays


def aggregate(
    view: str,
    manifest: dict,
    preflight: dict,
    entries: list[dict],
    arrays: list[dict[str, np.ndarray]],
    smoke: bool,
) -> dict:
    method_names = list(entries[0]["methods"])
    summary = {}
    y_oof = np.concatenate([item["y"] for item in arrays])
    for method in method_names:
        fold_scores = [
            entry["methods"][method]["test"]["macro"] for entry in entries
        ]
        probability = np.vstack([item[method] for item in arrays])
        pooled = evaluate(y_oof, probability, manifest["classes"])
        summary[method] = {
            "mean_fold_macro_ovr": float(np.mean(fold_scores)),
            "per_fold_macro_ovr": fold_scores,
            "pooled_oof": pooled,
        }
    contrasts = {}
    for name in entries[0]["contrasts"]:
        values = [entry["contrasts"][name] for entry in entries]
        contrasts[name] = {
            "mean": float(np.mean(values)),
            "per_fold": values,
            "positive_folds": int(np.sum(np.asarray(values) > 0)),
            "n_folds": len(values),
        }
    return {
        "status": (
            "SMOKE_INTERNAL_FIT_ONLY_DO_NOT_REPORT"
            if smoke
            else "COMPLETE"
        ),
        "view": view,
        "naming_authority": "camera_ready/scripts/analyses/README.md#three-class-protocol",
        "design_authority": "camera_ready/scripts/analyses/README.md#three-class-protocol",
        "head_authority": "camera_ready/scripts/analyses/README.md#three-class-protocol",
        "manifest_sha256": manifest["manifest_sha256"],
        "inner_assignment_sha256": preflight["assignment_sha256"],
        "scope": (
            "genuine K-class heads on frozen binary-trained backbones; "
            "not end-to-end multiclass experts"
        ),
        "classes": manifest["classes"],
        "experts": manifest["pool"],
        "folds": entries,
        "summary": summary,
        "contrasts": contrasts,
    }


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--view", choices=("B_arch6", "A_df40"), default="B_arch6"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    arguments = parser.parse_args()
    if arguments.smoke and arguments.resume:
        raise SystemExit("--smoke and --resume cannot be combined")
    if arguments.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = arguments.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    manifest, preflight, _ = BASE.load_inputs(arguments.view)
    CACHE_KEY.update({"view": arguments.view, "manifest_sha256": manifest["manifest_sha256"],
                      "inner_assignment_sha256": preflight["assignment_sha256"], "device": device,
                      "smoke": bool(arguments.smoke)})
    rows = manifest["manifest"]
    experts = manifest["pool"]
    classes = manifest["classes"]
    n_folds = 1 if arguments.smoke else manifest["n_folds"]
    print(
        f"paper-aligned K={len(classes)} comparison; view "
        f"{arguments.view}; device {device}; folds {n_folds}"
    )
    print(
        "labels: PS1-K3/PS2-K3/PS3-K3/PS4-K3, "
        "FS1-K3/FS2-K3, FSB and PS+"
    )
    entries = []
    arrays = []
    for fold in range(n_folds):
        fit_rows, select_rows, test_rows, inner = rows_for_fold(
            rows, preflight, fold, arguments.smoke
        )
        entry, fold_arrays = run_fold(
            arguments.view,
            fold,
            experts,
            classes,
            fit_rows,
            select_rows,
            test_rows,
            inner,
            device,
            arguments.smoke,
            arguments.resume,
            print,
        )
        scope = "SMOKE" if arguments.smoke else "FULL"
        fold_path = RESULTS / (
            f"NATIVE_K_ALIGNED_{scope}_{arguments.view}_F{fold}_RESULT.json"
        )
        oof_path = RESULTS / (
            f"NATIVE_K_ALIGNED_{scope}_{arguments.view}_F{fold}_OOF.npz"
        )
        fold_path.write_text(
            json.dumps(entry, indent=2), encoding="utf-8"
        )
        np.savez_compressed(oof_path, **fold_arrays)
        entries.append(entry)
        arrays.append(fold_arrays)

    result = aggregate(
        arguments.view,
        manifest,
        preflight,
        entries,
        arrays,
        arguments.smoke,
    )
    scope = "SMOKE" if arguments.smoke else "FULL"
    destination = RESULTS / (
        f"NATIVE_K_ALIGNED_{scope}_{arguments.view}_RESULT.json"
    )
    destination.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 72)
    for method, record in result["summary"].items():
        print(f"{method:<14} {record['mean_fold_macro_ovr']:.4f}")
    for name, record in result["contrasts"].items():
        print(
            f"{name:<40} {record['mean']:+.4f} "
            f"({record['positive_folds']}/{record['n_folds']})"
        )
    print(f"wrote {destination.name}")


if __name__ == "__main__":
    main()
