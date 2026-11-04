"""Shared feature-space training and evaluation runtime.

Provides the canonical implementation for training FeatureAssembly models
and evaluating them on test generators. Used by scripts 03, 06, 07, 10.

All hyperparameters default to values from experiments.yaml via config module.
"""

import numpy as np
import torch
import torch.nn as nn

from src.models.feature_space import FeatureAssembly
from src.utils.seed import set_seed
from src.utils.metrics import compute_auc
from src.utils.config import (
    FS_EPOCHS, FS_HIDDEN_DIMS, FS_DROPOUT, FS_EXPERT_DROPOUT,
    FS_LR, FS_WEIGHT_DECAY, FS_BATCH_SIZE,
)


def _class_weights(labels: np.ndarray):
    """Inverse-frequency class weights (mean weight == 1).

    Same convention as `prediction_space._train_mlp` so FS and PS MLP
    ladders share class-balancing. Returns None if one class is empty.
    """
    n_real = int((labels == 0).sum())
    n_fake = int((labels == 1).sum())
    if n_real == 0 or n_fake == 0:
        return None
    inv_real = len(labels) / (2.0 * n_real)
    inv_fake = len(labels) / (2.0 * n_fake)
    return torch.tensor([inv_real, inv_fake], dtype=torch.float32)


def _to_tensors(features_list):
    """Convert list of numpy arrays or tensors to tensors (zero-copy when possible)."""
    out = []
    for f in features_list:
        if isinstance(f, torch.Tensor):
            out.append(f)
        elif f.dtype == np.float32 and f.flags['C_CONTIGUOUS']:
            out.append(torch.from_numpy(f))
        else:
            out.append(torch.tensor(f, dtype=torch.float32))
    return out


@torch.no_grad()
def predict_fs(model, features_list, device="cpu", batch_size=2048):
    """Run FS model on per-expert features, return [N] P(fake) scores."""
    tensors = _to_tensors(features_list)
    N = tensors[0].shape[0]
    all_scores = []
    for start in range(0, N, batch_size):
        batch = [t[start:start + batch_size].to(device) for t in tensors]
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_scores.append(probs.cpu().numpy())
    return np.concatenate(all_scores)


def train_fs_model(
    train_feats, train_labels, feature_dims, proj_dim,
    *,
    val_feats=None, val_labels=None,
    seed=1024, device="cpu",
    classifier_type="mlp",
    epochs=None, batch_size=None, lr=None,
    weight_decay=None, expert_dropout=None,
    hidden_dims=None, dropout=None,
    log=None,
):
    """Train a FeatureAssembly model.

    Hyperparameters default to experiments.yaml. Returns
    (model, best_val_auc) when val_feats is given (best-SELECT
    checkpoint), else (model, None).

    Determinism: `set_seed(seed)` enables strict PyTorch determinism;
    `CUBLAS_WORKSPACE_CONFIG` must already be in os.environ before the
    first cuBLAS handle (handled by `scripts/_bootstrap.py`). Same
    torch/CUDA/GPU env => bit-identical FS outputs.
    """
    # Apply config defaults for any unset params
    if epochs is None:
        epochs = FS_EPOCHS
    if batch_size is None:
        batch_size = FS_BATCH_SIZE
    if lr is None:
        lr = FS_LR
    if weight_decay is None:
        weight_decay = FS_WEIGHT_DECAY
    if expert_dropout is None:
        expert_dropout = FS_EXPERT_DROPOUT
    if hidden_dims is None:
        hidden_dims = FS_HIDDEN_DIMS
    if dropout is None:
        dropout = FS_DROPOUT

    set_seed(seed)
    model = FeatureAssembly(
        feature_dims, proj_dim=proj_dim,
        classifier_type=classifier_type,
        expert_dropout=expert_dropout,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    # Class-balanced CE (no WeightedRandomSampler). Same convention as
    # src.models.prediction_space._train_mlp so FS and PS ladders use
    # identical balancing.
    class_w = _class_weights(train_labels)
    if class_w is not None:
        class_w = class_w.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    train_tensors = _to_tensors(train_feats)
    train_labels_t = torch.from_numpy(
        train_labels if train_labels.dtype == np.int64
        else train_labels.astype(np.int64))
    n_samples = len(train_labels_t)

    use_validation = val_feats is not None and val_labels is not None
    val_tensors = _to_tensors(val_feats) if use_validation else None

    best_val_auc = -1.0 if use_validation else None
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        # Uniform permutation over FIT. Class balance comes from the CE
        # weight, not from oversampling. Every sample is seen exactly once
        # per epoch.
        indices = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            batch_idx_t = indices[start:start + batch_size]
            batch_feats = [t[batch_idx_t].to(device)
                           for t in train_tensors]
            batch_labels = train_labels_t[batch_idx_t].to(device)

            optimiser.zero_grad()
            logits = model(batch_feats)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if use_validation:
            model.eval()
            val_scores = predict_fs(model, val_tensors, device)
            val_auc = compute_auc(val_labels, val_scores)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

            if log:
                log.info(f"    Epoch {epoch+1:3d}/{epochs}: "
                         f"loss={avg_loss:.4f}, val_auc={val_auc:.4f}"
                         f"{' *' if val_auc >= best_val_auc else ''}")
        elif log:
            log.info(f"    Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f}")

    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val_auc


