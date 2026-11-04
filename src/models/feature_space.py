"""Feature-Space methods (Section 3.3).

Composition interfaces that operate on expert penultimate features.

Methods:
    FS_1 (FS_1): deterministic LogReg on concatenated
        raw features. No learned projection, no hidden layers. The
        "interface, not capacity" anchor used throughout the paper.
    FeatureAssembly: per-expert projection + MLP/linear classifier (FS_2,
        FS_A). Trained via src/training/fs_runtime.py.
    RandomProjectionAssembly: fixed Gaussian projection + MLP (FS_C).

FeatureAssembly architecture:
    Per expert i: Linear(D_i, d) -> LayerNorm(d) -> ReLU
    Concatenation: [B, N*d]
    Classifier (MLP): Linear(N*d, 512) -> ReLU -> Dropout -> Linear(512, 256)
                       -> ReLU -> Dropout -> Linear(256, 2)
    Classifier (linear): Linear(N*d, 2)
    Expert dropout: per-sample Bernoulli with inverted scaling (p=0.25
        by default). At least one expert is guaranteed active per sample.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from typing import List, Optional

from src.utils.config import DS_CONFIG as _DS_CFG


def apply_expert_dropout(
    projected: List[torch.Tensor],
    expert_dropout: float,
    training: bool,
) -> List[torch.Tensor]:
    """Per-sample expert dropout with inverted scaling.

    Each sample draws its own Bernoulli mask over experts, with at least
    one expert guaranteed active. Masked positions zeroed; remaining
    positions scaled by n_experts / n_active_per_sample. Mutates and
    returns `projected` in place. No-op when not `training` or
    `expert_dropout <= 0`.
    """
    if not training or expert_dropout <= 0.0 or not projected:
        return projected

    n_experts = len(projected)
    B = projected[0].shape[0]
    device = projected[0].device

    keep = (torch.rand(n_experts, B, device=device) > expert_dropout).float()
    # Guarantee at least one expert active per sample.
    all_dropped = (keep.sum(dim=0) == 0)
    if all_dropped.any():
        rand_expert = torch.randint(0, n_experts, (B,), device=device)
        for col in torch.where(all_dropped)[0]:
            keep[rand_expert[col], col] = 1.0
    n_active = keep.sum(dim=0, keepdim=True)             # [1, B]
    scale = n_experts / n_active.clamp(min=1.0)          # [1, B]
    scale_col = scale.squeeze(0).unsqueeze(1)            # [B, 1]
    for i in range(n_experts):
        mask_i = keep[i].unsqueeze(1)                    # [B, 1]
        projected[i] = projected[i] * mask_i * scale_col
    return projected


class FeatureAssembly(nn.Module):
    """Feature-space assembly of frozen heterogeneous experts."""

    def __init__(
        self,
        feature_dims: List[int],
        proj_dim: int = 64,
        num_classes: int = 2,
        classifier_type: str = "mlp",
        expert_dropout: float = 0.25,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            feature_dims: Per-expert feature dimensions, e.g. [1024, 1024, 1024, 1024].
            proj_dim: Common projection dimension d.
            num_classes: Output classes (2 for binary).
            classifier_type: "mlp" or "linear".
            expert_dropout: Probability of zeroing an expert per sample during training.
            hidden_dims: MLP hidden layer sizes. Default [512, 256].
            dropout: Dropout rate in MLP classifier.
        """
        super().__init__()
        self.n_experts = len(feature_dims)
        self.proj_dim = proj_dim
        self.expert_dropout = expert_dropout
        if hidden_dims is None:
            from src.utils.config import FS_HIDDEN_DIMS
            hidden_dims = FS_HIDDEN_DIMS

        # Per-expert projection layers
        self.projections = nn.ModuleList()
        for dim in feature_dims:
            self.projections.append(nn.Sequential(
                nn.Linear(dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.ReLU(),
            ))

        cat_dim = self.n_experts * proj_dim

        # Classifier
        if classifier_type == "linear":
            self.classifier = nn.Linear(cat_dim, num_classes)
        else:
            layers = []
            in_dim = cat_dim
            for h in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                in_dim = h
            layers.append(nn.Linear(in_dim, num_classes))
            self.classifier = nn.Sequential(*layers)

    def forward(self, expert_features: List[torch.Tensor]) -> torch.Tensor:
        """Project each expert -> concat -> classify. Per-sample expert
        dropout is shared with RandomProjectionAssembly via the module
        helper.
        """
        projected = [proj(feat) for proj, feat in zip(self.projections, expert_features)]
        projected = apply_expert_dropout(
            projected, self.expert_dropout, self.training)
        z = torch.cat(projected, dim=1)  # [B, N*d]
        return self.classifier(z)


class RandomProjectionAssembly(nn.Module):
    """Fixed random projections (not learned) + trainable classifier.

    Table 3 ablation: tests whether *random* projection captures useful
    structure. The per-expert stack is Linear -> LayerNorm -> ReLU, matching
    FeatureAssembly, so the random-vs-learned comparison varies only in the
    choice of the linear map (random vs learned). Only the LayerNorm affine
    params and the classifier are trainable; the linear projection weight is
    frozen.

    Per-sample expert dropout is also wired so random-proj and learned-proj
    are trained under the same regulariser.
    """

    def __init__(
        self,
        feature_dims: List[int],
        proj_dim: int = 64,
        num_classes: int = 2,
        seed: int = 42,
        expert_dropout: Optional[float] = None,
    ):
        super().__init__()
        self.n_experts = len(feature_dims)
        self.proj_dim = proj_dim

        # Default to the same expert dropout as FeatureAssembly so the Table
        # 3a comparison is apples-to-apples.
        if expert_dropout is None:
            from src.utils.config import FS_EXPERT_DROPOUT
            expert_dropout = FS_EXPERT_DROPOUT
        self.expert_dropout = expert_dropout

        rng = torch.Generator().manual_seed(seed)
        self.projections = nn.ModuleList()
        for dim in feature_dims:
            proj = nn.Linear(dim, proj_dim, bias=False)
            nn.init.normal_(proj.weight, std=1.0 / (dim ** 0.5),
                            generator=rng)
            # Freeze the linear weights; LN affine stays trainable.
            proj.weight.requires_grad_(False)
            stack = nn.Sequential(
                proj,
                nn.LayerNorm(proj_dim),
                nn.ReLU(),
            )
            self.projections.append(stack)

        from src.utils.config import FS_HIDDEN_DIMS, FS_DROPOUT
        cat_dim = self.n_experts * proj_dim
        layers = []
        in_dim = cat_dim
        for h in FS_HIDDEN_DIMS:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(FS_DROPOUT)])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, expert_features: List[torch.Tensor]) -> torch.Tensor:
        projected = [proj(feat) for proj, feat in zip(self.projections, expert_features)]
        projected = apply_expert_dropout(
            projected, self.expert_dropout, self.training)
        z = torch.cat(projected, dim=1)
        return self.classifier(z)


# ── FS_1: Raw-feature LogReg (Section 4.2 headline anchor) ──────────────

# FS_1 memory budget. sklearn LogisticRegression(solver="lbfgs") upcasts
# the input feature matrix to float64 internally regardless of input
# dtype, so an N x D matrix occupies N * D * 8 bytes plus additional
# workspace. On 16 GiB-RAM hosts this OOMs for the full21 pool
# (N=26 591, D=33 792 -> 6.7 GiB input + ~5-7 GiB workspace). On
# >=24 GiB-RAM hosts the same fit completes normally. The budget is
# therefore made adaptive to the host's total physical RAM, using
# stdlib only (no psutil dependency). Set environment variable
# ``ALLOW_FS1_OOM=1`` to bypass the check entirely (the host will OOM
# if it does not actually have the memory). Set ``FS1_BUDGET_GIB``
# to a positive number to override the auto-detected budget.
FS_1_BUDGET_FRACTION = 0.40   # use up to 40% of total RAM for FS_1 input
FS_1_FALLBACK_BUDGET_GIB = 6.0  # if RAM detection fails


def _total_ram_gib() -> Optional[float]:
    """Return total physical RAM in GiB using stdlib only. Returns
    None if detection fails on this platform.

    Per-platform detection:
    - Windows: ctypes ``GlobalMemoryStatusEx`` (kernel32).
    - Linux:   /proc/meminfo MemTotal line.
    - macOS:   ``sysctl -n hw.memsize`` via subprocess.
    """
    import sys
    try:
        if sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(stat)
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(stat))
            if not ok:
                return None
            return stat.ullTotalPhys / (1024 ** 3)
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kib = int(line.split()[1])
                        return kib * 1024 / (1024 ** 3)
            return None
        if sys.platform == "darwin":
            import subprocess
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True).strip()
            return int(out) / (1024 ** 3)
    except Exception:
        return None
    return None


def fs1_budget_gib() -> float:
    """Return the FS_1 memory budget in GiB for the current host.

    Resolution order:
      1. ``FS1_BUDGET_GIB`` environment variable (explicit override).
      2. Auto-detected: ``FS_1_BUDGET_FRACTION * total_RAM_GiB``.
      3. ``FS_1_FALLBACK_BUDGET_GIB`` if detection fails.
    """
    import os
    env_val = os.environ.get("FS1_BUDGET_GIB", "").strip()
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass
    total = _total_ram_gib()
    if total is None:
        return FS_1_FALLBACK_BUDGET_GIB
    return total * FS_1_BUDGET_FRACTION


def fs1_memory_estimate_gib(n_samples: int, n_features: int) -> float:
    """Return the estimated GiB sklearn lbfgs would allocate for the
    input matrix of an N x D raw-feat LogReg fit. Used by callers to
    gate FS_1 on memory-tight hosts. The estimate is the input array
    in float64 (N * D * 8 bytes); peak workspace is roughly 1.5-2 x
    this, which is why the budget is a fraction of total RAM."""
    return n_samples * n_features * 8 / (1024 ** 3)


def fs1_over_budget(
    n_samples: int,
    n_features: int,
    budget_gib: Optional[float] = None,
) -> bool:
    """True iff the estimated FS_1 input-matrix size exceeds the
    host's adaptive memory budget, unless the user has set
    ``ALLOW_FS1_OOM=1`` to bypass the check entirely."""
    import os
    if os.environ.get("ALLOW_FS1_OOM", "").strip() in ("1", "true", "yes"):
        return False
    if budget_gib is None:
        budget_gib = fs1_budget_gib()
    return fs1_memory_estimate_gib(n_samples, n_features) > budget_gib


def fs1_skip_marker(
    n_samples: int,
    n_features: int,
    budget_gib: Optional[float] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Build the skip-marker dict written when FS_1 is over budget.
    Downstream verification (`scripts/review/reproduce_all.py
    ::compare_results`) recognises this shape and reports SKIP rather
    than MATCH/FAIL."""
    if budget_gib is None:
        budget_gib = fs1_budget_gib()
    needed = fs1_memory_estimate_gib(n_samples, n_features)
    total_ram = _total_ram_gib()
    marker = {
        "skipped": True,
        "reason": "memory_budget",
        "fs1_method": "raw_feature_logreg",
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "estimated_input_GiB": round(needed, 3),
        "budget_GiB": round(budget_gib, 3),
        "host_total_ram_GiB": (
            round(total_ram, 1) if total_ram is not None else None),
        "budget_fraction": FS_1_BUDGET_FRACTION,
        "override_env": "ALLOW_FS1_OOM=1 (or FS1_BUDGET_GIB=<gib>)",
    }
    if extra:
        marker.update(extra)
    return marker


def FS_1(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    test_features: np.ndarray,
    *,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """LogReg on concatenated raw features (no projection, no hidden layers).

    Paper's deterministic FS anchor for the "interface, not capacity"
    claim. Hyperparameters are read strictly from
    ``configs/experiments.yaml::raw_feature_logreg_*`` (paper App A);
    raises KeyError on any missing key to surface config drift instead
    of silently inheriting the generic ``logreg_*`` defaults.

    Logs the observed solver iterations and cap. Scikit-learn's
    convergence warnings remain visible.

    NOTE: this function is a pure scoring routine. Memory-budget gating
    and skip-marker writing live in the calling scripts (04 / 06); see
    ``fs1_over_budget`` and ``fs1_skip_marker``.
    """
    required = (
        "raw_feature_logreg_solver",
        "raw_feature_logreg_C",
        "raw_feature_logreg_class_weight",
        "raw_feature_logreg_max_iter",
        "raw_feature_logreg_random_state",
    )
    missing = [k for k in required if k not in _DS_CFG]
    if missing:
        raise KeyError(
            "FS_1 requires explicit config keys in "
            f"configs/experiments.yaml; missing: {missing}"
        )

    lr = LogisticRegression(
        solver=_DS_CFG["raw_feature_logreg_solver"],
        C=_DS_CFG["raw_feature_logreg_C"],
        class_weight=_DS_CFG["raw_feature_logreg_class_weight"],
        max_iter=_DS_CFG["raw_feature_logreg_max_iter"],
        random_state=_DS_CFG["raw_feature_logreg_random_state"],
    )
    lr.fit(fit_features, fit_labels)
    iterations = int(np.max(lr.n_iter_))
    log = logger if logger is not None else logging.getLogger(__name__)
    report = log.warning if iterations >= lr.max_iter else log.info
    report("FS1 solver iterations: %d/%d", iterations, lr.max_iter)
    return lr.predict_proba(test_features)[:, 1]
