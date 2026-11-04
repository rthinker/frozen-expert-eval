"""Prediction-Space Baselines (paper Section 3, item (ii); results Section 4.2).

10 PS functions operating on cached numpy arrays of expert logits/probs.
Each returns per-sample predicted probabilities for the positive class.

Main PS ladder (PS_1-PS_4, weakly monotone on clip4):
  PS_1: probability averaging (no fitting)
  PS_2: LogReg on logits (fitted on FIT)
  PS_C: LogReg-CV on logits (CV inside FIT)
  PS_3: MLP on logits (fitted on FIT, fixed epochs)
  PS_F: MLP on logits + early stopping (WD on SELECT, refit on FIT)
  PS_4: XGBoost (CV inside FIT)

Additional PS variants (specs in Appendix A Table A1; results in Appendix B Table A2):
  PS_A: calibrated averaging (fitted on SELECT via balanced-NLL)
  PS_B: learned per-expert weights (Nelder-Mead on SELECT AUC)
  PS_D: MLP on probabilities (fitted on FIT, fixed epochs)
  PS_E: MLP on probabilities + early stopping

Feature-space methods (FS_1, FeatureAssembly, RandomProjectionAssembly)
live in feature_space.py.
"""

import warnings
import numpy as np
from scipy.special import softmax as scipy_softmax
from scipy.special import log_softmax as scipy_log_softmax
from scipy.optimize import minimize_scalar, minimize
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from typing import Tuple, Optional

import torch
import torch.nn as nn

from src.utils.config import DS_CONFIG as _DS_CFG


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax along last axis."""
    return scipy_softmax(logits, axis=-1)


# ── PS_1: Probability Averaging ────────────────────────────────────────

def PS_1(test_logits: np.ndarray) -> np.ndarray:
    """Average softmax probabilities across experts.

    Args:
        test_logits: [N, E, 2] expert logits.

    Returns:
        [N] averaged P(fake) scores.
    """
    probs = _softmax(test_logits)  # [N, E, 2]
    return probs.mean(axis=1)[:, 1]  # [N]


# ── PS_A: Calibrated Probability Averaging ─────────────────────────────

def PS_A(
    select_logits: np.ndarray,
    select_labels: np.ndarray,
    test_logits: np.ndarray,
    max_iter: Optional[int] = None,
    return_diagnostics: bool = False,
):
    """Per-expert temperature scaling [Guo et al. 2017] then PS_1 average.

    Each expert i gets a scalar T_i fit on SELECT to minimize NLL with
    inverse-frequency sample weights (SELECT is unbalanced). The bounded
    search uses 0.01 <= T_i <= 100. Inference divides each expert's logits
    by T_i before softmax + average.

    Returns [N_test] P(fake), or ([N_test], {"temps": [...]}) if
    return_diagnostics=True.
    """
    if max_iter is None:
        max_iter = _DS_CFG.get("temp_max_iter", 50)
    n_experts = select_logits.shape[1]
    temps = np.ones(n_experts)

    # Balanced sample weights (mean weight == 1)
    n_real = int((select_labels == 0).sum())
    n_fake = int((select_labels == 1).sum())
    if n_real == 0 or n_fake == 0:
        sample_w = np.ones_like(select_labels, dtype=np.float64)
    else:
        w = np.where(select_labels == 1, 1.0 / n_fake, 1.0 / n_real)
        sample_w = w * len(select_labels) / w.sum()

    sel_idx = np.arange(len(select_labels))

    for i in range(n_experts):
        expert_logits = select_logits[:, i, :].astype(np.float64)  # [N, 2]

        def nll(t):
            # Numerically stable weighted NLL via scipy log_softmax.
            # Works even when expert logits are saturated (large magnitudes)
            # whereas sklearn.log_loss(softmax(.)) underflows.
            t_safe = max(float(t), 1e-4)
            log_probs = scipy_log_softmax(expert_logits / t_safe, axis=1)
            # Select the log-prob of the true class for each sample
            picked = log_probs[sel_idx, select_labels.astype(int)]
            return -float((picked * sample_w).sum() / sample_w.sum())

        # Scalar optimiser handles the 1-D bound and avoids finite-difference
        # dead zones that pinned L-BFGS-B at T=1.0 in the previous version.
        # Bounds chosen conservatively: T in (0.01, 100) covers everything
        # from "make logits sharper by 100x" to "flatten logits to uniform".
        # DF40 experts can be extremely overconfident, pushing T above 20.
        result = minimize_scalar(
            nll, bounds=(0.01, 100.0), method="bounded",
            options={"xatol": 1e-4, "maxiter": max_iter},
        )
        temps[i] = float(result.x)

    # Apply learned temperatures to test
    calibrated = np.zeros_like(test_logits)
    for i in range(n_experts):
        calibrated[:, i, :] = test_logits[:, i, :] / temps[i]
    probs = _softmax(calibrated)
    scores = probs.mean(axis=1)[:, 1]
    if return_diagnostics:
        return scores, {"temps": temps.tolist()}
    return scores


# ── PS_B: Learned Per-Expert Weights ───────────────────────────────────

def PS_B(
    select_logits: np.ndarray,
    select_labels: np.ndarray,
    test_logits: np.ndarray,
    max_iter: Optional[int] = None,
    return_diagnostics: bool = False,
):
    """Learned per-expert simplex weights on SELECT AUC, then weighted PS_1.

    Simplex constraint via softmax reparameterization (w = softmax(w_raw),
    optimized unconstrained). Nelder-Mead because AUC is a step function.
    Warm-started at uniform weights (== PS_1).

    Known failure: collapses to one-hot on SELECT-dominant experts and
    overfits OOD TEST (e.g. arch6, clip4). Appendix variant only.

    Returns [N_test] P(fake), or ([N_test], {"weights": [...]}) if
    return_diagnostics=True.
    """
    if max_iter is None:
        max_iter = _DS_CFG.get("weights_max_iter", 2000)
    n_experts = select_logits.shape[1]
    probs_sel = _softmax(select_logits)[:, :, 1]  # [N, E]
    probs_test = _softmax(test_logits)[:, :, 1]

    def neg_auc(w_raw):
        # scipy_softmax is numerically stable (subtracts max before exp);
        # raw np.exp(w_raw) overflows once Nelder-Mead pushes any w_raw
        # above ~700.
        w = scipy_softmax(w_raw)
        combined = (probs_sel * w[None, :]).sum(axis=1)
        return -roc_auc_score(select_labels, combined)

    result = minimize(neg_auc, x0=np.zeros(n_experts),
                      method="Nelder-Mead",
                      options={"maxiter": max_iter})
    w = scipy_softmax(result.x)
    scores = (probs_test * w[None, :]).sum(axis=1)
    if return_diagnostics:
        return scores, {"weights": w.tolist()}
    return scores


# ── PS_2: Logistic Regression on Concatenated Logits ───────────────────

def PS_2(
    fit_logits: np.ndarray,
    fit_labels: np.ndarray,
    test_logits: np.ndarray,
) -> np.ndarray:
    """L2-regularized LogReg on flat expert logits ([N, E, 2] -> [N, E*2]).

    Trained on FIT with class_weight="balanced", L-BFGS, C=1.0,
    random_state=1024. Returns [N_test] P(fake).
    """
    X_fit = fit_logits.reshape(len(fit_logits), -1)
    X_test = test_logits.reshape(len(test_logits), -1)

    lr = LogisticRegression(
        solver=_DS_CFG.get("logreg_solver", "lbfgs"),
        C=_DS_CFG.get("logreg_C", 1.0),
        class_weight=_DS_CFG.get("logreg_class_weight", "balanced"),
        max_iter=_DS_CFG.get("logreg_max_iter", 1000),
        random_state=_DS_CFG.get("logreg_random_state", 1024),
    )
    lr.fit(X_fit, fit_labels)
    return lr.predict_proba(X_test)[:, 1]


# ── PS_C: Regularised Logistic Regression (CV inside FIT) ──────────────

def PS_C(
    fit_logits: np.ndarray,
    fit_labels: np.ndarray,
    test_logits: np.ndarray,
    C_values=None,
    penalties=None,
    n_folds: Optional[int] = None,
    max_iter: Optional[int] = None,
    seed: Optional[int] = None,
    return_diagnostics: bool = False,
):
    """PS_2 with (penalty, C) chosen by 5-fold StratifiedKFold CV on FIT.

    Grid: penalties in {l1, l2, elasticnet} x C in {0.001..100} (defaults
    from config). Solver forced by penalty: saga for l1/elasticnet, lbfgs
    for l2. Best config refit on full FIT, then scores test. CV stays
    inside FIT (no SELECT consumption), keeping PS_C comparable to PS_2.

    Convergence warnings during CV are captured per-config and surfaced
    via diagnostics rather than streamed to stderr.

    Returns [N_test] P(fake), or ([N_test], diagnostics_dict) if
    return_diagnostics=True.
    """
    X_fit = fit_logits.reshape(len(fit_logits), -1)
    X_test = test_logits.reshape(len(test_logits), -1)
    if C_values is None:
        C_values = tuple(_DS_CFG.get(
            "logreg_cv_C", (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)))
    if penalties is None:
        penalties = tuple(_DS_CFG.get(
            "logreg_cv_penalties", ("l1", "l2", "elasticnet")))
    if n_folds is None:
        n_folds = _DS_CFG.get("logreg_cv_folds", 5)
    if max_iter is None:
        max_iter = _DS_CFG.get("logreg_cv_max_iter", 2000)
    if seed is None:
        seed = _DS_CFG.get("logreg_cv_random_state", 1024)

    best_auc = -1
    best_config = None
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    warned_configs = []
    convergence_warning_count = 0
    n_cv_fits = 0

    for pen in penalties:
        default_solver = _DS_CFG.get("logreg_cv_solver", "saga")
        solver = default_solver if pen in ("l1", "elasticnet") else "lbfgs"
        l1_ratio = (_DS_CFG.get("logreg_cv_l1_ratio", 0.5)
                    if pen == "elasticnet" else None)
        for C in C_values:
            aucs = []
            config_warning_count = 0
            for train_idx, val_idx in skf.split(X_fit, fit_labels):
                lr = LogisticRegression(
                    penalty=pen, C=C, solver=solver,
                    l1_ratio=l1_ratio,
                    class_weight=_DS_CFG.get("logreg_class_weight", "balanced"),
                    max_iter=max_iter, random_state=seed,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    lr.fit(X_fit[train_idx], fit_labels[train_idx])
                config_warning_count += sum(
                    1 for w in caught
                    if issubclass(w.category, ConvergenceWarning)
                )
                n_cv_fits += 1
                pred = lr.predict_proba(X_fit[val_idx])[:, 1]
                aucs.append(roc_auc_score(fit_labels[val_idx], pred))
            if config_warning_count:
                warned_configs.append({
                    "penalty": pen,
                    "C": float(C),
                    "solver": solver,
                    "l1_ratio": (None if l1_ratio is None else float(l1_ratio)),
                    "warning_count": int(config_warning_count),
                })
                convergence_warning_count += config_warning_count
            mean_auc = np.mean(aucs)
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_config = (pen, C, solver, l1_ratio)

    # Refit on full FIT with best config
    pen, C, solver, l1_ratio = best_config
    lr = LogisticRegression(
        penalty=pen, C=C, solver=solver,
        l1_ratio=l1_ratio,
        class_weight=_DS_CFG.get("logreg_class_weight", "balanced"),
        max_iter=max_iter, random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        lr.fit(X_fit, fit_labels)
    refit_warning_count = sum(
        1 for w in caught
        if issubclass(w.category, ConvergenceWarning)
    )
    scores = lr.predict_proba(X_test)[:, 1]

    if return_diagnostics:
        diagnostics = {
            "best_config": {
                "penalty": pen,
                "C": float(C),
                "solver": solver,
                "l1_ratio": (None if l1_ratio is None else float(l1_ratio)),
            },
            "best_cv_auc": float(best_auc),
            "n_cv_fits": int(n_cv_fits),
            "cv_convergence_warning_count": int(convergence_warning_count),
            "refit_convergence_warning_count": int(refit_warning_count),
            "total_convergence_warning_count": int(
                convergence_warning_count + refit_warning_count
            ),
            "warned_configs": warned_configs,
            "max_iter": int(max_iter),
        }
        return scores, diagnostics
    return scores


# ── PS_3/PS_D (+ PS_F/PS_E variants): MLP on Logits/Probs ──────────────

class _TinyMLP(nn.Module):
    """Tiny MLP for decision-space combination."""
    def __init__(self, in_dim, hidden=64, dropout=0.1, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden: int = None,
    dropout: float = None,
    lr: float = None,
    epochs: int = None,
    batch_size: int = 256,
    weight_decay: float = 0.0,
    seed: int = 1024,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    patience: Optional[int] = None,
) -> Tuple[_TinyMLP, int]:
    """Train _TinyMLP with class-weighted CE; returns (model, best_epoch).

    Hyperparameters default to the `decision_baselines` block of
    experiments.yaml. With (X_val, patience), uses SELECT-AUC early
    stopping and returns the best-epoch checkpoint; otherwise fixed-epoch.
    """
    if hidden is None:
        hidden = _DS_CFG.get("mlp_hidden", 64)
    if dropout is None:
        dropout = _DS_CFG.get("mlp_dropout", 0.1)
    if lr is None:
        lr = _DS_CFG.get("mlp_lr", 2e-4)
    if epochs is None:
        epochs = _DS_CFG.get("mlp_epochs", 30)
    torch.manual_seed(seed)
    in_dim = X_train.shape[1]
    model = _TinyMLP(in_dim, hidden, dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    # Inverse-frequency class weights so PS_3/PS_D/PS_F/PS_E see a balanced loss
    # surface on imbalanced FIT (Phase 2 policy: cache is unbalanced and
    # balancing happens at training time). This brings the MLP baselines in
    # line with sklearn paths (class_weight="balanced") and the FS combiner
    # (class-weighted CE). Without this, the MLPs under-weight the
    # minority "real" class on a 1:4 imbalanced pool.
    n_real = int((y_train == 0).sum())
    n_fake = int((y_train == 1).sum())
    if n_real > 0 and n_fake > 0:
        inv_real = len(y_train) / (2.0 * n_real)
        inv_fake = len(y_train) / (2.0 * n_fake)
        class_w = torch.tensor([inv_real, inv_fake], dtype=torch.float32)
    else:
        class_w = None
    criterion = nn.CrossEntropyLoss(weight=class_w)

    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_t = torch.as_tensor(y_train, dtype=torch.long)

    # Pre-tensorize val data if using early stopping (avoid per-epoch copy)
    val_t = None
    if X_val is not None:
        val_t = torch.as_tensor(X_val, dtype=torch.float32)

    best_auc = -1
    best_epoch = epochs
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        for start in range(0, len(X_t), batch_size):
            end = min(start + batch_size, len(X_t))
            idx = perm[start:end]
            optimiser.zero_grad()
            loss = criterion(model(X_t[idx]), y_t[idx])
            loss.backward()
            optimiser.step()

        # Early stopping check with best-checkpoint saving
        if X_val is not None and patience is not None:
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(model(val_t), dim=1)[:, 1].numpy()
            auc = roc_auc_score(y_val, probs)
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    # Restore best-epoch weights if early stopping was used
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model, best_epoch


def PS_3(
    fit_logits: np.ndarray,
    fit_labels: np.ndarray,
    test_logits: np.ndarray,
    epochs: Optional[int] = None,
    seed: int = 1024,
) -> np.ndarray:
    """Tiny MLP on flat expert logits ([N, E, 2] -> [N, E*2]).

    Fixed-epoch _TinyMLP with class-weighted CE, no early stopping, no
    hyperparameter search. Stochastic: script 04 averages per-sample
    scores across seeds. Returns [N_test] P(fake).
    """
    X_fit = fit_logits.reshape(len(fit_logits), -1)
    X_test = test_logits.reshape(len(test_logits), -1)
    model, _ = _train_mlp(X_fit, fit_labels, epochs=epochs, seed=seed)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(
            model(torch.as_tensor(X_test, dtype=torch.float32)), dim=1
        )[:, 1].numpy()
    return probs


def PS_D(
    fit_logits: np.ndarray,
    fit_labels: np.ndarray,
    test_logits: np.ndarray,
    epochs: Optional[int] = None,
    seed: int = 1024,
) -> np.ndarray:
    """PS_3 variant: same MLP recipe but inputs are softmaxed first.

    Discards logit-magnitude information; reported as an Appendix-A
    non-monotone variant (paired with PS_E). Returns [N_test] P(fake).
    """
    X_fit = _softmax(fit_logits).reshape(len(fit_logits), -1)
    X_test = _softmax(test_logits).reshape(len(test_logits), -1)
    model, _ = _train_mlp(X_fit, fit_labels, epochs=epochs, seed=seed)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(
            model(torch.as_tensor(X_test, dtype=torch.float32)), dim=1
        )[:, 1].numpy()
    return probs


def PS_F(
    fit_logits: np.ndarray,
    fit_labels: np.ndarray,
    select_logits: np.ndarray,
    select_labels: np.ndarray,
    test_logits: np.ndarray,
    wd_candidates=None,
    max_epochs: Optional[int] = None,
    patience: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Tuned MLP: WD search + SELECT-AUC early stopping, then FIT refit.

    Three-step protocol so the final test-predicting model trains only on
    FIT:
      1. For each WD, train on FIT with SELECT-AUC early stopping -> e_star.
      2. Pick WD whose best SELECT AUC wins.
      3. Refit on FIT for exactly e_star epochs (no early stopping).

    Stochastic (8 seeds averaged by script 04). Returns [N_test] P(fake).
    """
    if wd_candidates is None:
        wd_candidates = tuple(_DS_CFG.get(
            "mlp_es_weight_decay", (0.0, 1e-4, 1e-3, 1e-2)))
    if max_epochs is None:
        max_epochs = _DS_CFG.get("mlp_es_max_epochs", 200)
    if patience is None:
        patience = _DS_CFG.get("mlp_es_patience", 10)
    if seed is None:
        seed = _DS_CFG.get("logreg_random_state", 1024)

    X_fit = fit_logits.reshape(len(fit_logits), -1)
    X_sel = select_logits.reshape(len(select_logits), -1)
    X_test = test_logits.reshape(len(test_logits), -1)

    # Step 1-2: find best WD and its stopping epoch
    best_wd_auc = -1
    best_wd = 0.0
    best_e_star = max_epochs

    for wd in wd_candidates:
        model, e_star = _train_mlp(
            X_fit, fit_labels, epochs=max_epochs, weight_decay=wd,
            seed=seed, X_val=X_sel, y_val=select_labels, patience=patience,
        )
        # Evaluate on SELECT
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(
                model(torch.as_tensor(X_sel, dtype=torch.float32)), dim=1
            )[:, 1].numpy()
        auc = roc_auc_score(select_labels, probs)
        if auc > best_wd_auc:
            best_wd_auc = auc
            best_wd = wd
            best_e_star = e_star

    # Step 3: refit on FIT for exactly E* epochs (no early stopping)
    final_model, _ = _train_mlp(
        X_fit, fit_labels, epochs=best_e_star, weight_decay=best_wd, seed=seed,
    )
    final_model.eval()
    with torch.no_grad():
        probs = torch.softmax(
            final_model(torch.as_tensor(X_test, dtype=torch.float32)), dim=1
        )[:, 1].numpy()
    return probs


def PS_E(
    fit_logits: np.ndarray,
    fit_labels: np.ndarray,
    select_logits: np.ndarray,
    select_labels: np.ndarray,
    test_logits: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """PS_F variant: softmax probabilities instead of raw logits.

    Wrapper: softmaxes fit/select/test, then delegates to PS_F.
    Appendix-A non-monotone variant. Returns [N_test] P(fake).
    """
    fit_probs = _softmax(fit_logits)
    select_probs = _softmax(select_logits)
    test_probs = _softmax(test_logits)
    return PS_F(
        fit_probs.reshape(len(fit_probs), -1), fit_labels,
        select_probs.reshape(len(select_probs), -1), select_labels,
        test_probs.reshape(len(test_probs), -1), **kwargs,
    )


# ── PS_4: XGBoost on Concatenated Logits ───────────────────────────────

def PS_4(
    fit_logits: np.ndarray,
    fit_labels: np.ndarray,
    test_logits: np.ndarray,
    n_folds: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """CV-tuned XGBoost on flat expert logits ([N, E, 2] -> [N, E*2]).

    5-fold StratifiedKFold CV inside FIT over a 108-config grid
    (n_estimators / max_depth / learning_rate / subsample /
    colsample_bytree, defaults from config). Argmax mean CV AUC, refit
    once on full FIT, score test. scale_pos_weight = n_neg / n_pos for
    class imbalance.

    xgboost imported lazily; raises ImportError with install hint if
    absent. Returns [N_test] P(fake).
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("xgboost required for PS_4. pip install xgboost")

    X_fit = fit_logits.reshape(len(fit_logits), -1)
    X_test = test_logits.reshape(len(test_logits), -1)
    if n_folds is None:
        n_folds = _DS_CFG.get("xgb_cv_folds", 5)
    if seed is None:
        seed = _DS_CFG.get("xgb_random_state", 1024)

    # Scale pos weight for imbalanced data
    n_pos = (fit_labels == 1).sum()
    n_neg = (fit_labels == 0).sum()
    spw = n_neg / max(n_pos, 1)

    grid = [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr,
         "subsample": ss, "colsample_bytree": cs}
        for n in _DS_CFG.get("xgb_n_estimators", [50, 100, 200])
        for d in _DS_CFG.get("xgb_max_depth", [2, 3, 4])
        for lr in _DS_CFG.get("xgb_learning_rate", [0.05, 0.1, 0.3])
        for ss in _DS_CFG.get("xgb_subsample", [0.8, 1.0])
        for cs in _DS_CFG.get("xgb_colsample_bytree", [0.8, 1.0])
    ]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best_auc = -1
    best_params = grid[0]

    for params in grid:
        aucs = []
        for train_idx, val_idx in skf.split(X_fit, fit_labels):
            clf = XGBClassifier(
                **params,
                scale_pos_weight=spw,
                random_state=seed,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
            clf.fit(X_fit[train_idx], fit_labels[train_idx])
            pred = clf.predict_proba(X_fit[val_idx])[:, 1]
            aucs.append(roc_auc_score(fit_labels[val_idx], pred))
        mean_auc = np.mean(aucs)
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    # Refit on full FIT with best params
    clf = XGBClassifier(
        **best_params,
        scale_pos_weight=spw,
        random_state=seed,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X_fit, fit_labels)
    return clf.predict_proba(X_test)[:, 1]
