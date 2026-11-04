# C21: Measure the storage and fit costs of the two interfaces.
"""RUN D - the cost ledger.

Quantifies storage, memory and fitting cost for prediction and feature interfaces.

WHAT IS MEASURED
--------------------------------
Per interface, per pool:

  widths     floats per sample per expert and pool totals (features D_i
             vs logits K=2), plus bytes AS STORED in the released cache
             (features float16, logits float32) - read from the .npy
             headers, not assumed.
  storage    on-disk bytes of the cached features vs cached logits, for
             the pool's experts and for the full 21-expert release,
             across all ten split directories.
  combiner   training wall-clock and peak working set for the FS anchor
             (FS_1: L2 logistic on 9,984 raw dims) against the strongest
             PS combiner (PS_4: 5-fold-CV XGBoost on 12 logit dims), and
             the cheap PS representative (PS_2 logistic). FS_1 is timed
             via the frozen function; a same-config logistic is fitted
             once more to split fit/predict time, with a score-parity
             assertion so the split timing demonstrably measures the
             same computation.
  latency    per-sample, combiner-only inference over the 17,925-sample
             TEST set.
  extraction one-time cost, identical for both interfaces (the forward
             pass happens either way; reading the penultimate layer
             instead of the logit is free). Measured throughput can be
             recorded with --measured_extraction_imgs_per_sec when
             measured separately. Table A21 reports the shared extraction cost.

Usage (from the repository root)::

    python camera_ready/scripts/analyses/appI3_cost_ledger_v2.py --pool arch6                 # sizes only
    python camera_ready/scripts/analyses/appI3_cost_ledger_v2.py --pool arch6 --measure_fit   # + timings
    python camera_ready/scripts/analyses/appI3_cost_ledger_v2.py --pool arch6 --measure_fit \
        --measured_extraction_imgs_per_sec 41.2

Outputs -> camera_ready/output/results/analyses/run_d_cost_ledger_<pool>.json
Writes nothing outside camera_ready/output/.
"""

from __future__ import annotations

import _paths  # noqa: F401  # MUST be first

import argparse
import ctypes
import sys
if sys.platform == "win32":
    import ctypes.wintypes as wt
else:  # the memory probe below is Windows-only; this keeps the module importable elsewhere
    class wt:  # noqa: N801
        DWORD = ctypes.c_uint32
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import _common as common
from src.models.feature_space import FS_1  # noqa: E402
from src.models.prediction_space import PS_2, PS_4  # noqa: E402
from src.utils.config import DS_CONFIG, FIT_TAG, SELECT_TAG, get_pool_experts  # noqa: E402
from src.utils.dataset_helpers import TEST_GENERATORS  # noqa: E402

ALL_SPLITS = [FIT_TAG, SELECT_TAG] + list(TEST_GENERATORS)


# --------------------------------------------------------------------------
# Peak working set (Windows, stdlib ctypes; no psutil dependency)
# --------------------------------------------------------------------------

class _PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def _mem() -> Dict[str, float]:
    """Current and peak working set of this process, in GiB."""
    if sys.platform != "win32":
        return {"working_set_gib": None, "peak_working_set_gib": None}
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    # Explicit ctypes signatures preserve the process handle width on 64-bit Windows.
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    h = k32.GetCurrentProcess()
    # K32GetProcessMemoryInfo (kernel32) first; psapi alias as fallback.
    for fn in (getattr(k32, "K32GetProcessMemoryInfo", None),
               getattr(ctypes.WinDLL("psapi"), "GetProcessMemoryInfo", None)):
        if fn is None:
            continue
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), ctypes.c_ulong]
        fn.restype = ctypes.c_int
        if fn(h, ctypes.byref(pmc), pmc.cb):
            return {"working_set_gib": pmc.WorkingSetSize / 2**30,
                    "peak_working_set_gib": pmc.PeakWorkingSetSize / 2**30}
    return {"working_set_gib": None, "peak_working_set_gib": None}


# --------------------------------------------------------------------------
# Sizes
# --------------------------------------------------------------------------

def npy_meta(path: Path) -> Dict:
    """dtype/shape from the .npy header plus on-disk bytes, without loading."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        shape, _, dtype = np.lib.format._read_array_header(f, version)
    return {"shape": list(shape), "dtype": str(dtype),
            "bytes_on_disk": path.stat().st_size}


def measure_widths_and_storage(cache_root: Path, pool_experts: List[str],
                               log) -> Dict:
    cache_root = Path(cache_root)
    # widths from FIT split headers
    widths = {}
    for e in pool_experts:
        fm = npy_meta(cache_root / FIT_TAG / f"{e}_features.npy")
        lm = npy_meta(cache_root / FIT_TAG / f"{e}_logits.npy")
        d = fm["shape"][1]
        widths[e] = {
            "feature_floats": int(d),
            "feature_dtype": fm["dtype"],
            "feature_bytes_per_sample": int(d * np.dtype(fm["dtype"]).itemsize),
            "logit_floats": int(lm["shape"][1]),
            "logit_dtype": lm["dtype"],
            "logit_bytes_per_sample": int(
                lm["shape"][1] * np.dtype(lm["dtype"]).itemsize),
        }
    pool_feat_floats = sum(w["feature_floats"] for w in widths.values())
    pool_feat_bytes = sum(w["feature_bytes_per_sample"] for w in widths.values())
    pool_logit_floats = sum(w["logit_floats"] for w in widths.values())
    pool_logit_bytes = sum(w["logit_bytes_per_sample"] for w in widths.values())
    log(f"pool widths: features {pool_feat_floats} floats "
        f"({pool_feat_bytes} B/sample as stored) vs predictions "
        f"{pool_logit_floats} floats ({pool_logit_bytes} B/sample)")

    # storage across all ten split dirs
    def _sum(experts, suffix):
        total = 0
        for split in ALL_SPLITS:
            for e in experts:
                p = cache_root / split / f"{e}_{suffix}.npy"
                if not p.is_file():
                    raise SystemExit(f"missing cache file: {p}")
                total += p.stat().st_size
        return total

    all_experts = sorted({p.name.rsplit("_features.npy", 1)[0]
                          for p in (cache_root / FIT_TAG).glob("*_features.npy")})
    storage = {
        "pool_features_bytes": _sum(pool_experts, "features"),
        "pool_logits_bytes": _sum(pool_experts, "logits"),
        "release_features_bytes": _sum(all_experts, "features"),
        "release_logits_bytes": _sum(all_experts, "logits"),
        "n_release_experts": len(all_experts),
    }
    storage["pool_features_gib"] = storage["pool_features_bytes"] / 2**30
    storage["pool_logits_mib"] = storage["pool_logits_bytes"] / 2**20
    storage["release_features_gib"] = storage["release_features_bytes"] / 2**30
    storage["release_logits_mib"] = storage["release_logits_bytes"] / 2**20
    log(f"storage: pool features {storage['pool_features_gib']:.2f} GiB vs "
        f"pool logits {storage['pool_logits_mib']:.1f} MiB; release "
        f"features {storage['release_features_gib']:.2f} GiB")
    return {"per_expert": widths,
            "pool_totals": {
                "feature_floats": int(pool_feat_floats),
                "feature_bytes_per_sample_stored": int(pool_feat_bytes),
                "prediction_floats": int(pool_logit_floats),
                "prediction_bytes_per_sample_stored": int(pool_logit_bytes),
                "width_ratio_floats": float(pool_feat_floats /
                                            max(pool_logit_floats, 1)),
            },
            "storage": storage}


# --------------------------------------------------------------------------
# Combiner timing
# --------------------------------------------------------------------------

def measure_combiners(expert_names, cache_root, log) -> Dict:
    from sklearn.linear_model import LogisticRegression

    fit_feats, fit_logits3d, fit_labels = common.load_split(
        expert_names, FIT_TAG, cache_root)
    X_test, test_logits3d, test_labels, boundaries, gens = \
        common.load_test_generators(expert_names, cache_root,
                                    list(TEST_GENERATORS))
    X_fit = np.concatenate(fit_feats, axis=1)
    del fit_feats
    n_test = len(test_labels)
    out = {}

    # --- PS_2: cheap PS representative (logistic on logits) ---------------
    m0 = _mem()
    t0 = time.perf_counter()
    ps2_scores = np.asarray(PS_2(fit_logits3d, fit_labels, test_logits3d))
    t1 = time.perf_counter()
    out["PS_2_logreg_logits"] = {
        "total_fit_plus_predict_sec": t1 - t0,
        "input_dims": int(fit_logits3d.shape[1] * fit_logits3d.shape[2]),
        "peak_working_set_gib_after": _mem()["peak_working_set_gib"],
    }
    log(f"PS_2 fit+predict {t1 - t0:.1f}s")

    # --- PS_4: strongest PS (5-fold CV XGBoost on logits) -----------------
    t0 = time.perf_counter()
    ps4_scores = np.asarray(PS_4(fit_logits3d, fit_labels, test_logits3d))
    t1 = time.perf_counter()
    ps4_macro = common.macro_auc(common.per_generator_auc(
        test_labels, ps4_scores, boundaries, gens), gens)
    out["PS_4_xgboost_cv"] = {
        "total_fit_plus_predict_sec": t1 - t0,
        "includes": "5-fold CV grid as submitted",
        "macro_auroc": float(ps4_macro),
        "peak_working_set_gib_after": _mem()["peak_working_set_gib"],
    }
    log(f"PS_4 fit+predict {t1 - t0:.1f}s  macro={ps4_macro:.4f}")

    # --- FS_1 via the frozen function (total) -----------------------------
    t0 = time.perf_counter()
    fs1_scores = np.asarray(FS_1(X_fit, fit_labels, X_test))
    t1 = time.perf_counter()
    fs1_total = t1 - t0
    fs1_macro = common.macro_auc(common.per_generator_auc(
        test_labels, fs1_scores, boundaries, gens), gens)
    out["FS_1_frozen_total"] = {
        "total_fit_plus_predict_sec": fs1_total,
        "input_dims": int(X_fit.shape[1]),
        "macro_auroc": float(fs1_macro),
        "peak_working_set_gib_after": _mem()["peak_working_set_gib"],
        "note": "frozen FS_1 call; sklearn lbfgs upcasts the input to "
                "float64 internally (see codebase feature_space.py)",
    }
    log(f"FS_1 fit+predict {fs1_total:.1f}s  macro={fs1_macro:.4f}")

    # --- same-config logistic, split fit/predict, parity-checked ----------
    lr = LogisticRegression(
        solver=DS_CONFIG["raw_feature_logreg_solver"],
        C=DS_CONFIG["raw_feature_logreg_C"],
        class_weight=DS_CONFIG["raw_feature_logreg_class_weight"],
        max_iter=DS_CONFIG["raw_feature_logreg_max_iter"],
        random_state=DS_CONFIG["raw_feature_logreg_random_state"])
    t0 = time.perf_counter()
    lr.fit(X_fit, fit_labels)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    split_scores = lr.predict_proba(X_test)[:, 1]
    t_pred = time.perf_counter() - t0
    parity = float(np.max(np.abs(split_scores - fs1_scores)))
    out["FS_1_config_split_timing"] = {
        "fit_sec": t_fit,
        "predict_sec_total": t_pred,
        "predict_us_per_sample": 1e6 * t_pred / n_test,
        "score_parity_max_abs_diff_vs_frozen": parity,
        "parity_ok": bool(parity < 1e-8),
    }
    log(f"FS_1 split timing: fit {t_fit:.1f}s, predict {t_pred:.2f}s "
        f"({1e6 * t_pred / n_test:.1f} us/sample), parity {parity:.2e}")

    # --- PS predict-only latency (same-config PS_2 logistic) --------------
    flat_fit = fit_logits3d.reshape(len(fit_labels), -1)
    flat_test = test_logits3d.reshape(n_test, -1)
    lr_ps = LogisticRegression(
        solver=DS_CONFIG["logreg_solver"], C=DS_CONFIG["logreg_C"],
        max_iter=DS_CONFIG["logreg_max_iter"],
        random_state=DS_CONFIG["logreg_random_state"]) \
        if "logreg_solver" in DS_CONFIG else LogisticRegression(max_iter=1000)
    lr_ps.fit(flat_fit, fit_labels)
    t0 = time.perf_counter()
    lr_ps.predict_proba(flat_test)
    t_pred_ps = time.perf_counter() - t0
    out["PS_predict_only"] = {
        "predict_sec_total": t_pred_ps,
        "predict_us_per_sample": 1e6 * t_pred_ps / n_test,
        "head": "logistic on flat logits (PS_2 family)",
    }
    log(f"PS predict-only {1e6 * t_pred_ps / n_test:.2f} us/sample")
    return out


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="arch6")
    ap.add_argument("--measure_fit", action="store_true",
                    help="Also time combiner fits (loads the full feature "
                         "cache; run when no other heavy job is active).")
    ap.add_argument("--measured_extraction_imgs_per_sec", type=float,
                    default=None,
                    help="Extraction throughput in images per second, if "
                         "measured separately (the paper did not measure it).")
    args = ap.parse_args()

    log = common.make_logger("run_d_cost_ledger")
    expert_names = get_pool_experts(args.pool)
    cache_root = _paths.cache_root()
    log("RUN D - cost ledger")
    log(f"pool={args.pool}  experts={expert_names}")

    res = {
        "run": "D_cost_ledger",
        "pool": args.pool,
        "experts": list(expert_names),
        "widths_and_storage": measure_widths_and_storage(
            cache_root, list(expert_names), log),
        "extraction": {
            "note": "One-time cost, identical for both interfaces: the "
                    "expert forward pass happens under either interface and "
                    "reading the penultimate representation instead of the "
                    "logit is free. Where an expert is exposed only as a "
                    "scoring API, feature access is an access/contract cost, "
                    "not a compute cost.",
            "paper_reference": "Table A21: feature extraction throughput "
                               "was not measured",
            "measured_imgs_per_sec": args.measured_extraction_imgs_per_sec,
        },
        "host": {"platform": sys.platform, "python": sys.version.split()[0]},
    }
    if args.measure_fit:
        res["combiner_costs"] = measure_combiners(
            expert_names, cache_root, log)
    else:
        log("(sizes only; rerun with --measure_fit for combiner timings)")

    out = common.write_json(
        res, _paths.RESULTS_DIR / f"run_d_cost_ledger_{args.pool}.json")
    log(f"saved: {out}")


if __name__ == "__main__":
    main()
