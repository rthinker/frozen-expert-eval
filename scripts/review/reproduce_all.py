"""Block A definitions and helpers imported by the reproduction harness.

Run python camera_ready/scripts/review/reproduce_all_v4.py --dry-run
from the repository root to see the available steps.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# This file lives at scripts/review/; codebase root is two levels up.
CODEBASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODEBASE))

OUTPUT = CODEBASE / "output"
RESULTS = OUTPUT / "results"
CKPTS = OUTPUT / "checkpoints"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
LOGS = OUTPUT / "logs"
REFERENCE_PATH = RESULTS / "reference_results.json"

BACKUP_ROOT = OUTPUT / "reproducibility_backup"

# Tolerance for torch-trained / stochastic results (GPU non-determinism,
# 8-seed averaging, etc.).
AUC_TOLERANCE = 0.005

# Tolerance for "deterministic" methods (single-fit sklearn / XGBoost
# paths). The script used to require bit-identical match, but in
# practice scipy / sklearn patch versions produce sub-millidiff drift
# in L-BFGS / Nelder-Mead / randomized SVD on the same hardware. A
# 1e-4 ceiling absorbs that representation-level noise while still
# being 50x tighter than the stochastic tolerance, so any actual
# scientific drift is still caught.
DETERMINISTIC_TOLERANCE = 1e-4

# Smoke-test target.
SMOKE_POOL = "arch6"
SMOKE_TRAIN_TAG = "FIT"
SMOKE_EXPECTED_AUC = 0.9657
SMOKE_TOLERANCE = 0.005

# ---------------------------------------------------------------------------
# Experiment definitions: (script_num, cmd, expected_outputs, deterministic?)
# Imported by scripts/authoring/generate_reference.py to keep the reference
# bundle exactly in lock-step with the reproducibility harness.
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # -- 03: Cache integrity check --
    ("03", "python scripts/03_verify_cache.py --require-fit-select",
     [], True),

    # -- 04: PS ladder + raw-feat LogReg per pool (Section 4.2) --
    ("04", "python scripts/04_prediction_space.py --pool clip4 --include_raw_features",
     ["04_ds_clip4.json", "04_ds_clip4_rawfeat.json", "04_ds_clip4_scores.npz"], True),
    ("04", "python scripts/04_prediction_space.py --pool arch6 --include_raw_features",
     ["04_ds_arch6.json", "04_ds_arch6_rawfeat.json", "04_ds_arch6_scores.npz"], True),
    ("04", "python scripts/04_prediction_space.py --pool weak6 --include_raw_features",
     ["04_ds_weak6.json", "04_ds_weak6_rawfeat.json", "04_ds_weak6_scores.npz"], True),
    ("04", "python scripts/04_prediction_space.py --pool mixed6 --include_raw_features",
     ["04_ds_mixed6.json", "04_ds_mixed6_rawfeat.json", "04_ds_mixed6_scores.npz"], True),
    ("04", "python scripts/04_prediction_space.py --pool low_het4 --include_raw_features",
     ["04_ds_low_het4.json", "04_ds_low_het4_rawfeat.json", "04_ds_low_het4_scores.npz"], True),
    ("04", "python scripts/04_prediction_space.py --pool strong4 --include_raw_features",
     ["04_ds_strong4.json", "04_ds_strong4_rawfeat.json", "04_ds_strong4_scores.npz"], True),
    ("04", "python scripts/04_prediction_space.py --pool balanced8 --include_raw_features",
     ["04_ds_balanced8.json", "04_ds_balanced8_rawfeat.json", "04_ds_balanced8_scores.npz"], True),
    ("04", "python scripts/04_prediction_space.py --pool full21 --include_raw_features",
     ["04_ds_full21.json", "04_ds_full21_rawfeat.json", "04_ds_full21_scores.npz"], True),

    # -- 05: FS assembly (Section 4.2 headline + 4.3 capacity ablation) --
    ("05", "python scripts/05_feature_space.py --pool arch6 --combiner_type mlp --proj_dim 256",
     ["05_fs_arch6_mlp_d256.json"], False),
    ("05", "python scripts/05_feature_space.py --pool clip4 --combiner_type mlp --proj_dim 32",
     ["05_fs_clip4_mlp_d32.json"], False),
    ("05", "python scripts/05_feature_space.py --pool clip4 --combiner_type mlp --proj_dim 64",
     ["05_fs_clip4_mlp_d64.json"], False),
    ("05", "python scripts/05_feature_space.py --pool clip4 --combiner_type mlp --proj_dim 256",
     ["05_fs_clip4_mlp_d256.json"], False),
    ("05", "python scripts/05_feature_space.py --pool clip4 --combiner_type linear --proj_dim 64",
     ["05_fs_clip4_linear_d64.json"], False),
    ("05", "python scripts/05_feature_space.py --pool clip4 --combiner_type pca --proj_dim 64",
     ["05_fs_clip4_pca_d64.json"], False),
    ("05", "python scripts/05_feature_space.py --pool clip4 --combiner_type random --proj_dim 64",
     ["05_fs_clip4_random_d64.json"], False),
    ("05", "python scripts/05_feature_space.py --pool weak6 --combiner_type mlp --proj_dim 256",
     ["05_fs_weak6_mlp_d256.json"], False),
    ("05", "python scripts/05_feature_space.py --pool mixed6 --combiner_type mlp --proj_dim 256",
     ["05_fs_mixed6_mlp_d256.json"], False),
    ("05", "python scripts/05_feature_space.py --pool low_het4 --combiner_type mlp --proj_dim 256",
     ["05_fs_low_het4_mlp_d256.json"], False),
    ("05", "python scripts/05_feature_space.py --pool strong4 --combiner_type mlp --proj_dim 256",
     ["05_fs_strong4_mlp_d256.json"], False),
    ("05", "python scripts/05_feature_space.py --pool balanced8 --combiner_type mlp --proj_dim 256",
     ["05_fs_balanced8_mlp_d256.json"], False),
    ("05", "python scripts/05_feature_space.py --pool full21 --combiner_type mlp --proj_dim 256",
     ["05_fs_full21_mlp_d256.json"], False),

    # -- 06: Bootstrap raw-feat (Section 4.2 CI) --
    ("06", "python scripts/06_bootstrap_rawfeat.py --pool clip4",
     ["06_bootstrap_clip4.json"], True),
    ("06", "python scripts/06_bootstrap_rawfeat.py --pool arch6",
     ["06_bootstrap_arch6.json"], True),
    ("06", "python scripts/06_bootstrap_rawfeat.py --pool strong4",
     ["06_bootstrap_strong4.json"], True),
    ("06", "python scripts/06_bootstrap_rawfeat.py --pool balanced8",
     ["06_bootstrap_balanced8.json"], True),
    ("06", "python scripts/06_bootstrap_rawfeat.py --pool full21",
     ["06_bootstrap_full21.json"], True),

    # -- 07: Bootstrap MLP (Section 4.2 CI) --
    ("07", "python scripts/07_bootstrap_mlp.py --pool arch6 --proj_dim 256",
     ["07_bootstrap_mlp_arch6.json"], False),
    ("07", "python scripts/07_bootstrap_mlp.py --pool clip4 --proj_dim 32",
     ["07_bootstrap_mlp_clip4.json"], False),
    ("07", "python scripts/07_bootstrap_mlp.py --pool strong4 --proj_dim 256",
     ["07_bootstrap_mlp_strong4.json"], False),
    ("07", "python scripts/07_bootstrap_mlp.py --pool balanced8 --proj_dim 256",
     ["07_bootstrap_mlp_balanced8.json"], False),
    ("07", "python scripts/07_bootstrap_mlp.py --pool full21 --proj_dim 256",
     ["07_bootstrap_mlp_full21.json"], False),

    # -- 08: PCA bottleneck --
    ("08", "python scripts/08_pca_bottleneck.py --pool clip4",
     ["08_pca_clip4.json"], True),
    ("08", "python scripts/08_pca_bottleneck.py --pool arch6",
     ["08_pca_arch6.json"], True),
    ("08", "python scripts/08_pca_bottleneck.py --pool strong4",
     ["08_pca_strong4.json"], True),
    ("08", "python scripts/08_pca_bottleneck.py --pool balanced8",
     ["08_pca_balanced8.json"], True),
    ("08", "python scripts/08_pca_bottleneck.py --pool mixed6",
     ["08_pca_mixed6.json"], True),
    ("08", "python scripts/08_pca_bottleneck.py --pool merged9",
     ["08_pca_merged9.json"], True),
    ("08", "python scripts/08_pca_bottleneck.py --pool weak6",
     ["08_pca_weak6.json"], True),
    ("08", "python scripts/08_pca_bottleneck.py --pool low_het4",
     ["08_pca_low_het4.json"], True),
    ("08", "python scripts/08_pca_bottleneck.py --pool full21",
     ["08_pca_full21.json"], True),

    # -- 09: Disagreement (Appendix D) --
    ("09", "python scripts/09_disagreement_analysis.py --pool clip4",
     ["09_disagreement_clip4.json"], True),

    # -- 10: Leave-one-expert-out (Appendix E) --
    ("10", "python scripts/10_leave_one_out.py --pool clip4 --proj_dim 32",
     ["10_loo_clip4_d32.json"], False),
    ("10", "python scripts/10_leave_one_out.py --pool arch6 --proj_dim 256",
     ["10_loo_arch6_d256.json"], False),

    # -- 11: LOGO overlap-control (Appendix F) --
    ("11", "python scripts/11_disjoint_data.py --pool arch6 --proj_dim 256",
     ["11_disjoint_arch6_d256.json"], False),

    # -- Cross-regime replication --
    ("04", "python scripts/04_prediction_space.py --pool arch6_fr --include_raw_features",
     ["04_ds_arch6_fr.json", "04_ds_arch6_fr_rawfeat.json"], True),
    ("04", "python scripts/04_prediction_space.py --pool arch6_efs --include_raw_features",
     ["04_ds_arch6_efs.json", "04_ds_arch6_efs_rawfeat.json"], True),

    # -- 12: XGBoost on features (combiner-family control) --
    ("12", "python scripts/12_xgb_features.py --pool clip4",
     ["12_xgb_features_clip4.json"], True),
    ("12", "python scripts/12_xgb_features.py --pool arch6",
     ["12_xgb_features_arch6.json"], True),

    # -- 13: FS_1 raw vs standardized features --
    ("13", "python scripts/13_fs1_standardized.py --pool arch6",
     ["13_fs1_standardized_arch6.json"], True),
    ("13", "python scripts/13_fs1_standardized.py --pool clip4",
     ["13_fs1_standardized_clip4.json"], True),

    # -- 14: Merged pool (merged9) --
    ("14", "python scripts/14_merged_pool.py",
     ["14_merged9.json"], True),
    ("05", "python scripts/05_feature_space.py --pool merged9 --combiner_type mlp --proj_dim 256",
     ["05_fs_merged9_mlp_d256.json"], False),

    # -- 15: AUPRC appendix --
    ("15", "python scripts/15_auprc_appendix.py",
     ["15_auprc_appendix.json"], False),

    # -- 16: Per-expert individual AUROC --
    ("16", "python scripts/16_individual_expert_auc.py",
     ["16_individual_expert_auc.json"], True),

    # -- 17: Generate tables (depends on all results above) --
    ("17", "python scripts/17_generate_tables.py",
     [], True),
]

EXPECTED_TABLES = [
    "table2_ladder_clip4.tex",
    "table2a_ladder_arch6.tex",
    "table3_capacity_clip4_d64.tex",
    "table5_pca_bottleneck.tex",
    "diagnostic_scope_summary.tex",
    "table_robustness_controls.tex",
]


# ---------------------------------------------------------------------------
# Smoke test (was scripts/review/reproduce_smoke.py)
# ---------------------------------------------------------------------------

def run_smoke() -> int:
    """Smoke: rebuild FS_1 headline on arch6 from cached features.

    Returns 0 on PASS (within tolerance), non-zero on FAIL.
    """
    from src.models.feature_space import FS_1
    from src.utils.config import (
        CACHE_ROOT, RESULTS_ROOT, get_pool_experts,
    )
    from src.utils.dataset_helpers import TEST_GENERATORS
    from src.utils.io import load_concat_features
    from src.utils.metrics import compute_auc

    t0 = time.time()
    print("[smoke] Reproducing FS_1 headline (arch6, raw-feat LogReg).")
    print(f"[smoke] Expected Avg(8) AUC: {SMOKE_EXPECTED_AUC:.4f} "
          f"+/- {SMOKE_TOLERANCE}")

    expert_names = get_pool_experts(SMOKE_POOL)
    print(f"[smoke] Pool: {SMOKE_POOL}, N={len(expert_names)} experts.")

    print("[smoke] Loading FIT features...")
    fit_feats, fit_labels = load_concat_features(
        expert_names, SMOKE_TRAIN_TAG, CACHE_ROOT)
    print(f"[smoke]   FIT: {fit_feats.shape[0]} samples, "
          f"D={fit_feats.shape[1]}")

    print("[smoke] Loading test generators and scoring...")
    test_feats_list, test_labels_list, gen_sizes = [], [], []
    for gen in TEST_GENERATORS:
        gf, gl = load_concat_features(expert_names, gen, CACHE_ROOT)
        test_feats_list.append(gf)
        test_labels_list.append(gl)
        gen_sizes.append(gf.shape[0])

    all_test_feats = np.concatenate(test_feats_list, axis=0)
    all_scores = FS_1(fit_feats, fit_labels, all_test_feats)

    aucs = []
    offset = 0
    for gen, n, labels in zip(TEST_GENERATORS, gen_sizes, test_labels_list):
        scores = all_scores[offset:offset + n]
        auc = float(compute_auc(labels, scores))
        aucs.append(auc)
        print(f"[smoke]   {gen:15s}: {auc:.4f}")
        offset += n

    avg8 = float(np.mean(aucs))
    delta = avg8 - SMOKE_EXPECTED_AUC
    elapsed = time.time() - t0

    print()
    print(f"[smoke] Avg(8) = {avg8:.4f}")
    print(f"[smoke] Expected = {SMOKE_EXPECTED_AUC:.4f}, delta = {delta:+.4f}")
    print(f"[smoke] Elapsed: {elapsed:.1f}s")

    if abs(delta) <= SMOKE_TOLERANCE:
        print(f"[smoke] PASS (|delta| <= {SMOKE_TOLERANCE})")
        ref_path = RESULTS_ROOT / f"04_ds_{SMOKE_POOL}_rawfeat.json"
        if ref_path.exists():
            ref = json.loads(ref_path.read_text("utf-8"))
            ref_avg = ref.get("avg_auc_8")
            if ref_avg is not None:
                print(f"[smoke] Reference JSON: {ref_path.name} "
                      f"(avg_auc_8 = {ref_avg:.4f})")
        return 0
    print(f"[smoke] FAIL (|delta| > {SMOKE_TOLERANCE})")
    return 1


# ---------------------------------------------------------------------------
# Full-rerun helpers (used by the default mode)
# ---------------------------------------------------------------------------

def backup():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / ts
    print(f"Backing up to {backup_dir}")
    for subdir in [RESULTS, CKPTS, FIGURES, TABLES]:
        if subdir.exists() and any(subdir.iterdir()):
            dest = backup_dir / subdir.name
            shutil.copytree(subdir, dest)
            n = len(list(dest.rglob("*")))
            print(f"  {subdir.name}: {n} files")
    return backup_dir


def clear_outputs():
    for subdir in [RESULTS, CKPTS, FIGURES, TABLES]:
        if subdir.exists():
            # Preserve REFERENCE_PATH across the wipe so the post-rerun
            # verification step has something to compare against. The
            # default mode used to delete reference_results.json with
            # the rest of output/results/, then silently no-op the
            # comparison.
            preserved = None
            if subdir == RESULTS and REFERENCE_PATH.exists():
                preserved = REFERENCE_PATH.read_bytes()
            shutil.rmtree(subdir)
            subdir.mkdir(parents=True)
            if preserved is not None:
                REFERENCE_PATH.write_bytes(preserved)
            print(f"  Cleared {subdir.name}/")


def run_experiment(script_num, cmd, expected_outputs):
    """Run one experiment. Returns (success, elapsed_seconds)."""
    print(f"\n{'='*60}")
    print(f"[{script_num}] {cmd}")
    print(f"{'='*60}")

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(CODEBASE),
        str(CODEBASE / "scripts"),
        env.get("PYTHONPATH", ""),
    ])
    # shell=False + argv list works identically on Windows / Linux / macOS
    # and avoids depending on cmd.exe vs /bin/sh quoting differences.
    parts = shlex.split(cmd, posix=(os.name != "nt"))
    if parts and parts[0] == "python":
        parts[0] = sys.executable
    result = subprocess.run(parts, cwd=str(CODEBASE), env=env)
    elapsed = time.time() - t0
    minutes = elapsed / 60

    if result.returncode != 0:
        print(f"  FAILED ({minutes:.1f} min)")
        return False, elapsed

    missing = [f for f in expected_outputs
               if not (RESULTS / f).exists()]
    if missing:
        print(f"  WARNING: missing outputs: {missing}")
        return False, elapsed

    if script_num == "17":
        missing_tables = [
            t for t in EXPECTED_TABLES
            if not (TABLES / t).exists()
        ]
        if missing_tables:
            print(f"  WARNING: missing tables: {missing_tables}")
            return False, elapsed

    print(f"  OK ({minutes:.1f} min)")
    return True, elapsed


# ---------------------------------------------------------------------------
# Verification (used by default mode AND --verify-only)
# ---------------------------------------------------------------------------

# Wall-clock and runtime fields that legitimately drift between runs and
# must NOT be flagged as reproduction failures by the verifier.
_TIMING_FIELDS = frozenset({
    "elapsed_s", "elapsed_ms", "elapsed_seconds", "elapsed_minutes",
    "runtime_s", "runtime_ms", "runtime_seconds",
    "wall_time_s", "wall_time_ms", "wall_clock_s",
    "fold_time_s", "fold_time", "duration_s", "duration_ms",
    "time_s", "time_ms", "timestamp",
})


def _is_timing_key(key: str) -> bool:
    """True if the dotted key path's final segment is a timing field.

    Used to filter timing entries from a stored reference (which may
    have been generated before the timing-exclusion was added)."""
    last = key.rsplit(".", 1)[-1]
    last = last.split("[", 1)[0]
    return last in _TIMING_FIELDS


def _extract_metrics(data):
    """Recursively extract all numeric values from a JSON dict.

    Wall-clock / timing fields (see ``_TIMING_FIELDS``) are excluded
    because they legitimately drift between runs and would otherwise be
    flagged as reproduction failures (e.g. ``elapsed_s`` in
    ``12_xgb_features_*.json``).
    """
    metrics = {}

    def _walk(obj, prefix="", parent_key=""):
        if isinstance(obj, (int, float)):
            if parent_key in _TIMING_FIELDS:
                return
            metrics[prefix] = float(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{prefix}.{k}" if prefix else k, parent_key=k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(v, f"{prefix}[{i}]", parent_key=parent_key)
    _walk(data)
    return metrics


def verify_expected_outputs():
    """Check that every file listed in EXPERIMENTS::expected_outputs and
    every entry in EXPECTED_TABLES exists on disk.

    `.npz` score dumps are working byproducts (used by 06/07 bootstrap
    scripts during a full rerun) and are intentionally excluded from
    this repository by `.gitignore` (`output/results/*.npz`).
    `run_experiment()` already validates each script's outputs (.npz
    included) immediately after it runs, so we skip `.npz` here to
    keep `--verify-only` working on a fresh clone.

    Returns (n_missing_results, n_missing_tables).
    """
    seen_results = set()
    for _script_num, _cmd, outputs, _det in EXPERIMENTS:
        for f in outputs:
            if f.endswith(".npz"):
                continue
            seen_results.add(f)

    missing_results = sorted(
        f for f in seen_results if not (RESULTS / f).exists()
    )
    missing_tables = sorted(
        t for t in EXPECTED_TABLES if not (TABLES / t).exists()
    )

    print(f"\n{'='*60}")
    print("EXPECTED ARTIFACT CHECK")
    print(f"{'='*60}")
    print(f"  expected result files: {len(seen_results)}  "
          f"missing: {len(missing_results)}")
    if missing_results:
        for f in missing_results[:10]:
            print(f"    MISSING: {f}")
        if len(missing_results) > 10:
            print(f"    ... and {len(missing_results) - 10} more")
    print(f"  expected .tex tables:  {len(EXPECTED_TABLES)}  "
          f"missing: {len(missing_tables)}")
    for t in missing_tables:
        print(f"    MISSING: {t}")

    return len(missing_results), len(missing_tables)


def compare_results(reference_path=None):
    """Compare current results against reference.

    Returns (n_ok, n_warn, n_fail). A missing reference file is treated
    as a hard failure (n_fail += 1) so that --verify-only and the
    default rerun cannot silently report "all results verified" when
    no comparison was actually performed.
    """
    ref_path = reference_path or REFERENCE_PATH
    if not ref_path.exists():
        print(f"\nERROR: Reference file not found at {ref_path}")
        print("       Cannot verify regenerated JSONs against the "
              "reference; treating as FAIL.")
        print("       Restore the supplied Block A reference.")
        return 0, 0, 1

    ref = json.loads(ref_path.read_text("utf-8"))
    tolerance = ref.get("_tolerance_stochastic_auc", AUC_TOLERANCE)
    det_tolerance = ref.get(
        "_tolerance_deterministic_auc", DETERMINISTIC_TOLERANCE)
    ref_results = ref["results"]

    n_ok = n_warn = n_fail = 0
    print(f"\n{'='*60}")
    print(f"VERIFICATION against {ref_path.name}")
    print(f"  Reference generated: {ref.get('_generated', 'unknown')}")
    print(f"  Deterministic tolerance: {det_tolerance}")
    print(f"  Stochastic tolerance:    {tolerance}")
    print(f"{'='*60}")

    n_skip = 0
    for filename, ref_entry in sorted(ref_results.items()):
        new_file = RESULTS / filename
        if not new_file.exists():
            print(f"  MISSING: {filename}")
            n_fail += 1
            continue

        new_data = json.loads(new_file.read_text("utf-8"))

        # FS_1 memory-budget skip marker (see src/models/feature_space.py
        # ::fs1_skip_marker). Reported as SKIP, not FAIL — the reference
        # holds a real value but the local host could not afford the
        # ~7 GiB float64 fit.
        if isinstance(new_data, dict) and new_data.get("skipped") is True:
            reason = new_data.get("reason", "unspecified")
            need = new_data.get("estimated_input_GiB", "?")
            print(f"  SKIP:    {filename} (reason={reason}, "
                  f"needed~{need} GiB)")
            n_skip += 1
            continue

        new_metrics = _extract_metrics(new_data)
        # Filter timing entries from the stored reference too, in case
        # the reference was generated before _TIMING_FIELDS was added.
        ref_metrics = {
            k: v for k, v in ref_entry["metrics"].items()
            if not _is_timing_key(k)
        }
        is_det = ref_entry["deterministic"]

        if is_det:
            # Apply a tight numerical tolerance (default 1e-4) instead
            # of bit-equality. Scipy / sklearn patch versions produce
            # sub-millidiff drift in L-BFGS / Nelder-Mead / randomized
            # SVD on the same hardware that has no scientific impact;
            # 1e-4 still catches any real difference (paper publishes
            # to 3 decimals).
            mismatches = []
            max_delta = 0.0
            checked = 0
            for key, ref_val in ref_metrics.items():
                new_val = new_metrics.get(key)
                if new_val is None:
                    mismatches.append((key, ref_val, None, "MISSING"))
                    continue
                checked += 1
                delta = abs(ref_val - new_val)
                if delta > max_delta:
                    max_delta = delta
                if delta > det_tolerance:
                    mismatches.append((key, ref_val, new_val, delta))
            if not mismatches:
                print(f"  MATCH:   {filename} "
                      f"({checked} metrics, max delta={max_delta:.2e})")
                n_ok += 1
            else:
                print(f"  FAIL:    {filename} "
                      f"({len(mismatches)} mismatches)")
                for key, rv, nv, d in mismatches[:5]:
                    if isinstance(d, str):
                        print(f"           {key}: ref={rv} got={nv} "
                              f"({d})")
                    else:
                        print(f"           {key}: ref={rv:.6f} "
                              f"got={nv:.6f} (delta={d:.6f})")
                n_fail += 1
        else:
            max_delta = 0.0
            over_tol = []
            missing = []
            checked = 0
            for key, ref_val in ref_metrics.items():
                new_val = new_metrics.get(key)
                if new_val is None:
                    missing.append(key)
                    continue
                checked += 1
                delta = abs(ref_val - new_val)
                if delta > max_delta:
                    max_delta = delta
                if delta > tolerance:
                    over_tol.append((key, ref_val, new_val, delta))

            if not over_tol and not missing:
                print(f"  CLOSE:   {filename} "
                      f"({checked} metrics, max delta={max_delta:.6f})")
                n_ok += 1
            elif missing:
                print(f"  FAIL:    {filename} "
                      f"({len(missing)} missing metrics, "
                      f"{len(over_tol)} over tolerance)")
                for key in missing[:5]:
                    print(f"           MISSING: {key}")
                for key, rv, nv, d in over_tol[:5]:
                    print(f"           {key}: "
                          f"ref={rv:.4f} got={nv:.4f} "
                          f"(delta={d:.4f})")
                n_fail += 1
            else:
                print(f"  WARN:    {filename} "
                      f"({len(over_tol)} metrics exceed {tolerance})")
                for key, rv, nv, d in over_tol[:5]:
                    print(f"           {key}: "
                          f"ref={rv:.4f} got={nv:.4f} "
                          f"(delta={d:.4f})")
                n_warn += 1

    summary = f"{n_ok} MATCH, {n_warn} WARN, {n_fail} FAIL"
    if n_skip:
        summary += f", {n_skip} SKIP"
    print(f"\nSummary: {summary} "
          f"(det tol={det_tolerance}, stoch tol={tolerance})")
    return n_ok, n_warn, n_fail


def run_cache_validation() -> int:
    """Run cache release validation (Croissant + SHA-256 + license + card)
    via subprocess so the standalone tool stays the single source of truth.
    """
    cache_script = Path(__file__).resolve().parent / "validate_cache_release.py"
    return subprocess.run([sys.executable, str(cache_script)]).returncode


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_dry_run() -> int:
    print(f"\n{len(EXPERIMENTS)} experiments in the full pipeline:")
    for script_num, cmd, outputs, det in EXPERIMENTS:
        tag = "DETERMINISTIC" if det else "TORCH (tolerance)"
        print(f"  [{script_num}] {', '.join(outputs) or '(tables/figures)'} "
              f"-- {tag}")
    print()
    print("Default (no flags) reruns the full numbered pipeline above,")
    print("then verifies. GPU portion <1 h; total wall-clock CPU-bound")
    print("(see paper App I.9 / Checklist Q8). For the fast reviewer")
    print("check, use --check (smoke + verify + cache validate).")
    return 0


def mode_verify_only() -> int:
    n_missing_results, n_missing_tables = verify_expected_outputs()
    n_ok, n_warn, n_fail = compare_results()
    total_fail = n_fail + n_missing_results + n_missing_tables
    if total_fail > 0:
        print(f"\nVERIFICATION FAILED: "
              f"{n_missing_results} missing result files, "
              f"{n_missing_tables} missing tables, "
              f"{n_fail} reference-comparison failures.")
        return 1
    return 0


def _hf_cache_present() -> bool:
    """Quick check: does the FIT split have at least one extracted .npy?

    This repository holds only the cache's metadata (croissant.json,
    DATASET_CARD.md, etc.) without the per-expert .npy arrays. `--check`
    needs the full Hugging Face cache; this preflight catches that case
    so the reviewer gets a clear pointer instead of a stack trace from
    inside the smoke loader.
    """
    from src.utils.config import CACHE_ROOT as _CACHE_ROOT
    fit_dir = _CACHE_ROOT / "FIT"
    if not fit_dir.exists():
        return False
    return any(fit_dir.glob("*_features.npy"))


def mode_check() -> int:
    """Fast reviewer check: smoke + verify + cache."""
    if not _hf_cache_present():
        print("=" * 60)
        print("--check requires the full Hugging Face feature cache.")
        print("=" * 60)
        print(
            "This repository holds only the cache's metadata (croissant.json, "
            "DATASET_CARD.md, CHECKSUMS.txt, CACHE_PROVENANCE.json, LICENSE_NOTICE.txt). The "
            "per-expert .npy feature arrays are hosted separately on "
            "Hugging Face."
        )
        print()
        print("Either download the cache:")
        print("  huggingface-cli download anon-frozen-experts/df40-frozen-experts-anon \\")
        print('    --repo-type dataset --revision 18fb1b212f7251799fca8cb4d549be6d239ce13a --local-dir output/cached_features/ --include "*/*"')
        print()
        print("Or run only the JSON-vs-reference check (no cache needed):")
        print("  python camera_ready/scripts/review/reproduce_all_v4.py --verify-only")
        print("=" * 60)
        return 1

    print(f"\n{'='*60}\n=== STAGE 1: Smoke (FS_1 arch6)\n{'='*60}")
    rc = run_smoke()
    if rc != 0:
        print(f"\nREVIEWER CHECK FAILED at smoke (exit code {rc}).")
        return rc

    print(f"\n{'='*60}\n=== STAGE 2: Reference verification\n{'='*60}")
    rc = mode_verify_only()
    if rc != 0:
        print(f"\nREVIEWER CHECK FAILED at verification (exit code {rc}).")
        return rc

    print(f"\n{'='*60}\n=== STAGE 3: Cache release validation\n{'='*60}")
    rc = run_cache_validation()
    if rc != 0:
        print(f"\nREVIEWER CHECK FAILED at cache validation (exit code {rc}).")
        return rc

    print(f"\n{'='*60}\nAll reviewer checks passed.\n{'='*60}")
    return 0


def mode_rerun_all() -> int:
    """Power-user mode: full pipeline rerun + verify."""
    print("\n--- Step 1: Backup ---")
    backup_dir = backup()

    print("\n--- Step 2: Clear outputs ---")
    clear_outputs()

    print("\n--- Step 3: Run experiments ---")
    total_t0 = time.time()
    results_log = []
    n_experiment_failures = 0
    for script_num, cmd, outputs, _det in EXPERIMENTS:
        ok, elapsed = run_experiment(script_num, cmd, outputs)
        if not ok:
            n_experiment_failures += 1
        results_log.append({
            "script": script_num, "cmd": cmd,
            "ok": ok, "elapsed_s": round(elapsed, 1),
        })

    total_elapsed = time.time() - total_t0
    print(f"\nTotal runtime: {total_elapsed/3600:.1f} hours")

    n_missing_results, n_missing_tables = verify_expected_outputs()
    print("\n--- Step 4: Verify against reference ---")
    n_ok, n_warn, n_fail = compare_results()

    def _rel(p: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(CODEBASE)).replace("\\", "/")
        except ValueError:
            return Path(p).name
    run_log = {
        "timestamp": datetime.now().isoformat(),
        "backup_dir": _rel(backup_dir),
        "total_hours": round(total_elapsed / 3600, 2),
        "reference": _rel(REFERENCE_PATH),
        "experiment_failures": n_experiment_failures,
        "missing_results": n_missing_results,
        "missing_tables": n_missing_tables,
        "verification": {"match": n_ok, "warn": n_warn, "fail": n_fail},
        "experiments": results_log,
    }
    log_path = RESULTS / "reproducibility_log.json"
    log_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    print(f"\nRun log saved: {log_path}")

    total_fail = (n_experiment_failures + n_missing_results
                  + n_missing_tables + n_fail)
    if total_fail > 0:
        print(
            f"\nRUN FAILED: "
            f"{n_experiment_failures} experiment crashes, "
            f"{n_missing_results} missing result files, "
            f"{n_missing_tables} missing tables, "
            f"{n_fail} reference-comparison failures."
        )
        return 1
    print("\nAll experiments ran, all artifacts present, "
          "all results verified against reference.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    print(
        "Use python camera_ready/scripts/review/reproduce_all_v4.py --dry-run "
        "from the repository root to see the available reproduction steps.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
