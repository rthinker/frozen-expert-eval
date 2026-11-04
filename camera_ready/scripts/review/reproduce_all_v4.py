"""Reproduce and verify the DF40 experiments from the repository root.

    python camera_ready/scripts/review/reproduce_all_v4.py --dry-run
    python camera_ready/scripts/review/reproduce_all_v4.py --verify-only
    python camera_ready/scripts/review/reproduce_all_v4.py --check
    python camera_ready/scripts/review/reproduce_all_v4.py

--dry-run shows the plan. --verify-only compares stored results with the
references and checks archived score, checkpoint and ledger digests without
needing the feature cache. --check refits standard-split
arch6 FS1 as a smoke test, verifies the stored results and validates the cache.
The full run refits combiners from cached features. It covers the reviewed
experiments (A), the camera-ready LOGO runs (B) and the additional analyses,
Figure A2 and DF40 tables (C). Figure A1 and the Appendix A sensitivity checks
have separate commands in camera_ready/REPRODUCTION.md.

Use --only C09 for a single step, --from C09 to start at that step, or
--blocks A,B to select blocks. Use --df40-json to enable the steps that read
DF40 split lists, and --df40-raw for the source-image size check. The run
record identifies stored outputs retained for skipped checks.
--with-reproduction-gate adds the arch6 LOGO run without content exclusions
for the G0 and G2 comparisons.

The harness backs up existing results before a rerun. Run logs and coverage
reports are under camera_ready/output/logs/<run>/. Use --resume-run <run> to
continue with the recorded options. The harness validates completed outputs
and checkpoints against their current inputs before reuse.

See camera_ready/INSTALL.md for installation, camera_ready/REPRODUCTION.md for
execution and memory requirements, and camera_ready/RELEASE_MANIFEST.md
for the steps, outputs and comparison tolerances.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

CR = Path(__file__).resolve().parents[2]            # camera_ready/
ROOT = CR.parent                                    # repository root (Set 1, the reviewed code)
RESULTS2 = CR / "output" / "results"
MIT = RESULTS2 / "mitigation"
ANA = RESULTS2 / "analyses"
TK = ANA / "three_class"
LOGS2 = CR / "output" / "logs"
BACKUP2 = CR / "output" / "reproducibility_backup"
# Pinned camera-ready reference and comparison settings.
REFERENCE2 = CR / "reference" / "reference_results_camera_ready_v2.json"
INPUTS = CR / "inputs"
RUN_INPUTS = CR / "run_inputs"
STORED_LEDGER = CR / "provenance" / "05_number_provenance.csv"
STORED_EVIDENCE = CR / "reference" / "stored_evidence.sha256"
REVIEWED_HARNESS = ROOT / "scripts" / "review" / "reproduce_all.py"

DET_TOL = 1e-4          # as the reviewed harness: deterministic methods
STOCH_TOL = 0.005       # as the reviewed harness: seed-averaged neural methods
ENV_TOL = 0.005         # FS2 on a GPU: reported as ENV when exceeded (environment-dependent, App. F note)
# ENV_MAX bounds environment-dependent or stochastic differences. It is about three times the largest recorded
# change (0.033, FS2 on the 101-sample clip4 HeyGen fold across RTX 2070 Super and RTX 5080; gate G0log).
ENV_MAX = 0.1
# values that move in steps when one fold or seed changes: they need only be finite
ENV_STEPWISE = re.compile(r"(_p$|\.wins$|_wins$|\.positive_folds$|per_seed|best_epoch$)")   # fs_wins
# configuration recorded next to environment-dependent values: it must equal the reference
ENV_FIXED = re.compile(r"(\.metadata\.(dropout|expert_dropout|hidden|lr|proj_dim|seeds|epochs)(\[\d+\])?$"
                       r"|\.(seed|epochs|paper_aligned)$|FS2_seed_count$|\.n_folds$|\.n$)")
# environment-dependent values with documented alternatives: the unique training paths of DF40's aggregate list,
# 703,223 in the authors' copy (without e4e's 29,126 images) and 732,349 in the public DF40_all.json
ENV_DOCUMENTED = {"camera_ready/output/results/analyses/gate_expert_side_check.json":
                  {"expert_train_unique_paths.df40": (703223.0, 732349.0)}}
# result files that may be a skip marker, with the fields the marker must carry. The reviewed code writes a
# memory-budget marker for the full21 FS1 fit (Block A, checked by the reviewed harness); no step of blocks B and C
# writes one, so a marker there fails.
SKIP_PERMITTED = {}
# what each compared value may be, by its key (validated against every value of both references: none is
# outside its rule). p-values, counts and AUROC-type values; an AUROC-type value is bounded only when its reference
# lies in [0, 1] (a few leaf names, such as "test", name a count in one result and an AUROC in another).
VALUE_P = re.compile(r"^(p|p_value|pvalue|p_gt_zero)$|_p$|sign_test_p$|wilcoxon_p$|signflip_p$")
VALUE_COUNT = re.compile(r"^(wins|positive_folds|folds_positive|n|n_test|n_fit|n_select|n_folds|n_excluded|"
                         r"n_excluded_training_rows|best_epoch|epochs|n_runs|n_seeds)$|_wins$|_count$")
VALUE_NOT_PROB = re.compile(r"(delta|diff|minus|_vs_|ci95|ci_95|_ci$|bca|statistic|^z0$|acceleration|boot_mean|"
                            r"distortion|effect|margin|change|std|^sd_|^se$|range|temperature|tolerance)")
VALUE_PROB = re.compile(r"(auc|auroc|auprc|average_precision|avg8|^macro|_macro$|macro_|ceiling|best_ps|raw_feat|"
                        r"^fs_mlp|^fs[12]$|per_seed|^power$|prevalence|^select$|^test$)")
VALUE_PROB_PARENT = {"ps_means", "rungs", "auroc", "mean_auroc"}       # method -> AUROC
# AUROC- or AUPRC-named parents of values (auprc.per_fold_fs1[i], raw_feat_auc.<generator>)
VALUE_PROB_PARENT_NAME = re.compile(r"(auc|auroc|auprc|average_precision)")
# every status a result records must be one of these (compare_runs_v2 writes PASS/report/FAIL, run C and the audit
# PASS/FAIL, the three-class result COMPLETE)
STATUS_KEY = re.compile(r"(^|_)status$|^gate_G\d+$")
STATUS_OK = frozenset({"PASS", "report", "COMPLETE"})


def value_rule(key):
    """The validity rule of a compared value: "p", "count", "prob" or None."""
    parts = key.split(".")
    leaf = re.sub(r"\[\d+\]$", "", parts[-1])
    parent = re.sub(r"\[\d+\]$", "", parts[-2]) if len(parts) > 1 else ""
    if VALUE_P.search(leaf):
        return "p"
    if VALUE_COUNT.search(leaf):
        return "count"
    if VALUE_NOT_PROB.search(leaf) or any(VALUE_NOT_PROB.search(p) for p in parts[:-1]
                                          if p.startswith(("comparisons", "stats"))):
        return None
    if (parent in VALUE_PROB_PARENT or VALUE_PROB.search(leaf)
            or (VALUE_PROB_PARENT_NAME.search(parent) and not VALUE_NOT_PROB.search(parent))):
        return "prob"
    return None


def invalid_value(key, have, want, nums):
    """Why a finite compared value is impossible for its kind, or None. `nums`: every numeric value of the result."""
    rule = value_rule(key)
    if rule == "p" and not 0.0 <= have <= 1.0:
        return "p-value outside [0, 1]"
    if rule == "count":
        if have < 0 or not float(have).is_integer():
            return "count that is not a non-negative integer"
        leaf = re.sub(r"\[\d+\]$", "", key.split(".")[-1])
        if leaf.endswith("wins") or leaf in ("positive_folds", "folds_positive"):
            n = nums.get(key[:len(key) - len(leaf)] + "n", nums.get(key[:len(key) - len(leaf)] + "n_folds"))
            if n is not None and have > n:
                return f"more wins than folds ({n:g})"
        if leaf == "best_epoch":
            # the epochs of the value's own run (its nearest ancestor holding "epochs"), else any
            parts = key.split(".")
            own = next((nums[k] for k in (".".join(parts[:i] + ["epochs"]) for i in range(len(parts) - 1, -1, -1))
                        if k in nums), None)
            epochs = [own] if own is not None else [v for k, v in nums.items()
                                                    if k.endswith(".epochs") or k == "epochs"]
            if have < 1 or (epochs and have > max(epochs)):
                return "early-stopping epoch outside the epochs run"
    if rule == "prob" and 0.0 <= want <= 1.0 and not 0.0 <= have <= 1.0:
        return "AUROC or probability outside [0, 1]"
    return None


def bad_statuses(data):
    """Every recorded status that is not PASS, report or COMPLETE, as (key, value)."""
    out = []

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, str) and STATUS_KEY.search(str(k)):
                    if v not in STATUS_OK:
                        out.append((key, v))
                else:
                    walk(v, key)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{prefix}[{i}]")
    walk(data)
    return out


def rel(p) -> str:
    return Path(p).resolve().relative_to(ROOT).as_posix()


# ---------------------------------------------------------------------------
# The reviewed harness (Set 1), imported read-only
# ---------------------------------------------------------------------------

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reviewed():
    return load_module("reviewed_reproduce_all", REVIEWED_HARNESS)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    id: str
    block: str
    paper: str
    cmd: tuple                      # argv; "python" becomes this interpreter
    outputs: tuple = ()             # paths relative to the repository root
    gpu: bool = False               # uses the GPU when one is visible (FS2 or neural heads)
    needs: str = ""                 # "df40": runs only with --df40-json
    optional: str = ""              # "repro-gate": runs only with --with-reproduction-gate
    extra: dict = field(default_factory=dict, hash=False, compare=False)


def _r(*parts) -> str:
    return "/".join(parts)


CRS = "camera_ready/scripts"
MITR = "camera_ready/output/results/mitigation"
CD = f"{MITR}/content_disjoint"
MECH = f"{MITR}/mechanism"
EXCL = f"{MITR}/exclusions"
G0 = f"{MITR}/g0_reproduce"
ANAR = "camera_ready/output/results/analyses"
TKR = f"{ANAR}/three_class"
STANDARD_STATS = "camera_ready/output/results/full_ladder_stats.json"   # written by step C08
DF40_PREPARED = "camera_ready/output/df40_lists_cacheform"   # written by step C00 from --df40-json
DF40_LIST_NAMES = load_module("df40_list_preparer", CR / "scripts" / "analyses" / "prepare_df40_lists_v2.py").NAMES
POOLS_B = (("arch6", 256), ("clip4", 32), ("strong4", 256), ("weak6", 256))
GPU_SCRIPTS_A = {"05", "07", "10", "11", "15"}      # FS2 / MLP training in the reviewed pipeline


def steps_a(rv):
    out = []
    for i, (num, cmd, outputs, _det) in enumerate(rv.EXPERIMENTS, 1):
        argv = tuple(shlex.split(cmd))
        declared = tuple(f"output/results/{o}" for o in outputs)
        if num == "05":
            declared += block_a_checkpoints(argv)
        if num == "17":
            declared += tuple(f"output/tables/{name}" for name in rv.EXPECTED_TABLES)
        out.append(Step(f"A{i:02d}", "A", f"reviewed scripts/{num}", argv,
                        declared, gpu=num in GPU_SCRIPTS_A,
                        extra={"required_inputs": block_a_prerequisites(argv)}))
    return out


def block_a_checkpoints(cmd):
    """Exact outputs of script 05; the bootstrap consumes this same seed set."""
    variant = _arg(cmd, "--combiner_type") or "mlp"
    if variant == "pca":
        return ()  # the PCA branch writes its JSON only, before the neural seed loop
    import yaml
    cfg = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"))
    seeds = list(cfg["feature_assembly"]["seeds"])
    if "--seeds" in cmd:
        seeds = []
        for value in cmd[cmd.index("--seeds") + 1:]:
            if value.startswith("--"):
                break
            seeds.append(int(value))
    pool = _arg(cmd, "--pool")
    dim = _arg(cmd, "--proj_dim") or "64"
    tag = _arg(cmd, "--train_tag") or "FIT"
    suffix = "" if tag == "FIT" else f"_{tag}"
    return tuple(f"output/checkpoints/{pool}_{variant}_d{dim}_seed{seed}_best{suffix}.pt" for seed in seeds)


def block_a_prerequisites(cmd):
    script = next((Path(t).name for t in cmd if t.endswith(".py")), "")
    if script not in ("06_bootstrap_rawfeat.py", "07_bootstrap_mlp.py"):
        return ()
    pool = _arg(cmd, "--pool")
    tag = _arg(cmd, "--train_tag") or "FIT"
    suffix = "" if tag == "FIT" else f"_{tag}"
    needed = (f"output/results/04_ds_{pool}{suffix}.json", f"output/results/04_ds_{pool}_scores{suffix}.npz")
    return needed + (block_a_checkpoints(cmd) if script.startswith("07_") else ())


def check_plan_prerequisites(plan):
    """Refuse a partial selection whose generated inputs cannot be supplied."""
    supplied = set()
    for step in plan:
        for name in step.extra.get("required_inputs", ()):
            if name in supplied:
                continue
            path = ROOT / name
            why = "missing" if not path.is_file() else _damaged(path)
            if why:
                raise SystemExit(f"{step.id} requires {name} ({why}). Select the producing 04/05 steps "
                                 "as well, or provide their complete outputs. Nothing was changed.")
        supplied.update(step.outputs)


def steps_b():
    s = []
    n = 0

    def add(paper, cmd, outputs=(), gpu=False, optional="", extra=None):
        nonlocal n
        n += 1
        s.append(Step(f"B{n:02d}", "B", paper, tuple(cmd), tuple(outputs), gpu=gpu, optional=optional,
                      extra=extra or {}))

    py = "python"
    for pool, _ in POOLS_B:
        add(f"Sec. 5.1, App. F: training rows identical to a held-out test image ({pool}; gate G1)",
            [py, f"{CRS}/runs/logo_exclusions.py", "--pool", pool, "--codebase", ".",
             "--out", f"{EXCL}/logo_exclusions_{pool}.json"], [f"{EXCL}/logo_exclusions_{pool}.json"])
    # optional: the reproduction gate (no exclusion; must equal the stored reviewed run)
    add("App. F: arch6 LOGO without exclusion (gate G0 against the reviewed run)",
        [py, f"{CRS}/runs/11_disjoint_data_v4.py", "--pool", "arch6", "--proj_dim", "256", "--exclusion", "none",
         "--out-dir", G0, "--codebase", ".", "{gpu_name}", "{resume}"],
        [f"{G0}/11_disjoint_arch6_d256.json"], gpu=True, optional="repro-gate")
    add("App. F: arch6 richer inputs without exclusion (gate G0)",
        [py, f"{CRS}/runs/08_logo_extended_rungs_v3.py", "--pool", "arch6", "--regime", "both", "--exclusion",
         "none", "--out-dir", G0, "--codebase", ".", "{resume}"],
        [f"{G0}/A3c_logo_extended_rungs_arch6.json"], optional="repro-gate")
    add("App. F: gate G0 (reproduces the stored reviewed LOGO run)",
        [py, f"{CRS}/runs/compare_runs_v2.py", "g0", "--pool", "arch6", "--new-dir", G0],
        [f"{G0}/gate_G0_arch6.json"], optional="repro-gate")
    # arch6, content-disjoint: Table 1 LOGO row, Tables A11 to A13
    add("Table 1 and Tables A11-A13: arch6 LOGO, ten-method ladder, FS1 and FS2 (content-disjoint)",
        [py, f"{CRS}/runs/11_disjoint_data_v4.py", "--pool", "arch6", "--proj_dim", "256", "--exclusion", "content",
         "--manifest", f"{EXCL}/logo_exclusions_arch6.json", "--out-dir", CD, "--codebase", ".", "{gpu_name}",
         "{resume}"], [f"{CD}/11_disjoint_arch6_d256.json"], gpu=True)
    add("App. F: arch6 richer prediction-space inputs inside LOGO",
        [py, f"{CRS}/runs/08_logo_extended_rungs_v3.py", "--pool", "arch6", "--regime", "both", "--exclusion",
         "content", "--manifest", f"{EXCL}/logo_exclusions_arch6.json", "--out-dir", CD, "--codebase", ".",
         "{resume}"], [f"{CD}/A3c_logo_extended_rungs_arch6.json"])
    add("Table 1: arch6 LOGO ceiling over the full ladder",
        [py, f"{CRS}/runs/09_logo_combined_ceiling_v3.py", "--logo-json", f"{CD}/11_disjoint_arch6_d256.json",
         "--rungs-json", f"{CD}/A3c_logo_extended_rungs_arch6.json"], [f"{CD}/A3c_logo_combined_ceiling_arch6.json"])
    # G2 compares two runs of one environment; execute_plan skips it when only one of them is recomputed
    add("App. F: gate G2 (folds without excluded rows equal the run without exclusion)",
        [py, f"{CRS}/runs/compare_runs_v2.py", "g2", "--pool", "arch6", "--ref-dir", G0, "--new-dir", CD],
        [f"{CD}/gate_G2_arch6.json"], extra={"premise": "g2"})
    # mechanism: Table A9 LOGO columns, D.4
    add("App. D.4, Table A9: arch6 width controls on the standard split (gate G4)",
        [py, f"{CRS}/runs/05_run_c_width_matched_v5.py", "--pool", "arch6", "--protocol", "standard",
         "--out-dir", MECH, "--codebase", "."], [f"{MECH}/run_c_v2_standard_arch6.json"])
    add("App. D.4, Table A9: arch6 width controls under LOGO",
        [py, f"{CRS}/runs/05_run_c_width_matched_v5.py", "--pool", "arch6", "--protocol", "logo", "--exclusion",
         "content", "--manifest", f"{EXCL}/logo_exclusions_arch6.json", "--out-dir", MECH, "--codebase", ".",
         "{resume}"], [f"{MECH}/run_c_v2_logo_arch6.json"])
    add("App. D.4: gate G5 arch6 (PS_4 1e-9, FS1 1e-4)",
        [py, f"{CRS}/runs/compare_runs_v2.py", "g5", "--pool", "arch6", "--logo-dir", CD, "--mech-dir", MECH,
         "--fs1-tol", "1e-4", "--out", f"{MECH}/gate_G5_arch6_fs1tol1e-4.json"],
        [f"{MECH}/gate_G5_arch6_fs1tol1e-4.json"])
    # boundary pools: Table 1 rows, Sec. 5.3
    reviewed_evidence = {"clip4": "A4_logo_clip4.txt", "strong4": "A5_logo_strong4.txt", "weak6": "A4_logo_weak6.txt"}
    for pool, d in POOLS_B[1:]:
        add(f"Table 1, Sec. 5.3: {pool} LOGO, ten-method ladder, FS1 and FS2 (content-disjoint)",
            [py, f"{CRS}/runs/11_disjoint_data_v4.py", "--pool", pool, "--proj_dim", str(d), "--exclusion",
             "content", "--manifest", f"{EXCL}/logo_exclusions_{pool}.json", "--out-dir", CD, "--codebase", ".",
             "{gpu_name}", "{resume}"], [f"{CD}/11_disjoint_{pool}_d{d}.json"], gpu=True)
        add(f"Sec. 5.3: gate G0log {pool} (deterministic methods match the reviewed numerical record)",
            [py, f"{CRS}/runs/compare_runs_v2.py", "g0log", "--pool", pool, "--proj_dim", str(d), "--new-dir", CD,
             "--untouched-only", "--log", f"camera_ready/inputs/{reviewed_evidence[pool]}"],
            [f"{CD}/gate_G0log_{pool}.json"])
    for pool, d in POOLS_B[1:]:
        add(f"App. F: {pool} richer prediction-space inputs inside LOGO",
            [py, f"{CRS}/runs/08_logo_extended_rungs_v3.py", "--pool", pool, "--regime", "both", "--exclusion",
             "content", "--manifest", f"{EXCL}/logo_exclusions_{pool}.json", "--out-dir", CD, "--codebase", ".",
             "{resume}"], [f"{CD}/A3c_logo_extended_rungs_{pool}.json"])
        add(f"Table 1: {pool} LOGO ceiling over the full ladder",
            [py, f"{CRS}/runs/09_logo_combined_ceiling_v3.py", "--logo-json", f"{CD}/11_disjoint_{pool}_d{d}.json",
             "--rungs-json", f"{CD}/A3c_logo_extended_rungs_{pool}.json"],
            [f"{CD}/A3c_logo_combined_ceiling_{pool}.json"])
    add("App. D.4, Table A9: clip4 width controls on the standard split (gate G4)",
        [py, f"{CRS}/runs/05_run_c_width_matched_v5.py", "--pool", "clip4", "--protocol", "standard",
         "--out-dir", MECH, "--codebase", "."], [f"{MECH}/run_c_v2_standard_clip4.json"])
    add("App. D.4, Table A9: clip4 width controls under LOGO",
        [py, f"{CRS}/runs/05_run_c_width_matched_v5.py", "--pool", "clip4", "--protocol", "logo", "--exclusion",
         "content", "--manifest", f"{EXCL}/logo_exclusions_clip4.json", "--out-dir", MECH, "--codebase", ".",
         "{resume}"], [f"{MECH}/run_c_v2_logo_clip4.json"])
    add("App. D.4: gate G5 clip4 (PS_4 1e-9, FS1 1e-4)",
        [py, f"{CRS}/runs/compare_runs_v2.py", "g5", "--pool", "clip4", "--proj_dim", "32", "--logo-dir", CD,
         "--mech-dir", MECH, "--fs1-tol", "1e-4"], [f"{MECH}/gate_G5_clip4.json"])
    add("App. F: FS1 score precision of the saved arrays (no fitting)",
        [py, f"{CRS}/review/check_fs1_score_precision_v2.py", "--out", f"{MITR}/fs1_score_precision_check.json"],
        [f"{MITR}/fs1_score_precision_check.json"])
    add("Statistics of the LOGO runs: self-test against the stored reviewed runs",
        [py, f"{CRS}/authoring/full_ladder_stats_v13.py", "--self-test"])
    # C08a recomputes the Table 2 statistics after C08. B32 can defer the comparison until then.
    add("Table 1 LOGO rows, Table A9 LOGO columns, Tables A11-A13, Sec. 5: statistics",
        [py, f"{CRS}/authoring/full_ladder_stats_v13.py", "--run-dir", CD, "--mech-dir", MECH, "--out-dir", MITR,
         "--standard-stats", STANDARD_STATS, "--reviewed-results", "output/results", "{defer_audit}"],
         [f"{MITR}/mitigation_stats.json", f"{MITR}/mitigation_numbers.md"], extra={"requires_steps": ("B31",)})
    return s


# Expected public authoring products; independent of the files the builder emits.
TABLE_LOCATION_REFERENCE = CR / "provenance" / "paper_table_locations.json"
CHECK_TABLES = (
    "t01_logo_arch6_primary", "t02_expanded_ladder", "t03_width_controls", "t04_k3_heads",
    "t05_k3_fixed_output_views", "t06_cost_access_ledger", "t07_reuse_per_generator",
    "t08_strict_refit_arms", "t09_logo_boundary_pools", "t10_appendix_h_expert_means",
    "t11_appendix_h_individual_experts", "t12_release_level_reuse", "t13_integrity_counts",
    "t14_expert_exposure_by_training_list", "t15_expert_exposure_by_pool", "t16_main_results",
    "t17_pool_deltas", "t18_full_ladder", "t19_logo_arch6_folds", "t20_logo_inferential",
    "t21_rq2_both_protocols",
)


def table_deliverables():
    """Relative paths for 25 paper tables, 3 additional tables, 21 checks and 3 indexes/reports."""
    locations = json.loads(TABLE_LOCATION_REFERENCE.read_text(encoding="utf-8"))
    tables = locations["tables"]
    assert sum(v["status"] == "printed" for v in tables.values()) == 25
    assert sum(v["status"] == "additional" for v in tables.values()) == 3
    return ("05_number_provenance.csv", "06_posted_number_check.md", "paper_table_locations.json",
            *(f"tables/{name}.tex" for name in CHECK_TABLES),
            *(f"tables/{'paper' if value['status'] == 'printed' else 'additional'}/{label}.tex"
              for label, value in sorted(tables.items())))


def check_table_deliverables(output, since=None):
    """Reject missing/empty/stale products, wrong current table sets and an inconsistent location map."""
    errors = []
    expected = table_deliverables()
    for name in expected:
        p = output / name
        if not p.is_file() or p.stat().st_size == 0:
            errors.append(f"missing or empty: {name}")
        elif since is not None and p.stat().st_mtime < since - 2:
            errors.append(f"not rebuilt in this run: {name}")
    for group in ("paper", "additional"):
        want = {Path(name).name for name in expected if name.startswith(f"tables/{group}/")}
        got = {p.name for p in (output / "tables" / group).glob("*.tex")}
        if got != want:
            errors.append(f"{group} table set differs: missing={sorted(want-got)}, unexpected={sorted(got-want)}")
    try:
        got = json.loads((output / "paper_table_locations.json").read_text(encoding="utf-8"))
        want = json.loads(TABLE_LOCATION_REFERENCE.read_text(encoding="utf-8"))
        if got != want:
            errors.append("paper_table_locations.json differs from the current public location map")
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"invalid location map: {exc}")
    return errors


def steps_c():
    s = []
    n = 0

    def add(paper, cmd, outputs=(), gpu=False, needs=""):
        nonlocal n
        n += 1
        s.append(Step(f"C{n:02d}", "C", paper, tuple(cmd), tuple(outputs), gpu=gpu, needs=needs))

    py = "python"
    A = f"{CRS}/analyses"
    # DF40's public split lists, rewritten in the cache's path form for the steps that read them
    s.append(Step("C00", "C", "DF40's public split lists in the cache's path form (for C04 and C14-C16)",
                  (py, f"{A}/prepare_df40_lists_v2.py", "--public", "{df40_public}", "--out", DF40_PREPARED),
                  tuple(f"{DF40_PREPARED}/{name}.json" for name in (*DF40_LIST_NAMES, "PREPARED")), needs="df40"))
    # Appendix A
    # C01 copies counts from the stored result when the DF40 lists or image root are unavailable.
    add("App. A, overlap between combiner training and test data, with its table: reuse counts (feature cache; "
        "DF40 lists and image root when given)",
        [py, f"{CRS}/review/check_split_reuse_v3.py", "--cache", "output/cached_features", "--ids",
         "camera_ready/output/ids", "--out", "camera_ready/output/results/split_reuse_check.json",
         "--carry-from", "camera_ready/inputs/split_reuse_check.json", "{reuse_routes}"],
        ["camera_ready/output/results/split_reuse_check.json"])
    add("App. A: train/test overlap of the standard split by path level",
        [py, f"{A}/appF_strict_refit.py", "--pool", "arch6", "--diagnose"], [f"{ANAR}/run_b_diagnose_arch6.json"])
    add("App. A: rows reused by content but not by path (12,893 of 17,925)",
        [py, f"{A}/appA_content_reuse.py"], [f"{ANAR}/A4_content_only_rows.json"])
    add("App. A: expert-side exposure (DF40 training lists vs held-out generators)",
        [py, f"{A}/appA_expert_side_v3.py", "--json_dir", "{df40}"], [f"{ANAR}/gate_expert_side_check.json"],
        needs="df40")
    add("App. A: bit identity of the reused rows in features and logits (11,817 of 11,886 identical on every "
        "arch6 expert)",
        [py, f"{A}/appA_bit_identity.py"], [f"{ANAR}/verify_bit_identity.json"])
    # Appendix B
    add("App. B, Table A3: richer prediction-space inputs, arch6",
        [py, f"{A}/appB_richer_inputs.py", "--pool", "arch6", "--xgb_mode", "cv"],
        [f"{ANAR}/run_a_extended_ladder_arch6.json"])
    add("App. B, Table A3: richer prediction-space inputs, clip4",
        [py, f"{A}/appB_richer_inputs.py", "--pool", "clip4", "--xgb_mode", "cv"],
        [f"{ANAR}/run_a_extended_ladder_clip4.json"])
    add("Table 1 standard row, Tables 2, A13, A14 and A15; additional protocol examples: statistics against the full ladder (standard split)",
        [py, f"{CRS}/authoring/full_ladder_stats_standard_v3.py", "--codebase", ".", "--run-a-dir", ANAR,
         "--logo-ceiling", "camera_ready/inputs/A3c_logo_combined_ceiling_arch6.json",
         "--out", STANDARD_STATS],
        [STANDARD_STATS])
    # Table 2's audit after its inputs. B32 derives it from the standard-split statistics on disk; this
    # step recomputes it from those C08 has just written and checks the identities between its inputs (the id keeps
    # the numbers of the later steps)
    s.append(Step("C08a", "C", "Table 2: standard-LOGO contrast and baseline inflation, from the "
                               "standard-split statistics of C08 (identities checked)",
                  (py, f"{CRS}/authoring/full_ladder_stats_v13.py", "--audit-only", "--stats",
                   f"{MITR}/mitigation_stats.json", "--standard-stats", STANDARD_STATS, "--reviewed-results",
                   "output/results"),
                  (f"{MITR}/mitigation_stats.json", f"{MITR}/mitigation_numbers.md")))
    # Appendix D.4
    add("App. D.4, Table A9 standard split: width-matched controls",
        [py, f"{A}/appD_width_controls.py", "--pool", "arch6"], [f"{ANAR}/run_c_width_matched_arch6.json"])
    # Appendix F
    add("Additional analysis: simulated detection thresholds at 8 folds (power simulation, rebuilt)",
        [py, f"{A}/appF_power_mde_v2.py"], [f"{ANAR}/appF_power_mde.json"])
    add("App. F, Table A14, Sec. 5.1: strict path-disjoint refit",
        [py, f"{A}/appF_strict_refit.py", "--pool", "arch6", "--levels", "family", "--stratify"],
        [f"{ANAR}/run_b_path_disjoint_arch6_family.json", f"{ANAR}/run_b_diagnose_arch6.json"])  # it
    # rewrites C02's diagnosis file too, which therefore counts as its output (not an input of C08 and its readers)
    # Appendix H
    add("App. H, Table A16: individual experts under the corrected score convention",
        [py, f"{CRS}/authoring/regenerate_appendix_h_v3.py"],
        ["camera_ready/output/results/16_individual_expert_auc_corrected.json"])
    add("App. H: published versus corrected convention (A2)",
        [py, f"{A}/appH_logit_convention_diff_v2.py"], [f"{ANAR}/A2_appendix_h_diff.json"])
    # Appendix I.1
    add("App. I.1: three-class folds, construction A (DF40 split lists)",
        [py, f"{A}/appI1_k3_build_v2.py", "--df40-json", "{df40}"], [f"{TKR}/K3_folds_df40.json"], needs="df40")
    add("App. I.1: expert exposure of the three-class rows (DF40 split lists)",
        [py, f"{A}/appI1_k3_exposure_audit_v2.py", "--df40-json", "{df40}"], [f"{TKR}/K3_exposure_audit.json"],
        needs="df40")
    add("App. I.1: frozen three-class manifests, constructions A and B (DF40 split lists)",
        [py, f"{A}/appI1_k3_freeze_v2.py", "--df40-json", "{df40}"],
        [f"{TKR}/K3_FROZEN_A_df40.json", f"{TKR}/K3_FROZEN_B_arch6.json"], needs="df40")
    # the out-of-fold arrays these steps write are declared outputs (not inputs of their own steps)
    add("App. I.1, Table A19: three-class heads on fixed outputs, construction A",
        [py, f"{A}/appI1_k3_fit.py", "--view", "A_df40"],
        [f"{TKR}/K3_RESULT_A_df40.json", f"{TKR}/K3_OOF_A_df40.npz"])
    add("App. I.1, Table A19: three-class heads on fixed outputs, construction B",
        [py, f"{A}/appI1_k3_fit.py", "--view", "B_arch6"],
        [f"{TKR}/K3_RESULT_B_arch6.json", f"{TKR}/K3_OOF_B_arch6.npz"])
    add("App. I.1: inner cross-fitting assignments for Table A18",
        [py, f"{A}/appI1_native_k_preflight.py"],
        [f"{TKR}/NATIVE_K_PREFLIGHT_B_arch6.json", f"{TKR}/NATIVE_K_PREFLIGHT_A_df40.json"])
    add("App. I.1, Table A18: paper-aligned three-class comparison, construction B",
        [py, f"{A}/appI1_native_k_aligned_v2.py", "--view", "B_arch6", "{resume}"],
         [f"{TKR}/NATIVE_K_ALIGNED_FULL_B_arch6_RESULT.json",
          *(f"{TKR}/NATIVE_K_ALIGNED_FULL_B_arch6_F{fold}_{suffix}"
            for fold in range(3) for suffix in ("RESULT.json", "OOF.npz"))], gpu=True)
    # Appendix I.3
    add("App. I.3, Table A21: cost and access ledger",
        [py, f"{A}/appI3_cost_ledger_v2.py", "--pool", "arch6", "--measure_fit"],
        [f"{ANAR}/run_d_cost_ledger_arch6.json"])
    # Appendix F checks that read the refit and the ledger
    add("App. F: checks behind the strict-refit sentences",
        [py, f"{A}/appF_strict_refit_claims_v2.py"], [f"{ANAR}/verify_strict_arm_claims.json"])
    # derived
    add("Figure A2 (best feature-space combiner against the full ladder)",
        [py, f"{CRS}/authoring/figure_a2_full_ladder_v4.py", "--codebase", ".", "--stats",
         "camera_ready/output/results/full_ladder_stats.json", "--fallback", "camera_ready/inputs", "--out",
         "camera_ready/output/figures/figureA2_bottleneck_scatter_v3"],
        ["camera_ready/output/figures/figureA2_bottleneck_scatter_v3.pdf",
         "camera_ready/output/figures/figureA2_bottleneck_scatter_v3.png"])
    s.append(Step(f"C{n + 1:02d}", "C",
                  "The 25 DF40 tables, three additional analyses and the number "
                  "ledger (checks the 96 numbers posted in the discussion)",
                  (py, f"{CRS}/authoring/generate_camera_ready_tables_v18.py", "--inputs", "camera_ready/run_inputs",
                   "--out", "camera_ready/output/tables", "--no-fail"),   # its checks are classified at the end
                  tuple(f"camera_ready/output/tables/{name}" for name in table_deliverables()),
                  extra={"stage_inputs": True}))
    return s


# ---------------------------------------------------------------------------
# Reference specification used by generate_reference_v3.py.
# ---------------------------------------------------------------------------

# Timing and host fields: recorded, never compared (they legitimately change between runs).
TIMING_FIELDS = frozenset({
    "elapsed_s", "elapsed_ms", "elapsed_seconds", "elapsed_minutes", "elapsed_sec", "runtime_s", "runtime_ms",
    "runtime_seconds", "wall_time_s", "wall_time_ms", "wall_clock_s", "fold_time_s", "fold_time", "duration_s",
    "duration_ms", "time_s", "time_ms", "timestamp", "fold_seconds", "total_fit_plus_predict_sec", "fit_sec",
    "predict_sec_total", "predict_us_per_sample", "working_set_gib", "peak_working_set_gib",
    "peak_working_set_gib_after", "measured_imgs_per_sec",
})
# FS2-dependent values include the LOGO deltas, fold tests and confidence intervals, and the standard-split
# dimension sweep.
FS2_KEYS = (r"(fs_mlp|fs2|FS2|FS_2|FS2-K3|PS3-K3|(^|\.)delta_fs_vs_ps$|(^|\.)fs_(wins|sign_test_p|wilcoxon_p)$"
            r"|(^|\.)fs_fold_bootstrap_ci\[\d\]$|^dim_sweep_clip4\.)")


def _res(path, kind="det", tol=None, env=None, strings=None, key_map=None, partial_ok=False, step="", exclude=None):
    return {"path": path, "kind": kind, "tol": tol, "env": env, "strings": strings, "key_map": key_map or {},
            "partial_ok": partial_ok, "step": step, "exclude": exclude}


RESULTS_SPEC = [
    # Block B
    *[_res(f"{EXCL}/logo_exclusions_{p}.json", strings=r"gate_G1$", step="B") for p, _ in POOLS_B],
    *[_res(f"{CD}/11_disjoint_{p}_d{d}.json", env=FS2_KEYS, step="B") for p, d in POOLS_B],
    *[_res(f"{CD}/A3c_logo_extended_rungs_{p}.json", step="B") for p, _ in POOLS_B],
    *[_res(f"{CD}/A3c_logo_combined_ceiling_{p}.json", env=FS2_KEYS, step="B") for p, _ in POOLS_B],
    _res(f"{CD}/gate_G2_arch6.json", kind="gate", strings=r"status$", step="B"),
    *[_res(f"{CD}/gate_G0log_{p}.json", kind="gate", strings=r"status$", step="B") for p, _ in POOLS_B[1:]],
    *[_res(f"{MECH}/run_c_v2_{proto}_{p}.json", env=FS2_KEYS, step="B")
      for proto in ("standard", "logo") for p in ("arch6", "clip4")],
    _res(f"{MECH}/gate_G5_arch6_fs1tol1e-4.json", kind="gate", strings=r"status$", step="B"),
    _res(f"{MECH}/gate_G5_clip4.json", kind="gate", strings=r"status$", step="B"),
    _res(f"{MITR}/fs1_score_precision_check.json", step="B"),
    _res(f"{MITR}/mitigation_stats.json", env=FS2_KEYS, step="B"),
    # Block C
    _res("camera_ready/output/results/split_reuse_check.json", partial_ok=True, step="C"),
    _res(f"{ANAR}/run_b_diagnose_arch6.json", step="C"),
    _res(f"{ANAR}/A4_content_only_rows.json", strings=r"content_expert$", step="C"),
    # the number of unique training paths depends on the version of DF40's lists (the authors' local copy, without
    # e4e's 29,126 images: 703,223; the public DF40_all.json: 732,349); the overlap counts the paper quotes do not
    _res(f"{ANAR}/gate_expert_side_check.json", env=r"^expert_train_unique_paths", step="C"),
    _res(f"{ANAR}/verify_bit_identity.json", step="C"),
    _res(f"{ANAR}/run_a_extended_ladder_arch6.json", step="C"),
    _res(f"{ANAR}/run_a_extended_ladder_clip4.json", step="C"),
    _res("camera_ready/output/results/full_ladder_stats.json", env=FS2_KEYS, step="C"),    # FS2 values
    _res(f"{ANAR}/run_c_width_matched_arch6.json", step="C"),
    # rebuilt power simulation (seeded; every power is a count of simulated sets, so 1e-3 allows a few sets to
    # flip across platforms)
    _res(f"{ANAR}/appF_power_mde.json", tol=1e-3, step="C"),
    _res(f"{ANAR}/run_b_path_disjoint_arch6_family.json", step="C"),
    _res("camera_ready/output/results/16_individual_expert_auc_corrected.json", step="C"),
    _res(f"{ANAR}/A2_appendix_h_diff.json", step="C"),
    _res(f"{TKR}/K3_folds_df40.json", strings=r"content_expert$", step="C", exclude=r"^manifest\["),
    _res(f"{TKR}/K3_exposure_audit.json", step="C"),
    _res(f"{TKR}/K3_FROZEN_A_df40.json", strings=r"(manifest_sha256|content_expert)$", step="C",
         exclude=r"^manifest\["),   # the rows are covered by manifest_sha256
    _res(f"{TKR}/K3_FROZEN_B_arch6.json", strings=r"(manifest_sha256|content_expert)$", step="C",
         exclude=r"^manifest\["),
    _res(f"{TKR}/K3_RESULT_A_df40.json", tol=1e-3, strings=r"digest$", step="C"),
    _res(f"{TKR}/K3_RESULT_B_arch6.json", tol=1e-3, strings=r"digest$", step="C",
         key_map={"pooled_oof.FS-LR(std)-K3": "pooled_oof.FS_1",
                  "pooled_oof.PS-bin-to-K3 ceiling": "pooled_oof.PS_ceiling",
                  "pooled_oof.delta_feature_lr_minus_PS": "pooled_oof.delta_FS1_minus_PS"}),
    _res(f"{TKR}/NATIVE_K_PREFLIGHT_B_arch6.json", strings=r"(sha256|manifest_sha256)$", step="C",
         exclude=r"\.assignments\["),   # the assignments are covered by assignment_sha256
    _res(f"{TKR}/NATIVE_K_PREFLIGHT_A_df40.json", strings=r"(sha256|manifest_sha256)$", step="C",
         exclude=r"\.assignments\["),
    # the seed-averaged heads trained on a GPU depend on it: the stored run used an RTX 2070 Super (July 2026); on an
    # RTX 5080 two FS2-K3 seeds stop one epoch earlier and the FS2-K3 mean moves by 0.0007
    _res(f"{TKR}/NATIVE_K_ALIGNED_FULL_B_arch6_RESULT.json", kind="stoch", env=FS2_KEYS,
         strings=r"(manifest_sha256|inner_assignment_sha256)$", step="C"),
    _res(f"{ANAR}/run_d_cost_ledger_arch6.json", step="C"),
    _res(f"{ANAR}/verify_strict_arm_claims.json", tol=1e-3, step="C"),
]


def extract(data, strings=None, exclude=None):
    """Numeric leaves (as the reviewed harness: bools count, timing fields skipped) plus selected string leaves."""
    nums, strs = {}, {}
    pat = re.compile(strings) if strings else None
    skip = re.compile(exclude) if exclude else None

    def walk(obj, prefix="", key=""):
        if skip is not None and skip.search(prefix):
            return
        if isinstance(obj, (int, float)):
            if key not in TIMING_FIELDS:
                nums[prefix] = float(obj)
        elif isinstance(obj, str):
            if pat is not None and pat.search(prefix):
                strs[prefix] = obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k), str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{prefix}[{i}]", key)
    walk(data)
    return nums, strs


def _renamed(d, key_map):
    if not key_map:
        return d
    out = {}
    for k, v in d.items():
        for new, old in key_map.items():
            if k == new or k.startswith(new + ".") or k.startswith(new + "["):
                k = old + k[len(new):]
                break
        out[k] = v
    return out


def required_keys(path, keys, coverage=None):
    """The reference keys a partial result must hold, from the run, never from the result itself.
    `coverage`: None for a stored result (every key of its reference), or the routes of the run that wrote it,
    {"df40_lists": bool, "df40_images": bool}. split_reuse_check.json always holds the checks of the feature cache
    (route B without the image-file checks); a run given DF40's lists adds route A, one given the DF40 image root the
    image-file checks (`route_b.raw_files`). Any other partial result needs every key."""
    if not path.endswith("split_reuse_check.json") or coverage is None:
        return set(keys)
    need = [r"^route_b\.(?!raw_files\.)"]
    if coverage.get("df40_lists"):
        need.append(r"^route_a\.")
    if coverage.get("df40_images"):
        need.append(r"^route_b\.raw_files\.")
    return {k for k in keys if any(re.search(p, k) for p in need)}


def read_reference(path, expected=None):
    """Read a comparison reference and check it against the independent output inventory."""
    ref = json.loads(path.read_text(encoding="utf-8"))
    entries = ref.get("results") if isinstance(ref, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise ValueError("missing or empty results inventory")
    for name, value in ref.items():
        if name.startswith("_tolerance_") and (type(value) not in (int, float)
                                                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name}: invalid tolerance")
    if expected is not None and set(entries) != expected:
        raise ValueError(f"results inventory differs: missing={sorted(expected - entries.keys())}, "
                         f"unexpected={sorted(entries.keys() - expected)}")
    for name, entry in entries.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("metrics"), dict):
            raise ValueError(f"{name}: missing or invalid metrics")
        metrics, strings = entry["metrics"], entry.get("strings_ref", {})
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"{name}: nonfinite or invalid reference metric")
        if not isinstance(strings, dict) or any(not isinstance(value, str) for value in strings.values()):
            raise ValueError(f"{name}: invalid reference strings")
        if expected is not None and not (metrics or strings):
            raise ValueError(f"{name}: no reference values")
    return entries


def compare_v4(reference_path=None, only_steps=None, run=None):
    """Blocks B and C against the camera-ready reference. Returns (ok, warn, fail, skip, lines).

    For a gate file the run did not rewrite, compare every reference value. A stored file whose rows are excluded from the comparison must be the reference's
    source byte for byte (its SHA-256).

    Require finite values valid for their kind, every recorded
    status PASS, report or COMPLETE, permitted skip markers only, the required keys of partial results, a bound on
    environment-dependent and stochastic differences, configuration values equal to the reference, and a failure for
    a result with nothing to compare. `run`: None to verify the stored results, or {"start": epoch seconds,
    "df40_lists": bool, "df40_images": bool} for a run, whose options set the keys a result it rewrote must hold."""
    expected = {spec["path"] for spec in RESULTS_SPEC} if reference_path in (None, REFERENCE2) else None
    reference_path = REFERENCE2 if reference_path is None else reference_path
    lines = []
    try:
        entries = read_reference(reference_path, expected)
    except (OSError, ValueError, OverflowError) as exc:
        return 0, 0, 1, 0, [f"ERROR: invalid reference {reference_path.name}: {exc}"]
    ok = warn = fail = skip = 0
    for path, entry in sorted(entries.items()):
        if only_steps and entry.get("step") not in only_steps:
            continue
        tol = entry.get("tol")
        if (entry.get("kind") not in ("det", "stoch", "gate")
                or not isinstance(entry.get("key_map"), dict)
                or (tol is not None and (type(tol) not in (int, float) or not math.isfinite(tol) or tol <= 0))):
            lines.append(f"  FAIL:    {path} (invalid reference comparison settings)")
            fail += 1
            continue
        try:
            for field in ("env", "strings", "exclude"):
                if entry.get(field) is not None:
                    re.compile(entry[field])
        except (TypeError, re.error) as exc:
            lines.append(f"  FAIL:    {path} (invalid reference pattern: {exc})")
            fail += 1
            continue
        f = ROOT / path
        if not f.exists():
            lines.append(f"  MISSING: {path}")
            fail += 1
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            lines.append(f"  FAIL:    {path} (unreadable: {exc})")
            fail += 1
            continue
        if isinstance(data, dict) and data.get("skipped") is True:
            need = SKIP_PERMITTED.get(path)
            if need is not None and all(data.get(k) == v for k, v in need.items()):
                lines.append(f"  SKIP:    {path} (reason={data.get('reason', 'unspecified')})")
                skip += 1
            else:
                lines.append(f"  FAIL:    {path} (skip marker, reason={data.get('reason', 'unspecified')}; no "
                             f"skip is permitted for this result)")
                fail += 1
            continue
        nums, strs = extract(data, entry.get("strings"), entry.get("exclude"))
        nums, strs = _renamed(nums, entry["key_map"]), _renamed(strs, entry["key_map"])
        kind = entry["kind"]
        tol = entry.get("tol") or (STOCH_TOL if kind == "stoch" else DET_TOL)
        env = re.compile(entry["env"]) if entry.get("env") else None
        documented = ENV_DOCUMENTED.get(path, {})
        strings_ref = entry.get("strings_ref", {})
        rewritten = run is not None and f.stat().st_mtime >= run["start"] - 2      # rewritten by this run
        # a gate file's numbers (its recorded differences) are compared when it was not rewritten
        metrics = entry["metrics"] if (kind != "gate" or not rewritten) else {}
        every = set(metrics) | set(strings_ref)
        coverage = None
        if rewritten:
            coverage = {"df40_lists": bool(run.get("df40_lists")), "df40_images": bool(run.get("df40_images"))}
        required = required_keys(path, every, coverage) if entry.get("partial_ok") else every
        hard, soft, envd, missing, checked, checked_s, max_d = [], [], [], [], 0, 0, 0.0
        hard += [(k, "PASS, report or COMPLETE", v, "status") for k, v in bad_statuses(data)]
        if kind == "gate" and rewritten:
            # A new difference need not equal the historical difference. Its evidence must
            # nevertheless cover the same metrics and be finite, with checked row counts.
            missing.extend(k for k in entry["metrics"] if k not in nums)
            hard.extend((k, "finite", v, "gate evidence") for k, v in nums.items() if not math.isfinite(v))
            checked += len(entry["metrics"].keys() & nums.keys())
            expected_counts = {"g0": {"logo": 96, "rungs": 96},
                               "g2": {"logo": 96, "rungs": 96},
                               "g0log": {"logo": 64}, "g5": {"mechanism": 16}}
            counts = data.get("checked") or {}
            folds = counts.get("folds", [])
            expected_folds = {"whichisreal", "CollabDiff", "MidJourney", "heygen", "starganv2",
                              "deepfacelab", "stargan", "styleclip"}
            contract = expected_counts.get(data.get("gate"))
            if (contract is None or len(folds) != 8 or set(folds) != expected_folds
                    or any(type(counts.get(k)) is not int or counts[k] != n for k, n in (contract or {}).items())):
                hard.append(("checked", "complete gate evidence", counts, "coverage"))
        if entry.get("exclude") and not rewritten and entry.get("source_sha256"):
            digest = _file_hash(f)
            if digest != entry["source_sha256"] and _crlf_hash(f) != entry["source_sha256"]:
                hard.append(("(file bytes)", entry["source_sha256"][:16], digest[:16],
                             "differs from the reference's source; its excluded rows are compared only this way"))
        for key, want in strings_ref.items():
            have = strs.get(key)
            if have is None:
                if key in required:
                    missing.append(key)
                continue
            checked_s += 1
            if have != want:
                hard.append((key, want, have, "string"))
        for key, want in metrics.items():
            have = nums.get(key)
            if have is None:
                if key in required:
                    missing.append(key)
                continue
            checked += 1
            if not (math.isfinite(have) and math.isfinite(want)):
                if not (have == want or (math.isnan(have) and math.isnan(want))):
                    hard.append((key, want, have, "not finite"))
                continue
            why = invalid_value(key, have, want, nums)                     # valid for its kind
            if why:
                hard.append((key, want, have, why))
                continue
            d = abs(have - want)
            max_d = max(max_d, d)
            if key in documented:
                if d > tol:
                    (envd if any(abs(have - a) <= tol for a in documented[key]) else hard).append((key, want, have, d))
            elif env is not None and env.search(key):
                if ENV_FIXED.search(key):
                    if d > 0:
                        hard.append((key, want, have, d))
                elif d > ENV_MAX and not ENV_STEPWISE.search(key):
                    hard.append((key, want, have, d))
                elif d > ENV_TOL:
                    envd.append((key, want, have, d))
            elif d > tol:
                (soft if kind == "stoch" and d <= ENV_MAX else hard).append((key, want, have, d))
        if checked == 0 and checked_s == 0 and not missing:
            lines.append(f"  FAIL:    {path} (nothing to compare)")
            fail += 1
        elif hard or missing:
            lines.append(f"  FAIL:    {path} ({len(hard)} mismatches, {len(missing)} missing)")
            for k, w, h, d in hard[:5]:
                lines.append(f"           {k}: ref={w} got={h} ({d if isinstance(d, str) else f'delta={d:.6g}'})")
            for k in missing[:5]:
                lines.append(f"           MISSING: {k}")
            fail += 1
        elif soft or envd:
            tag = "ENV" if envd and not soft else "WARN"
            lines.append(f"  {tag}:{' ' * (8 - len(tag))}{path} ({len(soft)} over {tol}, {len(envd)} environment-dependent "
                         f"over {ENV_TOL})")
            for k, w, h, d in (soft + envd)[:5]:
                lines.append(f"           {k}: ref={w:.6f} got={h:.6f} (delta={d:.6f})")
            warn += 1
        else:
            word = "MATCH" if kind != "stoch" else "CLOSE"
            what = (("verdict" + (f" and {checked} metrics" if checked else "")) if kind == "gate"
                    else f"{checked} metrics, max delta={max_d:.2e}")
            lines.append(f"  {word}:   {path} ({what})")
            ok += 1
    lines.append(f"Summary (blocks B and C): {ok} MATCH, {warn} WARN/ENV, {fail} FAIL, {skip} SKIP")
    return ok, warn, fail, skip, lines


compare_v3 = compare_v4       # Compatibility alias.


# guard_block_a supplements the reviewed comparison with checks for non-finite values, invalid skip markers and
# large stochastic differences. The reviewed full21 FS1 fit skips when its 6.7 GiB float64 input needs more than
# 40% of host RAM (about 16.7 GiB total).
# The reviewed LOGO report's inferential values can jump when a fold changes sign.
BLOCK_A_STEPWISE = {"fs_wins", "rawfeat_wins", "fs_sign_test_p", "rawfeat_sign_test_p",
                    "fs_wilcoxon_p", "rawfeat_wilcoxon_p"}

BLOCK_A_SKIP_PERMITTED = {name: {"reason": "memory_budget", "fs1_method": "raw_feature_logreg", "pool": "full21"}
                          for name in ("04_ds_full21_rawfeat.json", "06_bootstrap_full21.json")}


def guard_block_a(rv):
    """The Block A results against the reviewed reference, for what the reviewed comparison lets through:
    unreadable files, non-finite values, values impossible for their kind (invalid_value), recorded statuses other
    than PASS, report or COMPLETE, skip markers other than the documented ones, and stochastic differences above
    ENV_MAX, except valid discrete inferential changes from the stochastic LOGO result.
    Missing files and deterministic differences stay with the reviewed comparison. Returns (fail, lines)."""
    lines, fail = [], 0
    if not rv.REFERENCE_PATH.exists():
        return 0, lines                       # the reviewed comparison already fails on a missing reference
    ref = json.loads(rv.REFERENCE_PATH.read_text(encoding="utf-8"))
    for name, entry in sorted(ref["results"].items()):
        f = rv.RESULTS / name
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            lines.append(f"  FAIL:    output/results/{name} (unreadable: {exc})")
            fail += 1
            continue
        if isinstance(data, dict) and data.get("skipped") is True:
            need = BLOCK_A_SKIP_PERMITTED.get(name)
            if need is None or any(data.get(k) != v for k, v in need.items()):
                lines.append(f"  FAIL:    output/results/{name} (skip marker, reason="
                             f"{data.get('reason', 'unspecified')}; only the full21 memory-budget skips are permitted)")
                fail += 1
            continue
        have_all = rv._extract_metrics(data)
        bad = [f"{k}: {v} (a status must be PASS, report or COMPLETE)" for k, v in bad_statuses(data)]
        for key, want in entry["metrics"].items():
            have = have_all.get(key)
            if have is None or rv._is_timing_key(key):
                continue
            if not (math.isfinite(have) and math.isfinite(want)):
                if not (have == want or (math.isnan(have) and math.isnan(want))):
                    bad.append(f"{key}: ref={want} got={have} (not finite)")
            elif invalid_value(key, have, want, have_all):
                bad.append(f"{key}: ref={want} got={have} ({invalid_value(key, have, want, have_all)})")
            elif not entry.get("deterministic") and abs(have - want) > ENV_MAX:
                if (name == "11_disjoint_arch6_d256.json" and key in BLOCK_A_STEPWISE
                        and (not key.endswith("_wins") or have <= have_all.get("n_folds", 0))):
                    lines.append(f"  ENV:     output/results/{name}: {key}: ref={want} got={have}")
                else:
                    bad.append(f"{key}: ref={want} got={have} (delta={abs(have - want):.6g} above {ENV_MAX})")
        if bad:
            lines.append(f"  FAIL:    output/results/{name} ({len(bad)} values)")
            lines += [f"           {b}" for b in bad[:5]]
            fail += 1
    lines.append(f"Block A, version 3 checks: {fail} FAIL")
    return fail, lines


# The 96 public discussion-check identifiers, independent of a generated report.
EXPECTED_POSTED_IDS = (
    'logo.fs1.delta',
    'logo.fs1.ci_lo',
    'logo.fs1.ci_hi',
    'logo.fs1.wins',
    'logo.fs1.signflip_p',
    'logo.fs1.sign_p',
    'logo.fs1.wilcoxon_p',
    'logo.ceiling',
    'arch6.ps4_submitted',
    'arch6.deployable_ceiling',
    'arch6.ext.delta',
    'arch6.ext.ci_lo',
    'arch6.ext.ci_hi',
    'arch6.ext.wins',
    'arch6.ext.p',
    'arch6.oracle_domain',
    'clip4.deployable_ceiling',
    'clip4.ext.delta',
    'clip4.ext.ci_lo',
    'clip4.ext.ci_hi',
    'logo_refit.ceiling_before',
    'logo_refit.ceiling_after',
    'logo_refit.delta_before',
    'logo_refit.delta_after',
    'logo_refit.extended_wins',
    'width.pca_d2.macro',
    'width.ps4',
    'width.pca_d2.delta',
    'width.pca_d2.p',
    'width.probe_d2.macro',
    'k3.PS1-K3',
    'k3.PS2-K3',
    'k3.PS3-K3',
    'k3.PS4-K3',
    'k3.FS1-K3',
    'k3.FS2-K3',
    'k3.FSB-K3-18',
    'k3.PS+-K3',
    'k3.fs2_minus_select',
    'k3.fs1_minus_select',
    'k3.fs1_minus_select.pos_folds',
    'k3.fs2_minus_select.pos_folds',
    'k3.select_chosen',
    'k3.heads_at_cap',
    'k3_view_a.delta',
    'k3_view_a.pos',
    'k3_view_b.delta',
    'k3_view_b.pos',
    'cost.feature_floats',
    'cost.feature_bytes',
    'cost.pred_floats',
    'cost.pred_bytes',
    'cost.pool_gib',
    'cost.pool_mib',
    'cost.fs1_fit_s',
    'cost.ps4_cv_s',
    'cost.ps2_s',
    'cost.fs1_us',
    'cost.ps_us',
    'strict.full.delta',
    'strict.full.ci_lo',
    'strict.full.ci_hi',
    'strict.strict.delta',
    'strict.strict.ci_lo',
    'strict.strict.ci_hi',
    'strict.strict.fs1',
    'strict.strict.ps',
    'strict.matched.mean',
    'strict.matched.seeds',
    'strict.minus_matched',
    'strict.real_rows_full',
    'strict.real_rows_strict',
    'strict.families_full',
    'strict.families_strict',
    'reuse.total_n',
    'reuse.total_path',
    'reuse.path_pct',
    'reuse.total_content',
    'reuse.content_pct',
    'reuse.fake_generators',
    'stargan.full',
    'stargan.matched_min',
    'stargan.matched_max',
    'stargan.strict',
    'stargan.full_margin',
    'pool.clip4.delta',
    'pool.clip4.ceiling',
    'pool.strong4.delta',
    'pool.strong4.ceiling',
    'pool.weak6.delta',
    'pool.weak6.ceiling',
    'pool.clip4.large_mean',
    'pool.strong4.large_mean',
    'pool.weak6.large_mean',
    'exp.arch6_best.published',
    'exp.arch6_best.corrected',
)
POSTED_CHECK = CR / "output" / "tables" / "06_posted_number_check.md"
POSTED_ROW = re.compile(r"^\| `([^`]+)` \|([^|]*)\|([^|]*)\|([^|]*)\|([^|]+)\|$")


def _change(a, b):
    """Absolute change between two numbers given as text ('' when either is not a number)."""
    try:
        return f"{abs(float(a.replace(',', '')) - float(b.replace(',', ''))):.4f}"
    except ValueError:
        return ""


def _number(text):
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def ledger_problems(a, b):
    """Rebuilt ledger rows that are not valid numbers: not a number or not finite where the stored row is a
    number, or impossible for their kind (invalid_value). A row read from a result file (kind "json") is judged by its
    source key; a derived row by its quantity id, for p-values and counts only. Rows that hold a name (a ceiling
    method, for example) are compared by their display alone."""
    out = []
    for k, row in b.items():
        want = _number(a[k]["value"]) if k in a else None
        have = _number(row["value"])
        if want is None:
            if k not in a and row["value"].strip().lower() in ("nan", "inf", "-inf", "+inf"):
                out.append((k, "", row["value"], "not finite"))
            continue
        if have is None:
            out.append((k, a[k]["value"], row["value"], "not a number"))
        elif not math.isfinite(have):
            out.append((k, a[k]["value"], row["value"], "not finite"))
        else:
            source = (row.get("key") or "") if row.get("kind") == "json" else ""
            if re.search(r"\s[-+*/]\s|[()]", source):      # an expression of several keys: finite is all we know
                continue
            key = source or k[1]
            why = invalid_value(key, have, want, {}) if key else None
            if why and (source or value_rule(key) in ("p", "count")):
                out.append((k, a[k]["value"], row["value"], why))
    return out


# Classify each rebuilt value by its source key: "machine" for a time or memory measurement (Table A21 and the posted timings, measured again in
# every run: recorded, never compared), "environment" for an FS2 statistic or a contrast with a three-class head
# trained on a GPU (reported with its change), "exact" otherwise.
GPU_CONTRAST_SOURCES = re.compile(r"NATIVE_K_ALIGNED")
# every value of the reviewed FS2-type outputs (scripts 05, 07 and 10 of the reviewed pipeline, neural
# heads trained on the GPU when one is visible) depends on the environment whatever its key (mean_avg8,
# per_generator_seed_averaged, loo[...].avg8_mean, ...), except their settings and counts
FS2_SOURCES = re.compile(r"(^|/)(05_fs_|07_bootstrap_mlp_|10_loo_)[^/]*\.json$")
FS2_SETTINGS = re.compile(r"proj_dim|n_trainable_params|^len\(")


def ledger_class(row):
    """"machine", "environment" or "exact" for one ledger row, by its source key, kind and source file."""
    key = (row.get("key") or "").strip()
    leaf = re.sub(r"\[\d+\]$", "", key.split(".")[-1]) if key else ""
    if leaf in TIMING_FIELDS:
        return "machine"
    if key and (re.search(FS2_KEYS, key) or (row.get("kind") == "derived" and re.search(r"\bfs2\b", key))):
        return "environment"
    if GPU_CONTRAST_SOURCES.search(row.get("source") or "") and key.startswith("contrasts."):
        return "environment"
    if FS2_SOURCES.search(row.get("source") or "") and not FS2_SETTINGS.search(key):
        return "environment"
    return "exact"


def ledger_classes(path=CR / "output" / "tables" / "05_number_provenance.csv"):
    """Quantity id -> ledger_class, from a ledger file ({} when it does not exist)."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return {r["quantity"]: ledger_class(r) for r in csv.DictReader(fh)}


def compare_ledgers(stored=STORED_LEDGER, new=CR / "output" / "tables" / "05_number_provenance.csv"):
    """Compare all public ledger entries at their printed precision.

    Use ledger_class to distinguish changes in display. Report machine-dependent and environment-dependent
    changes separately from other differences. Check every rebuilt number with ledger_problems."""
    if not new.exists():
        return None

    def load(p):
        with open(p, encoding="utf-8") as fh:
            return {(r["table"], r["quantity"]): r for r in csv.DictReader(fh)}
    a, b = load(stored), load(new)
    same = [k for k in a if k in b and a[k]["display"] == b[k]["display"]]
    moved = [k for k in a if k in b and a[k]["display"] != b[k]["display"]]
    row = lambda k: (k, a[k]["display"], b[k]["display"], _change(a[k]["value"], b[k]["value"]))  # noqa: E731
    gone = [k for k in a if k not in b]
    cls = {k: ledger_class(a[k]) for k in moved}
    return {"rows": len(a), "same_display": len(same),
            "environment_dependent": [row(k) for k in moved if cls[k] == "environment"],
            "machine_dependent": [row(k) for k in moved if cls[k] == "machine"],
            "different_display": [row(k) for k in moved if cls[k] == "exact"],
            "missing": gone,
            "new_rows": [k for k in b if k not in a], "invalid": ledger_problems(a, b)}


def compare_posted(path=POSTED_CHECK, classes=None, expected_ids=EXPECTED_POSTED_IDS):
    """Check the numbers posted in the discussion. A posted number that differs
    is classified by its quantity's ledger row (`classes`: ledger_classes of the rebuilt ledger); otherwise, by FS2_KEYS on the quantity id."""
    errors = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        lines = []
        errors.append(f"missing or unreadable report: {exc}")
    rows = [m.groups() for m in map(POSTED_ROW.match, lines) if m]
    actual = [r[0].strip() for r in rows]
    want = set(expected_ids)
    missing, unexpected = sorted(want-set(actual)), sorted(set(actual)-want)
    duplicates = sorted(q for q in set(actual) if actual.count(q) > 1)
    if missing:
        errors.append(f"missing posted IDs: {missing}")
    if unexpected:
        errors.append(f"unexpected posted IDs: {unexpected}")
    if duplicates:
        errors.append(f"duplicate posted IDs: {duplicates}")
    if len(rows) != len(want):
        errors.append(f"expected {len(want)} unique posted rows, found {len(rows)}")
    out = {"rows": len(rows), "agree": 0, "environment_dependent": [], "machine_dependent": [],
           "different": [], "coverage_errors": errors}
    for qid, _, posted, got, status in (tuple(c.strip() for c in r) for r in rows):
        if status == "PASS":
            out["agree"] += 1
            continue
        c = (classes or {}).get(qid) or ("environment" if re.search(FS2_KEYS, qid) else "exact")
        if status == "FAIL" and c in ("environment", "machine"):
            out[f"{c}_dependent"].append((qid, posted, got, _change(posted, got)))
        else:
            out["different"].append((qid, posted, got, status))
    return out


# ---------------------------------------------------------------------------
# Environment record
# ---------------------------------------------------------------------------

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
               "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")
PACKAGES = ("numpy", "scipy", "scikit-learn", "xgboost", "torch", "torchvision", "matplotlib", "PyYAML",
            "threadpoolctl", "joblib")


def _cpu_name():
    try:
        if sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True).stdout.strip()
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001  best effort
        pass
    return platform.processor() or "unknown"


def _ram_gib():
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 2**30, 1)
        if sys.platform == "darwin":
            return round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                            text=True).stdout) / 2**30, 1)
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 2**20, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def _nvidia_smi():
    if not shutil.which("nvidia-smi"):
        return None
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return [dict(zip(("index", "name", "memory", "driver"), (x.strip() for x in line.split(","))))
            for line in r.stdout.splitlines() if line.strip()]


def _child_json(code, env):
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"error": (r.stderr or r.stdout)[-300:]}


def environment_record(env):
    from importlib import metadata
    pkgs = {}
    for name in PACKAGES:
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    torch_code = ("import json, torch; c = torch.cuda.is_available(); print(json.dumps({'torch': torch.__version__, "
                  "'cuda_build': torch.version.cuda, 'cuda_available': c, 'devices': [torch.cuda.get_device_name(i) "
                  "for i in range(torch.cuda.device_count())] if c else []}))")
    blas_code = ("import json, numpy, threadpoolctl; print(json.dumps([{k: d.get(k) for k in "
                 "('internal_api', 'num_threads', 'version')} for d in threadpoolctl.threadpool_info()]))")
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "cpu": _cpu_name(), "cpu_count_logical": os.cpu_count(), "ram_gib": _ram_gib(),
            "gpus_nvidia_smi": _nvidia_smi(), "torch_in_run_env": _child_json(torch_code, env),
            "blas_in_run_env": _child_json(blas_code, env), "packages": pkgs,
            "env": {k: env.get(k) for k in THREAD_VARS}}


def child_env(gpu_name=None):
    env = os.environ.copy()
    for name in ("CR_RESULTS_DIR", "CR_CODEBASE", "MITIGATION_CODEBASE", "DF40_JSON_DIR",
                 "REPRO_ATTEMPT_ID", "REPRO_FINGERPRINT", "REPRO_VALIDATED_RESUME"):
        env.pop(name, None)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "scripts"), env.get("PYTHONPATH", "")])
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if gpu_name:
        gpus = _nvidia_smi() or []
        hits = [g for g in gpus if gpu_name.lower() in g["name"].lower()]
        if len(hits) != 1:
            raise SystemExit(f"--gpu-name {gpu_name}: {len(hits)} matching GPU(s); visible: "
                             f"{[g['name'] for g in gpus]}")
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = hits[0]["index"]
        torch_device = _child_json(TORCH_DEVICE_PROBE, env)
        if not isinstance(torch_device, str) or gpu_name.lower() not in torch_device.lower():
            raise SystemExit(f"--gpu-name {gpu_name}: PyTorch cannot use the selected CUDA GPU "
                             f"(reported {torch_device!r}). Nothing was changed.")
        _DEVICES[gpu_name] = torch_device
    return env


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def resolve_cmd(step, args, resuming):
    out = []
    for tok in step.cmd:
        if tok == "python":
            out.append(sys.executable)
            if "{resume}" in step.cmd:      # a step that keeps checkpoints runs through the launcher
                out.append(str(STAMPER))
        elif tok == "{gpu_name}":
            if args.gpu_name:
                out += ["--gpu-name", args.gpu_name]
        elif tok == "{resume}":
            if resuming:
                out.append("--resume")
        elif tok == "{df40}":            # the lists rewritten by step C00
            out.append(DF40_PREPARED)
        elif tok == "{df40_public}":     # the lists as published by DF40
            out.append(str(Path(args.df40_json).resolve()))
        elif tok == "{defer_audit}":     # B32 defers an inconsistent audit to C08a when C08a is in the run
            if getattr(args, "audit_later", False):
                out.append("--defer-audit")
        elif tok == "{reuse_routes}":
            if args.df40_json:
                out += ["--df40-json", str(Path(args.df40_json).resolve())]
            if args.df40_raw:
                out += ["--raw-root", str(Path(args.df40_raw).resolve())]
            if args.df40_json and args.df40_raw:
                out += ["--expect", "camera_ready/tests/expected_split_reuse.json"]
        else:
            out.append(tok)
    return out


def run_step(step, argv, env, run_dir):
    log_path = run_dir / f"{step.id}.log"
    if log_path.exists():          # an earlier, interrupted attempt: keep its console output
        n = 1
        while (run_dir / f"{step.id}.attempt{n}.log").exists():
            n += 1
        log_path.rename(run_dir / f"{step.id}.attempt{n}.log")
    start = datetime.now()
    t0 = time.time()
    print(f"\n{'=' * 72}\n[{step.id}] {step.paper}\n$ {' '.join(argv)}\n{'=' * 72}", flush=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"# step {step.id} (block {step.block}): {step.paper}\n# $ {shlex.join(argv)}\n"
                 f"# cwd: repository root; start {start.isoformat(timespec='seconds')}\n\n")
        proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
            fh.flush()             # an interrupted step keeps its console output up to its last line
        rc = proc.wait()
        elapsed = time.time() - t0
        fh.write(f"\n# exit code {rc}; elapsed {elapsed:.1f} s\n")
    stale, output_hashes = [], {}
    for o in step.outputs:
        p = ROOT / o
        if not p.is_file():
            stale.append(f"missing {o}")
        elif p.stat().st_mtime < t0 - 2:
            stale.append(f"not rewritten {o}")
        else:
            damage = _damaged(p)
            if damage:
                stale.append(f"invalid {o}: {damage}")
            else:
                digest = _file_hash(p, fresh=True)
                if re.fullmatch(r"[0-9a-f]{64}", digest):
                    output_hashes[o] = digest
                else:
                    stale.append(f"{o}: {digest}")
    status = "OK" if rc == 0 and not stale else ("FAILED" if rc != 0 else "STALE")
    print(f"  {status} ({elapsed / 60:.1f} min){' ' + '; '.join(stale) if stale else ''}", flush=True)
    return {"id": step.id, "block": step.block, "paper": step.paper, "cmd": shlex.join(
        [Path(argv[0]).name] + argv[1:]), "gpu": step.gpu, "start": start.isoformat(timespec="seconds"),
        "end": datetime.now().isoformat(timespec="seconds"), "elapsed_s": round(elapsed, 1), "exit_code": rc,
        "status": status, "output_problems": stale, "log": rel(log_path),
        "output_sha256": output_hashes if status == "OK" else {}}


def _is_skip_marker(p):
    """A JSON result that is a skip marker ({"skipped": true, ...})."""
    try:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  not JSON, or unreadable: not a marker
        return False
    return isinstance(d, dict) and d.get("skipped") is True


PINNED_TABLE_INPUTS = frozenset({
    "11_disjoint_arch6_d256.json", "A3c_logo_extended_rungs_arch6.json",
    "A3c_logo_combined_ceiling_arch6.json",
})


def stage_run_inputs(run_start, record):
    """Inputs of the table builder: the regenerated file where this run wrote one, else the stored copy. A
    regenerated documented memory-budget skip marker (the full21 FS1 fit on hosts below about 16.7 GiB of RAM) is not
    staged; its stored copy is, and the run log lists it."""
    if RUN_INPUTS.exists():
        aside = BACKUP2 / f"run_inputs_{datetime.now():%Y%m%d_%H%M%S}"
        aside.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(RUN_INPUTS), str(aside))
    RUN_INPUTS.mkdir(parents=True)
    candidates = [ANA, TK, RESULTS2, ROOT / "output" / "results", ROOT / "configs"]
    chosen, kept = {}, []
    for f in sorted(INPUTS.iterdir()):
        src = f
        for c in (() if f.name in PINNED_TABLE_INPUTS else candidates):
            p = c / f.name
            if p.exists() and p.stat().st_mtime >= run_start - 2:
                if f.name in BLOCK_A_SKIP_PERMITTED and _is_skip_marker(p):
                    kept.append(f.name)              # the stored copy (src = f) is staged
                else:
                    src = p
                break
        shutil.copy2(src, RUN_INPUTS / f.name)
        chosen[f.name] = rel(src)
    record["table_inputs"] = chosen
    record["table_inputs_pinned_historical"] = sorted(PINNED_TABLE_INPUTS & chosen.keys())
    if kept:
        record["table_inputs_stored_for_skip_markers"] = kept
        print(f"  table inputs: the stored copies of {', '.join(kept)} (this run wrote memory-budget skip markers)",
              flush=True)
    n_new = sum(1 for v in chosen.values() if not v.startswith("camera_ready/inputs/"))
    print(f"  table inputs: {n_new} regenerated in this run, {len(chosen) - n_new} stored", flush=True)


CHECKPOINT_PARTS = ("folds", "rung_folds", "mechanism_folds", "scores")    # what a backup moves aside


def backup_v2(blocks, resuming, move_aside=True):
    if resuming:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP2 / ts
    if RESULTS2.exists():
        shutil.copytree(RESULTS2, dest / "results")
    if "B" in blocks and move_aside:   # fold checkpoints would otherwise be reused; move them aside (nothing deleted)
        # only the resume files and the per-sample predictions; the stored results stay in place
        # until a step rewrites them, so a step this run skips (gate G2 without --with-reproduction-gate) keeps its
        # stored verdict for the closing verification
        for sub in ("content_disjoint", "mechanism"):
            for part in CHECKPOINT_PARTS:
                p = MIT / sub / part
                if p.exists():
                    (dest / "moved_aside" / sub).mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(dest / "moved_aside" / sub / part))
    print(f"Backed up camera_ready/output/results to {rel(dest)}", flush=True)
    return dest


def selected_steps(args, rv):
    blocks = [b.strip().upper() for b in args.blocks.split(",") if b.strip()]
    if not blocks or any(b not in ("A", "B", "C") for b in blocks):
        raise SystemExit(f"--blocks {args.blocks!r}: give one or more of A, B and C, separated by commas")
    plan = []
    if "A" in blocks:
        plan += steps_a(rv)
    if "B" in blocks:
        plan += [s for s in steps_b() if not s.optional or (s.optional == "repro-gate" and args.with_reproduction_gate)]
    if "C" in blocks:
        plan += steps_c()
    return blocks, plan


def with_df40_preparation(plan, full, args):
    """Steps reading DF40's split lists use the files written by step C00: add C00 when they run without it."""
    users = [i for i, s in enumerate(plan) if "{df40}" in s.cmd]
    if not users or not args.df40_json or any(s.id == "C00" for s in plan):
        return plan
    print(f"Step C00 added: {', '.join(plan[i].id for i in users)} read the lists it writes.", flush=True)
    return plan[:users[0]] + [s for s in full if s.id == "C00"] + plan[users[0]:]


def timing_report(record, path, rv=None):
    """Write TIMINGS.md with the steps the run covered. A run is complete only
    when every step of a default full run finished in it (the steps that need DF40's lists, when they are not given,
    and gate G2 without the reproduction gate keep their stored outputs, as in a default full run), and otherwise lists
    the steps not run, skipped or not finished, whose stored outputs were kept and not recomputed. It lists the steps
    that ran but skipped their computation with a marker, names what no run of this harness covers, compares a time
    the paper states only with a run that covers it, and names the steps behind each GPU time (whole steps, CPU parts
    included). `rv`: the reviewed harness, which lists the steps of block A (without it, those of the record)."""
    st = list(latest_records(record).values())
    by_id = {s["id"]: s for s in st}
    rerun = sorted({r["id"] for r in record["steps"] if r.get("superseded")})
    sessions = record.get("sessions", [])
    start = record.get("run_start_epoch", 0)
    block_a = (steps_a(rv) if rv is not None else
               [Step(s["id"], "A", s.get("paper", ""), (), gpu=bool(s.get("gpu"))) for s in st if s["block"] == "A"])
    full = block_a + [s for s in steps_b() if not s.optional] + steps_c()
    optional = {s.id for s in steps_b() if s.optional}

    def status(i):
        return by_id.get(i, {}).get("status")

    def as_default(s):
        """A skip that a default full run makes too."""
        return status(s.id) == "SKIPPED" and (s.needs == "df40" or bool(s.extra.get("premise")))

    def reason(s):
        return by_id[s.id].get("reason") or "no reason recorded"

    def ids(steps):
        return ", ".join(s.id for s in steps)

    def wall(steps):
        return sum(by_id[s.id].get("elapsed_s", 0.0) for s in steps if s.id in by_id)

    done = [s for s in full if status(s.id) == "OK"]
    default_skips = [s for s in full if as_default(s)]
    not_run = [s for s in full if s.id not in by_id]
    skipped = [s for s in full if status(s.id) == "SKIPPED" and not as_default(s)]
    unfinished = [s for s in full if s.id in by_id and status(s.id) not in ("OK", "SKIPPED")]
    complete = not (not_run or skipped or unfinished)
    markers = []
    for s in done:
        for o in s.outputs:
            p = ROOT / o
            if p.exists() and p.stat().st_mtime >= start - 2 and _is_skip_marker(p):
                try:
                    why = json.loads(p.read_text(encoding="utf-8")).get("reason") or "no reason recorded"
                except (OSError, ValueError):
                    why = "no reason recorded"
                markers.append(f"{s.id} ({o}: {why})")
    lines = ["# Timings of this run", "",
             f"Run `{record['run_id']}` on {record['environment']['cpu']}, {record['environment']['ram_gib']} GiB RAM, "
             f"GPU {', '.join(record['environment']['torch_in_run_env'].get('devices') or ['none'])}.", "",
             f"Sessions: {len(sessions)} (resumed {max(len(sessions) - 1, 0)} times). Times below are the "
             f"wall-clock of each step's last complete attempt"
             + (f"; steps rerun after an interruption or failure: {', '.join(rerun)}." if rerun else "."), "",
             "## Coverage", ""]
    if complete:
        lines.append(f"Complete: every step of a full run of blocks A, B and C ({len(full)} steps) finished in this run"
                     + (f", except {len(default_skips)} that a default full run also skips (below)." if default_skips
                        else "."))
    else:
        lines += [f"Partial: {len(done)} of the {len(full)} steps of a full run of blocks A, B and C finished in this "
                  "run. The outputs of the others are the stored ones, kept and not recomputed:", ""]
        if not_run:
            lines.append(f"- not run (outside the blocks, --from or --only of this run): {ids(not_run)}")
        lines += [f"- skipped: {s.id} ({reason(s)})" for s in skipped]
        if unfinished:
            lines.append("- not finished: " + ", ".join(f"{s.id} ({status(s.id)})" for s in unfinished))
    if default_skips:
        lines += ["", "Skipped as in a default full run (their stored outputs are kept): "
                  + "; ".join(f"{s.id} ({reason(s)})" for s in default_skips)]
    ran_optional = [i for i in by_id if i in optional]
    lines += ["", "Optional steps (the reproduction gate, --with-reproduction-gate): "
              + (f"run: {', '.join(ran_optional)}." if ran_optional else "not run.")]
    if markers:
        lines += ["", "Steps that ran but skipped their computation, as their result records: " + "; ".join(markers)]
    lines += ["", "Additional commands outside this harness: "
              "the t-SNE view of Figure A1 (camera_ready/scripts/authoring/figure_a1_tsne.py) and the sensitivity checks of Appendix A "
              "(camera_ready/scripts/sensitivity/, run on their own).", "",
              "| Block | Steps finished (of a full run) | Wall-clock |", "|---|---|---|"]
    for b in ("A", "B", "C"):
        mine = [s for s in full if s.block == b]
        if any(s.id in by_id for s in mine):
            fin = [s for s in mine if status(s.id) == "OK"]
            lines.append(f"| {b} | {len(fin)} of {len(mine)} | {wall(fin) / 3600:.2f} h |")
    lines += ["", "## Against the times the paper states", "",
              "A time is compared only when this run covers what the paper times. Wall-clock is that of whole steps.",
              "", "| Claim (paper) | Measured |", "|---|---|"]
    lines.append("| A full run about 36 h on a 16 GiB i9-10980HK laptop: blocks A, B and C, with the "
                 "Figure A1 and Appendix A sensitivity commands run separately (App. A, Q8) | "
                 + (f"{wall(done) / 3600:.2f} h (every step of blocks A, B and C) |" if complete else
                    f"not comparable: {len(done)} of the {len(full)} steps finished in this run |"))
    lines.append("| Rerunning the sensitivity checks of Appendix A about 15 h more (App. A, Q8) | not measured: their "
                 "scripts are not steps of this harness |")
    for b, estimate in (("A", 14), ("B", 14), ("C", 8)):
        mine = [s for s in full if s.block == b]
        if not any(s.id in by_id for s in mine):
            continue
        fin = [s for s in mine if status(s.id) == "OK"]
        whole = all(status(s.id) == "OK" or as_default(s) for s in mine)
        lines.append(f"| Block {b} about {estimate} h (camera_ready/REPRODUCTION.md, estimate) | "
                     + (f"{wall(fin) / 3600:.2f} h |" if whole else
                        f"not comparable: {len(fin)} of the {len(mine)} steps of block {b} finished |"))
    gpu_a = [s for s in full if s.block == "A" and s.gpu]
    if any(s.id in by_id for s in gpu_a):
        fin = [s for s in gpu_a if status(s.id) == "OK"]
        lines.append("| GPU portion (FS2 training and bootstraps) about 2 h (App. A, Q8) | "
                     + (f"{wall(fin) / 3600:.2f} h: steps {ids(fin)}, the block A steps that train FS2 (whole steps, "
                        "CPU parts included) |" if len(fin) == len(gpu_a) else
                        f"not comparable: {len(fin)} of the {len(gpu_a)} block A steps that train FS2 finished "
                        f"({ids(fin) or 'none'}) |"))
    gpu_bc = [s for s in full if s.block in ("B", "C") and s.gpu and status(s.id) == "OK"]
    if gpu_bc:
        lines.append(f"| (no time stated) block B and C steps that train on the GPU | {wall(gpu_bc) / 3600:.2f} h: "
                     f"steps {ids(gpu_bc)} (whole steps, CPU parts included) |")
    ledger = ROOT / ANAR / "run_d_cost_ledger_arch6.json"
    cost_steps = [s for s in st if s["id"] == "C21" and s.get("status") == "OK"]
    if cost_steps and ledger.exists() and ledger.stat().st_mtime >= start - 2:
        cc = json.loads(ledger.read_text(encoding="utf-8")).get("combiner_costs", {})
        at = f" (step {cost_steps[0]['id']} of this run)"

        def g(method, key, fmt):
            v = cc.get(method, {}).get(key)
            return "not recorded" if v is None else format(v, fmt)
        lines += [f"| FS1 fit and predict 53 s (Table A21) | {g('FS_1_frozen_total', 'total_fit_plus_predict_sec', '.1f')} s{at} |",
                  f"| PS4 with cross-validation 62 s (Table A21) | {g('PS_4_xgboost_cv', 'total_fit_plus_predict_sec', '.1f')} s{at} |",
                  f"| PS2 0.16 s (Table A21) | {g('PS_2_logreg_logits', 'total_fit_plus_predict_sec', '.2f')} s{at} |",
                  f"| FS1 inference 35 us per sample (Table A21) | "
                  f"{g('FS_1_config_split_timing', 'predict_us_per_sample', '.1f')} us{at} |",
                  f"| PS2 inference 0.1 us per sample (Table A21) | {g('PS_predict_only', 'predict_us_per_sample', '.2f')} us{at} |"]
    lines += ["", "## Steps", "", "| Step | Block | Paper item | Wall-clock | Status |", "|---|---|---|---|---|"]
    for s in st:
        why = f" ({s['reason']})" if s.get("reason") else ""
        lines.append(f"| {s['id']} | {s['block']} | {s['paper']} | {s.get('elapsed_s', 0.0) / 60:.1f} min | "
                     f"{s['status']}{why} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_dry_run(args, rv):
    blocks, plan, _ = select_plan(args, rv)            # with --only, --from and C00 where added
    if args.markdown:
        print("\n| Step | Paper item | Command (from the repository root) | Output |")
        print("|---|---|---|---|")
        shown = {"{gpu_name}": "", "{resume}": "", "{df40}": DF40_PREPARED, "{df40_public}": "<DF40 lists>",
                 "{reuse_routes}": "", "{defer_audit}": ""}
        for s in plan:
            cmd = " ".join(t for t in (shown.get(t, t) for t in _shown_cmd(s)) if t)
            note = " (needs `--df40-json`)" if s.needs else (" (optional)" if s.optional else "")
            if s.extra.get("premise"):
                note += " (runs when its reference run is recomputed too, or neither is)"
            outs = "<br>".join(f"`{o}`" for o in s.outputs) or "-"
            print(f"| {s.id} | {s.paper}{note} | `{cmd}` | {outs} |")
        return 0
    print(f"\n{len(plan)} steps in blocks {','.join(blocks)}:")
    for s in plan:
        tags = [t for t in ("GPU" if s.gpu else "", "needs --df40-json" if s.needs == "df40" else "",
                            "with --with-reproduction-gate" if s.optional else "",
                            "runs when its reference run is recomputed too, or neither is"
                            if s.extra.get("premise") else "") if t]
        print(f"  [{s.id}] {s.paper}" + (f"  ({'; '.join(tags)})" if tags else ""))
        shown = {"{gpu_name}": "[--gpu-name <name>]", "{resume}": "[--resume when resuming]",
                 "{df40}": DF40_PREPARED, "{df40_public}": "<DF40 split lists>",
                 "{reuse_routes}": "[--df40-json <dir> --raw-root <dir> --expect ...]",
                 "{defer_audit}": "[--defer-audit when C08a is in the run]"}
        print("         " + " ".join(shown.get(t, shlex.quote(t)) for t in _shown_cmd(s)))
    return 0


def _shown_cmd(step):
    """The command of a step as the plan shows it: with the stamping launcher for a step that keeps checkpoints."""
    cmd = list(step.cmd)
    return cmd[:1] + [STAMPER_SHOWN] + cmd[1:] if "{resume}" in cmd and cmd[:1] == ["python"] else cmd


def step_script(step):
    return next((Path(t).name for t in step.cmd if t.endswith(".py")), "")


CAMERA_READY_CACHE_READERS = frozenset({
    "logo_exclusions.py", "11_disjoint_data_v4.py", "08_logo_extended_rungs_v3.py",
    "05_run_c_width_matched_v5.py", "check_split_reuse_v3.py", "appF_strict_refit.py",
    "appA_content_reuse.py", "appA_bit_identity.py", "appB_richer_inputs.py", "appD_width_controls.py",
    "regenerate_appendix_h_v3.py", "appI1_k3_build_v2.py", "appI1_k3_exposure_audit_v2.py",
    "appI1_k3_freeze_v2.py", "appI1_k3_fit.py", "appI1_native_k_preflight.py",
    "appI1_native_k_aligned_v2.py", "appI3_cost_ledger_v2.py",
})


def reads_cache(step):
    script = step_script(step)
    if step.block == "A":
        return script.startswith(tuple(f"{i:02d}_" for i in range(3, 17)))
    return script in CAMERA_READY_CACHE_READERS


def preflight_plan(plan, args, env):
    """Authenticate the cache before touching outputs, using the same paths as the children."""
    active = [s for s in plan if not (s.needs == "df40" and not args.df40_json)]
    needs_cache = any(reads_cache(s) for s in active)
    if not needs_cache and not any(step_script(s) == "17_generate_tables.py" for s in active):
        return
    output = configured_output_root()
    if needs_cache:
        authenticate_cache(output / "cached_features", env)


def configured_output_root():
    """Reject a configuration whose writes would fall outside the harness backup."""
    import yaml

    output = ROOT / "output"
    config = ROOT / "configs" / "paths.yaml"
    if not config.exists():
        config = config.with_name("paths.yaml.example")
    paths = yaml.safe_load(config.read_text(encoding="utf-8"))
    configured = (ROOT / paths.get("output_dir", "./output")).resolve()
    if configured != output.resolve():
        raise SystemExit(f"The harness requires output_dir to resolve to {output}. "
                         f"The configuration resolves it to {configured}. Nothing was changed.")
    return configured


def authenticate_cache(cache, env):
    validator = ROOT / "scripts" / "review" / "validate_cache_release.py"
    rc = subprocess.run([sys.executable, str(validator), "--cache-root", str(cache)],
                        cwd=str(ROOT), env=env).returncode
    if rc:
        raise SystemExit(f"Feature cache authentication failed at {cache}. Nothing was changed.")


def verify_stored_evidence():
    """Check the archived score arrays, fold checkpoints and provenance ledger."""
    try:
        entries = {}
        for line in STORED_EVIDENCE.read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            if (not re.fullmatch(r"[0-9a-f]{64}", digest) or name in entries
                    or not name.startswith(("camera_ready/output/results/", "output/results/",
                                            "camera_ready/provenance/"))):
                raise ValueError(f"invalid entry: {name}")
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or path.relative_to(ROOT).as_posix() != name:
                raise ValueError(f"invalid repository path: {name}")
            entries[name] = (path, digest)
        if not entries:
            raise ValueError("empty manifest")
    except (OSError, ValueError) as exc:
        print(f"  FAIL:    {STORED_EVIDENCE.name} ({exc})")
        return 1

    failures = 0
    for name, (path, digest) in entries.items():
        if not path.is_file():
            print(f"  FAIL:    {name} (missing stored evidence)")
        elif _file_hash(path, fresh=True) != digest:
            print(f"  FAIL:    {name} (stored evidence SHA-256 differs)")
        else:
            continue
        failures += 1
    print(f"Stored evidence: {len(entries) - failures} MATCH, {failures} FAIL")
    return failures


def mode_verify_only(rv, run=None):
    """`run`: None for the stored results; for a rerun, its start and DF40 options."""
    print(f"\n{'=' * 72}\nBLOCK A: reviewed results against the reviewed reference\n{'=' * 72}")
    n_mr, n_mt = rv.verify_expected_outputs()
    try:
        expected = {name for _, _, outputs, _ in rv.EXPERIMENTS for name in outputs if name.endswith(".json")}
        read_reference(rv.REFERENCE_PATH, expected)
        g_fail, g_lines = guard_block_a(rv)
        a_ok, a_warn, a_fail = rv.compare_results()
    except Exception as exc:  # noqa: BLE001
        print(f"  the reviewed comparison stopped: {type(exc).__name__}: {exc}")
        g_fail, g_lines = 0, []
        a_ok, a_warn, a_fail = 0, 0, 1
    print("\n".join(g_lines))
    print(f"\n{'=' * 72}\nBLOCKS B and C: camera-ready results against {rel(REFERENCE2)}\n{'=' * 72}")
    ok, warn, fail, skip, lines = compare_v4(run=run)
    print("\n".join(lines))
    evidence_fail = verify_stored_evidence() if run is None else None
    total = n_mr + n_mt + a_fail + g_fail + fail + (evidence_fail or 0)
    evidence = "" if evidence_fail is None else f"; stored evidence {evidence_fail} FAIL"
    print(f"\nVerification: block A {a_ok} MATCH, {a_warn} WARN, {a_fail} FAIL, {g_fail} FAIL in the version 3 checks; "
          f"blocks B and C {ok} MATCH, {warn} WARN/ENV, {fail} FAIL, {skip} SKIP{evidence}.")
    return {"a": {"match": a_ok, "warn": a_warn, "fail": a_fail, "fail_v3_checks": g_fail, "missing_results": n_mr,
                  "missing_tables": n_mt},
            "bc": {"match": ok, "warn": warn, "fail": fail, "skip": skip},
            "stored_evidence_failures": evidence_fail}, (1 if total else 0)


def mode_check(args, rv):
    env = child_env()
    authenticate_cache(configured_output_root() / "cached_features", env)
    print(f"\n{'=' * 72}\n=== STAGE 1: smoke (reviewed: FS1 arch6 from the cached features)\n{'=' * 72}")
    if rv.run_smoke() != 0:
        return 1
    print(f"\n{'=' * 72}\n=== STAGE 2: smoke (camera-ready: saved LOGO score arrays, no fitting)\n{'=' * 72}")
    rc = subprocess.run([sys.executable, str(CR / "scripts" / "review" / "check_fs1_score_precision_v2.py")],
                        cwd=str(ROOT), env=env).returncode
    if rc != 0:
        return rc
    print(f"\n{'=' * 72}\n=== STAGE 3: reference verification (blocks A, B, C)\n{'=' * 72}")
    _, rc = mode_verify_only(rv)
    if rc != 0:
        return rc
    return 0


LOG_NAME = "reproducibility_log_v4.json"
ARG_KEYS = ("blocks", "gpu_name", "df40_json", "df40_raw", "with_reproduction_gate", "from_step", "only")


def write_log(run_dir, record):
    """Write the run log atomically (temporary file, then rename), so an interruption cannot damage it."""
    tmp = run_dir / (LOG_NAME + ".tmp")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / LOG_NAME)


def latest_records(record):
    """Step id -> its current record (the last one not superseded by a rerun), in the order steps first ran."""
    order, latest = [], {}
    for r in record["steps"]:
        if r["id"] not in order:
            order.append(r["id"])
        if not r.get("superseded"):
            latest[r["id"]] = r
    return {i: latest[i] for i in order if i in latest}


def _damaged(p):
    try:
        if p.stat().st_size == 0:
            return "empty file"
        if p.suffix == ".json":
            json.loads(p.read_text(encoding="utf-8"))
        elif p.suffix == ".npz":
            with zipfile.ZipFile(p) as z:
                members = z.namelist()
                if not members or len(members) != len(set(members)):
                    return "missing or duplicate archive members"
                required = set()
                if p.name.startswith("04_ds_"):
                    required = {"labels", "gen_boundaries", "generators"}
                elif p.name.startswith("K3_OOF_"):
                    required = {"y", "fs", "ps", "pca", "classes"}
                elif p.name.startswith("NATIVE_K_ALIGNED_FULL_B_arch6_") and p.name.endswith("_OOF.npz"):
                    required = {"y", "PS1-K3", "PS2-K3", "PS3-K3", "PS4-K3", "FS1-K3", "FS2-K3", "FSB-K3-18", "PS+-K3"}
                missing = {f"{key}.npy" for key in required} - set(members)
                if missing:
                    return f"missing archive members {sorted(missing)}"
                bad = z.testzip()
                if bad is not None:
                    return f"bad member {bad}"
        elif p.suffix == ".pt":
            import torch
            checkpoint = torch.load(p, map_location="cpu", weights_only=True)
            state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
            if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
                return "checkpoint has no tensor model_state_dict"
        return None
    except Exception as exc:  # noqa: BLE001  any unreadable file counts as damaged
        return f"{type(exc).__name__}: {exc}"


# the fold checkpoints of the runners that resume, each checked together with the score file it names.
# Folder -> (runners, checkpoint name, score file name, the per-fold field that names the methods).
FAMILIES = {
    "folds": (("11_disjoint_data_v2.py", "11_disjoint_data_v3.py", "11_disjoint_data_v4.py"), re.compile(r"^(?P<pool>[^_]+)_d(?P<d>\d+)_(?P<fold>[^_]+)\.json$"),
              "scores/{pool}_d{d}_{fold}.npz", "ps_means"),
    "rung_folds": (("08_logo_extended_rungs_v2.py", "08_logo_extended_rungs_v3.py"), re.compile(r"^(?P<pool>[^_]+)_(?P<fold>[^_]+)\.json$"),
                   "scores/{pool}_rungs_{fold}.npz", "rungs"),
    "mechanism_folds": (("05_run_c_width_matched_v2.py", "05_run_c_width_matched_v3.py",
                         "05_run_c_width_matched_v4.py", "05_run_c_width_matched_v5.py"), re.compile(r"^(?P<pool>[^_]+)_(?P<fold>[^_]+)\.json$"),
                        "scores/{pool}_mechanism_{fold}.npz", "auroc"),
}
K3_CACHE = re.compile(r"^NATIVE_K_ALIGNED_(FULL|SMOKE)_.+_F\d+_.+\.(npz|json)$")   # three-class resume files, in pairs
# the per-fold results of the three-class runner, which are not resume files
K3_FOLD_OUTPUT = re.compile(r"^NATIVE_K_ALIGNED_(FULL|SMOKE)_.+_F\d+_(RESULT\.json|OOF\.npz)$")


def _arg(cmd, flag):
    cmd = list(cmd)
    return cmd[cmd.index(flag) + 1] if flag in cmd[:-1] else None


def _expected_methods(family, pool, d, ref):
    """The methods (and, for the LOGO ladder, the number of FS2 seeds) a fold checkpoint must hold: those of the
    reference result of the same pool."""
    if family == "folds":
        m = ref.get(f"{CD}/11_disjoint_{pool}_d{d}.json", {}).get("metrics", {})
        return ({k.split(".")[2] for k in m if k.startswith("per_fold[0].ps_means.")},
                sum(1 for k in m if k.startswith("per_fold[0].fs_mlp_per_seed[")))
    if family == "rung_folds":
        m = ref.get(f"{CD}/A3c_logo_extended_rungs_{pool}.json", {}).get("metrics", {})
        return {re.sub(r"^folds\.[^.]+\.rungs\.", "", k) for k in m if re.match(r"^folds\.[^.]+\.rungs\.", k)}, None
    m = ref.get(f"{MECH}/run_c_v2_logo_{pool}.json", {}).get("metrics", {})
    return {k.split(".")[2] for k in m if k.startswith("per_fold[0].auroc.")}, None


def _excluded_rows(manifest):
    """Excluded training rows per held-out generator in this run's exclusion list ({} when it does not read)."""
    try:
        man = json.loads((ROOT / manifest).read_text(encoding="utf-8"))
        return {f["held_out"]: int(f["n_excluded"]) for f in man["folds"]}
    except Exception:  # noqa: BLE001  no list: no checkpoint of this pool can be checked, so none is reused
        return {}


def step_checkpoints(s):
    """The resume files of step `s`, in the sets that are reused or moved aside together: each fold
    checkpoint with its score file, as (family, files, name match), and each three-class npz/json pair."""
    if "{resume}" not in s.cmd:
        return []
    script = step_script(s)
    if script in ("appI1_native_k_aligned.py", "appI1_native_k_aligned_v2.py"):
        pairs = {}
        for p in (sorted(TK.iterdir()) if TK.exists() else []):
            if (K3_CACHE.match(p.name) and not K3_FOLD_OUTPUT.match(p.name)          # not the results
                    and f"_{_arg(s.cmd, '--view')}_F" in p.name):
                pairs.setdefault(p.with_suffix(""), []).append(p)
        return [("three_class", files, None) for files in pairs.values()]
    family = next((f for f, spec in FAMILIES.items() if script in spec[0]), None)
    if family is None or _arg(s.cmd, "--out-dir") is None:
        return []
    run_dir = ROOT / _arg(s.cmd, "--out-dir")
    pool, d = _arg(s.cmd, "--pool"), _arg(s.cmd, "--proj_dim")
    sets = []
    for ckpt in sorted((run_dir / family).glob("*.json")) if (run_dir / family).exists() else []:
        m = FAMILIES[family][1].match(ckpt.name)
        if not m or m["pool"] != pool or (family == "folds" and m["d"] != d):
            continue
        scores = run_dir / FAMILIES[family][2].format(**m.groupdict())
        sets.append((family, [ckpt] + ([scores] if scores.exists() else []), m))
    return sets


# ownership of resume files. Every attempt of a step that keeps checkpoints runs its script through
# stamp_checkpoints_v2.py, which writes the attempt's id (random) and fingerprint digest, and the digest of the array file,
# into every resume file it writes. A resume file is reused only by its own step in its own run (attempts_of).
STAMPER = Path(__file__).resolve().parent / "stamp_checkpoints_v2.py"         # Checkpoint launcher.
STAMPER_SHOWN = "camera_ready/scripts/review/stamp_checkpoints_v2.py"


def attempts_of(record, step_id):
    """The attempts of a step in this run, by attempt id (from the run log)."""
    return {r["attempt_id"]: r for r in record.get("steps", []) if r.get("id") == step_id and r.get("attempt_id")}


# ---------------------------------------------------------------------------
# fingerprints. What an attempt depends on, recorded when it starts and compared before anything of it is
# reused: its command (settings), the repository's code and configuration, the Python packages, the feature cache
# and the content of the input files it reads.
# ---------------------------------------------------------------------------

FP_CODE_SUFFIXES = (".py", ".yaml", ".yml", ".toml", ".cfg", ".lock")
FP_SKIP_DIRS = frozenset({".git", "output", "tests", "__pycache__", ".pytest_cache", "run_inputs", ".venv", "venv",
                          "reproducibility_backup", "logs"})
FP_SKIP_INPUT_DIRS = frozenset({"folds", "rung_folds", "mechanism_folds", "scores", "logs", "reproducibility_backup"})
_HASHES, _GLOBAL_FP = {}, {}


def _name(p):
    """A file's path relative to the repository root (absolute when outside it)."""
    try:
        return rel(p)
    except ValueError:
        return Path(p).resolve().as_posix()


def _file_hash(p, *, fresh=False):
    """SHA-256 of a file's content. Re-read resume-verified outputs even when file stats match."""
    try:
        key = None
        if not fresh:
            st = p.stat()
            key = (str(p), st.st_size, st.st_mtime_ns)
            if key in _HASHES:
                return _HASHES[key]
        import hashlib
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        if key is not None:
            _HASHES[key] = digest
        return digest
    except OSError as exc:
        return f"unreadable ({type(exc).__name__})"


def resume_output_problem(path, digest, run_start):
    if not path.is_file() or path.stat().st_mtime < run_start - 2:
        return True
    return (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or _file_hash(path, fresh=True) != digest or bool(_damaged(path)))


def _crlf_hash(p):
    """SHA-256 of a file's content with its line endings made CRLF, as the stored files have them. A
    checkout with LF endings, or a file rewritten on Linux or macOS, holds the same content with other bytes."""
    try:
        import hashlib
        b = p.read_bytes()
        return hashlib.sha256(b.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()
    except OSError as exc:
        return f"unreadable ({type(exc).__name__})"


def _tree_hash(files):
    import hashlib
    h = hashlib.sha256()
    for p in sorted(files, key=lambda q: q.as_posix()):
        h.update(f"{_name(p)}\0{_file_hash(p)}\n".encode("utf-8"))
    return h.hexdigest()


def _walk(base, skip_dirs):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def global_fingerprint():
    """Code, configuration and Python packages (computed once per process)."""
    if not _GLOBAL_FP:
        from importlib import metadata
        code = [p for p in _walk(ROOT, FP_SKIP_DIRS)
                if p.suffix in FP_CODE_SUFFIXES or (p.name.startswith("requirements") and p.suffix == ".txt")]
        versions = {}
        for name in PACKAGES:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        _GLOBAL_FP.update({"code": _tree_hash(code),
                           "packages": json.dumps({"python": platform.python_version(), **versions}, sort_keys=True)})
    return {key: value for key, value in _GLOBAL_FP.items() if key != "cache"}


def fingerprint_inputs(step, plan, args=None):
    """The input files of a step: every file it reads (step_reads), a folder counting with the files in it except the
    step's own outputs, the outputs of later steps of the plan, checkpoint, score and log folders, three-class resume
    files, console logs and notes (they change while the run goes on and are checked on their own)."""
    own = set(step.outputs)
    later = {o for t in plan[next((i for i, t in enumerate(plan) if t.id == step.id), len(plan)) + 1:]
             for o in t.outputs}
    files = set()
    for r in step_reads(step, args):
        bases = [INPUTS, RESULTS2, ROOT / "output" / "results"] if r == "" else [ROOT / r]
        for p in bases:
            if p.is_file():
                if _name(p) not in own and _name(p) not in later:        # nor a later step's output
                    files.add(p)
            elif p.is_dir():
                for f in _walk(p, FP_SKIP_INPUT_DIRS):
                    name = _name(f)
                    # Reviewed evidence is under inputs/; generated console logs sit in logs/ folders.
                    if name in own or name in later or K3_CACHE.match(f.name) or f.suffix in (".md", ".tmp"):
                        continue
                    files.add(f)
    return sorted(files, key=lambda q: q.as_posix())


def df40_images_digest(step, args):
    """Hash the image-file metadata a step reads from --df40-raw, or return None if it reads none.
    Step C01 ({reuse_routes}) looks up <image root>/<id> for every id of its --ids
    lists and reads only whether the file exists and its size (check_split_reuse_v3.py opens no image), so the digest
    covers exactly that, list by list and id by id; a missing file or an unreadable entry counts as such. About 51,000
    lookups (8 to 20 seconds on the authors' laptop), each time C01's fingerprint is taken."""
    raw = getattr(args, "df40_raw", None) if args is not None else None
    if not raw or "{reuse_routes}" not in step.cmd:
        return None
    import hashlib
    image_root, ids = Path(raw).resolve(), _arg(step.cmd, "--ids")
    lists = sorted((ROOT / ids).glob("*.txt")) if ids and (ROOT / ids).is_dir() else []
    h = hashlib.sha256()
    for lst in lists:
        for line in lst.read_text(encoding="utf-8").splitlines():
            try:
                size = str((image_root / line).stat().st_size)
            except FileNotFoundError:
                size = "missing"
            except OSError as exc:
                size = f"unreadable ({type(exc).__name__})"
            h.update(f"{lst.name}\0{line}\0{size}\n".encode("utf-8"))
    return h.hexdigest()


def fingerprint(step, plan, args):
    """The fingerprint of an attempt of `step` with the run's options `args` (the command without --resume). With
    --df40-raw, step C01's also holds the DF40 image files it looks up (df40_images_digest)."""
    import hashlib
    argv = ["python"] + [STAMPER_SHOWN if t == str(STAMPER) else t for t in resolve_cmd(step, args, resuming=False)[1:]]
    fp = {"settings": hashlib.sha256(shlex.join(argv).encode("utf-8")).hexdigest(), **global_fingerprint(),
          "inputs": _tree_hash(fingerprint_inputs(step, plan, args))}
    if reads_cache(step):
        if "cache" not in _GLOBAL_FP:
            _GLOBAL_FP["cache"] = _tree_hash(_walk(ROOT / "output" / "cached_features", ()))
        fp["cache"] = _GLOBAL_FP["cache"]
    images = df40_images_digest(step, args)
    if images is not None:
        fp["df40_images"] = images
    if step.gpu:                    # the device it trains on
        fp["device"] = run_device(getattr(args, "gpu_name", None))
    return fp


_DEVICES = {}
TORCH_DEVICE_PROBE = ("import json, torch; print(json.dumps(torch.cuda.get_device_name(0) "
                      "if torch.cuda.is_available() else 'cpu'))")


def run_device(gpu_name=None):
    """The device a step that trains on the GPU uses in this run's environment, as torch
    sees it: the name of the first visible CUDA device, or "cpu". A resume compares it with the recorded one, so it
    never mixes folds trained on a GPU with folds trained on the CPU or on another GPU. Asked once per GPU choice."""
    if gpu_name not in _DEVICES:
        name = _child_json(TORCH_DEVICE_PROBE, child_env(gpu_name))
        _DEVICES[gpu_name] = name if isinstance(name, str) else "unknown"
    return _DEVICES[gpu_name]


def fingerprint_digest(fp):
    """One SHA-256 for a whole fingerprint (what stamp_checkpoints_v2.py writes into each resume file)."""
    import hashlib
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode("utf-8")).hexdigest() if fp else None


def fingerprint_change(recorded, now):
    """What differs between a recorded fingerprint and the current one ('' when nothing does)."""
    if not recorded:
        return "the fingerprint (not recorded)"
    words = {"settings": "its command", "code": "the code or configuration", "packages": "the Python packages",
             "cache": "the feature cache", "inputs": "its input files",
             "df40_images": "the DF40 image files it looks up (existence and size)",
             "device": "the device it trains on (GPU or CPU)"}
    return ", ".join(words.get(k, k) for k in sorted(set(recorded) | set(now)) if recorded.get(k) != now.get(k))


# the AUROC of a saved score array must equal the one its checkpoint records within this tolerance. The
# runners save float32 scores; the cast changed an AUROC by at most 7.5e-6 in the stored runs
# (camera_ready/output/results/mitigation/fs1_score_precision_check.json).
SCORE_AUC_TOL = 1e-4


def cache_labels(generator):
    """The labels of a held-out generator's test set in the feature cache, in its order (None when absent): every fold
    checkpoint of that generator must hold exactly these."""
    import numpy as np
    try:
        return np.asarray(np.load(ROOT / "output" / "cached_features" / generator / "labels.npy", allow_pickle=False),
                          dtype=np.int64)
    except (OSError, ValueError):
        return None


def scores_problem(family, rec, scores, truth):
    """Why the score file of a fold checkpoint does not match it, or None: its labels must be those of the held-out
    generator in the feature cache (`truth`), in the same order, and every array the checkpoint records an AUROC for
    must be one-dimensional, of the fold's test size and finite, with that AUROC."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    try:
        with np.load(scores, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    except Exception as exc:  # noqa: BLE001
        return f"score file unreadable ({type(exc).__name__})"
    y, n = arrays.get("labels"), rec.get("n_test")
    if y is None or y.ndim != 1 or len(y) != n:
        return "score file lacks the labels of the fold's test size"
    if set(np.unique(y).tolist()) != {0, 1}:
        return "labels are not two-class 0/1"
    if truth is None:
        return "the feature cache holds no labels for this generator, so the labels cannot be checked"
    if len(truth) != len(y) or not np.array_equal(np.asarray(y, dtype=np.int64), truth):
        return "labels differ from the held-out generator's labels in the feature cache"
    recorded = dict(rec.get(FAMILIES[family][3]) or {})
    if family == "folds":
        recorded["FS_1_raw_feat_logreg"] = rec.get("raw_feat_auc")
        if rec.get("fs_mlp_mean") is not None:
            recorded["FS_2_mlp_seed_avg"] = rec["fs_mlp_mean"]
    for name, auc in sorted(recorded.items()):
        s = arrays.get(name)
        if s is None:
            return f"score file lacks {name}"
        if s.ndim != 1 or len(s) != n:
            return f"{name}: scores of shape {tuple(s.shape)}, expected ({n},)"
        if not np.all(np.isfinite(s)):
            return f"{name}: scores not finite"
        if not isinstance(auc, (int, float)) or not abs(roc_auc_score(y, s) - auc) <= SCORE_AUC_TOL:
            return f"{name}: the AUROC of the saved scores differs from the recorded {auc}"
    return None


class _Mover:
    """Moves files into one backup folder (nothing is deleted) and records every move."""

    def __init__(self, label):
        self.dest = BACKUP2 / f"{label}_{datetime.now():%Y%m%d_%H%M%S}"
        self.moved, self.steps = [], set()

    def aside(self, files, why, step_id=None):
        for i, p in enumerate(files):
            target = self.dest / p.relative_to(RESULTS2)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(target))
            reason = why if i == 0 else "moved with the file above"
            self.moved.append({"file": rel(p), "reason": reason, "moved_to": rel(target), "step": step_id})
            print(f"  checkpoint moved aside: {rel(p)} ({reason})", flush=True)
        if step_id:
            self.steps.add(step_id)


def stamp_problem(stamp, attempts, fp_now, array_file, digest_key):
    """Why a resume file cannot be bound to an attempt of its own step in this run, or None. `stamp`: its provenance
    field (stamp_checkpoints_v2.py); `attempts`: attempts_of the step; `fp_now`: the step's fingerprint now (None: not
    compared); `array_file`: the array file written with it, whose digest the stamp records under `digest_key`."""
    if not isinstance(stamp, dict) or not stamp.get("attempt"):
        return "not stamped by an attempt of this run"
    attempt = attempts.get(stamp["attempt"])
    if attempt is None:
        return "stamped by another run or step"
    if stamp.get("fingerprint") != fingerprint_digest(attempt.get("fingerprint")):
        return "its fingerprint is not that of the attempt it names"
    changed = fingerprint_change(attempt.get("fingerprint"), fp_now) if fp_now is not None else ""
    if changed:
        return f"{changed} changed since the attempt that wrote it"
    if array_file is None or not array_file.is_file() or _file_hash(array_file) != stamp.get(digest_key):
        return "its array file is not the one written with it"
    return None


def fold_problem(family, files, m, attempts, ref, excluded, fp_now=None):
    """Why a fold checkpoint cannot be reused, or None. `files`: the checkpoint and, when it exists, its score file;
    `attempts`: the attempts of its step in this run (attempts_of); `excluded`: held-out generator -> excluded
    training rows (None: the run without exclusion, gate G0); `fp_now`: the step's fingerprint now."""
    ckpt = files[0]
    try:
        rec = json.loads(ckpt.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"unreadable ({type(exc).__name__})"
    scores_rel = FAMILIES[family][2].format(**m.groupdict())
    if rec.get("scores_file") != scores_rel or len(files) != 2:
        return f"score file {scores_rel} missing"
    scores = files[1]
    why = stamp_problem(rec.get("provenance"), attempts, fp_now, scores, "scores_sha256")
    if why:
        return why
    methods_ref, seeds_ref = _expected_methods(family, m["pool"], m.groupdict().get("d"), ref)
    methods = set(rec.get(FAMILIES[family][3]) or {})
    if not methods_ref or methods != methods_ref:
        return "methods differ from the reference result"
    if family == "folds" and (not isinstance(rec.get("fs_mlp_mean"), (int, float))
                              or len(rec.get("fs_mlp_per_seed") or []) != seeds_ref):
        return "FS2 or its number of seeds differs from the reference result"
    want = 0 if excluded is None else excluded.get(m["fold"])
    if want is None or rec.get("n_excluded_training_rows") != want:
        return "excluded training rows differ from this run's exclusion list"
    return scores_problem(family, rec, scores, cache_labels(m["fold"]))   # every saved array, labels and AUROCs


def k3_arrays(files):
    """The arrays of a three-class resume pair, and why they cannot be used (None when they can): readable, finite,
    and, for the class probabilities ("select", "test"), rows of probabilities; the seed in the metadata must be the
    one in the file name."""
    import numpy as np
    npz = next((p for p in files if p.suffix == ".npz"), None)
    meta = next((p for p in files if p.suffix == ".json"), None)
    if len(files) != 2 or npz is None or meta is None:
        return None, "incomplete pair"
    try:
        info = json.loads(meta.read_text(encoding="utf-8"))
        with np.load(npz, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    except Exception as exc:  # noqa: BLE001
        return None, f"unreadable ({type(exc).__name__})"
    if not arrays:
        return None, "no arrays"
    for k, a in sorted(arrays.items()):
        if a.size == 0 or not np.issubdtype(a.dtype, np.number) or not np.all(np.isfinite(a)):
            return None, f"{k}: empty, not numeric or not finite"
    for k in ("select", "test"):
        a = arrays.get(k)
        if a is not None and (a.ndim != 2 or a.min() < 0 or a.max() > 1
                              or not np.allclose(a.sum(axis=1), 1.0, atol=1e-4)):
            return None, f"{k}: not rows of class probabilities"
    seed = re.search(r"_S(\d+)$", npz.stem)
    if seed and isinstance(info, dict) and info.get("seed") is not None and int(info["seed"]) != int(seed.group(1)):
        return None, "the seed in the metadata differs from the file name"
    return arrays, None


def k3_expected(view):
    """Rows of each part of each fold of a three-class view and its number of classes, from the frozen manifest (as
    appI1_native_k_aligned.py splits it: test = the fold's rows; select and fit = the other folds' SELECT and FIT
    rows). {} when the manifest does not read."""
    try:
        man = json.loads((TK / f"K3_FROZEN_{view}.json").read_text(encoding="utf-8"))
        rows, k = man["manifest"], len(man["classes"])
    except Exception:  # noqa: BLE001
        return {}
    return {f: {"test": sum(r["fold"] == f for r in rows),
                "select": sum(r["fold"] != f and r["split"] == "SELECT" for r in rows),
                "fit": sum(r["fold"] != f and r["split"] == "FIT" for r in rows), "classes": k}
            for f in sorted({r["fold"] for r in rows})}


K3_PARTS = (("select", "select"), ("test", "test"), ("select_logits", "select"), ("test_logits", "test"),
            ("fit_logits", "fit"))


def k3_problems(sets, attempts, fp_now, expected):
    """Why each three-class resume pair of one step cannot be reused ({index: why}): k3_arrays, the stamp of its
    attempt (stamp_problem, with the digest of its array file), and the rows and classes of its fold in the frozen
    manifest (`expected`, k3_expected) for every array: class probabilities and head outputs alike."""
    problems = {}
    for i, (_family, files, _m) in enumerate(sets):
        arrays, why = k3_arrays(files)
        if not why:
            npz = next(p for p in files if p.suffix == ".npz")
            info = json.loads(next(p for p in files if p.suffix == ".json").read_text(encoding="utf-8"))
            why = stamp_problem(info.get("provenance") if isinstance(info, dict) else None, attempts, fp_now, npz,
                                "arrays_sha256")
        if not why:
            exp = expected.get(int(re.search(r"_F(\d+)_", npz.name).group(1)))
            if exp is None:
                why = "the frozen manifest of the view does not read or lacks this fold"
            for key, part in K3_PARTS:
                a = arrays.get(key)
                if why or a is None:
                    continue
                if len(a) != exp[part]:
                    why = f"{key}: {len(a)} rows, while the fold's {part} part has {exp[part]}"
                elif a.shape[-1] != exp["classes"]:
                    why = f"{key}: {a.shape[-1]} classes, while the view has {exp['classes']}"
        if why:
            problems[i] = why
    return problems


def validate_checkpoints(record, plan, args=None):
    """Before resuming, check the resume files of every step that keeps them. A set that
    fails a check is moved into the backup folder (nothing is deleted), and its step runs again.
    `args`: the run's options, for the fingerprints
    (None: fingerprints are not compared). Returns (moved, step ids)."""
    mover = _Mover("checkpoints")
    ref = {}
    if REFERENCE2.exists():
        ref = json.loads(REFERENCE2.read_text(encoding="utf-8")).get("results", {})
    for s in plan:
        sets = step_checkpoints(s)
        if not sets:
            continue
        attempts = attempts_of(record, s.id)
        fp_now = fingerprint(s, plan, args) if args is not None else None
        excluded = None if _arg(s.cmd, "--exclusion") == "none" else _excluded_rows(
            _arg(s.cmd, "--manifest") or f"{EXCL}/logo_exclusions_{_arg(s.cmd, '--pool')}.json")
        k3 = (k3_problems(sets, attempts, fp_now, k3_expected(_arg(s.cmd, "--view")))
              if sets[0][0] == "three_class" else {})
        for i, (family, files, m) in enumerate(sets):
            if family == "three_class":
                why = k3.get(i)
            else:
                why = fold_problem(family, files, m, attempts, ref, excluded, fp_now)
            if why:
                mover.aside(files, why, s.id)
    return mover.moved, mover.steps


def invalidate_checkpoints(plan, stale):
    """A step whose inputs are recomputed (it reads an output of a step that runs again) recomputes in full,
    so its resume files are moved aside before it runs. Returns the moves."""
    mover = _Mover("invalidated")
    for s in plan:
        if s.id in stale:
            for _family, files, _m in step_checkpoints(s):
                mover.aside(files, f"{s.id} reads an output that is recomputed", s.id)
    return mover.moved


# what a step's script reads besides the paths on its command line (from the scripts themselves)
READS = {
    "15_auprc_appendix.py": ("output/results/11_disjoint_arch6_d256.json",),
    "17_generate_tables.py": ("output/results",),
    "check_fs1_score_precision.py": (CD, MECH),
    "check_fs1_score_precision_v2.py": (CD, MECH, f"{CD}/scores", f"{MECH}/scores"),
    "appF_strict_refit_claims.py": (ANAR, "output/results/06_bootstrap_arch6.json"),     # and this
    "appF_strict_refit_claims_v2.py": (ANAR, "output/results/06_bootstrap_arch6.json"),
    "appH_logit_convention_diff.py": ("camera_ready/output/results/16_individual_expert_auc_corrected.json",
                                      "output/results/16_individual_expert_auc.json"),
    "appH_logit_convention_diff_v2.py": ("camera_ready/output/results/16_individual_expert_auc_corrected.json",
                                         "output/results/16_individual_expert_auc.json"),
    "regenerate_appendix_h_v2.py": ("output/results/16_individual_expert_auc.json",),
    "regenerate_appendix_h_v3.py": ("output/results/16_individual_expert_auc.json",),
    "full_ladder_stats_standard.py": ("output/results",),
    "full_ladder_stats_standard_v2.py": ("output/results",),
    "full_ladder_stats_standard_v3.py": ("output/results",),
    "appI1_k3_exposure_audit.py": (TKR,),
    "appI1_k3_exposure_audit_v2.py": (TKR,),
    "appI1_k3_freeze.py": (TKR,),
    "appI1_k3_freeze_v2.py": (TKR,),
    "appI1_k3_fit.py": (TKR,),
    "appI1_native_k_preflight.py": (TKR,),
    "appI1_native_k_aligned.py": (TKR,),
    "appI1_native_k_aligned_v2.py": (TKR,),
    "generate_camera_ready_tables_v7.py": ("",),     # the table step stages every stored result
    "generate_camera_ready_tables_v8.py": ("",),
    "generate_camera_ready_tables_v9.py": ("",),
    "generate_camera_ready_tables_v10.py": ("",),
    "generate_camera_ready_tables_v11.py": ("",),
    "generate_camera_ready_tables_v12.py": ("",),
    "generate_camera_ready_tables_v13.py": ("",),
    "generate_camera_ready_tables_v14.py": ("",),
    "generate_camera_ready_tables_v18.py": ("",),
    "full_ladder_stats_v11.py": ("camera_ready/provenance/trusted_run_pairs.json",),
    "full_ladder_stats_v12.py": ("camera_ready/provenance/trusted_run_pairs.json",),
    "full_ladder_stats_v13.py": ("camera_ready/provenance/trusted_run_pairs.json",),
    "09_logo_combined_ceiling_v3.py": ("camera_ready/provenance/trusted_run_pairs.json",),
    "figure_a2_full_ladder_v2.py": ("output/results",),     # Block A's results (--codebase .)
    "figure_a2_full_ladder_v3.py": ("output/results",),
    "figure_a2_full_ladder_v4.py": ("output/results",),
}


def step_reads(step, args=None):
    """The repository paths a step reads: those on its command line except where it writes (--out, --out-dir), and
    those in READS for its script. If the run options supply the DF40 split lists, include those and the
    expected counts the reuse check compares with. The DF40 image root is not walked: step C01 reads
    only the existence and size of the files it looks up there, which its fingerprint holds (df40_images_digest)."""
    cmd, reads = list(step.cmd), []
    for i, t in enumerate(cmd):
        if i and cmd[i - 1] in ("--out", "--out-dir"):
            continue
        if t == "{df40}":
            reads.append(DF40_PREPARED)
        elif t in ("{df40_public}", "{reuse_routes}"):
            if args is not None and getattr(args, "df40_json", None):
                reads.append(Path(args.df40_json).resolve().as_posix())
                if t == "{reuse_routes}" and getattr(args, "df40_raw", None):
                    reads.append("camera_ready/tests/expected_split_reuse.json")
        elif "/" in t and not t.startswith(("-", "{")):
            reads.append(t)
    script = step_script(step)
    if script == "full_ladder_stats_v13.py":
        if run_dir := _arg(cmd, "--run-dir"):
            reads.append(f"{run_dir}/scores")
        if "--self-test" in cmd:
            reads.append("camera_ready/inputs")
    if script == "compare_runs_v2.py" and ("g2" in cmd or "--untouched-only" in cmd):
        reads.append(f"{EXCL}/logo_exclusions_{_arg(cmd, '--pool')}.json")
    return reads + list(READS.get(script, ())) + list(step.extra.get("required_inputs", ()))


def _covers(prefix, path):
    prefix = prefix.rstrip("/")
    return prefix == "" or path == prefix or path.startswith(prefix + "/")


def failed_prerequisite(step, plan, latest):
    reads = step_reads(step)
    for producer in plan:
        if producer.id == step.id:
            break
        status = latest.get(producer.id, {}).get("status")
        if status in (None, "OK", "SKIPPED"):
            continue
        if (step.block == producer.block == "A"
                or (step_script(producer) == "03_verify_cache.py" and reads_cache(step))
                or producer.id in step.extra.get("requires_steps", ())
                or any(_covers(r, o) for r in reads for o in producer.outputs)):
            return producer.id
    return None


def steps_to_redo(plan, unfinished):
    """Which steps run again on a resume. `unfinished` gives the steps that must (id -> why); a later step
    runs again too when it reads an output of a step that runs again, and in Block A every step after the first one
    that runs again does (the reviewed pipeline runs in order, and its reads are not listed here). Returns (why,
    stale): step id -> why it runs again, and the steps that read an output that is recomputed, which therefore rerun
    in full (also when they would have run again anyway)."""
    why, stale = dict(unfinished), set()
    first_a = None
    for i, s in enumerate(plan):
        if s.block == "A" and first_a is not None:
            why.setdefault(s.id, f"follows {first_a} in Block A")
        reads = step_reads(s)
        upstream = next((t.id for t in plan[:i] if t.id in why
                         and (t.id in s.extra.get("requires_steps", ())
                              or set(s.outputs).intersection(t.outputs)
                              or any(_covers(r, o) for r in reads for o in t.outputs))), None)
        if upstream:
            stale.add(s.id)
            relation = "depends on" if upstream in s.extra.get("requires_steps", ()) else "reads an output of"
            why.setdefault(s.id, f"{relation} {upstream}")
        if s.block == "A" and first_a is None and s.id in why:
            first_a = s.id
    return why, stale


def write_runbook(run_dir, record):
    gpu = record["arguments"].get("gpu_name")
    gpu_line = (f"\nBefore resuming, check that `nvidia-smi` shows the GPU named \"{gpu}\"; the harness refuses to run "
                f"without it, so the run cannot switch GPU silently.\n" if gpu else "")
    (run_dir / "RUNBOOK.md").write_text(f"""# Run {record['run_id']}

Started {record['started']} with:

    {record['command']}

## If the run is interrupted (reboot, crash, power loss, closed terminal)

From the repository root:

    python camera_ready/scripts/review/reproduce_all_v4.py --resume-run {record['run_id']}

This reuses the options above and skips the steps already finished. It passes --resume to the runners that keep
checkpoints (an interrupted LOGO run loses at most its current fold). It first checks every checkpoint with the score
file it names: stamped by an attempt of their own step in this run (stamp_checkpoints_v2.py) whose fingerprint of
settings, code, packages, feature cache and input files is unchanged, the score file still the one stamped, matching
the exclusion list, with the held-out generator's labels from the feature cache, and every score array of the fold's
size, finite and with the AUROC the checkpoint records. It moves any checkpoint that fails a check into
camera_ready/output/reproducibility_backup/, so its fold is recomputed. A step whose outputs do not read or whose
fingerprint changed runs again; a step that reads an output of a step that runs again reruns in full. Every earlier
record of this run is kept. Do not run two runs at once in one copy of the repository: they write the same files.
{gpu_line}
## Progress

- `{LOG_NAME}` in this folder: every step with its status (RUNNING while it runs), start, end and time.
- `<step>.log`: the console output of each step; an interrupted attempt is kept as `<step>.attempt<N>.log`.
- `TIMINGS.md`: written when the run finishes.
""", encoding="utf-8")


# gate G2 compares the content-disjoint arch6 run with the run without exclusion, in the folds that lost no
# training rows, at 1e-9 for every method (FS2 and the MLP and XGBoost heads included): both runs must come from one
# environment. The stored pair comes from one session on the authors' RTX 5080.
G2_NEW = (f"{CD}/11_disjoint_arch6_d256.json", f"{CD}/A3c_logo_extended_rungs_arch6.json")
G2_REF = (f"{G0}/11_disjoint_arch6_d256.json", f"{G0}/A3c_logo_extended_rungs_arch6.json")


def _run_environment(path):
    """The library versions and device a LOGO run recorded with its result (None when it cannot be read)."""
    try:
        env = json.loads((ROOT / path).read_text(encoding="utf-8")).get("environment") or {}
    except (OSError, ValueError, AttributeError):
        return None
    keys = ("python", "numpy", "scipy", "sklearn", "xgboost", "torch", "cuda")
    return {**{k: env.get(k) for k in keys}, "system": str(env.get("platform") or "").split("-")[0]}


def premise_problem(step, full_plan):
    """Why a step cannot test its premise in this run (None when it can). Gate G2 runs when this run recomputes both
    runs in each pair, or uses a stored pair that records one environment. This applies to both the LOGO ladders
    and the richer-input runs."""
    if step.extra.get("premise") != "g2":
        return None
    outs = {o for s in full_plan for o in s.outputs}
    for new, ref in zip(G2_NEW, G2_REF):
        if (new in outs) != (ref in outs):
            which = "the content-disjoint arch6 run" if new in outs else "the arch6 run without exclusion"
            name = Path(new if new in outs else ref).name
            return (f"gate G2 compares two runs of one environment, and this run recomputes only {which} ({name}); "
                    f"rerun with --with-reproduction-gate, from step B05 or earlier, to recompute both. Its stored "
                    f"verdict stays")
        if new not in outs:
            a, b = _run_environment(new), _run_environment(ref)
            if a is not None and b is not None and a != b:
                return (f"gate G2 compares two runs of one environment, and the stored {Path(new).name} of the "
                        f"content-disjoint run and of the run without exclusion record different environments; "
                        f"rerun with --with-reproduction-gate to recompute both. Its stored verdict stays")
    return None


def execute_plan(plan, args, record, run_dir, env, run_start, resuming, full_plan=None):
    """Run `plan` in order. `full_plan`: the run's whole plan (default `plan`), against which fingerprints are taken."""
    full_plan = full_plan or plan
    latest = latest_records(record)
    for step in plan:
        why = premise_problem(step, full_plan)
        if why:
            print(f"\n[{step.id}] SKIPPED: {why}")
            record["steps"].append({"id": step.id, "block": step.block, "paper": step.paper, "gpu": step.gpu,
                                    "elapsed_s": 0.0, "status": "SKIPPED", "reason": why})
            write_log(run_dir, record)
            continue
        if step.needs == "df40" and not args.df40_json:
            print(f"\n[{step.id}] SKIPPED (needs --df40-json): {step.paper}")
            record["steps"].append({"id": step.id, "block": step.block, "paper": step.paper, "gpu": step.gpu,
                                    "elapsed_s": 0.0, "status": "SKIPPED", "reason": "needs --df40-json"})
            write_log(run_dir, record)
            continue
        blocked = failed_prerequisite(step, full_plan, latest)
        if blocked is not None:
            why = f"prerequisite {blocked} did not complete successfully"
            print(f"\n[{step.id}] BLOCKED: {why}")
            rec = {"id": step.id, "block": step.block, "paper": step.paper, "gpu": step.gpu,
                   "elapsed_s": 0.0, "status": "BLOCKED", "reason": why}
            record["steps"].append(rec)
            latest[step.id] = rec
            write_log(run_dir, record)
            continue
        if step.extra.get("stage_inputs"):
            stage_run_inputs(run_start, record)
        argv = resolve_cmd(step, args, resuming)
        # what this attempt depends on, and its id, recorded before it starts; the stamping launcher
        # writes both into every resume file of the attempt
        fp, attempt_id = fingerprint(step, full_plan, args), uuid.uuid4().hex
        rec = {"id": step.id, "block": step.block, "paper": step.paper, "gpu": step.gpu, "status": "RUNNING",
               "start": datetime.now().isoformat(timespec="seconds"), "elapsed_s": 0.0, "attempt_id": attempt_id,
               "fingerprint": fp}
        record["steps"].append(rec)
        write_log(run_dir, record)       # a crash during the step leaves it marked RUNNING
        attempt_env = {**env, "REPRO_ATTEMPT_ID": attempt_id, "REPRO_FINGERPRINT": fingerprint_digest(fp),
                       "REPRO_VALIDATED_RESUME": "1" if resuming else "0"}
        rec.update(run_step(step, argv, attempt_env, run_dir))
        latest[step.id] = rec
        write_log(run_dir, record)


def finish_run(record, run_dir, rv):
    latest = latest_records(record)
    record["finished"] = datetime.now().isoformat(timespec="seconds")
    record["step_hours"] = round(sum(r.get("elapsed_s", 0.0) for r in latest.values()) / 3600, 3)
    record["hours_since_start"] = round((time.time() - record["run_start_epoch"]) / 3600, 3)
    print(f"\nSum of step times: {record['step_hours']:.2f} h (since the run started: "
          f"{record['hours_since_start']:.2f} h)")
    verification, rc = mode_verify_only(rv, run={"start": record["run_start_epoch"],
                                                 "df40_lists": record.get("df40_lists_given"),
                                                 "df40_images": record.get("df40_image_root_given")})
    record["verification"] = verification
    # what step C01 carried over instead of recomputing, from the result it wrote in this run
    reuse = RESULTS2 / "split_reuse_check.json"
    if reuse.exists() and reuse.stat().st_mtime >= record["run_start_epoch"] - 2:
        try:
            carried = json.loads(reuse.read_text(encoding="utf-8")).get("_carried_over")
        except (UnicodeDecodeError, json.JSONDecodeError):
            carried = None
        if carried:
            record["split_reuse_carried_over"] = carried
            print(f"\nApp. A reuse check: {', '.join(carried['routes'])} carried over from {carried['from']} "
                  f"(not recomputed: DF40's lists or image root were not given)")
    # the paper numbers are judged only when the table step rebuilt them in this run
    table_ids = {s.id for s in steps_c() if s.extra.get("stage_inputs")}
    new_ledger = CR / "output" / "tables" / "05_number_provenance.csv"
    table_completed = any(latest.get(i, {}).get("status") == "OK" for i in table_ids)
    deliverable_errors = (check_table_deliverables(new_ledger.parent, record["run_start_epoch"])
                          if table_completed else [])
    record["table_deliverable_errors"] = deliverable_errors
    for error in deliverable_errors:
        print(f"  TABLE OUTPUT: {error}")
    rebuilt = (table_completed and new_ledger.is_file()
               and new_ledger.stat().st_mtime >= record["run_start_epoch"] - 2)
    ledger = compare_ledgers() if rebuilt else None
    posted = (compare_posted(path=new_ledger.parent / "06_posted_number_check.md",
                             classes=ledger_classes(new_ledger) if rebuilt else None)
              if table_completed else None)
    if not rebuilt:
        record["paper_numbers"] = "not rebuilt in this run (the table step did not run or did not finish)"
        print("\nPaper numbers: not rebuilt in this run (the table step did not run or did not finish); not judged")
    as_text = lambda d: {k: (v if not isinstance(v, list) else [list(map(str, x)) if isinstance(x, (tuple, list)) else str(x) for x in v])  # noqa: E731
                         for k, v in d.items()}
    largest = lambda items: max((float(x[3]) for x in items if x[3]), default=0.0)  # noqa: E731
    if ledger is not None:
        record["paper_numbers"] = as_text(ledger)
        env, mach = ledger["environment_dependent"], ledger["machine_dependent"]
        print(f"\nPaper numbers (rebuilt ledger vs stored, printed precision): {ledger['same_display']} of "
              f"{ledger['rows']} identical; {len(env)} environment-dependent (heads trained on a GPU, largest change "
              f"{largest(env):.4f}); {len(mach)} machine-dependent (timings, not compared); "
              f"{len(ledger['different_display'])} different; {len(ledger['missing'])} missing; "
              f"{len(ledger['invalid'])} not valid numbers")
        for tag, items in (("ENV", env), ("TIMING", mach), ("DIFFERENT", ledger["different_display"])):
            for k, a, b, d in items[:20]:
                print(f"  {tag}: {k[0]} / {k[1]}: stored {a}, rebuilt {b}" + (f" (change {d})" if d else ""))
        for k in ledger["missing"][:20]:
            print(f"  MISSING: {k[0]} / {k[1]}")
        for k, a, b, why in ledger["invalid"][:20]:
            print(f"  INVALID: {k[0]} / {k[1]}: stored {a}, rebuilt {b} ({why})")
    if posted is not None:
        record["posted_numbers"] = as_text(posted)
        print(f"Numbers posted in the discussion ({rel(new_ledger.parent / '06_posted_number_check.md')}): "
              f"{posted['agree']} of {posted['rows']} agree; "
              f"{len(posted['environment_dependent'])} environment-dependent; "
              f"{len(posted['machine_dependent'])} machine-dependent (timings); {len(posted['different'])} different")
        for tag in ("environment", "machine"):
            for q, a, b, d in posted[f"{tag}_dependent"]:
                print(f"  {'ENV' if tag == 'environment' else 'TIMING'}: {q}: posted {a}, rebuilt {b} (change {d})")
        for q, a, b, status in posted["different"]:
            print(f"  {status}: {q}: posted {a}, rebuilt {b}")
        for error in posted["coverage_errors"]:
            print(f"  POSTED COVERAGE: {error}")
    numbers_bad = (bool(deliverable_errors)
                   or bool(ledger and (ledger["different_display"] or ledger["missing"] or ledger["invalid"]))
                   or bool(posted and (posted["different"] or posted["coverage_errors"])))
    record["paper_numbers_ok"] = not numbers_bad if table_completed else None
    if sys.platform != "win32":          # the references come from the authors' Windows machine
        note = (f"This run is on {platform.system()}; the references were computed on Windows. The deterministic "
                f"L-BFGS fits (FS1) can move by up to about 1e-3 on another platform (camera_ready/REPRODUCTION.md, "
                f"Exactness), so FS1-based values, and gates that compare FS1 at 1e-4 or tighter, may be flagged.")
        record["platform_note"] = note
        print(f"\n{note}")
    failures = [r for r in latest.values() if r.get("status") not in ("OK", "SKIPPED")]
    record["step_failures"] = len(failures)
    options = record.get("arguments", {})
    blocks = record.get("blocks", ("A", "B", "C"))
    partial = options.get("only") or options.get("from_step") or set(blocks) != {"A", "B", "C"}
    record["scope"] = "partial" if partial else "full"
    write_log(run_dir, record)
    timing_report(record, run_dir / "TIMINGS.md", rv)
    print(f"\nRun log: {rel(run_dir / LOG_NAME)}\nTimings: {rel(run_dir / 'TIMINGS.md')}")
    if failures:
        print(f"Steps not OK: {', '.join(r['id'] for r in failures)}. To retry them: "
              f"python camera_ready/scripts/review/reproduce_all_v4.py --resume-run {record['run_id']}")
    requested = {x.strip() for x in (options.get("only") or "").split(",") if x.strip()}
    no_completed_target = bool(requested) and not any(latest.get(i, {}).get("status") == "OK" for i in requested)
    if no_completed_target:
        print("No explicitly requested step completed successfully.")
    failed = bool(failures or rc or numbers_bad or no_completed_target)
    scope = "Partial run (selected steps only)" if partial else "Full workflow run"
    print(f"{scope}: {'failed' if failed else 'completed'}; see the run log for recomputed and skipped steps.")
    return int(failed)


def select_plan(args, rv):
    """The steps a run with these options executes: the blocks, then --only or --from, then C00 where the steps that
    read DF40's lists need it. --dry-run shows the same plan."""
    blocks, full = selected_steps(args, rv)
    plan = full
    if args.only is not None:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        if not wanted:
            raise SystemExit("--only must name at least one step")
        unknown = sorted(set(wanted) - {s.id for s in plan})
        if unknown:
            raise SystemExit(f"--only: no such step(s) {unknown} in blocks {','.join(blocks)} (see --dry-run)")
        plan = [s for s in plan if s.id in wanted]
    if args.from_step:
        ids = [s.id for s in plan]
        if args.from_step not in ids:
            raise SystemExit(f"--from {args.from_step}: no such step (see --dry-run)")
        plan = plan[ids.index(args.from_step):]
    return blocks, with_df40_preparation(plan, full, args), full


def mode_run(args, rv):
    blocks, plan, full = select_plan(args, rv)
    check_plan_prerequisites(plan)
    if args.only and not any(
            not (s.needs == "df40" and not args.df40_json)
            and not (s.optional and not args.with_reproduction_gate) for s in plan):
        raise SystemExit("The explicit --only selection has no runnable steps. Supply its prerequisites "
                         "(--df40-json or --with-reproduction-gate) or select a runnable step. Nothing was changed.")
    partial = bool(args.from_step or args.only)   # a new run of some steps: recomputed fully, nothing cleared
    args.audit_later = any(s.id == "C08a" for s in plan)      # B32 defers its audit to C08a
    env = child_env(args.gpu_name)
    preflight_plan(plan, args, env)
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = LOGS2 / run_id
    if (run_dir / LOG_NAME).exists():
        raise SystemExit(f"run {run_id} already exists; continue it with --resume-run {run_id}, or choose another "
                         f"--run-id")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_start = time.time()
    env_rec = environment_record(env)
    record = {"run_id": run_id, "started": datetime.now().isoformat(timespec="seconds"), "run_start_epoch": run_start,
              "command": shlex.join(["python", Path(sys.argv[0]).as_posix()] + sys.argv[1:]),
              "blocks": blocks, "arguments": {k: getattr(args, k) for k in ARG_KEYS},
              "df40_lists_given": bool(args.df40_json), "df40_image_root_given": bool(args.df40_raw),
              "environment": env_rec,
              "sessions": [{"kind": "start", "started": datetime.now().isoformat(timespec="seconds"),
                            "steps_planned": [s.id for s in plan]}],
              "steps": [], "setup": {"status": "preparing"}}
    print(json.dumps(env_rec, indent=2))
    write_log(run_dir, record)
    write_runbook(run_dir, record)
    print(f"\nRun {run_id}. If it is interrupted: python camera_ready/scripts/review/reproduce_all_v4.py "
          f"--resume-run {run_id}  (see {rel(run_dir / 'RUNBOOK.md')})", flush=True)
    stepped = {s.block for s in plan}
    if "A" in stepped:
        print("\n--- Block A: backup" + ("" if partial else " and clear") + " (reviewed harness) ---")
        record["backup_block_a"] = rel(rv.backup())
        write_log(run_dir, record)
        if not partial:
            rv.clear_outputs()
    if "B" in stepped or "C" in stepped:
        dest = backup_v2(blocks, False, move_aside=not partial)
        record["backup_blocks_bc"] = rel(dest) if dest else None
    record["setup"] = {"status": "complete", "committed": datetime.now().isoformat(timespec="seconds")}
    write_log(run_dir, record)
    execute_plan(plan, args, record, run_dir, env, run_start, resuming=False)
    return finish_run(record, run_dir, rv)


def mode_resume(args, rv):
    """Continue an interrupted run: same options, finished steps skipped, --resume passed, records kept.
    Checkpoints are checked first. A step whose outputs cannot be read or whose fingerprint changed is
    not finished, and a step that reads an output of a step that runs again runs again too."""
    run_dir = LOGS2 / args.resume_run
    if not (run_dir / LOG_NAME).exists():
        raise SystemExit(f"no run log at {rel(run_dir / LOG_NAME)}")
    record = json.loads((run_dir / LOG_NAME).read_text(encoding="utf-8"))
    if record.get("setup", {}).get("status") != "complete":
        raise SystemExit("Cannot resume: run setup did not commit after its backups completed. "
                         "Preserve the interrupted run and backup folders, restore affected outputs from a "
                         "complete backup if setup moved them, then start with a new --run-id. Nothing was changed.")
    stored = record.get("arguments", {})
    eff = argparse.Namespace(**vars(args))
    for k in ARG_KEYS:
        setattr(eff, k, stored.get(k))
    eff.blocks = stored.get("blocks") or args.blocks
    blocks, full = selected_steps(eff, rv)
    plan = full
    if eff.only:
        wanted = {x.strip() for x in eff.only.split(",")}
        plan = [s for s in plan if s.id in wanted]
    if eff.from_step:
        ids = [s.id for s in plan]
        plan = plan[ids.index(eff.from_step):]
    plan = with_df40_preparation(plan, full, eff)
    check_plan_prerequisites(plan)
    eff.audit_later = any(s.id == "C08a" for s in plan)
    run_start = record["run_start_epoch"]
    # what a resume needs is checked before any checkpoint is moved or any record superseded
    # DF40's lists and image root given to the run must be there too; without them, the steps that read
    # them would count as changed and the checkpoints of their readers would be moved aside
    absent = [f"{opt} {val}" for opt, val in (("--df40-json", eff.df40_json), ("--df40-raw", eff.df40_raw))
              if val and not (Path(val) if Path(val).is_absolute() else ROOT / val).exists()]
    if absent:
        raise SystemExit(f"Not resuming run {record['run_id']}: {'; '.join(absent)} not found. Make it available (for "
                         f"example, connect the drive that holds it) and resume again; nothing was changed.")
    env = child_env(eff.gpu_name)       # refuses here, before any move, if the run's GPU is not visible
    preflight_plan(plan, eff, env)
    moved, from_checkpoints = validate_checkpoints(record, plan, eff)   # before anything else
    latest = latest_records(record)
    owners = {output: s.id for s in plan if latest.get(s.id, {}).get("status") == "OK" for output in s.outputs}

    unfinished = {}
    for i, s in enumerate(plan):
        status = latest.get(s.id, {}).get("status")
        if status not in ("OK", "SKIPPED"):
            unfinished[s.id] = "not finished"
        elif status == "OK":
            for output in s.outputs:
                if owners[output] != s.id:
                    continue
                path = ROOT / output
                digest = latest[s.id].get("output_sha256", {}).get(output)
                if resume_output_problem(path, digest, run_start):
                    unfinished[s.id] = f"output {output} is missing, damaged or changed"
                    # C08a reads B32's statistics before rewriting that same file.
                    if any(_covers(r, output) for r in step_reads(s)):
                        for previous in reversed(plan[:i]):
                            if output in previous.outputs:
                                unfinished[previous.id] = f"{s.id} needs its unmodified input {output}"
                                break
            changed = fingerprint_change(latest[s.id].get("fingerprint"), fingerprint(s, plan, eff))
            if changed and s.id not in unfinished:
                unfinished[s.id] = f"{changed} changed since it ran"
    unfinished.update({i: "a checkpoint was moved aside" for i in from_checkpoints if i not in unfinished})
    why, stale = steps_to_redo(plan, unfinished)
    todo = [s for s in plan if s.id in why]
    print(f"\nResuming run {record['run_id']}: {len(plan) - len(todo)} of {len(plan)} steps finished; "
          f"next: {todo[0].id if todo else 'none'}")
    for s in todo:
        note = "" if why[s.id] == "not finished" else why[s.id]
        if s.id in stale and step_checkpoints(s):
            note = (note + "; " if note else "") + "an input is recomputed, so it reruns in full"
        if note:
            print(f"  {s.id} runs again: {note}")
    if not todo:
        if moved:
            record["sessions"].append({"kind": "resume", "started": datetime.now().isoformat(timespec="seconds"),
                                       "steps_planned": [], "checkpoints_moved": moved})
            write_log(run_dir, record)
        return finish_run(record, run_dir, rv)
    moved += invalidate_checkpoints(plan, stale)
    redo = {s.id for s in todo}
    for r in record["steps"]:
        if r["id"] in redo:
            r["superseded"] = True
    record["sessions"].append({"kind": "resume", "started": datetime.now().isoformat(timespec="seconds"),
                               "environment": environment_record(env), "steps_planned": [s.id for s in todo],
                               "why": {s.id: why[s.id] for s in todo}, "rerun_in_full": sorted(stale),
                               "checkpoints_moved": moved})
    write_log(run_dir, record)
    execute_plan(todo, eff, record, run_dir, env, run_start, resuming=True, full_plan=plan)
    return finish_run(record, run_dir, rv)


def main() -> int:
    p = argparse.ArgumentParser(description="Reproducibility entry point for the camera-ready paper (blocks A, B, C).",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="show the plan without running anything")
    p.add_argument("--markdown", action="store_true", help="with --dry-run: print the plan as a markdown table")
    g.add_argument("--verify-only", action="store_true", help="stored results against the references only")
    g.add_argument("--check", action="store_true", help="smoke + verify + cache validation")
    p.add_argument("--blocks", default="A,B,C", help="blocks to run (default A,B,C)")
    p.add_argument("--from", dest="from_step",
                   help="new run from this step id on (backup first; recomputed fully; nothing cleared)")
    p.add_argument("--only", help="new run of one step id, or several separated by commas, in plan order "
                                  "(backup first; recomputed fully; nothing cleared)")
    p.add_argument("--gpu-name", help="pin every step to the one GPU whose name contains this text (e.g. 5080)")
    p.add_argument("--df40-json", help="folder with DF40's split lists (enables the steps that read them)")
    p.add_argument("--df40-raw", help="DF40 image root (file sizes only; completes the Appendix A reuse check)")
    p.add_argument("--with-reproduction-gate", action="store_true",
                   help="also rerun arch6 LOGO without exclusion and check it equals the stored reviewed run "
                        "(gate G0); gate G2 then compares two runs of this environment")
    p.add_argument("--run-id", help="name of the log folder (default run_<time>)")
    p.add_argument("--resume-run", metavar="RUN_ID",
                   help="continue an interrupted run with its own options (see RUNBOOK.md in its log folder)")
    args = p.parse_args()
    if args.resume_run and (args.from_step or args.only is not None or args.dry_run or args.verify_only or args.check):
        p.error("--resume-run takes the run's own options; use it without --from, --only or another mode")
    # Reject options that do not apply to the selected mode.
    if args.markdown and not args.dry_run:
        p.error("--markdown applies to --dry-run")
    if args.verify_only or args.check:
        run_options = (
            ("--from", args.from_step is not None), ("--only", args.only is not None),
            ("--gpu-name", args.gpu_name is not None), ("--df40-json", args.df40_json is not None),
            ("--df40-raw", args.df40_raw is not None), ("--blocks", args.blocks != "A,B,C"),
            ("--with-reproduction-gate", args.with_reproduction_gate), ("--run-id", args.run_id is not None),
        )
        ignored = [option for option, given in run_options if given]
        if ignored:
            mode = "--verify-only" if args.verify_only else "--check"
            p.error(f"{mode} does not accept run options: {', '.join(ignored)}")
    if args.resume_run:
        ignored = [o for o, given in (("--gpu-name", args.gpu_name), ("--df40-json", args.df40_json),
                                      ("--df40-raw", args.df40_raw), ("--blocks", args.blocks != "A,B,C"),
                                      ("--with-reproduction-gate", args.with_reproduction_gate),
                                      ("--run-id", args.run_id)) if given]
        if ignored:
            p.error(f"--resume-run keeps the options the run started with and would ignore {', '.join(ignored)}; "
                    "resume without them, or start a new run (--from <step> or --only <step>)")
    print("=" * 72 + "\nREPRODUCIBILITY (camera-ready, version 4)\n" + f"Repository: {ROOT}\nTime: {datetime.now()}\n"
          + "=" * 72)
    rv = reviewed()
    if args.dry_run:
        return mode_dry_run(args, rv)
    if args.verify_only:
        return mode_verify_only(rv)[1]
    if args.check:
        return mode_check(args, rv)
    if args.resume_run:
        return mode_resume(args, rv)
    return mode_run(args, rv)


if __name__ == "__main__":
    sys.exit(main())
