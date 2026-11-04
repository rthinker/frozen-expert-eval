"""Synthetic tests of the camera-ready harness using supplied results and fixtures."""
import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

CR = Path(__file__).resolve().parents[1]
HARNESS = CR / "scripts" / "review" / "reproduce_all_v4.py"


def _harness():
    spec = importlib.util.spec_from_file_location("reproduce_all_v4", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shared_cases():
    spec = importlib.util.spec_from_file_location("harness_test_cases", CR / "tests" / "harness_test_cases.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._harness = _harness
    return mod


CASES = _shared_cases()
for _name in sorted(dir(CASES)):
    if _name.startswith("test_"):
        globals()[_name] = getattr(CASES, _name)


_verdicts = CASES._verdicts


def test_verify_only_accepts_the_supplied_release(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(HARNESS), "--verify-only"], cwd=tmp_path,
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Verification: block A 64 MATCH" in result.stdout
    assert "blocks B and C 52 MATCH" in result.stdout


def test_a_failed_child_makes_the_run_exit_nonzero(tmp_path, monkeypatch):
    h = _harness()
    CASES._sandbox(h, monkeypatch, tmp_path)
    monkeypatch.setattr(h, "reviewed", lambda: None)
    failed = h.Step("B01", "B", "failing child", (sys.executable, "-c", "raise SystemExit(7)"))
    blocked = h.Step("B02", "B", "dependent child", (sys.executable, "-c", "raise SystemExit(0)"),
                     extra={"requires_steps": (failed.id,)})
    monkeypatch.setattr(h, "selected_steps", lambda *args: (["B"], [failed, blocked]))
    monkeypatch.setattr(sys, "argv", [str(HARNESS), "--blocks", "B", "--run-id", "failed-child"])

    assert h.main() == 1
    run = h.LOGS2 / "failed-child"
    record = json.loads((run / h.LOG_NAME).read_text(encoding="utf-8"))
    assert [(step["id"], step["status"]) for step in record["steps"]] == [("B01", "FAILED"), ("B02", "BLOCKED")]
    assert record["steps"][0]["exit_code"] == 7
    assert not (run / "B02.log").exists()


def _compare(h, tmp_path, cases, run=None):
    ref = {"results": {}}
    for name, (data, entry) in cases.items():
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
        ref["results"][str(tmp_path / name)] = {"kind": "det", "tol": None, "env": None, "key_map": {},
                                                "partial_ok": False, "step": "C", "metrics": {}, "strings_ref": {},
                                                **entry}
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(ref), encoding="utf-8")
    return h.compare_v4(path, run=run)


# Gate G2

def test_gate_g2_runs_only_when_both_runs_it_compares_come_from_this_run():
    h = _harness()
    b = {s.id: s for s in h.steps_b()}
    g2 = next(s for s in b.values() if s.extra.get("premise") == "g2")
    default = [s for s in h.steps_b() if not s.optional]                       # a full run without the gate option
    assert "only the content-disjoint arch6 run" in h.premise_problem(g2, default)
    assert h.premise_problem(g2, h.steps_b()) is None                           # --with-reproduction-gate
    assert h.premise_problem(g2, [g2]) is None                                  # --only: the stored pair
    reference_only = [s for s in h.steps_b() if any(o in h.G2_REF for o in s.outputs)] + [g2]
    assert "only the arch6 run without exclusion" in h.premise_problem(g2, reference_only)
    assert all(h.premise_problem(s, default) is None for s in default if s.id != g2.id)


def test_a_skipped_gate_is_recorded_with_its_reason(tmp_path, monkeypatch):
    h = _harness()
    CASES._sandbox(h, monkeypatch, tmp_path)
    write = CASES.WRITE
    new = h.Step("B08", "B", "content run", ("python", "-c", write, h.G2_NEW[0], "{}"), (h.G2_NEW[0],))
    gate = h.Step("B11", "B", "gate G2", ("python", "-c", write, f"{h.CD}/gate_G2_arch6.json", "{}"),
                  (f"{h.CD}/gate_G2_arch6.json",), extra={"premise": "g2"})
    record = {"steps": []}
    run_dir = tmp_path / "logs"
    run_dir.mkdir()
    args = argparse.Namespace(gpu_name=None, df40_json=None, df40_raw=None)
    h.execute_plan([new, gate], args, record, run_dir, os.environ.copy(), time.time(), resuming=False)
    status = {r["id"]: r["status"] for r in record["steps"]}
    assert status == {"B08": "OK", "B11": "SKIPPED"}, record["steps"]
    assert "--with-reproduction-gate" in record["steps"][-1]["reason"]
    assert not (tmp_path / h.CD / "gate_G2_arch6.json").exists()


# Analysis and authoring steps

def test_the_steps_use_the_new_versions_of_their_scripts():
    h = _harness()
    c = {s.id: s for s in h.steps_c()}
    assert c["C01"].cmd[1].endswith("check_split_reuse_v3.py")
    assert c["C04"].cmd[1].endswith("appA_expert_side_v3.py") and c["C13"].cmd[1].endswith("appH_logit_convention_diff_v2.py")
    assert "--carry-from" in c["C01"].cmd and "camera_ready/inputs/split_reuse_check.json" in c["C01"].cmd
    fig = next(s for s in c.values() if "Figure A2" in s.paper)
    assert fig.cmd[1].endswith("figure_a2_full_ladder_v4.py") and "--fallback" in fig.cmd
    tables = next(s for s in c.values() if s.extra.get("stage_inputs"))
    assert tables.cmd[1].endswith("generate_camera_ready_tables_v18.py")
    # Steps using versioned analysis and authoring scripts.
    want = {"C00": "prepare_df40_lists_v2.py", "C08": "full_ladder_stats_standard_v3.py", "C12": "regenerate_appendix_h_v3.py", "C10": "appF_power_mde_v2.py",
            "C14": "appI1_k3_build_v2.py", "C15": "appI1_k3_exposure_audit_v2.py", "C16": "appI1_k3_freeze_v2.py",
            "C20": "appI1_native_k_aligned_v2.py", "C21": "appI3_cost_ledger_v2.py",
            "C22": "appF_strict_refit_claims_v2.py"}
    for sid, script in want.items():
        assert c[sid].cmd[1].endswith(script), (sid, c[sid].cmd[1])
    b = {s.id: s for s in h.steps_b()}
    runners = {s.cmd[1].rsplit("/", 1)[-1] for s in b.values()}
    assert {"11_disjoint_data_v4.py", "05_run_c_width_matched_v5.py", "check_fs1_score_precision_v2.py"} <= runners
    assert not runners & {"11_disjoint_data_v3.py", "05_run_c_width_matched_v4.py", "check_fs1_score_precision.py"}
    assert h.STAMPER.name == "stamp_checkpoints_v2.py" and h.STAMPER.exists()
    assert "Tables 2, A13, A14 and A15; additional protocol examples" in c["C08"].paper
    assert f"{h.TKR}/K3_OOF_A_df40.npz" in c["C17"].outputs and f"{h.TKR}/K3_OOF_B_arch6.npz" in c["C18"].outputs
    assert h.REFERENCE2.name == "reference_results_camera_ready_v2.json" and h.REFERENCE2.exists()


def test_a_memory_budget_skip_marker_is_not_staged_for_the_tables(tmp_path, monkeypatch):
    h = _harness()
    root = tmp_path / "repo"
    out = root / "camera_ready" / "output"
    for name, value in (("ROOT", root), ("INPUTS", root / "camera_ready" / "inputs"),
                        ("RUN_INPUTS", root / "camera_ready" / "run_inputs"), ("RESULTS2", out / "results"),
                        ("ANA", out / "results" / "analyses"), ("TK", out / "results" / "analyses" / "three_class"),
                        ("BACKUP2", out / "reproducibility_backup")):
        monkeypatch.setattr(h, name, value)
    h.INPUTS.mkdir(parents=True)
    stored = {"name": "fs_1_raw_features_logreg", "pool": "full21", "avg_auc_8": 0.972}
    for name in ("04_ds_full21_rawfeat.json", "04_ds_arch6_rawfeat.json"):
        (h.INPUTS / name).write_text(json.dumps(stored), encoding="utf-8")
    start = time.time() - 5
    reviewed = root / "output" / "results"
    reviewed.mkdir(parents=True)
    marker = {"skipped": True, "reason": "memory_budget", "fs1_method": "raw_feature_logreg", "pool": "full21"}
    (reviewed / "04_ds_full21_rawfeat.json").write_text(json.dumps(marker), encoding="utf-8")
    (reviewed / "04_ds_arch6_rawfeat.json").write_text(json.dumps({"avg_auc_8": 0.966}), encoding="utf-8")
    record = {}
    h.stage_run_inputs(start, record)
    staged = json.loads((h.RUN_INPUTS / "04_ds_full21_rawfeat.json").read_text(encoding="utf-8"))
    assert staged == stored and record["table_inputs_stored_for_skip_markers"] == ["04_ds_full21_rawfeat.json"]
    assert record["table_inputs"]["04_ds_arch6_rawfeat.json"] == "output/results/04_ds_arch6_rawfeat.json"


# Paper numbers

def _ledger(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["table", "quantity", "value", "display", "source", "key", "kind",
                                           "sha256_16", "note"])
        w.writeheader()
        for r in rows:
            w.writerow({"sha256_16": "", "note": "", **r})


def test_timings_and_gpu_heads_are_not_counted_as_different_paper_numbers(tmp_path):
    h = _harness()
    base = [
        {"table": "t06", "quantity": "cost.fs1_fit_s", "value": "53.1", "display": "53", "source": "cost.json",
         "key": "combiner_costs.FS_1_frozen_total.total_fit_plus_predict_sec", "kind": "json"},
        {"table": "Table A10", "quantity": "A10.heygen.delta2", "value": "-0.0388", "display": "-0.039",
         "source": "mitigation_stats.json", "key": "logo_arch6.stats.fs2_full.per_fold[3]", "kind": "json"},
        {"table": "Table A15", "quantity": "A15.arch6.delta2", "value": "0.109", "display": "0.109",
         "source": "05_fs_arch6_mlp_d256.json", "key": "fs2 - best PS (full precision)", "kind": "derived"},
        {"table": "t04", "quantity": "k3.fs1_minus_select", "value": "-0.0211", "display": "-0.021",
         "source": "NATIVE_K_ALIGNED_FULL_B_arch6_RESULT.json",
         "key": "contrasts.FS1_minus_select_chosen_core_PS.mean", "kind": "json"},
        {"table": "Table A13", "quantity": "A13.arch6.fs2.ps", "value": "0.8603", "display": "0.860",
         "source": "full_ladder_stats.json", "key": "standard.arch6.ps_ceiling", "kind": "json"},
    ]
    moved = [dict(r) for r in base]
    for r, new in zip(moved, ("55", "-0.031", "0.101", "-0.022", "0.870")):
        r["display"] = new
    _ledger(tmp_path / "stored.csv", base)
    _ledger(tmp_path / "new.csv", moved)
    got = h.compare_ledgers(tmp_path / "stored.csv", tmp_path / "new.csv")
    assert [k[1] for k, *_ in got["machine_dependent"]] == ["cost.fs1_fit_s"]
    assert sorted(k[1] for k, *_ in got["environment_dependent"]) == [
        "A10.heygen.delta2", "A15.arch6.delta2", "k3.fs1_minus_select"]
    assert [k[1] for k, *_ in got["different_display"]] == ["A13.arch6.fs2.ps"]   # an exact value, despite its id
    posted = tmp_path / "posted.md"
    posted.write_text("| `cost.fs1_fit_s` | t | 53 | 55 | FAIL |\n| `A13.arch6.fs2.ps` | t | 0.860 | 0.870 | FAIL |\n"
                      "| `A10.heygen.delta2` | t | -0.039 | -0.031 | FAIL |\n| `x` | t | 1 | 1 | PASS |\n",
                      encoding="utf-8")
    p = h.compare_posted(posted, classes=h.ledger_classes(tmp_path / "new.csv"))
    assert (p["agree"], len(p["machine_dependent"]), len(p["environment_dependent"]), len(p["different"])) == (
        1, 1, 1, 1)


def test_paper_numbers_are_judged_only_when_the_tables_were_rebuilt_in_the_run(tmp_path, monkeypatch):
    h = _harness()
    CASES._sandbox(h, monkeypatch, tmp_path)

    def must_not_run(*a, **k):
        raise AssertionError("the stored ledger of an earlier run must not be judged")
    monkeypatch.setattr(h, "compare_ledgers", must_not_run)
    monkeypatch.setattr(h, "compare_posted", must_not_run)
    run_dir = tmp_path / "logs"
    run_dir.mkdir()
    record = {"run_id": "t", "run_start_epoch": time.time(), "environment": {"cpu": "c", "ram_gib": 1,
              "torch_in_run_env": {"devices": []}}, "steps": [
              {"id": "C11", "block": "C", "paper": "p", "gpu": False, "elapsed_s": 1.0, "status": "OK"}]}
    assert h.finish_run(record, run_dir, None) == 0
    assert record["paper_numbers"].startswith("not rebuilt in this run") and record["paper_numbers_ok"] is None


# FS2 statistics

def test_statistics_derived_from_fs2_are_environment_dependent(tmp_path):
    h = _harness()
    entry = {"env": h.FS2_KEYS, "metrics": {"delta_fs_vs_ps": 0.079, "fs_wins": 6, "fs_sign_test_p": 0.289,
                                            "fs_wilcoxon_p": 0.055, "fs_fold_bootstrap_ci[0]": 0.022,
                                            "delta_rawfeat_vs_ps": 0.081}}
    other_gpu = {"delta_fs_vs_ps": 0.071, "fs_wins": 7, "fs_sign_test_p": 0.07, "fs_wilcoxon_p": 0.039,
                 "fs_fold_bootstrap_ci": [0.015], "delta_rawfeat_vs_ps": 0.081}
    fs1_moved = dict(other_gpu, delta_rawfeat_vs_ps=0.09)
    got = _compare(h, tmp_path, {"gpu.json": (other_gpu, entry), "fs1.json": (fs1_moved, entry)})
    assert _verdicts(got[4]) == {"gpu.json": "ENV", "fs1.json": "FAIL"}, got[4]
    spec = {s["path"]: s for s in h.RESULTS_SPEC}
    assert spec["camera_ready/output/results/full_ladder_stats.json"]["env"] == h.FS2_KEYS
    import re
    for key in ("standard.arch6.fs2_ci95[0]", "logo_arch6.stats.fs2.wins", "dim_sweep_clip4.64.auroc"):
        assert re.search(h.FS2_KEYS, key), key
    for key in ("standard.arch6.fs1", "per_fold[0].raw_feat_auc", "mechanism_logo.mean_auroc.FS_pca2_logreg"):
        assert not re.search(h.FS2_KEYS, key), key


def test_the_stored_results_match_the_reference_v2():
    h = _harness()
    ok, warn, fail, skip, lines = h.compare_v4()
    assert (ok, warn, fail, skip) == (52, 0, 0, 0), "\n".join(lines)
    gates = [ln for ln in lines if "/gate_G" in ln]          # the six gate files (not gate_expert_side_check)
    assert len(gates) == 6 and all("verdict and" in ln and "metrics" in ln for ln in gates), gates


# Verification

def test_gate_numbers_are_compared_in_a_stored_result(tmp_path):
    h = _harness()
    entry = {"kind": "gate", "metrics": {"untouched.PS_1.max_abs_diff": 0.0}, "strings_ref": {"status": "PASS"},
             "strings": "status$"}
    moved = {"untouched": {"PS_1": {"max_abs_diff": 0.5}}, "status": "PASS"}
    assert _verdicts(_compare(h, tmp_path, {"gate.json": (moved, entry)})[4]) == {"gate.json": "FAIL"}
    rewritten = {"start": time.time() - 60}           # new differences, with complete checked evidence
    moved.update(gate="g2", checked={"folds": ["whichisreal", "CollabDiff", "MidJourney", "heygen",
        "starganv2", "deepfacelab", "stargan", "styleclip"], "logo": 96, "rungs": 96})
    assert _verdicts(_compare(h, tmp_path, {"gate.json": (moved, entry)}, run=rewritten)[4]) == {"gate.json": "MATCH"}


def test_a_stored_file_with_excluded_rows_must_be_the_references_source(tmp_path):
    h = _harness()
    data = {"content_expert": "x", "n": 4, "manifest": [{"fold": 0}, {"fold": 1}]}
    (tmp_path / "probe.json").write_text(json.dumps(data), encoding="utf-8")
    digest = h._file_hash(tmp_path / "probe.json")
    entry = {"metrics": {"n": 4}, "exclude": r"^manifest\[", "source_sha256": digest}
    assert _verdicts(_compare(h, tmp_path, {"same.json": (data, entry)})[4]) == {"same.json": "MATCH"}
    edited = dict(data, manifest=[{"fold": 0}, {"fold": 2}])                  # a changed excluded row
    assert _verdicts(_compare(h, tmp_path, {"edited.json": (edited, entry)})[4]) == {"edited.json": "FAIL"}


def test_an_early_stopping_epoch_is_checked_against_its_own_run():
    h = _harness()
    nums = {"FS2-K3.epochs": 10, "PS3-K3.epochs": 50, "FS2-K3.per_seed.7.best_epoch": 20}
    assert h.invalid_value("FS2-K3.per_seed.7.best_epoch", 20, 9, nums) == "early-stopping epoch outside the epochs run"
    assert h.invalid_value("PS3-K3.per_seed.7.best_epoch", 20, 9, {**nums, "PS3-K3.per_seed.7.best_epoch": 20}) is None


# Resume and fingerprints

def test_a_resume_needs_the_feature_cache_before_anything_moves(tmp_path, monkeypatch):
    h = _harness()
    root = tmp_path / "repo"
    root.mkdir()
    CASES._sandbox(h, monkeypatch, root)
    (root / "configs").mkdir()
    (root / "configs" / "paths.yaml.example").write_text("output_dir: ./output\n", encoding="utf-8")
    runner = root / "logo_exclusions.py"
    runner.write_text(CASES.WRITE, encoding="utf-8")
    steps = [h.Step("B01", "B", "one", ("python", str(runner), "out/a.txt"), ("out/a.txt",))]
    monkeypatch.setattr(h.subprocess, "run", lambda argv, **_: SimpleNamespace(returncode=not Path(argv[-1]).is_dir()))
    monkeypatch.setattr(h, "selected_steps", lambda args, rv: (["B"], list(steps)))
    args = argparse.Namespace(blocks="B", gpu_name=None, df40_json=None, df40_raw=None, with_reproduction_gate=False,
                              from_step=None, only=None, run_id="t3", resume_run=None)
    assert h.mode_run(args, None) == 0
    cache = root / "output" / "cached_features"
    cache.rename(root / "output" / "cache_unplugged")
    args.resume_run = "t3"
    with pytest.raises(SystemExit, match="Feature cache authentication failed"):
        h.mode_resume(args, None)
    assert not h.BACKUP2.exists() or not any(p.name.startswith("checkpoints_") for p in h.BACKUP2.iterdir())
    log = json.loads((root / "camera_ready" / "output" / "logs" / "t3" / h.LOG_NAME).read_text(encoding="utf-8"))
    assert len(log["sessions"]) == 1 and not any(r.get("superseded") for r in log["steps"])


def test_preflight_blocks_cache_readers_but_leaves_stored_only_plans_cache_free(tmp_path, monkeypatch):
    h = _harness()
    monkeypatch.setattr(h, "ROOT", tmp_path)
    args = SimpleNamespace(df40_json=None)
    table = next(s for s in h.steps_c() if s.id == "C24")
    reader = next(s for s in h.steps_b() if s.id == "B08")
    monkeypatch.setattr(h.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1))
    h.preflight_plan([table], args, {})
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "paths.yaml.example").write_text("output_dir: ./output\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Feature cache authentication failed"):
        h.preflight_plan([reader], args, {})
    assert not (tmp_path / "output").exists()


def test_failed_cache_gate_blocks_readers_and_failed_preparation_blocks_list_consumers():
    h = _harness()
    gate = h.Step("A01", "A", "cache", ("python", "scripts/03_verify_cache.py"))
    reader = next(s for s in h.steps_b() if s.id == "B08")
    prepare, exposure = (next(s for s in h.steps_c() if s.id == name) for name in ("C00", "C04"))
    assert h.failed_prerequisite(reader, [gate, reader], {"A01": {"status": "FAILED"}}) == "A01"
    assert h.failed_prerequisite(exposure, [prepare, exposure], {"C00": {"status": "STALE"}}) == "C00"
    self_test, stats = (next(s for s in h.steps_b() if s.id == name) for name in ("B31", "B32"))
    assert h.failed_prerequisite(stats, [self_test, stats], {"B31": {"status": "FAILED"}}) == "B31"


def test_reference_inventory_and_reference_numbers_cannot_silently_reduce_coverage(tmp_path, monkeypatch):
    h = _harness()
    path = tmp_path / "reference.json"
    monkeypatch.setattr(h, "REFERENCE2", path)
    path.write_text('{"results": {}}', encoding="utf-8")
    assert "empty results inventory" in h.compare_v4()[4][0]
    path.write_text(json.dumps({"results": {"test.json": {"metrics": {"auc": 0.5}}}}), encoding="utf-8")
    assert "inventory differs" in h.compare_v4()[4][0]
    path.write_text(json.dumps({"results": {"test.json": {"metrics": {"auc": float("nan")}}}}), encoding="utf-8")
    monkeypatch.setattr(h, "RESULTS_SPEC", [{"path": "test.json"}])
    assert "nonfinite" in h.compare_v4()[4][0]


def test_resume_rejects_changed_json_and_valid_zip_with_missing_arrays(tmp_path):
    h = _harness()
    for path in (tmp_path / "probe.json", tmp_path / "K3_OOF_B_arch6.npz"):
        if path.suffix == ".json":
            path.write_text('{"auc": 0.5}', encoding="utf-8")
        else:
            np.savez(path, y=[0], fs=[1], ps=[1], pca=[1], classes=[1])
        start, digest = time.time(), h._file_hash(path)
        assert not h.resume_output_problem(path, digest, start)
        if path.suffix == ".json":
            path.write_text('{"auc": 0.6}', encoding="utf-8")
        else:
            np.savez(path, y=[0])
            assert "missing archive members" in h._damaged(path)
        assert h.resume_output_problem(path, digest, start)


def test_resume_tracks_the_latest_writer_of_a_shared_report(tmp_path, monkeypatch):
    h = _harness()
    CASES._sandbox(h, monkeypatch, tmp_path)
    output = "out/report.json"
    steps = [h.Step(step_id, "B", step_id, ("python", "-c", CASES.WRITE, output, value), (output,))
             for step_id, value in (("B01", '{"base": 1}'), ("B02", '{"final": 1}'))]
    monkeypatch.setattr(h, "selected_steps", lambda args, rv: (["B"], steps))
    args = argparse.Namespace(blocks="B", gpu_name=None, df40_json=None, df40_raw=None,
                              with_reproduction_gate=False, from_step=None, only=None, run_id="shared", resume_run=None)
    assert h.mode_run(args, None) == 0
    args.resume_run = "shared"
    assert h.mode_resume(args, None) == 0
    log = h.LOGS2 / "shared" / h.LOG_NAME
    assert len(json.loads(log.read_text(encoding="utf-8"))["sessions"]) == 1

    (tmp_path / output).write_text('{"changed": 1}', encoding="utf-8")
    assert h.mode_resume(args, None) == 0
    sessions = json.loads(log.read_text(encoding="utf-8"))["sessions"]
    assert sessions[-1]["steps_planned"] == ["B01", "B02"]
    assert json.loads((tmp_path / output).read_text(encoding="utf-8")) == {"final": 1}


def test_standalone_overrides_are_not_inherited_and_pinned_gpu_must_be_usable(monkeypatch):
    h = _harness()
    for name in ("CR_RESULTS_DIR", "CR_CODEBASE", "MITIGATION_CODEBASE", "DF40_JSON_DIR"):
        monkeypatch.setenv(name, "outside-the-clone")
        assert name not in h.child_env()
    monkeypatch.setattr(h, "_nvidia_smi", lambda: [{"name": "RTX 5080", "index": "0"}])
    monkeypatch.setattr(h, "_child_json", lambda *_: "cpu")
    with pytest.raises(SystemExit, match="PyTorch cannot use"):
        h.child_env("5080")


def test_a_later_steps_output_named_on_the_command_line_is_not_an_input(tmp_path, monkeypatch):
    h = _harness()
    root = tmp_path / "repo"
    root.mkdir()
    CASES._sandbox(h, monkeypatch, root)
    later_out = "camera_ready/output/results/full_ladder_stats.json"
    (root / later_out).parent.mkdir(parents=True, exist_ok=True)
    (root / later_out).write_text("{}", encoding="utf-8")
    (root / "camera_ready" / "inputs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(h, "INPUTS", root / "camera_ready" / "inputs")
    reader = h.Step("B32", "B", "stats", ("python", "x.py", "--standard-stats", later_out), ("out/s.json",))
    writer = h.Step("C08", "C", "standard", ("python", "y.py", "--out", later_out), (later_out,))
    names = [h._name(p) for p in h.fingerprint_inputs(reader, [reader, writer])]
    assert later_out not in names
    assert later_out in [h._name(p) for p in h.fingerprint_inputs(reader, [reader])]


# Table 2 audit and timing reports

def test_the_audit_is_deferred_to_c08a_in_a_full_run():
    h = _harness()
    b = {s.id: s for s in h.steps_b()}
    c = h.steps_c()
    ids = [s.id for s in c]
    assert ids.index("C08a") == ids.index("C08") + 1 and ids[-1] == "C24"
    audit = c[ids.index("C08a")]
    assert audit.cmd[1].endswith("full_ladder_stats_v13.py") and "--audit-only" in audit.cmd
    assert b["B31"].cmd[1].endswith("full_ladder_stats_v13.py")
    stats = b["B32"]
    assert stats.cmd[1].endswith("full_ladder_stats_v13.py") and "{defer_audit}" in stats.cmd
    # The harness requires complete statistics inputs.
    # analysis
    assert not any(t.startswith("--allow-") for s in (audit, b["B31"], stats) for t in s.cmd)
    later = argparse.Namespace(gpu_name=None, df40_json=None, df40_raw=None, audit_later=True)
    alone = argparse.Namespace(gpu_name=None, df40_json=None, df40_raw=None, audit_later=False)
    assert "--defer-audit" in h.resolve_cmd(stats, later, False)
    assert "--defer-audit" not in h.resolve_cmd(stats, alone, False)


def test_the_timing_report_names_the_times_the_paper_states(tmp_path):
    h = _harness()
    steps = [{"id": f"{b}01", "block": b, "paper": "p", "gpu": b == "A", "elapsed_s": 3600.0, "status": "OK"}
             for b in "ABC"]
    record = {"run_id": "t", "environment": {"cpu": "c", "ram_gib": 16, "torch_in_run_env": {"devices": []}},
              "sessions": [{}], "steps": steps}
    h.timing_report(record, tmp_path / "TIMINGS.md")
    text = (tmp_path / "TIMINGS.md").read_text(encoding="utf-8")
    assert "about 36 h" in text and "about 2 h" in text and "Block B about 14 h" in text
    assert "under 1 h" not in text and "about 14 h on a 16 GiB" not in text


def test_cost_timing_report_survives_table_renumbering(tmp_path):
    h = _harness()
    h.ROOT = tmp_path
    cost = tmp_path / h.ANAR / 'run_d_cost_ledger_arch6.json'
    cost.parent.mkdir(parents=True)
    cost.write_text(json.dumps({'combiner_costs': {
        'FS_1_frozen_total': {'total_fit_plus_predict_sec': 53.1}}}), encoding='utf-8')
    record = {'run_id': 'cost', 'environment': {'cpu': 'c', 'ram_gib': 16,
        'torch_in_run_env': {'devices': []}}, 'sessions': [{}], 'steps': [
        {'id': 'C21', 'block': 'C', 'paper': 'App. I.3, Table A21: cost and access ledger',
         'gpu': False, 'elapsed_s': 1, 'status': 'OK'}]}
    h.timing_report(record, tmp_path / 'TIMINGS.md')
    report = (tmp_path / 'TIMINGS.md').read_text(encoding='utf-8')
    assert 'FS1 fit and predict 53 s (Table A21) | 53.1 s (step C21 of this run)' in report


# DF40 input fingerprints

def _df40_sandbox(h, monkeypatch, tmp_path):
    """A sandbox repository with DF40's lists and image root outside it, and one id list for C01."""
    root = tmp_path / "repo"
    root.mkdir()
    CASES._sandbox(h, monkeypatch, root)
    lists = tmp_path / "df40_lists"
    lists.mkdir()
    (lists / "DF40_all.json").write_text('["fake/a.png"]', encoding="utf-8")
    images = tmp_path / "df40_images"
    (images / "fake").mkdir(parents=True)
    (images / "fake" / "a.png").write_bytes(b"12345")
    (images / "fake" / "unused.png").write_bytes(b"1")
    ids = root / "camera_ready" / "output" / "ids"
    ids.mkdir(parents=True)
    (ids / "FIT.txt").write_text("fake/a.png\nfake/b.png\n", encoding="utf-8")       # fake/b.png has no file yet
    return root, lists, images


def _df40_steps(h):
    """Stand-ins for C00, C01, a step reading C00's lists (as C04), an unrelated step and a step reading the results
    folder (as the table step), with the command-line tokens of the real steps."""
    w, reuse = CASES.WRITE, "camera_ready/output/results/split_reuse_check.json"
    return [h.Step("C00", "C", "lists", ("python", "-c", w, f"{h.DF40_PREPARED}/PREPARED.json", "{}", "{df40_public}"),
                   (f"{h.DF40_PREPARED}/PREPARED.json",), needs="df40"),
            h.Step("C01", "C", "reuse", ("python", "-c", w, reuse, "{}", "--ids", "camera_ready/output/ids",
                                         "{reuse_routes}"), (reuse,)),
            h.Step("C04", "C", "expert side", ("python", "-c", w, "out/c04.txt", "x", "--json_dir", "{df40}"),
                   ("out/c04.txt",), needs="df40"),
            h.Step("C05", "C", "other", ("python", "-c", w, "out/c05.txt"), ("out/c05.txt",)),
            h.Step("C22", "C", "tables",
                   ("python", "-c", w, "out/c22.txt", "x", "--from", "camera_ready/output/results"), ("out/c22.txt",))]


def test_df40_lists_and_the_image_files_c01_looks_up_are_in_the_fingerprints(tmp_path, monkeypatch):
    """Changes at the same DF40 paths invalidate finished steps. C00 and C01 fingerprint the split lists.
    With --df40-raw, C01 also fingerprints existence and size for every image path used by the check."""
    h = _harness()
    _, lists, images = _df40_sandbox(h, monkeypatch, tmp_path)
    plan = _df40_steps(h)
    c00, c01 = plan[0], plan[1]
    args = argparse.Namespace(gpu_name=None, df40_json=str(lists), df40_raw=str(images))
    fp00, fp01 = h.fingerprint(c00, plan, args), h.fingerprint(c01, plan, args)
    assert "df40_images" in fp01 and "df40_images" not in fp00
    no_root = argparse.Namespace(gpu_name=None, df40_json=str(lists), df40_raw=None)
    assert "df40_images" not in h.fingerprint(c01, plan, no_root)
    (lists / "DF40_all.json").write_text('["fake/a.png", "fake/b.png"]', encoding="utf-8")   # same path, new content
    assert h.fingerprint_change(fp00, h.fingerprint(c00, plan, args)) == "its input files"
    fp01b = h.fingerprint(c01, plan, args)
    assert h.fingerprint_change(fp01, fp01b) == "its input files"
    (images / "fake" / "unused.png").write_bytes(b"123")              # an image no id names: no change
    assert h.fingerprint_change(fp01b, h.fingerprint(c01, plan, args)) == ""
    (images / "fake" / "a.png").write_bytes(b"1234567")               # an image C01 looks up changes size
    fp01c = h.fingerprint(c01, plan, args)
    assert h.fingerprint_change(fp01b, fp01c) == "the DF40 image files it looks up (existence and size)"
    (images / "fake" / "b.png").write_bytes(b"")                      # an id without a file gets one (zero bytes)
    assert h.fingerprint_change(fp01c, h.fingerprint(c01, plan, args)) == (
        "the DF40 image files it looks up (existence and size)")


def test_a_resume_reruns_the_df40_steps_and_their_readers_when_df40s_files_change(tmp_path, monkeypatch):
    h = _harness()
    root, lists, images = _df40_sandbox(h, monkeypatch, tmp_path)
    steps = _df40_steps(h)
    monkeypatch.setattr(h, "selected_steps", lambda args, rv: (["C"], list(steps)))
    args = argparse.Namespace(blocks="C", gpu_name=None, df40_json=str(lists), df40_raw=str(images),
                              with_reproduction_gate=False, from_step=None, only=None, run_id="t6", resume_run=None)
    assert h.mode_run(args, None) == 0
    log = root / "camera_ready" / "output" / "logs" / "t6" / h.LOG_NAME

    def resume():
        args.resume_run = "t6"
        assert h.mode_resume(args, None) == 0
        return json.loads(log.read_text(encoding="utf-8"))["sessions"]

    (images / "fake" / "unused.png").write_bytes(b"123")              # not looked up by C01: nothing runs again
    assert len(resume()) == 1
    (images / "fake" / "a.png").write_bytes(b"1234567")               # looked up: C01 and the step reading its output
    s = resume()[-1]
    assert s["steps_planned"] == ["C01", "C22"], s
    assert s["why"] == {"C01": "the DF40 image files it looks up (existence and size) changed since it ran",
                        "C22": "reads an output of C01"}
    (lists / "DF40_all.json").write_text('["fake/a.png", "fake/b.png"]', encoding="utf-8")   # the lists
    s = resume()[-1]
    assert s["steps_planned"] == ["C00", "C01", "C04", "C22"], s
    assert s["why"]["C00"] == s["why"]["C01"] == "its input files changed since it ran"
    assert s["why"]["C04"] == "reads an output of C00" and s["why"]["C22"] == "reads an output of C01"


def test_the_steps_reading_c00_or_c01_outputs_run_again_with_them():
    """In the real plan: the steps that read the lists C00 writes ({df40}) and the table step run again when C00 does;
    the table step runs again when C01 does."""
    h = _harness()
    c = h.steps_c()
    readers = {s.id for s in c if "{df40}" in s.cmd}
    tables = {s.id for s in c if s.extra.get("stage_inputs")}
    assert readers and tables
    why, _ = h.steps_to_redo(c, {"C00": "its input files changed since it ran"})
    assert readers | tables <= set(why), why
    assert all(why[i] == "reads an output of C00" for i in readers)
    why, _ = h.steps_to_redo(c, {"C01": "the DF40 image files it looks up (existence and size) changed since it ran"})
    assert tables <= set(why) and not readers & set(why), why


# Run isolation, gate compatibility and ledger classes

def test_a_full_run_moves_only_resume_files_so_a_skipped_gate_keeps_its_stored_verdict(tmp_path, monkeypatch):
    """F1: a full run moved the whole content_disjoint/ folder aside, with the stored G2 verdict; G2, skipped without
    --with-reproduction-gate, never wrote it back, and the closing verification reported it MISSING (exit 1). Now
    only the resume files and per-sample predictions move, and the stored verdict is verified as stored."""
    h = _harness()
    root = tmp_path / "repo"
    root.mkdir()
    CASES._sandbox(h, monkeypatch, root)
    cd, mech, excl = root / h.CD, root / h.MECH, root / h.EXCL
    for f, text in ((cd / "gate_G2_arch6.json", '{"status": "PASS", "worst": 0.0}'),
                    (cd / "11_disjoint_arch6_d256.json", "{}"), (cd / "folds" / "arch6_d256_f1.json", "{}"),
                    (cd / "scores" / "arch6_d256_f1.npz", "x"), (mech / "mechanism_folds" / "f1.json", "{}"),
                    (mech / "run_c_v2_logo_arch6.json", "{}"), (excl / "logo_exclusions_arch6.json", "{}")):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    old = time.time() - 86400                                   # stored before the run started
    os.utime(cd / "gate_G2_arch6.json", (old, old))
    dest = h.backup_v2(["B", "C"], False, move_aside=True)
    for kept in (cd / "gate_G2_arch6.json", cd / "11_disjoint_arch6_d256.json", mech / "run_c_v2_logo_arch6.json",
                 excl / "logo_exclusions_arch6.json"):
        assert kept.exists(), kept
    for moved, sub in ((cd / "folds", "content_disjoint/folds"), (cd / "scores", "content_disjoint/scores"),
                       (mech / "mechanism_folds", "mechanism/mechanism_folds")):
        assert not moved.exists() and (dest / "moved_aside" / sub).is_dir(), sub
    ref = {"results": {f"{h.CD}/gate_G2_arch6.json": {
        "kind": "gate", "tol": None, "env": None, "key_map": {}, "partial_ok": False, "step": "B",
        "metrics": {"worst": 0.0}, "strings_ref": {"status": "PASS"}, "strings": "status$"}}}
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(ref), encoding="utf-8")
    ok, warn, fail, skip, lines = h.compare_v4(path, run={"start": time.time(), "df40_lists": False,
                                                          "df40_images": False})
    assert (ok, fail) == (1, 0), lines


def test_gate_g2_premise_is_judged_pair_by_pair_and_by_the_recorded_environment(tmp_path, monkeypatch):
    """F4: with --with-reproduction-gate --from B06 the run recomputes the richer-input runs of both but the ladder of
    the content-disjoint run only; judged side by side, G2 then compared environments at 1e-9. A stored pair must
    also record the same environment."""
    h = _harness()
    full = h.steps_b()                                          # with the steps of --with-reproduction-gate
    ids = [s.id for s in full]
    g2 = next(s for s in full if s.extra.get("premise") == "g2")
    why = h.premise_problem(g2, full[ids.index("B06"):])
    assert why and "only the content-disjoint arch6 run" in why and "11_disjoint_arch6_d256.json" in why, why
    assert h.premise_problem(g2, full) is None
    root = tmp_path / "repo"
    monkeypatch.setattr(h, "ROOT", root)
    env = {"python": "3.11.3", "numpy": "1.26.4", "scipy": "1.11.4", "sklearn": "1.6.0", "xgboost": "3.2.0",
           "torch": "2.10.0+cu128", "cuda": "NVIDIA GeForce RTX 5080", "platform": "Windows-10-10.0.26300-SP0"}
    for path in h.G2_NEW + h.G2_REF:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(json.dumps({"environment": env}), encoding="utf-8")
    assert h.premise_problem(g2, [g2]) is None                  # the stored pair: one environment
    other = dict(env, platform="Linux-5.15.0-x86_64-with-glibc2.35", cuda=None)
    (root / h.G2_NEW[0]).write_text(json.dumps({"environment": other}), encoding="utf-8")
    why = h.premise_problem(g2, [g2])
    assert why and "record different environments" in why, why


def test_every_value_of_the_reviewed_fs2_outputs_is_environment_dependent_in_the_ledger():
    """F2: the paper-number check classed rows by key text only, so the FS2 values of the reviewed outputs (keys such
    as mean_avg8 or per_generator_seed_averaged) counted as exact, and a full rerun on another GPU or the CPU ended
    with exit 1. On the stored ledger: every row from scripts 05, 07 and 10 of the reviewed pipeline is
    environment-dependent except settings and counts; FS1 values of the LOGO runs stay exact."""
    h = _harness()
    with open(CR / "provenance" / "05_number_provenance.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    fs2 = [r for r in rows if h.FS2_SOURCES.search(r["source"])]
    assert len(fs2) > 100
    exact = {r["key"] for r in fs2 if h.ledger_class(r) == "exact"}
    assert exact == {"4 experts x proj_dim", "len(per_generator_seed_averaged)", "n_trainable_params", "proj_dim",
                     "proj_dim + 1 (PCA weights + intercept)"}, exact
    by_quantity = {r["quantity"]: h.ledger_class(r) for r in rows}
    assert by_quantity["A15.arch6.fs2"] == by_quantity["A15.arch6.delta2"] == "environment"
    assert by_quantity["T2.reporting.fs2_delta"] == "environment"
    logo_fs1 = [r for r in rows if r["source"].endswith("11_disjoint_arch6_d256.json")
                and r["key"].endswith("raw_feat_auc")]
    assert logo_fs1 and all(h.ledger_class(r) == "exact" for r in logo_fs1)


# Checkpoint classification, fingerprints and CLI options
def test_the_per_fold_results_of_the_three_class_runner_are_not_resume_pairs(tmp_path, monkeypatch):
    """Keep _F<k>_RESULT.json and _F<k>_OOF.npz out of the checkpoint-pair inventory. They are final fold outputs."""
    h = _harness()
    tk = tmp_path / "three_class"
    tk.mkdir()
    for name in ("NATIVE_K_ALIGNED_FULL_B_arch6_F0_PS3_K3_S7.npz", "NATIVE_K_ALIGNED_FULL_B_arch6_F0_PS3_K3_S7.json",
                 "NATIVE_K_ALIGNED_FULL_B_arch6_F0_HEADS.npz", "NATIVE_K_ALIGNED_FULL_B_arch6_F0_HEADS.json",
                 "NATIVE_K_ALIGNED_FULL_B_arch6_F0_RESULT.json", "NATIVE_K_ALIGNED_FULL_B_arch6_F0_OOF.npz"):
        (tk / name).write_bytes(b"x")
    monkeypatch.setattr(h, "TK", tk)
    c20 = next(s for s in h.steps_c() if s.id == "C20")
    sets = h.step_checkpoints(c20)
    stems = sorted(sorted(p.name for p in files) for _, files, _ in sets)
    assert stems == [["NATIVE_K_ALIGNED_FULL_B_arch6_F0_HEADS.json", "NATIVE_K_ALIGNED_FULL_B_arch6_F0_HEADS.npz"],
                     ["NATIVE_K_ALIGNED_FULL_B_arch6_F0_PS3_K3_S7.json",
                      "NATIVE_K_ALIGNED_FULL_B_arch6_F0_PS3_K3_S7.npz"]]


def test_c11_declares_the_diagnosis_file_it_rewrites_so_c08_does_not_fingerprint_it():
    """C11 rewrites C02's diagnosis file. C08 omits the later producer's output from its input fingerprint."""
    h = _harness()
    c = {s.id: s for s in h.steps_c()}
    diag = f"{h.ANAR}/run_b_diagnose_arch6.json"
    assert diag in c["C11"].outputs and diag in c["C02"].outputs
    plan = h.steps_c()
    args = argparse.Namespace(gpu_name=None, df40_json=None, df40_raw=None)
    inputs = {p.name for p in h.fingerprint_inputs(c["C08"], plan, args)}
    assert "run_b_diagnose_arch6.json" not in inputs


def test_the_fingerprint_of_a_gpu_step_holds_its_device(monkeypatch):
    """The device fingerprint prevents reuse across CPU and GPU runs."""
    h = _harness()
    monkeypatch.setattr(h, "global_fingerprint", lambda: {})
    monkeypatch.setattr(h, "fingerprint_inputs", lambda step, plan, args: [])
    monkeypatch.setattr(h, "run_device", lambda gpu_name=None: "NVIDIA GeForce RTX 5080")
    args = argparse.Namespace(gpu_name=None, df40_json=None, df40_raw=None, audit_later=False)
    gpu_step = next(s for s in h.steps_b() if s.gpu)
    cpu_step = next(s for s in h.steps_c() if not s.gpu and not s.needs and "{reuse_routes}" not in s.cmd)
    fp = h.fingerprint(gpu_step, [gpu_step], args)
    assert fp["device"] == "NVIDIA GeForce RTX 5080" and "device" not in h.fingerprint(cpu_step, [cpu_step], args)
    assert h.fingerprint_change(fp, {**fp, "device": "cpu"}) == "the device it trains on (GPU or CPU)"


def test_options_a_mode_would_ignore_are_refused(monkeypatch):
    """An unknown block or an option inapplicable to its mode must cause a nonzero exit."""
    h = _harness()
    for argv in (["--blocks", "A B"], ["--blocks", "D"], ["--markdown"], ["--resume-run", "x", "--gpu-name", "5080"],
                 ["--resume-run", "x", "--blocks", "C"], ["--resume-run", "x", "--with-reproduction-gate"]):
        monkeypatch.setattr(sys, "argv", ["reproduce_all_v4.py"] + argv)
        with pytest.raises(SystemExit) as e:
            h.main()
        assert e.value.code != 0, argv


def test_a_dry_run_shows_the_plan_of_only_and_from(capsys, monkeypatch):
    h = _harness()
    monkeypatch.setattr(sys, "argv", ["reproduce_all_v4.py", "--dry-run", "--only", "C08,C24"])
    assert h.main() == 0
    out = capsys.readouterr().out
    assert "2 steps in blocks A,B,C" in out and "[C08]" in out and "[C24]" in out and "[C09]" not in out
    monkeypatch.setattr(sys, "argv", ["reproduce_all_v4.py", "--dry-run", "--blocks", "C", "--from", "C23"])
    assert h.main() == 0
    out = capsys.readouterr().out
    assert "2 steps in blocks C" in out and "[C23]" in out and "[C22]" not in out
