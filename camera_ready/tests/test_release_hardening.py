"""Release IO and harness regression tests. No models are fitted."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CR = ROOT / "camera_ready"


def module(rel, name=None):
    spec = importlib.util.spec_from_file_location(name or Path(rel).stem, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def harness():
    return module("camera_ready/scripts/review/reproduce_all_v4.py")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("trusted", [False, True])
def test_object_identifiers_never_enable_pickle(tmp_path, monkeypatch, trusted):
    ids = module("src/utils/cache_ids.py")
    cache = tmp_path / "FIT/sample_ids.npy"
    cache.parent.mkdir()
    np.save(cache, np.asarray(["row-a", "row-b"], dtype=object))
    anchor = tmp_path / "code_ids"
    anchor.mkdir()
    (anchor / "FIT.txt").write_text("row-a\nrow-b\n", encoding="utf-8")
    dump(anchor / "MANIFEST.json", {"splits": {"FIT": {"rows": 2,
        "source_sha256": digest(cache) if trusted else "0"*64,
        "txt_sha256": digest(anchor / "FIT.txt")}}})
    monkeypatch.setattr(ids, "IDS_ROOT", anchor)
    real_load = np.load
    calls = []
    def checked_load(*args, **kwargs):
        calls.append(kwargs.get("allow_pickle"))
        assert kwargs.get("allow_pickle") is False
        return real_load(*args, **kwargs)
    monkeypatch.setattr(ids.np, "load", checked_load)
    if trusted:
        assert ids.load_sample_ids(cache).tolist() == ["row-a", "row-b"]
    else:
        with pytest.raises(ValueError, match="Unverified object"):
            ids.load_sample_ids(cache)
    assert calls == [False]


def test_unicode_identifiers_are_supported(tmp_path):
    ids = module("src/utils/cache_ids.py")
    path = tmp_path / "ids.npy"
    np.save(path, np.asarray(["custom-a", "custom-b"], dtype=str))
    assert ids.load_sample_ids(path).tolist() == ["custom-a", "custom-b"]


def test_local_cache_cannot_replace_its_trust_anchor(tmp_path):
    validator = module("scripts/review/validate_cache_release.py")
    shutil.copy2(CR / "dataset/CACHE_PROVENANCE.json", tmp_path)
    doc = json.loads((tmp_path / "CACHE_PROVENANCE.json").read_text())
    doc["data_manifest"]["sha256"] = "a" * 64
    dump(tmp_path / "CACHE_PROVENANCE.json", doc)
    with pytest.raises(validator.CheckFailure, match="trusted code release"):
        validator.check_provenance(tmp_path)


def test_historical_table_trio_stays_pinned(tmp_path, monkeypatch):
    h = harness()
    for key, value in {"ROOT": tmp_path, "INPUTS": tmp_path/"inputs", "RUN_INPUTS": tmp_path/"staged",
                       "BACKUP2": tmp_path/"backup", "ANA": tmp_path/"ana", "TK": tmp_path/"tk",
                       "RESULTS2": tmp_path/"results"}.items():
        monkeypatch.setattr(h, key, value)
    for name in h.PINNED_TABLE_INPUTS | {"other.json"}:
        dump(h.INPUTS/name, {"raw_feat_auc": 0.8})
        dump(h.RESULTS2/name, {"raw_feat_auc": 0.800001})
    record = {}
    h.stage_run_inputs(time.time()-1, record)
    for name in h.PINNED_TABLE_INPUTS:
        assert (h.RUN_INPUTS/name).read_bytes() == (h.INPUTS/name).read_bytes()
    assert json.loads((h.RUN_INPUTS/"other.json").read_text())["raw_feat_auc"] == 0.800001
    assert len(record["table_inputs_pinned_historical"]) == 3


@pytest.mark.parametrize("during", [False, True])
def test_resume_refuses_interrupted_backup_setup(tmp_path, monkeypatch, during):
    h = harness()
    step = h.Step("B01", "B", "test", ("python", "unused.py"), ("result.json",))
    monkeypatch.setattr(h, "ROOT", tmp_path)
    monkeypatch.setattr(h, "LOGS2", tmp_path/"logs")
    monkeypatch.setattr(h, "selected_steps", lambda *args: (["B"], [step]))
    monkeypatch.setattr(h, "environment_record", lambda *args: {})
    monkeypatch.setattr(h, "write_runbook", lambda *args: None)
    executed = []
    monkeypatch.setattr(h, "execute_plan", lambda *args, **kwargs: executed.append(True))
    def interrupted(*args, **kwargs):
        if during:
            (tmp_path/"partial_backup").mkdir()
            (tmp_path/"partial_backup/one.txt").write_text("copied one file")
        raise InterruptedError("interrupted backup")
    monkeypatch.setattr(h, "backup_v2", interrupted)
    args = argparse.Namespace(blocks="B", gpu_name=None, df40_json=None, df40_raw=None,
                              with_reproduction_gate=False, from_step=None, only=None, run_id="interrupted",
                              resume_run=None)
    with pytest.raises(InterruptedError):
        h.mode_run(args, None)
    record = json.loads((h.LOGS2/"interrupted"/h.LOG_NAME).read_text())
    assert record["setup"]["status"] == "preparing"
    args.resume_run = "interrupted"
    with pytest.raises(SystemExit, match="run setup did not commit"):
        h.mode_resume(args, None)
    assert not executed


def test_block_a_outputs_and_prerequisites_cover_every_seed(tmp_path, monkeypatch):
    h = harness()
    plan = h.steps_a(h.reviewed())
    pca = next(s for s in plan if "--combiner_type pca" in " ".join(s.cmd))
    assert not any(p.endswith(".pt") for p in pca.outputs)
    producers = [s for s in plan if any(p.endswith(".pt") for p in s.outputs)]
    assert producers
    for step in producers:
        assert len([p for p in step.outputs if p.endswith(".pt")]) == 8
    readers = [s for s in plan if "07_bootstrap_mlp.py" in " ".join(s.cmd)]
    for step in readers:
        required = step.extra["required_inputs"]
        assert len([p for p in required if p.endswith(".pt")]) == 8
        assert any(p.endswith("_scores.npz") for p in required)
        assert set(required) <= set(p for s in plan for p in s.outputs)
    monkeypatch.setattr(h, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="requires"):
        h.check_plan_prerequisites([readers[0]])


def test_safe_checkpoint_validation_rejects_damage(tmp_path):
    import torch
    h = harness()
    path = tmp_path/"model.pt"
    torch.save({"model_state_dict": {"weight": torch.zeros(2, 2)}, "config": {"seed": 7}}, path)
    assert h._damaged(path) is None
    path.write_bytes(b"partial checkpoint")
    assert h._damaged(path)


@pytest.mark.parametrize("name", ["table.tex", "table.csv", "figure.png", "figure.pdf", "notes.md", "result.json",
                                  "scores.npz", "model.pt"])
def test_a_zero_byte_output_counts_as_damaged(tmp_path, name):
    path = tmp_path/name
    path.write_bytes(b"")
    assert harness()._damaged(path) == "empty file"


@pytest.mark.parametrize("name", ["table.tex", "table.csv", "figure.png", "notes.md"])
def test_an_output_with_content_and_no_format_check_passes(tmp_path, name):
    path = tmp_path/name
    path.write_bytes(b"content")
    assert harness()._damaged(path) is None


@pytest.mark.parametrize("stamps", [{}, {"REPRO_ATTEMPT_ID": "x"},
    {"REPRO_ATTEMPT_ID": "x", "REPRO_FINGERPRINT": "x"}])
def test_resume_requires_harness_validation(monkeypatch, stamps):
    paths = module("camera_ready/scripts/runs/_cr_paths.py")
    for key in ("REPRO_ATTEMPT_ID", "REPRO_FINGERPRINT", "REPRO_VALIDATED_RESUME"):
        monkeypatch.delenv(key, raising=False)
    for key, value in stamps.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(SystemExit, match="requires checkpoint validation"):
        paths.require_validated_resume(True)
    paths.require_validated_resume(False)
    monkeypatch.setenv("REPRO_ATTEMPT_ID", "a")
    monkeypatch.setenv("REPRO_FINGERPRINT", "f")
    monkeypatch.setenv("REPRO_VALIDATED_RESUME", "1")
    paths.require_validated_resume(True)


@pytest.mark.parametrize("fault", ["empty", "fold", "method", "nan", "inf", "duplicate"])
def test_gates_reject_incomplete_or_nonfinite_inputs(fault):
    gate = module("camera_ready/scripts/runs/compare_runs_v2.py")
    ref = {g: dict.fromkeys(gate.ALL_METHODS, 0.5) for g in gate.GENS}
    new = copy.deepcopy(ref)
    if fault == "empty":
        new.clear()
    elif fault == "fold":
        new.pop(gate.GENS[-1])
    elif fault == "method":
        new[gate.GENS[-1]].pop("PS_4_xgboost")
    elif fault in ("nan", "inf"):
        new[gate.GENS[-1]]["PS_4_xgboost"] = float(fault)
    else:
        fold = {"held_out": "whichisreal", "ps_means": {}, "raw_feat_auc": 0.5}
        with pytest.raises(ValueError, match="duplicate"):
            gate.logo_values({"per_fold": [fold, fold]})
        return
    with pytest.raises(ValueError):
        gate.diff_table(new, ref)


def test_nan_cannot_evade_maximum_reduction():
    gate = module("camera_ready/scripts/runs/compare_runs_v2.py")
    row = dict(fold="a", method="PS_4_xgboost", group="xgboost", new=0.5, ref=0.5, abs_diff=0.)
    bad = {**row, "fold": "b", "new": float("nan"), "abs_diff": float("nan")}
    with pytest.raises(ValueError, match="nonfinite"):
        gate.summarise([row, bad], {"xgboost": 1e-3})


@pytest.mark.parametrize("fault", ["counts", "missing", "nan"])
def test_rewritten_gate_cannot_pass_without_evidence(tmp_path, fault):
    h = harness()
    gate = module("camera_ready/scripts/runs/compare_runs_v2.py")
    data = {"gate": "g5", "status": "PASS", "g5": {"max_abs_diff": 0.0},
            "checked": {"folds": list(gate.GENS), "mechanism": 16}}
    if fault == "counts": data.pop("checked")
    if fault == "missing": data.pop("g5")
    if fault == "nan": data["g5"]["max_abs_diff"] = float("nan")
    path = tmp_path/"gate.json"
    dump(path, data)
    ref = tmp_path/"reference.json"
    dump(ref, {"results": {str(path): {"kind": "gate", "step": "B", "metrics": {"g5.max_abs_diff": 0.0},
        "key_map": {}, "strings": "status$", "strings_ref": {"status": "PASS"}}}})
    assert h.compare_v4(ref, run={"start": time.time()-2})[2] == 1


def test_packaged_exclusion_location_exists():
    paths = module("camera_ready/scripts/runs/_cr_paths.py")
    assert paths.EXCLUSION_ROOT == CR/"output/results/mitigation/exclusions"
    assert (paths.EXCLUSION_ROOT/"logo_exclusions_arch6.json").is_file()


def test_atomic_report_preserves_destination_on_replace_failure(tmp_path, monkeypatch):
    paths = module("camera_ready/scripts/runs/_cr_paths.py")
    report = tmp_path/"report.json"
    report.write_text("original")
    def fail(*args): raise OSError("interrupted replace")
    monkeypatch.setattr(paths.os, "replace", fail)
    with pytest.raises(OSError): paths.atomic_write_text(report, "new")
    assert report.read_text() == "original"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]


@pytest.mark.parametrize("fault", ["missing", "labels", "length", "nan", "method"])
def test_sensitivity_scores_are_required_and_aligned(tmp_path, fault):
    summary = module("camera_ready/scripts/sensitivity/summarize_v2.py")
    y = np.asarray([0, 1, 0, 1])
    base_keys = list(summary.ladder_checks.BASE_METHODS)+["FS_1_raw_feat_logreg", "FS_2_mlp_seed_avg"]
    rich_keys = list(summary.ladder_checks.RICHER_CANDIDATES)
    b = {k: np.asarray([0.1, 0.9, 0.2, 0.8]) for k in base_keys}
    r = {k: np.asarray([0.1, 0.9, 0.2, 0.8]) for k in rich_keys}
    rec = {"held_out": summary.GENS[0], "n_test": 4, "scores_file": "ladder.npz", "fs_mlp_mean": 1.0}
    rr = {"scores_file": "rungs.npz", "rungs": dict.fromkeys(rich_keys, 1.0)}
    np.savez(tmp_path/"ladder.npz", labels=y, **b)
    if fault == "labels": y = y[::-1]
    if fault == "length": r[rich_keys[0]] = np.zeros(3)
    if fault == "nan": r[rich_keys[0]][0] = np.nan
    if fault == "method": r.pop(rich_keys[0])
    if fault != "missing": np.savez(tmp_path/"rungs.npz", labels=y, **r)
    assert summary.score_gaps(tmp_path, {"per_fold": [rec]}, {"folds": {summary.GENS[0]: rr}})


def test_explicit_empty_or_unrunnable_plan_is_rejected():
    h = harness()
    for selection in ("", " ", ",", "C04"):
        args = argparse.Namespace(blocks="A,B,C", gpu_name=None, df40_json=None, df40_raw=None,
                                  with_reproduction_gate=False, from_step=None, only=selection, run_id="unused")
        with pytest.raises(SystemExit, match="at least one step|no runnable"):
            h.mode_run(args, h.reviewed())


@pytest.mark.parametrize("mode, option, value", [
    ("--verify-only", "--only", "C04"),
    ("--verify-only", "--blocks", "C"),
    ("--check", "--only", ""),
    ("--check", "--gpu-name", "5080"),
])
def test_verification_modes_reject_run_options(mode, option, value, monkeypatch, capsys):
    h = harness()
    monkeypatch.setattr(sys, "argv", ["reproduce_all_v4.py", mode, option, value])
    monkeypatch.setattr(h, "reviewed", lambda: pytest.fail("an invalid command should not dispatch"))
    with pytest.raises(SystemExit) as exc:
        h.main()
    assert exc.value.code == 2
    assert f"{mode} does not accept run options: {option}" in capsys.readouterr().err


@pytest.mark.parametrize("extension, fault", [
    ("pt", "missing"), ("pt", "corrupt"), ("pt", "changed"), ("pdf", "changed"), ("png", "changed"),
])
def test_resume_rebuilds_artifact_producer_and_consumer(tmp_path, monkeypatch, extension, fault):
    h = harness()
    helpers = module("camera_ready/tests/harness_test_cases.py", "resume_test_helpers")
    helpers._sandbox(h, monkeypatch, tmp_path)
    artifact = f"artifact.{extension}"
    producer = h.Step("A01", "A", "producer", ("python", "producer.py"), ("train.json", artifact))
    consumer = h.Step("A02", "A", "consumer", ("python", "consumer.py"), ("bootstrap.json",),
                      extra={"required_inputs": ("train.json", artifact)})
    monkeypatch.setattr(h, "selected_steps", lambda *args: (["A"], [producer, consumer]))
    executed = []

    def write_artifact(path, value):
        if extension == "pt":
            import torch
            torch.save({"model_state_dict": {"weight": torch.full((2,), value)}}, path)
        else:
            path.write_bytes(bytes([value]))

    def run_step(step, argv, env, run_dir):
        executed.append(step.id)
        for rel in step.outputs:
            path = tmp_path / rel
            if rel == artifact:
                write_artifact(path, 1)
            else:
                dump(path, {"valid": True})
        return {"id": step.id, "block": "A", "paper": "fixture", "status": "OK", "elapsed_s": 0,
                "output_sha256": {p: h._file_hash(tmp_path/p) for p in step.outputs}}
    monkeypatch.setattr(h, "run_step", run_step)
    monkeypatch.setattr(h, "write_runbook", lambda *args: None)
    monkeypatch.setattr(h, "timing_report", lambda *args: None)
    rv = argparse.Namespace(backup=lambda: tmp_path/"backup")
    args = argparse.Namespace(blocks="A", gpu_name=None, df40_json=None, df40_raw=None,
            with_reproduction_gate=False, from_step=None, only="A01,A02", run_id="checkpoint", resume_run=None)
    assert h.mode_run(args, rv) == 0
    path = tmp_path / artifact
    if fault == "missing":
        path.unlink()
    elif fault == "corrupt":
        path.write_bytes(b"cut off")
    else:
        original_stat = path.stat()
        original_digest = digest(path)
        write_artifact(path, 0)
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        replacement_stat = path.stat()
        assert replacement_stat.st_size == original_stat.st_size
        assert replacement_stat.st_mtime_ns == original_stat.st_mtime_ns
        assert digest(path) != original_digest
    executed.clear()
    args.resume_run = "checkpoint"
    assert h.mode_resume(args, rv) == 0
    assert executed == ["A01", "A02"]


def test_explicit_request_with_only_skips_cannot_finish_successfully(tmp_path, monkeypatch):
    h = harness()
    monkeypatch.setattr(h, "ROOT", tmp_path)
    monkeypatch.setattr(h, "CR", tmp_path)
    monkeypatch.setattr(h, "RESULTS2", tmp_path/"results")
    monkeypatch.setattr(h, "mode_verify_only", lambda *args, **kwargs: ({}, 0))
    monkeypatch.setattr(h, "timing_report", lambda *args: None)
    record = {"run_id": "skip", "run_start_epoch": time.time(), "arguments": {"only": "C04"},
              "steps": [{"id": "C04", "status": "SKIPPED"}]}
    assert h.finish_run(record, tmp_path, None) == 1



def test_split_reuse_failed_expectation_does_not_replace_report(tmp_path, monkeypatch):
    mod = module("camera_ready/scripts/review/check_split_reuse_v3.py")
    stored = json.loads((CR/"output/results/split_reuse_check.json").read_text())
    monkeypatch.setattr(mod, "route_b", lambda *args: stored["route_b"])
    target = tmp_path/"report.json"
    target.write_text("previous validated report")
    wrong = copy.deepcopy(stored)
    wrong["route_b"]["content_reuse"]["totals"]["n"] += 1
    expected = tmp_path/"expected.json"
    dump(expected, wrong)
    monkeypatch.setattr(sys, "argv", ["check", "--cache", str(tmp_path/"unused"),
                        "--expect", str(expected), "--out", str(target)])
    assert mod.main() == 1
    assert target.read_text() == "previous validated report"



def test_new_split_ids_are_safe_and_keep_row_identity(tmp_path):
    import logging
    mod = module("scripts/02_resplit_fit_select.py")
    ids = np.asarray(["first", "second", "third"], dtype=object)
    labels = np.asarray([0, 1, 0], dtype=np.int32)
    rows = np.asarray([2, 1])
    output = tmp_path/"new_split"
    mod._slice_cache(tmp_path, output, [], rows, labels, ids, {}, logging.getLogger("test"))
    written = np.load(output/"sample_ids.npy", allow_pickle=False)
    assert written.tolist() == ["third", "second"]
    assert (output/"fingerprint.txt").read_text() == mod._fingerprint(labels[rows], ids[rows])
    safe = module("src/utils/cache_ids.py")
    assert safe.load_sample_ids(output/"sample_ids.npy").tolist() == written.tolist()
