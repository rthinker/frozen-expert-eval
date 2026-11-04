"""Check reference digests across line endings, required inputs on resume and standalone output guards."""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

CR = Path(__file__).resolve().parents[1]
AUTH = CR / "scripts" / "authoring"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V4 = _load(CR / "tests" / "test_reproduce_all_v4.py", "harness_tests_for_output_guards")     # its harness loader and sandbox


def test_a_byte_checked_file_with_other_line_endings_is_still_the_references_source(tmp_path):
    h = V4._harness()
    data = {"n": 4, "manifest": [{"fold": 0}, {"fold": 1}]}
    crlf = json.dumps(data, indent=2).replace("\n", "\r\n").encode("utf-8")
    edited = json.dumps(dict(data, manifest=[{"fold": 0}, {"fold": 2}]), indent=2).encode("utf-8")
    files = {"crlf.json": crlf, "lf.json": crlf.replace(b"\r\n", b"\n"), "edited.json": edited}
    entry = {"kind": "det", "tol": None, "env": None, "key_map": {}, "partial_ok": False, "step": "C",
             "metrics": {"n": 4}, "strings_ref": {}, "exclude": r"^manifest\[",
             "source_sha256": hashlib.sha256(crlf).hexdigest()}
    ref = {"results": {}}
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
        ref["results"][str(tmp_path / name)] = dict(entry)
    (tmp_path / "ref.json").write_text(json.dumps(ref), encoding="utf-8")
    lines = h.compare_v4(tmp_path / "ref.json", run=None)[4]
    assert V4._verdicts(lines) == {"crlf.json": "MATCH", "lf.json": "MATCH", "edited.json": "FAIL"}


def test_prediction_space_rerun_uses_current_output_schema(tmp_path, monkeypatch):
    h = V4._harness()
    rv = h.reviewed()
    source = rv.RESULTS
    ref = json.loads(rv.REFERENCE_PATH.read_text(encoding="utf-8"))
    prediction_files = [name for name in ref["results"]
                        if name.startswith("04_ds_") and not name.endswith("_rawfeat.json")]
    assert prediction_files
    ref["results"] = {name: ref["results"][name] for name in prediction_files}
    monkeypatch.setattr(rv, "RESULTS", tmp_path)
    monkeypatch.setattr(rv, "REFERENCE_PATH", tmp_path / "reference.json")
    rv.REFERENCE_PATH.write_text(json.dumps(ref), encoding="utf-8")

    for name in prediction_files:
        rerun = json.loads((source / name).read_text(encoding="utf-8"))
        for legacy in ("method_stds", "method_per_generator", "method_n_runs", "best_ds_mean"):
            rerun.pop(legacy, None)
        # The current writer builds its single-run aggregate from the run itself.
        ps_b = next(row for row in rerun["baselines"] if row["name"] == "PS_B_learned_weights")
        rerun["method_means"]["PS_B_learned_weights"]["per_generator"]["deepfacelab"] = (
            ps_b["per_generator"]["deepfacelab"]
        )
        (tmp_path / name).write_text(json.dumps(rerun), encoding="utf-8")

    assert rv.compare_results() == (len(prediction_files), 0, 0)
    assert h.guard_block_a(rv)[0] == 0

    result = tmp_path / "04_ds_arch6.json"
    rerun = json.loads(result.read_text(encoding="utf-8"))
    rerun["method_means"]["PS_3_mlp_logits"]["mean_avg8"] -= 0.001
    result.write_text(json.dumps(rerun), encoding="utf-8")
    assert rv.compare_results() == (len(prediction_files) - 1, 0, 1)


def test_a_resume_needs_the_df40_folders_the_run_was_started_with(tmp_path, monkeypatch, capsys):
    h = V4._harness()
    root = tmp_path / "repo"
    root.mkdir()
    V4.CASES._sandbox(h, monkeypatch, root)
    lists = tmp_path / "df40_lists"
    lists.mkdir()
    steps = [h.Step("B01", "B", "one", ("python", "-c", V4.CASES.WRITE, "out/a.txt"), ("out/a.txt",))]
    monkeypatch.setattr(h, "selected_steps", lambda args, rv: (["B"], list(steps)))
    args = argparse.Namespace(blocks="B", gpu_name=None, df40_json=str(lists), df40_raw=None,
                              with_reproduction_gate=False, from_step=None, only=None, run_id="t9", resume_run=None)
    assert h.mode_run(args, None) == 0
    assert "Partial run (selected steps only): completed" in capsys.readouterr().out
    lists.rename(tmp_path / "df40_unplugged")
    args.resume_run = "t9"
    with pytest.raises(SystemExit, match="Not resuming run t9: --df40-json .* not found"):
        h.mode_resume(args, None)
    log = json.loads((root / "camera_ready" / "output" / "logs" / "t9" / h.LOG_NAME).read_text(encoding="utf-8"))
    assert len(log["sessions"]) == 1 and not any(r.get("superseded") for r in log["steps"])
    assert log["scope"] == "partial"


@pytest.mark.parametrize("name", ["full_ladder_stats_v13.py", "full_ladder_stats_standard_v3.py",
                                  "regenerate_appendix_h_v3.py"])
def test_the_authoring_scripts_refuse_a_stored_path_by_hand(name, tmp_path, monkeypatch):
    mod = _load(AUTH / name, f"output_guard_{name[:-3]}")
    stored = CR / "output" / "results" / "mitigation" / "mitigation_stats.json"
    monkeypatch.delenv("REPRO_ATTEMPT_ID", raising=False)
    with pytest.raises(SystemExit, match="among the stored results"):
        mod.refuse_stored_write(stored)
    mod.refuse_stored_write(tmp_path / "elsewhere.json")              # another folder: allowed
    monkeypatch.setenv("REPRO_ATTEMPT_ID", "test")
    mod.refuse_stored_write(stored)                                   # under the harness: allowed


def test_the_statistics_write_nothing_when_refused(monkeypatch):
    mod = _load(AUTH / "full_ladder_stats_v13.py", "stats_write_guard")
    monkeypatch.delenv("REPRO_ATTEMPT_ID", raising=False)
    target = CR / "output" / "results" / "mitigation" / "output_guard_probe_never_written.json"
    with pytest.raises(SystemExit):
        mod.write_atomic(target, "{}")
    assert not target.exists() and not Path(str(target) + ".tmp").exists()


def test_the_appendix_h_table_refuses_its_default_output_before_any_work():
    env = {k: v for k, v in os.environ.items() if k != "REPRO_ATTEMPT_ID"}
    r = subprocess.run([sys.executable, "-B", str(AUTH / "regenerate_appendix_h_v3.py")], capture_output=True,
                       text=True, env=env, cwd=str(CR.parent))
    assert r.returncode != 0 and "among the stored results" in r.stdout + r.stderr



def test_stored_evidence_flags_a_changed_score_and_ledger_and_a_deleted_checkpoint(tmp_path, monkeypatch, capsys):
    h = V4._harness()
    monkeypatch.setattr(h, "ROOT", tmp_path)
    manifest = tmp_path / "stored_evidence.sha256"
    monkeypatch.setattr(h, "STORED_EVIDENCE", manifest)
    names = (
        "camera_ready/output/results/mitigation/scores/fold.npz",
        "camera_ready/output/results/mitigation/folds/fold.json",
        "camera_ready/provenance/05_number_provenance.csv",
    )
    payloads = (b"score archive", b'{"fold": 1}', b"quantity,value\nfs1,0.8\n")
    for name, content in zip(names, payloads):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest.write_text("".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n"
                                for name, data in zip(names, payloads)), encoding="utf-8")
    assert h.verify_stored_evidence() == 0

    (tmp_path / names[0]).write_bytes(bytes(len(payloads[0])))
    (tmp_path / names[1]).unlink()
    (tmp_path / names[2]).write_bytes(payloads[2].replace(b"0.8", b"0.7"))
    assert h.verify_stored_evidence() == 3
    output = capsys.readouterr().out
    assert all(f"{name} (" in output for name in names)


def _clean_verification(monkeypatch, h, evidence_failures):
    """A verification whose only possible failures are the stored-evidence ones."""
    rv = types.SimpleNamespace(verify_expected_outputs=lambda: (0, 0), EXPERIMENTS=[], REFERENCE_PATH="reference",
                               compare_results=lambda: (64, 0, 0))
    monkeypatch.setattr(h, "read_reference", lambda *a, **k: {})
    monkeypatch.setattr(h, "guard_block_a", lambda rv: (0, []))
    monkeypatch.setattr(h, "compare_v4", lambda run=None: (52, 0, 0, 0, []))
    monkeypatch.setattr(h, "verify_stored_evidence", lambda: evidence_failures)
    return rv


@pytest.mark.parametrize("failures,code", [(0, 0), (3, 1)])
def test_the_closing_summary_reports_stored_evidence_failures(monkeypatch, capsys, failures, code):
    h = V4._harness()
    record, rc = h.mode_verify_only(_clean_verification(monkeypatch, h, failures))
    closing = capsys.readouterr().out.strip().splitlines()[-1]
    assert closing.startswith("Verification: block A 64 MATCH")
    assert closing.endswith(f"52 MATCH, 0 WARN/ENV, 0 FAIL, 0 SKIP; stored evidence {failures} FAIL.")
    assert rc == code and record["stored_evidence_failures"] == failures


def test_the_closing_summary_of_a_rerun_has_no_stored_evidence_clause(monkeypatch, capsys):
    h = V4._harness()
    rv = _clean_verification(monkeypatch, h, None)

    def refuse():
        raise AssertionError("a rerun compares its own outputs, not the stored evidence")
    monkeypatch.setattr(h, "verify_stored_evidence", refuse)
    record, rc = h.mode_verify_only(rv, run={"start": 0})
    closing = capsys.readouterr().out.strip().splitlines()[-1]
    assert closing.endswith("52 MATCH, 0 WARN/ENV, 0 FAIL, 0 SKIP.")
    assert rc == 0 and record["stored_evidence_failures"] is None


def test_reviewed_logo_stepwise_values_do_not_hide_impossible_counts_or_metric_drift(tmp_path, monkeypatch):
    h = V4._harness()
    rv = h.reviewed()
    monkeypatch.setattr(rv, "RESULTS", tmp_path)
    monkeypatch.setattr(rv, "REFERENCE_PATH", tmp_path / "reference.json")
    name = "11_disjoint_arch6_d256.json"
    values = {"n_folds": 8, "fs_sign_test_p": 0.2, "fs_wins": 5, "avg_auc_8": 0.8}
    rv.REFERENCE_PATH.write_text(json.dumps({"results": {name: {"deterministic": False, "metrics": values}}}),
                                 encoding="utf-8")
    path = tmp_path / name
    changed = {**values, "fs_sign_test_p": 0.7, "fs_wins": 6}
    path.write_text(json.dumps(changed), encoding="utf-8")
    failures, messages = h.guard_block_a(rv)
    assert failures == 0 and sum("ENV:" in line for line in messages) == 2
    for invalid in ({"fs_wins": 9}, {"fs_sign_test_p": 1.2}, {"avg_auc_8": 0.5}):
        path.write_text(json.dumps({**changed, **invalid}), encoding="utf-8")
        assert h.guard_block_a(rv)[0] == 1


def test_legacy_logo_statistics_must_still_have_complete_folds_and_ladders():
    builder = _load(AUTH / "generate_camera_ready_tables_v18.py", "builder_legacy_completeness")
    path = CR / "output" / "results" / "mitigation" / "mitigation_stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    assert "input_completeness" not in stats
    assert builder.statistics_problem(stats) is None
    stats["logo_boundary"]["strong4"]["ladder"] = "ten"
    assert "expected full-ladder statistics" in builder.statistics_problem(stats)
    stats["logo_boundary"]["strong4"]["ladder"] = "full"
    stats["logo_boundary"]["strong4"]["folds"].pop()
    assert "expected all 8 held-out generators" in builder.statistics_problem(stats)



def test_reference_export_refuses_unverified_sources_and_existing_destination(tmp_path, monkeypatch):
    gen = _load(AUTH / "generate_reference_v3.py", "guarded_reference_export")
    h = V4._harness()
    name = "result.json"
    result, approved, out = (tmp_path / path for path in (name, "shipped.json", "export.json"))
    result.write_bytes(b'{"auc": 0.8}')
    spec = {"path": name, "kind": "det", "tol": None, "env": None, "strings": None,
            "key_map": None, "partial_ok": False, "step": "C", "exclude": None}
    approved.write_text(json.dumps({"results": {name: {
        "metrics": {"auc": 0.8}, "source_sha256": hashlib.sha256(result.read_bytes()).hexdigest()
    }}}), encoding="utf-8")
    monkeypatch.setattr(h, "RESULTS_SPEC", [spec])
    monkeypatch.setattr(gen, "ROOT", tmp_path)
    monkeypatch.setattr(gen, "OUT", approved)
    monkeypatch.setattr(gen, "load_harness", lambda: h)
    monkeypatch.setattr(sys, "argv", ["export", "--out", str(out)])

    result.write_bytes(b'{"auc": 0.7}')
    with pytest.raises(SystemExit, match="source digest differs"):
        gen.main()
    assert not out.exists()
    result.write_bytes(b'{"auc": 0.8}')
    assert gen.main() == 0
    assert json.loads(out.read_bytes())["results"][name]["metrics"] == {"auc": 0.8}
    with pytest.raises(SystemExit, match="not a new destination"):
        gen.main()
