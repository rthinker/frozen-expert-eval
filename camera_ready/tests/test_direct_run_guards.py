"""Check that standalone scripts protect the supplied results.

The harness backs up the results before setting REPRO_ATTEMPT_ID for its steps. Standalone analyses and runners use the shared path guards. The expert-side and Appendix H checks require an explicit output. Tests use a temporary copy of the analyses folder.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CR = Path(__file__).resolve().parents[1]
ROOT = CR.parent
ANALYSES = CR / "scripts" / "analyses"
RUNS = CR / "scripts" / "runs"
PREPARE = ANALYSES / "prepare_df40_lists_v2.py"


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ("REPRO_ATTEMPT_ID", "CR_RESULTS_DIR", "DF40_JSON_DIR")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra)
    return env


def _copy_analyses(tmp_path):
    """The analyses folder in a layout of its own, with this repository's reviewed code base (read only)."""
    dst = tmp_path / "camera_ready" / "scripts" / "analyses"
    shutil.copytree(ANALYSES, dst)
    return dst


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_public_lists(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    prep = _load(PREPARE, "prepare_under_test")
    for name in prep.LIST_FILES:
        (public / name).write_text('["deepfakes_detection_datasets/DF40_train/fs/frames/video/1.png"]',
                                   encoding="utf-8")
    return public, prep.LIST_FILES


def _prepare(public, out):
    return subprocess.run([sys.executable, "-B", str(PREPARE), "--public", str(public), "--out", str(out)],
                          capture_output=True, text=True, env=_env(), cwd=public.parent)


def test_an_analysis_run_by_hand_refuses_before_any_work(tmp_path):
    dst = _copy_analyses(tmp_path)
    for script in ("appF_strict_refit_claims_v2.py", "appA_content_reuse.py", "appB_richer_inputs.py",
                   "appI1_k3_freeze_v2.py"):
        r = subprocess.run([sys.executable, "-B", str(dst / script)], capture_output=True, text=True,
                           env=_env(CR_CODEBASE=str(ROOT)), cwd=str(tmp_path))
        assert r.returncode != 0 and "among the stored results" in r.stderr, (script, r.stderr[-400:])
    assert not (tmp_path / "camera_ready" / "output").exists()


def test_an_analysis_without_options_prints_its_description_for_help(tmp_path):
    dst = _copy_analyses(tmp_path)
    for script in ("appF_strict_refit_claims_v2.py", "appA_bit_identity.py", "appI1_k3_freeze_v2.py",
                   "appI1_native_k_preflight.py"):
        r = subprocess.run([sys.executable, "-B", str(dst / script), "--help"], capture_output=True, text=True,
                           env=_env(CR_CODEBASE=str(ROOT)), cwd=str(tmp_path))
        assert r.returncode == 0 and len(r.stdout.strip()) > 40, (script, r.stdout[-300:], r.stderr[-300:])
    assert not (tmp_path / "camera_ready" / "output").exists()


def test_the_harness_and_a_folder_of_ones_own_pass_the_guard(tmp_path):
    dst = _copy_analyses(tmp_path)
    (dst / "probe.py").write_text('"""Probe."""\nimport _paths\nprint("RESULTS", _paths.RESULTS_DIR)\n',
                                  encoding="utf-8")

    def run(**extra):
        return subprocess.run([sys.executable, "-B", str(dst / "probe.py")], capture_output=True, text=True,
                              env=_env(CR_CODEBASE=str(ROOT), **extra), cwd=str(tmp_path))
    assert "among the stored results" in run().stderr
    assert run(REPRO_ATTEMPT_ID="t").stdout.startswith("RESULTS")
    own = tmp_path / "mine"
    r = run(CR_RESULTS_DIR=str(own))
    assert r.returncode == 0 and own.is_dir(), r.stderr[-400:]


def test_df40_lists_must_be_the_prepared_ones(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ANALYSES))
    m = _load(ANALYSES / "_paths.py", "paths_under_test")
    monkeypatch.delenv("DF40_JSON_DIR", raising=False)
    public, _ = _make_public_lists(tmp_path)
    prepared = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", ["x", "--df40-json", str(public)])
    with pytest.raises(SystemExit, match="prepare_df40_lists_v2.py"):
        m.df40_json_dir()
    r = _prepare(public, prepared)
    assert r.returncode == 0, r.stderr
    monkeypatch.setattr(sys, "argv", ["x", "--df40-json", str(prepared)])
    assert m.df40_json_dir() == prepared.resolve()
    (prepared / "DF40_all.json").write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit, match="SHA-256 mismatch for DF40_all.json"):
        m.df40_json_dir()


def test_preparing_df40_lists_preserves_source_and_last_valid_output_on_bad_input(tmp_path):
    public, names = _make_public_lists(tmp_path)
    initial = {p.name: p.read_bytes() for p in public.iterdir()}
    same = _prepare(public, public / ".." / public.name)
    assert same.returncode != 0 and "--out must be separate" in same.stderr
    assert {p.name: p.read_bytes() for p in public.iterdir()} == initial

    prepared = tmp_path / "prepared"
    first = _prepare(public, prepared)
    assert first.returncode == 0, first.stderr
    initial = {p.name: p.read_bytes() for p in prepared.iterdir()}
    (public / names[0]).write_text("[]", encoding="utf-8")
    (public / names[-1]).write_text("{", encoding="utf-8")
    failed = _prepare(public, prepared)
    assert failed.returncode != 0 and f"Invalid DF40 list {public / names[-1]}" in failed.stderr
    assert {p.name: p.read_bytes() for p in prepared.iterdir()} == initial


def test_a_runner_or_gate_started_by_hand_refuses_a_stored_output_path(tmp_path, monkeypatch):
    m = _load(RUNS / "_cr_paths.py", "cr_paths_under_test")
    monkeypatch.delenv("REPRO_ATTEMPT_ID", raising=False)
    refused = [CR / "output" / "results" / "mitigation" / "mechanism" / "gate_G5_arch6.json",
               CR / "output" / "results" / "mitigation" / "content_disjoint" / "logs" / "x",
               CR / "output" / "results" / "split_reuse_check.json",
               ROOT / "output" / "results" / "04_ds_arch6.json"]
    if (CR / "output" / "results" / "MITIGATION").exists():           # a case-insensitive file system
        refused.append(CR / "output" / "results" / "MITIGATION" / "x.json")
    for p in refused:
        with pytest.raises(SystemExit, match="among the stored results"):
            m.out_path(p)                                            # raises before creating anything
    assert m.stored_problem(CR / "output" / "results" / "sensitivity_rerun" / "dedup" / "x.json") is None
    assert m.stored_problem(tmp_path / "x.json") is None
    monkeypatch.setenv("REPRO_ATTEMPT_ID", "t")
    assert m.stored_problem(CR / "output" / "results" / "mitigation" / "x.json") is None


def test_the_expert_side_check_needs_prepared_lists_and_an_out_by_hand(tmp_path):
    script = ANALYSES / "appA_expert_side_v3.py"
    raw = tmp_path / "raw"
    raw.mkdir()
    r = subprocess.run([sys.executable, "-B", str(script), "--json_dir", str(raw)], capture_output=True, text=True,
                       env=_env(), cwd=str(tmp_path))
    assert r.returncode != 0 and "PREPARED.json" in r.stderr, r.stderr[-400:]
    public, _ = _make_public_lists(tmp_path)
    prepared = _prepare(public, raw)
    assert prepared.returncode == 0, prepared.stderr
    r = subprocess.run([sys.executable, "-B", str(script), "--json_dir", str(raw)], capture_output=True, text=True,
                       env=_env(), cwd=str(tmp_path))
    assert r.returncode != 0 and "writes only where --out says" in r.stderr, r.stderr[-400:]


def test_the_appendix_h_diff_parses_its_arguments_and_writes_only_where_told(tmp_path):
    script = ANALYSES / "appH_logit_convention_diff_v2.py"
    stored = CR / "output" / "results" / "analyses" / "A2_appendix_h_diff.json"
    before = stored.read_bytes()
    r = subprocess.run([sys.executable, "-B", str(script), "--help"], capture_output=True, text=True, env=_env())
    assert r.returncode == 0 and "usage:" in r.stdout
    r = subprocess.run([sys.executable, "-B", str(script)], capture_output=True, text=True, env=_env())
    assert r.returncode != 0 and "writes only where --out says" in r.stderr
    out = tmp_path / "A2.json"
    r = subprocess.run([sys.executable, "-B", str(script), "--out", str(out)], capture_output=True, text=True,
                       env=_env())
    assert r.returncode == 0, r.stderr[-400:]
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(before.decode("utf-8"))
    assert stored.read_bytes() == before
