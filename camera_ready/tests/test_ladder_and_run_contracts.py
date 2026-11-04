"""Check ladder completeness, run pairing, checkpoint fingerprints and timing reports.

These tests use supplied results and synthetic data without the feature cache.
"""
import argparse
import copy
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

CR = Path(__file__).resolve().parents[1]
ROOT = CR.parent
RES = CR / "output" / "results"
CD = RES / "mitigation" / "content_disjoint"
MECH = RES / "mitigation" / "mechanism"
SENS = RES / "sensitivity"
RUNS = CR / "scripts" / "runs"
AUTH = CR / "scripts" / "authoring"
SENSITIVITY = CR / "scripts" / "sensitivity"
GENS = ["whichisreal", "CollabDiff", "MidJourney", "heygen", "starganv2", "deepfacelab", "stargan", "styleclip"]
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
sys.path.insert(0, str(RUNS))
import ladder_checks as lc  # noqa: E402


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _run(*args):
    return subprocess.run([sys.executable, "-B", *map(str, args)], cwd=ROOT, env=ENV, capture_output=True, text=True)


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fingerprinted(run):
    out = copy.deepcopy(run)
    for g, f in lc.fold_records(out).items():
        f["fit_fingerprint"], f["select_fingerprint"] = f"fit-{g}", f"select-{g}"
    return out


def _capped_only(rungs):
    out = copy.deepcopy(rungs)
    for f in out["folds"].values():
        for k in [k for k in f["rungs"] if k.endswith("_xgb_widened")]:
            del f["rungs"][k]
    return out


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# LOGO statistics

def _stats():
    return _module(AUTH / "full_ladder_stats_v13.py", "logo_stats_under_test")


def _opts(**kw):
    return argparse.Namespace(**{"allow_partial_folds": False, "allow_ten_methods": False,
                                 "allow_incomplete_ladder": False, **kw})


def test_the_statistics_take_the_stored_runs_with_their_full_ladder(tmp_path):
    m = _stats()
    for name in ("11_disjoint_arch6_d256.json", "A3c_logo_extended_rungs_arch6.json"):
        shutil.copy2(CD / name, tmp_path / name)
    notes = []
    _, _, block = m.logo_inputs(tmp_path, "arch6", "256", _opts(), notes)
    assert block == {"ladder": "full", "folds_missing": []} and notes == []


def test_the_statistics_refuse_an_incomplete_ladder_unless_asked_and_then_record_it(tmp_path):
    m = _stats()
    _write(tmp_path / "11_disjoint_arch6_d256.json", _fingerprinted(load(CD / "11_disjoint_arch6_d256.json")))
    _write(tmp_path / "A3c_logo_extended_rungs_arch6.json",
           _fingerprinted(_capped_only(load(CD / "A3c_logo_extended_rungs_arch6.json"))))
    with pytest.raises(SystemExit, match="the prediction-space ladder is not complete: whichisreal: a capped-only"):
        m.logo_inputs(tmp_path, "arch6", "256", _opts(), [])
    notes = []
    _, _, block = m.logo_inputs(tmp_path, "arch6", "256", _opts(allow_incomplete_ladder=True), notes)
    assert block["ladder"] == "incomplete" and len(notes) == 8
    assert all(n.endswith("(--allow-incomplete-ladder)") for n in notes)
    assert m.arch6_basis({**block, "folds": GENS}).startswith("an incomplete ladder (--allow-incomplete-ladder")


def test_the_statistics_refuse_runs_that_do_not_belong_together_whatever_the_options(tmp_path):
    m = _stats()
    shutil.copy2(CD / "11_disjoint_arch6_d256.json", tmp_path)
    shutil.copy2(SENS / "dedup" / "A3c_logo_extended_rungs_arch6.json", tmp_path)
    every = _opts(allow_partial_folds=True, allow_ten_methods=True, allow_incomplete_ladder=True)
    with pytest.raises(SystemExit, match="fold whichisreal: n_fit 10995 in its richer-input run, 13535 in the LOGO run"):
        m.logo_inputs(tmp_path, "arch6", "256", every, [])
    run_dir, mech_dir = tmp_path / "runs", tmp_path / "mech"
    run_dir.mkdir()
    mech_dir.mkdir()
    for name in ("11_disjoint_arch6_d256.json", "11_disjoint_clip4_d32.json"):
        shutil.copy2(CD / name, run_dir)
    shutil.copy2(SENS / "dedup" / "run_c_v2_logo_arch6.json", mech_dir)
    shutil.copy2(MECH / "run_c_v2_logo_clip4.json", mech_dir)
    with pytest.raises(SystemExit, match="n_fit 10995 in the mechanism run, 13535 in the LOGO run"):
        m.mechanism_inputs(mech_dir, run_dir, every, [])


def test_the_audit_takes_an_incomplete_arch6_block_only_with_its_option(tmp_path):
    stats = load(RES / "mitigation" / "mitigation_stats.json")
    stats["logo_arch6"]["ladder"] = "incomplete"
    p = _write(tmp_path / "mitigation_stats.json", stats)
    common = ["--audit-only", "--stats", p, "--standard-stats", RES / "full_ladder_stats.json",
              "--reviewed-results", ROOT / "output" / "results"]
    r = _run(AUTH / "full_ladder_stats_v13.py", *common)
    assert r.returncode != 0
    assert "computed over 'incomplete', not the full ladder; --allow-incomplete-ladder accepts it" in r.stdout + r.stderr
    r = _run(AUTH / "full_ladder_stats_v13.py", *common, "--allow-incomplete-ladder")
    assert r.returncode == 0, r.stdout + r.stderr
    assert load(p)["audit"]["logo_arch6_basis"].startswith("an incomplete ladder (--allow-incomplete-ladder")


# Combined ceiling

def test_the_combined_ceiling_checks_its_two_runs(tmp_path):
    cc = RUNS / "09_logo_combined_ceiling_v3.py"
    lp, rp = CD / "11_disjoint_clip4_d32.json", CD / "A3c_logo_extended_rungs_clip4.json"
    r = _run(cc, "--logo-json", lp, "--rungs-json", rp, "--out", tmp_path / "cc.json")
    assert r.returncode == 0, r.stdout + r.stderr
    new, old = load(tmp_path / "cc.json"), load(CD / "A3c_logo_combined_ceiling_clip4.json")
    assert {k: v for k, v in new.items() if k not in ("version", "inputs")} == {
        k: v for k, v in old.items() if k not in ("version", "inputs")}
    r = _run(cc, "--logo-json", lp, "--rungs-json", SENS / "dedup" / "A3c_logo_extended_rungs_clip4.json",
             "--out", tmp_path / "mixed.json", "--allow-incomplete-ladder")
    assert r.returncode != 0 and "n_fit" in r.stdout + r.stderr and not (tmp_path / "mixed.json").exists()
    logo = _write(tmp_path / "logo.json", _fingerprinted(load(lp)))
    rungs = _write(tmp_path / "rungs.json", _fingerprinted(_capped_only(load(rp))))
    r = _run(cc, "--logo-json", logo, "--rungs-json", rungs, "--out", tmp_path / "capped.json")
    assert r.returncode != 0 and "capped-only" in r.stdout + r.stderr and not (tmp_path / "capped.json").exists()
    r = _run(cc, "--logo-json", logo, "--rungs-json", rungs, "--out", tmp_path / "capped.json",
             "--allow-incomplete-ladder")
    assert r.returncode == 0, r.stdout + r.stderr
    ic = load(tmp_path / "capped.json")["input_completeness"]
    assert ic["complete"] is False and len(ic["notes"]) == 8


# Sensitivity summary

def _sensitivity_copy(where):
    d = where / "sensitivity"
    shutil.copytree(SENS, d)
    for name in ("SUMMARY.md", "summary.json"):
        (d / name).unlink()
    return d


def test_the_summary_stops_on_a_missing_fold_and_with_its_option_leaves_that_part_out(tmp_path):
    d = _sensitivity_copy(tmp_path)
    (d / "paper_heads" / "clip4" / "clip4_heygen.json").unlink()
    r = _run(SENSITIVITY / "summarize_v2.py", "--dir", d, "--out-dir", d)
    assert r.returncode != 0 and "paper_heads clip4: " in r.stderr and "clip4_heygen.json not found" in r.stderr
    assert not (d / "summary.json").exists() and not (d / "SUMMARY.md").exists()
    r = _run(SENSITIVITY / "summarize_v2.py", "--dir", d, "--out-dir", d, "--allow-partial")
    assert r.returncode == 0, r.stderr
    s = load(d / "summary.json")
    assert s["input_completeness"]["complete"] is False and list(s["paper_heads"]) == ["arch6", "strong4", "weak6"]
    md = (d / "SUMMARY.md").read_text(encoding="utf-8")
    assert "**Not complete.**" in md and "### clip4\n\nnot complete (--allow-partial): " in md


def test_the_summary_refuses_runs_that_do_not_belong_together_even_with_its_option(tmp_path):
    d = _sensitivity_copy(tmp_path / "mixture")
    shutil.copy2(CD / "11_disjoint_arch6_d256.json", d / "dedup" / "11_disjoint_arch6_d256.json")
    r = _run(SENSITIVITY / "summarize_v2.py", "--dir", d, "--out-dir", d, "--allow-partial")
    assert r.returncode != 0 and not (d / "summary.json").exists()
    assert ("is not camera_ready/output/results/mitigation/content_disjoint/11_disjoint_arch6_d256.json with its "
            "training duplicates removed") in r.stderr
    d = _sensitivity_copy(tmp_path / "pool")
    shutil.copy2(d / "dedup" / "run_c_v2_logo_clip4.json", d / "dedup" / "run_c_v2_logo_arch6.json")
    r = _run(SENSITIVITY / "summarize_v2.py", "--dir", d, "--out-dir", d, "--allow-partial")
    assert r.returncode != 0 and "its mechanism run: pool 'clip4', the deduplicated LOGO run: 'arch6'" in r.stderr


# Table builder

def _current_builder():
    m = _module(AUTH / "generate_camera_ready_tables_v18.py", "table_builder_under_test")
    m.INPUTS = CR / "inputs"
    return m


def _patched(monkeypatch, m, key, value):
    original = m.load_json
    monkeypatch.setattr(m, "load_json", lambda k: value if k == key else original(k))


def test_the_builder_takes_the_stored_summary_and_runs():
    m = _current_builder()
    assert m.summary_problem(m.load_json("sens_summary")) is None
    assert m.runs_problem() is None


def test_the_builder_refuses_a_partial_summary_or_one_without_a_printed_part():
    m = _current_builder()
    s = m.load_json("sens_summary")
    partial = {**s, "input_completeness": {"complete": False, "notes": ["paper_heads clip4: a fold not found"]}}
    assert m.summary_problem(partial) == "it is marked as not complete (paper_heads clip4: a fold not found)"
    lacking = copy.deepcopy(s)
    del lacking["paper_heads"]["clip4"]
    lacking["audit"] = {}
    assert m.summary_problem(lacking).startswith("it lacks paper_heads.clip4, audit: overlap distortion")


def test_the_builder_refuses_runs_that_do_not_belong_together(monkeypatch):
    m = _current_builder()
    _patched(monkeypatch, m, "cd_rungs_arch6", load(SENS / "dedup" / "A3c_logo_extended_rungs_arch6.json"))
    assert "fold whichisreal: n_fit 10995 in its richer-input run, 13535 in the arch6 LOGO run" in m.runs_problem()


def test_the_builder_refuses_an_incomplete_ladder_or_combined_ceiling(monkeypatch):
    m = _current_builder()
    _patched(monkeypatch, m, "cd_rungs_arch6", _capped_only(load(CD / "A3c_logo_extended_rungs_arch6.json")))
    assert "the full ladder is not complete (whichisreal: a capped-only richer-input run" in m.runs_problem()
    m = _current_builder()
    cc = load(CR / "inputs" / "A3c_logo_combined_ceiling_arch6.json")
    cc["input_completeness"] = {"complete": False, "notes": ["a gap"]}
    _patched(monkeypatch, m, "logo_combined", cc)
    assert m.runs_problem().endswith("is marked as not complete (a gap)")


# ---------------------------------------------------------------- the runners: resume, dry run

class _Stop(Exception):
    """Raised where a runner would compute a fold."""


def _runner(name):
    return _module(RUNS / name, f"runner_under_test_{name[:-3]}")


def _sandbox(monkeypatch, mod):
    """Outputs go where the test says; the cache loader gives empty generators; computing a fold stops the run."""
    import logo_common

    def out_path(p):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        return Path(p)

    def stop(*a, **k):
        raise _Stop
    monkeypatch.setattr(mod._cr_paths, "out_path", out_path)
    monkeypatch.setattr(mod._cr_paths, "make_logger", lambda *a, **k: logging.getLogger("runner_contract_test"))
    monkeypatch.setattr(logo_common, "load_generators", lambda *a, **k: {g: (None, None, [0]) for g in GENS})
    monkeypatch.setattr(logo_common, "fold_view", stop)


RESUME = [("11_disjoint_data_v4.py", CD / "folds" / "arch6_d256_heygen.json", "folds/arch6_d256_heygen.json",
           ["--proj_dim", "256"]),
          ("08_logo_extended_rungs_v3.py", CD / "rung_folds" / "arch6_heygen.json", "rung_folds/arch6_heygen.json", []),
          ("05_run_c_width_matched_v5.py", MECH / "mechanism_folds" / "arch6_heygen.json",
           "mechanism_folds/arch6_heygen.json", ["--protocol", "logo"])]


@pytest.mark.parametrize("script,stored,ckpt,extra", RESUME, ids=[r[0] for r in RESUME])
def test_a_resume_reuses_only_checkpoints_with_membership_fingerprints(tmp_path, monkeypatch, script, stored, ckpt,
                                                                       extra):
    mod = _runner(script)
    _sandbox(monkeypatch, mod)
    out = tmp_path / "run"
    (out / ckpt).parent.mkdir(parents=True)
    shutil.copy2(stored, out / ckpt)                       # written by an earlier runner: no fingerprints
    argv = [script, "--codebase", str(ROOT), "--pool", "arch6", "--exclusion", "none", "--out-dir", str(out),
            "--folds", "heygen", "--resume", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="requires checkpoint validation"):
        mod.main()
    # The reuse branch is reachable only with the harness's validation context.
    monkeypatch.setenv("REPRO_VALIDATED_RESUME", "1")
    monkeypatch.setenv("REPRO_ATTEMPT_ID", "test-attempt")
    monkeypatch.setenv("REPRO_FINGERPRINT", "test-fingerprint")
    with pytest.raises(_Stop):                             # recomputed, not reused
        mod.main()
    rec = load(out / ckpt)
    rec["fit_fingerprint"], rec["select_fingerprint"] = "a", "b"
    _write(out / ckpt, rec)
    assert mod.main() in (0, None)                         # reused: no fold to compute (one of eight: no summary)


def test_a_dry_run_of_the_richer_input_runner_writes_nothing_even_in_a_stored_folder(tmp_path, monkeypatch):
    mod = _runner("08_logo_extended_rungs_v3.py")
    import logo_common

    def refuse(*a, **k):
        raise AssertionError("a dry run asked for an output path")
    y, yt = np.array([0, 1] * 20), np.array([0, 1] * 4)
    monkeypatch.setattr(mod._cr_paths, "out_path", refuse)
    monkeypatch.setattr(logo_common, "load_generators", lambda *a, **k: {})
    monkeypatch.setattr(logo_common, "fold_view", lambda data, held, excl=None: (
        [np.arange(80, dtype=np.float32).reshape(40, 2)], np.zeros((40, 2, 2)), y, None, np.zeros((8, 2, 2)), yt, 0))
    stored = tmp_path / "stored"
    (stored / "rung_folds").mkdir(parents=True)
    (stored / "rung_folds" / "arch6_heygen.json").write_text("{}", encoding="utf-8")
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    for out in (stored, tmp_path / "new"):
        monkeypatch.setattr(sys, "argv", ["08_logo_extended_rungs_v3.py", "--codebase", str(ROOT), "--pool", "arch6",
                                          "--exclusion", "none", "--out-dir", str(out), "--dry-run"])
        assert mod.main() == 0
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == before


# Sensitivity reruns

def test_the_deduplicated_reruns_run_the_runners_that_record_fingerprints(tmp_path, monkeypatch):
    mod = _module(SENSITIVITY / "dedup_logo_v2.py", "dedup_logo_under_test")
    seen = []
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda cmd, **k: (seen.append(cmd), types.SimpleNamespace(returncode=0))[1])
    mod.run_pool("arch6", str(tmp_path), None, lambda msg: None)
    assert [Path(c[c.index("--wrap") + 1]).name for c in seen] == ["11_disjoint_data_v4.py",
                                                                    "08_logo_extended_rungs_v3.py"]


def test_the_mechanism_rerun_needs_logo_runs_with_fingerprints(tmp_path, monkeypatch):
    mod = _module(SENSITIVITY / "dedup_followup_v2.py", "dedup_followup_under_test")
    assert mod.WRAPPER.name == "dedup_logo_v2.py"
    for name in ("11_disjoint_arch6_d256.json", "11_disjoint_clip4_d32.json"):
        shutil.copy2(SENS / "dedup" / name, tmp_path / name)
    monkeypatch.setattr(sys, "argv", ["dedup_followup_v2.py", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="records no membership fingerprints"):
        mod.main()


def test_the_standard_split_heads_record_and_require_the_fingerprints(tmp_path):
    mod = _module(SENSITIVITY / "paper_heads_logo_v2.py", "paper_heads_under_test")
    assert RUNS / "ladder_checks.py" in mod.CODE
    source = (SENSITIVITY / "paper_heads_logo_v2.py").read_text(encoding="utf-8")
    assert '"fit_fingerprint": membership_fingerprint(_ff)' in source
    assert '"select_fingerprint": membership_fingerprint(_sf)' in source
    rec = {"pool": "arch6", "held_out": "heygen", "provenance": {"fingerprint": "x", "scores_sha256": "y"}}
    (tmp_path / "arch6").mkdir()
    _write(tmp_path / "arch6" / "arch6_heygen.json", rec)
    assert mod.check_fold(tmp_path, "arch6", "heygen", "x") == (
        "record without the membership fingerprints of FIT and SELECT (written by version 1)")


# ---------------------------------------------------------------- the harness: versions and TIMINGS.md

def _harness():
    return _module(CR / "scripts" / "review" / "reproduce_all_v4.py", "harness_under_test")


def _rv():
    return types.SimpleNamespace(EXPERIMENTS=[("03", "python scripts/03_x.py", ["03_x.json"], True),
                                              ("05", "python scripts/05_x.py", ["05_x.json"], False)])


def _record(steps, start):
    return {"run_id": "t", "environment": {"cpu": "c", "ram_gib": 16, "torch_in_run_env": {"devices": []}},
            "sessions": [{}], "steps": steps, "run_start_epoch": start}


def _rec(s, status="OK", elapsed=600.0, reason=None):
    r = {"id": s.id, "block": s.block, "paper": s.paper, "gpu": s.gpu, "elapsed_s": elapsed, "status": status}
    return {**r, "reason": reason} if reason else r


def test_the_harness_runs_the_new_versions_and_fingerprints_the_mapping():
    h = _harness()
    b = {s.id: s for s in h.steps_b()}
    c = {s.id: s for s in h.steps_c()}
    runners = {s.cmd[1].rsplit("/", 1)[-1] for s in b.values()}
    assert {"11_disjoint_data_v4.py", "08_logo_extended_rungs_v3.py", "05_run_c_width_matched_v5.py",
            "09_logo_combined_ceiling_v3.py", "full_ladder_stats_v13.py"} <= runners
    assert not runners & {"11_disjoint_data_v3.py", "08_logo_extended_rungs_v2.py", "05_run_c_width_matched_v4.py",
                          "09_logo_combined_ceiling_v2.py", "full_ladder_stats_v10.py", "full_ladder_stats_v11.py",
                          "full_ladder_stats_v12.py"}
    assert c["C08a"].cmd[1].endswith("full_ladder_stats_v13.py")
    assert c["C24"].cmd[1].endswith("generate_camera_ready_tables_v18.py")
    for family, runner in (("folds", "11_disjoint_data_v4.py"), ("rung_folds", "08_logo_extended_rungs_v3.py"),
                           ("mechanism_folds", "05_run_c_width_matched_v5.py")):
        assert runner in h.FAMILIES[family][0]
    mapping = "camera_ready/provenance/trusted_run_pairs.json"
    for s in [b["B32"], c["C08a"]] + [s for s in b.values() if s.cmd[1].endswith("09_logo_combined_ceiling_v3.py")]:
        assert mapping in [str(p).replace("\\", "/") for p in h.step_reads(s)], s.id


def test_a_full_run_is_complete_with_the_skips_of_a_default_run(tmp_path):
    h = _harness()
    full = h.steps_a(_rv()) + [s for s in h.steps_b() if not s.optional] + h.steps_c()
    steps = []
    for s in full:
        if s.needs == "df40":
            steps.append(_rec(s, "SKIPPED", 0.0, "needs --df40-json"))
        elif s.extra.get("premise"):
            steps.append(_rec(s, "SKIPPED", 0.0, "gate G2 compares two runs of one environment"))
        else:
            steps.append(_rec(s))
    h.timing_report(_record(steps, time.time() + 1000), tmp_path / "T.md", _rv())
    text = (tmp_path / "T.md").read_text(encoding="utf-8")
    ok = sum(1 for r in steps if r["status"] == "OK")
    assert (f"Complete: every step of a full run of blocks A, B and C ({len(full)} steps) finished in this run, "
            f"except {len(steps) - ok} that a default full run also skips (below).") in text
    assert f"| {ok * 600 / 3600:.2f} h (every step of blocks A, B and C) |" in text
    assert "Skipped as in a default full run (their stored outputs are kept): " in text
    assert "(needs --df40-json)" in text and "Optional steps (the reproduction gate, --with-reproduction-gate): not run." in text
    assert ("| GPU portion (FS2 training and bootstraps) about 2 h (App. A, Q8) | 0.17 h: steps A02, the block A "
            "steps that train FS2 (whole steps, CPU parts included) |") in text
    # The sensitivity checks are in Appendix A.
    assert ("Additional commands outside this harness: the "
            "t-SNE view of Figure A1 (camera_ready/scripts/authoring/figure_a1_tsne.py) and the sensitivity checks of Appendix A") in text
    assert "| Rerunning the sensitivity checks of Appendix A about 15 h more (App. A, Q8) | not measured" in text


def test_a_partial_run_lists_what_it_did_not_recompute(tmp_path):
    h = _harness()
    c = h.steps_c()
    steps = [_rec(s) for s in c if not s.needs]
    steps[-1]["status"] = "FAILED"
    h.timing_report(_record(steps, time.time() + 1000), tmp_path / "T.md", _rv())
    text = (tmp_path / "T.md").read_text(encoding="utf-8")
    n_full = len(h.steps_a(_rv())) + len([s for s in h.steps_b() if not s.optional]) + len(c)
    ok = len(steps) - 1
    assert (f"Partial: {ok} of the {n_full} steps of a full run of blocks A, B and C finished in this run. The "
            "outputs of the others are the stored ones, kept and not recomputed:") in text
    assert "- not run (outside the blocks, --from or --only of this run): A01, A02, B01, " in text
    assert f"- not finished: {steps[-1]['id']} (FAILED)" in text
    assert f"not comparable: {ok} of the {n_full} steps finished in this run |" in text
    assert "| Block C about 8 h (camera_ready/REPRODUCTION.md, estimate) | not comparable: " in text
    assert "GPU portion" not in text and "| C |" in text and "| A |" not in text


def test_a_skipped_step_and_a_skip_marker_are_named(tmp_path, monkeypatch):
    h = _harness()
    monkeypatch.setattr(h, "ROOT", tmp_path)
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    rv = _rv()
    a = h.steps_a(rv)
    marker = tmp_path / a[0].outputs[0]
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"skipped": True, "reason": "memory budget"}), encoding="utf-8")
    b = [s for s in h.steps_b() if not s.optional]
    steps = [_rec(s) for s in a] + [_rec(b[0], "SKIPPED", 0.0, "a reason of this run")]
    h.timing_report(_record(steps, time.time() - 60), tmp_path / "T.md", rv)
    text = (tmp_path / "T.md").read_text(encoding="utf-8")
    assert f"- skipped: {b[0].id} (a reason of this run)" in text
    assert (f"Steps that ran but skipped their computation, as their result records: {a[0].id} "
            f"({a[0].outputs[0]}: memory budget)") in text
    assert "| Block A about 14 h (camera_ready/REPRODUCTION.md, estimate) | 0.33 h |" in text
    assert f"| {b[0].id} | B | {b[0].paper} | 0.0 min | SKIPPED (a reason of this run) |" in text
