"""Check fold construction, twin counts, combined ceilings, posted-number classification, input precedence,
resume device requirements and regenerated standard-split statistics from stored inputs.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

CR = Path(__file__).resolve().parents[1]
ROOT = CR.parent
RUNS, AUTH = CR / "scripts" / "runs", CR / "scripts" / "authoring"
MIT = CR / "output" / "results" / "mitigation"
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
ENV.pop("REPRO_ATTEMPT_ID", None)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _harness():
    return _load(CR / "scripts" / "review" / "reproduce_all_v4.py", "harness_gaps")


def test_the_camera_ready_folds_equal_the_reviewed_ones():
    """logo_common.fold_view and split_fit_select (no exclusion) give the training data and the FIT/SELECT split of the
    reviewed scripts/11_disjoint_data.py (merge_generators, _split_fit_select) for every held-out generator."""
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(RUNS)]
    import logo_common as lc
    from src.utils.dataset_helpers import TEST_GENERATORS
    reviewed = _load(ROOT / "scripts" / "11_disjoint_data.py", "reviewed_11")
    rng = np.random.default_rng(0)
    data = {}
    for g in TEST_GENERATORS:
        n = int(rng.integers(30, 60))
        data[g] = ([rng.normal(size=(n, 3)) for _ in range(2)], [rng.normal(size=(n, 2)) for _ in range(2)],
                   np.array([0, 1] * (n // 2) + [1] * (n % 2)))
    for held in TEST_GENERATORS:
        tr_f, tr_lg, tr_y, te_f, te_lg, te_y, n_excl = lc.fold_view(data, held)
        a = lc.split_fit_select(tr_f, tr_lg, tr_y)
        mf, ml, my = reviewed.merge_generators([data[g] for g in TEST_GENERATORS if g != held])
        b = reviewed._split_fit_select(mf, reviewed._stack_logits(ml), my, 0.15, 1024)
        assert n_excl == 0
        for x, y in zip(a, b):
            if isinstance(x, list):
                assert all(np.array_equal(u, v) for u, v in zip(x, y))
            else:
                assert np.array_equal(x, y)


def test_fold_view_excludes_training_rows_in_all_arrays(monkeypatch):
    from src.utils import dataset_helpers

    monkeypatch.setattr(dataset_helpers, "TEST_GENERATORS", ("g1", "g2", "g3"))
    logo = _load(RUNS / "logo_common.py", "fold_exclusions")

    def example(ids):
        ids = np.asarray(ids)
        features = [ids[:, None], (ids + 100)[:, None]]
        logits = [np.column_stack((x[:, 0], -x[:, 0])) for x in features]
        return features, logits, ids % 2

    data = {"g1": example([10, 11, 12]), "g2": example([20, 21, 22]), "g3": example([30, 31, 32])}
    train_f, train_lg, train_y, test_f, test_lg, test_y, count = logo.fold_view(
        data, "g2", {"g1": [1], "g3": [0, 2]}
    )
    assert count == 3
    for got, expected in (((train_f, train_lg, train_y), example([10, 12, 31])),
                          ((test_f, test_lg, test_y), example([20, 21, 22]))):
        features, logits, labels = expected
        np.testing.assert_array_equal(got[0], features)
        np.testing.assert_array_equal(got[1], np.stack(logits, axis=1))
        np.testing.assert_array_equal(got[2], labels)


def test_the_twin_counts_of_the_sensitivity_check():
    tw = _load(CR / "scripts" / "sensitivity" / "count_twins.py", "count_twins_gaps")
    fit = [("a", 0, "h1", 0), ("a", 1, "h1", 0), ("b", 2, "h2", 1), ("b", 3, "h3", 0)]
    sel = [("a", 4, "h1", 1), ("b", 5, "h4", 0), ("b", 6, "h2", 1), ("b", 7, "h2", 1)]
    c = tw.twin_counts(fit, sel)
    assert c == {"fit_rows": 4, "select_rows": 4, "select_rows_with_fit_twin": 3, "share": 0.75,
                 "twins_by_generator": {"a": 1, "b": 2}, "twins_with_other_label": 1,
                 "extra_copies_inside_fit": 1, "extra_copies_inside_select": 1}


def test_the_combined_ceiling_on_the_stored_runs_and_with_ties(tmp_path):
    script = RUNS / "09_logo_combined_ceiling_v3.py"
    cd = MIT / "content_disjoint"
    out = tmp_path / "ceiling.json"
    r = subprocess.run([sys.executable, "-B", str(script), "--logo-json", str(cd / "11_disjoint_arch6_d256.json"),
                        "--rungs-json", str(cd / "A3c_logo_extended_rungs_arch6.json"), "--out", str(out)],
                       capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr[-400:]
    got, ref = (json.loads(p.read_text(encoding="utf-8")) for p in (out, cd / "A3c_logo_combined_ceiling_arch6.json"))
    assert got["combined_ceiling"] == ref["combined_ceiling"] and got["folds_where_extended_wins"] == 1
    assert [f["ceiling_method"] for f in got["per_fold"]] == [f["ceiling_method"] for f in ref["per_fold"]]
    assert all(len(json.loads((cd / "A3c_logo_extended_rungs_arch6.json").read_text(encoding="utf-8"))
                   ["folds"][g]["rungs"]) == 12 for g in [f["fold"] for f in got["per_fold"]])
    # a tie between the ladder and a richer input goes to the ladder, as in the statistics
    membership = {"fit_fingerprint": "fit", "select_fingerprint": "select"}
    logo = {"pool": "p", "avg_raw_feat_logreg": 0.9, "avg_best_ps": 0.8,
            "per_fold": [{"held_out": "g", "best_ps_name": "PS_2_logreg", "best_ps_auc": 0.8, "raw_feat_auc": 0.9,
                          "n_test": 10, **membership}]}
    rungs = {"folds": {"g": {"rungs": {"disagreement_logreg": 0.8, "quadratic_logreg": 0.7}, **membership}}}
    (tmp_path / "l.json").write_text(json.dumps(logo), encoding="utf-8")
    (tmp_path / "r.json").write_text(json.dumps(rungs), encoding="utf-8")
    r = subprocess.run([sys.executable, "-B", str(script), "--logo-json", str(tmp_path / "l.json"), "--rungs-json",
                        str(tmp_path / "r.json"), "--out", str(tmp_path / "tie.json"), "--allow-incomplete-ladder"],
                       capture_output=True, text=True, env=ENV)
    tie = json.loads((tmp_path / "tie.json").read_text(encoding="utf-8"))
    assert r.returncode == 0 and tie["per_fold"][0]["ceiling_method"] == "PS_2_logreg"
    assert tie["folds_where_extended_wins"] == 0


def test_the_posted_numbers_are_classified_by_their_status_and_class(tmp_path):
    h = _harness()
    md = tmp_path / "posted.md"
    md.write_text("\n".join([
        "| Quantity | Posted text | Posted | Recomputed | Status |",
        "| `a.exact` | x | 0.123 | 0.123 | PASS |",
        "| `a.rounded` | x | 0.12 | 0.1234 | PASS (log rounding) |",
        "| `a.gone` | x | 0.5 | - | MISSING |",
        "| `logo.fs2.delta` | x | 0.079 | 0.081 | FAIL |",
        "| `cost.seconds` | x | 12 | 15 | FAIL |",
        "| `a.moved` | x | 0.400 | 0.410 | FAIL |"]) + "\n", encoding="utf-8")
    ids = ("a.exact", "a.rounded", "a.gone", "logo.fs2.delta", "cost.seconds", "a.moved")
    out = h.compare_posted(md, classes={"cost.seconds": "machine"}, expected_ids=ids)
    assert out["rows"] == 6 and out["agree"] == 1 and out["coverage_errors"] == []
    assert [r[0] for r in out["environment_dependent"]] == ["logo.fs2.delta"]
    assert [r[0] for r in out["machine_dependent"]] == ["cost.seconds"]
    assert [(r[0], r[3]) for r in out["different"]] == [
        ("a.rounded", "PASS (log rounding)"), ("a.gone", "MISSING"), ("a.moved", "FAIL")]


def test_the_table_inputs_prefer_this_runs_files_in_folder_order(tmp_path, monkeypatch):
    h = _harness()
    root = tmp_path / "repo"
    out = root / "camera_ready" / "output"
    names = {"ROOT": root, "INPUTS": root / "camera_ready" / "inputs", "RUN_INPUTS": root / "camera_ready" / "run_inputs",
             "RESULTS2": out / "results", "ANA": out / "results" / "analyses",
             "TK": out / "results" / "analyses" / "three_class", "BACKUP2": out / "reproducibility_backup"}
    for name, value in names.items():
        monkeypatch.setattr(h, name, value)
    for d in names.values():
        d.mkdir(parents=True, exist_ok=True)
    for f in ("both.json", "results_only.json", "old.json"):
        (names["INPUTS"] / f).write_text('"stored"', encoding="utf-8")
    old = names["RESULTS2"] / "old.json"
    old.write_text('"stale"', encoding="utf-8")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    run_start = time.time() - 60
    (names["ANA"] / "both.json").write_text('"analyses"', encoding="utf-8")
    (names["RESULTS2"] / "both.json").write_text('"results"', encoding="utf-8")
    (names["RESULTS2"] / "results_only.json").write_text('"results"', encoding="utf-8")
    record = {}
    h.stage_run_inputs(run_start, record)
    staged = {f: json.loads((names["RUN_INPUTS"] / f).read_text(encoding="utf-8")) for f in ("both.json",
                                                                                           "results_only.json",
                                                                                           "old.json")}
    assert staged == {"both.json": "analyses", "results_only.json": "results", "old.json": "stored"}


def test_a_resume_refuses_to_start_without_the_runs_gpu(tmp_path, monkeypatch):
    h = _harness()
    monkeypatch.setattr(h, "LOGS2", tmp_path)
    (tmp_path / "r1").mkdir()
    (tmp_path / "r1" / h.LOG_NAME).write_text(json.dumps({
        "run_id": "r1", "arguments": {"gpu_name": "NoSuchGPU", "blocks": "C"}, "blocks": ["C"],
        "run_start_epoch": 0, "steps": [], "sessions": [], "setup": {"status": "complete"}}), encoding="utf-8")
    monkeypatch.setattr(h, "_nvidia_smi", lambda: [{"name": "Some Other GPU", "index": 0}])
    args = argparse.Namespace(resume_run="r1", **{k: None for k in h.ARG_KEYS})
    args.blocks = "A,B,C"
    with pytest.raises(SystemExit, match="NoSuchGPU"):
        h.mode_resume(args, None)


def test_the_standard_split_statistics_are_recomputed_from_the_stored_inputs(tmp_path):
    """The stored inputs reproduce the standard-split statistics and the eight-generator LOGO statistics."""
    out = tmp_path / "full_ladder_stats.json"
    r = subprocess.run([sys.executable, "-B", str(AUTH / "full_ladder_stats_standard_v3.py"), "--codebase", str(ROOT),
                        "--run-a-dir", str(CR / "output" / "results" / "analyses"), "--logo-ceiling",
                        str(CR / "inputs" / "A3c_logo_combined_ceiling_arch6.json"), "--out", str(out)],
                       capture_output=True, text=True, env=ENV, cwd=str(ROOT))
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]
    got = json.loads(out.read_text(encoding="utf-8"))
    ref = json.loads((CR / "output" / "results" / "full_ladder_stats.json").read_text(encoding="utf-8"))
    assert got["standard"] == ref["standard"] and got["logo_arch6"] == ref["logo_arch6"]
