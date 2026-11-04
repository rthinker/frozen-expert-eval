"""Cache-free synthetic verification and recovery cases for the current harness."""
import argparse
import importlib.util
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

CR = Path(__file__).resolve().parents[1]
ENV = "(fs_mlp|fs2|FS2|FS_2|FS2-K3|PS3-K3)"


def _harness():
    spec = importlib.util.spec_from_file_location("reproduce_all_v4", CR / "scripts" / "review" / "reproduce_all_v4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compare(h, tmp_path, cases, run=None):
    """cases: file name -> (stored result, reference entry fields). Returns compare_v4's (ok, warn, fail, skip, lines)."""
    ref = {"results": {}}
    for name, (data, entry) in cases.items():
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
        ref["results"][str(tmp_path / name)] = {"kind": "det", "tol": None, "env": None, "key_map": {},
                                                "partial_ok": False, "step": "C", "metrics": {}, "strings_ref": {},
                                                **entry}
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(ref), encoding="utf-8")
    return h.compare_v4(path, run=run)


def _verdicts(lines):
    """File name -> verdict, from the result lines of compare_v4 (indented by two spaces; details by more)."""
    pattern = re.compile(r"^ {2}(\w+):\s+(.+?\.json)(?= \(|$)")
    matches = (pattern.match(line) for line in lines)
    return {Path(match[2]).name: match[1] for match in matches if match}


def test_a_skip_marker_needs_permission(tmp_path, monkeypatch):
    h = _harness()
    cases = {"skip.json": ({"skipped": True, "reason": "memory_budget"}, {"metrics": {"a": 0.5}})}
    assert _verdicts(_compare(h, tmp_path, cases)[4]) == {"skip.json": "FAIL"}
    monkeypatch.setattr(h, "SKIP_PERMITTED", {str(tmp_path / "skip.json"): {"reason": "memory_budget"}})
    assert _verdicts(_compare(h, tmp_path, cases)[4]) == {"skip.json": "SKIP"}


def test_a_partial_result_must_hold_the_routes_of_its_run(tmp_path):
    """A stored result needs every reference key. A regenerated result needs the routes selected by the run."""
    h = _harness()
    entry = {"partial_ok": True, "metrics": {"route_b.x": 1, "route_b.raw_files.y": 2, "route_a.z": 3}}
    full = {"route_a": {"z": 3}, "route_b": {"x": 1, "raw_files": {"y": 2}}}
    cases = {"empty_split_reuse_check.json": ({}, entry),
             "cache_split_reuse_check.json": ({"route_b": {"x": 1}}, entry),
             "claims_split_reuse_check.json": ({"_inputs": {"DF40_all_json_sha256": "h"}, "route_b": {"x": 1}}, entry),
             "full_split_reuse_check.json": (full, entry)}
    stored = _verdicts(_compare(h, tmp_path, cases)[4])
    assert stored == {"empty_split_reuse_check.json": "FAIL", "cache_split_reuse_check.json": "FAIL",
                      "claims_split_reuse_check.json": "FAIL", "full_split_reuse_check.json": "MATCH"}, stored
    start = datetime.now().timestamp() - 60          # the files above were written after this "run" started
    plain = _verdicts(_compare(h, tmp_path, cases, run={"start": start})[4])
    assert plain == {"empty_split_reuse_check.json": "FAIL", "cache_split_reuse_check.json": "MATCH",
                     "claims_split_reuse_check.json": "MATCH", "full_split_reuse_check.json": "MATCH"}, plain
    lists = _verdicts(_compare(h, tmp_path, cases, run={"start": start, "df40_lists": True})[4])
    assert lists["cache_split_reuse_check.json"] == "FAIL" and lists["full_split_reuse_check.json"] == "MATCH"
    images = _verdicts(_compare(h, tmp_path, cases, run={"start": start, "df40_images": True})[4])
    assert images["claims_split_reuse_check.json"] == "FAIL" and images["full_split_reuse_check.json"] == "MATCH"
    # a result the run did not rewrite is a stored result: every key
    later = _verdicts(_compare(h, tmp_path, cases, run={"start": datetime.now().timestamp() + 600})[4])
    assert later == stored


def test_values_must_be_valid_for_their_kind(tmp_path):
    """Check metric ranges before classifying environment-dependent differences. P-values and AUROCs must lie
    in [0, 1], counts must be non-negative integers, wins cannot exceed folds, and early stopping must occur
    within the epochs run."""
    h = _harness()
    seeds = {"env": ENV, "metrics": {"fs_mlp_per_seed[0]": 0.9, "fs_mlp_per_seed[1]": 0.8}}
    got = _compare(h, tmp_path, {
        "seed_high.json": ({"fs_mlp_per_seed": [2.0, 0.8]}, seeds),
        "seed_low.json": ({"fs_mlp_per_seed": [-1.0, 0.8]}, seeds),
        "seed_moved.json": ({"fs_mlp_per_seed": [0.5, 0.8]}, seeds),            # valid change reported as ENV
        "fs2_over_one.json": ({"fs2": 1.05}, {"env": ENV, "metrics": {"fs2": 0.97}}),   # within ENV_MAX
        "p.json": ({"fs2": {"sign_p": 1.5}}, {"env": ENV, "metrics": {"fs2.sign_p": 0.29}}),
        "wins.json": ({"fs2": {"wins": 9, "n": 8}}, {"env": ENV, "metrics": {"fs2.wins": 5, "fs2.n": 8}}),
        "count.json": ({"fs2": {"wins": 2.5, "n": 8}}, {"env": ENV, "metrics": {"fs2.wins": 2, "fs2.n": 8}}),
        "epoch.json": ({"FS2-K3": {"epochs": 10, "per_seed": {"7": {"best_epoch": 11}}}},
                       {"env": ENV, "metrics": {"FS2-K3.epochs": 10, "FS2-K3.per_seed.7.best_epoch": 9}}),
        "valid.json": ({"auc": 0.9, "fold": {"p_value": 0.03, "n": 8, "wins": 6}},
                       {"metrics": {"auc": 0.9, "fold.p_value": 0.03, "fold.n": 8, "fold.wins": 6}}),
        "auprc.json": ({"fs2": {"auprc": {"per_fold_fs1": [1.2], "delta": -0.3}}},          # AUPRC of a fold > 1
                       {"env": ENV, "metrics": {"fs2.auprc.per_fold_fs1[0]": 0.9, "fs2.auprc.delta": -0.3}}),
    })
    assert _verdicts(got[4]) == {"seed_high.json": "FAIL", "seed_low.json": "FAIL", "seed_moved.json": "ENV",
                                 "fs2_over_one.json": "FAIL", "p.json": "FAIL", "wins.json": "FAIL",
                                 "count.json": "FAIL", "epoch.json": "FAIL", "valid.json": "MATCH",
                                 "auprc.json": "FAIL"}, got[4]


def test_a_failed_status_fails_whatever_the_numbers(tmp_path):
    """Matching metrics cannot override a failed status. Each status must be PASS, report or COMPLETE."""
    h = _harness()
    entry = {"metrics": {"macro.FS_1": 0.96}}
    got = _compare(h, tmp_path, {
        "g4_fail.json": ({"macro": {"FS_1": 0.96}, "gate_G4_status": "FAIL"}, entry),
        "nested_fail.json": ({"macro": {"FS_1": 0.96}, "gate_G4": {"FS_1": {"status": "FAIL"}}}, entry),
        "audit_fail.json": ({"macro": {"FS_1": 0.96}, "audit": {"status": "FAIL"}}, entry),
        "fine.json": ({"macro": {"FS_1": 0.96}, "gate_G4_status": "PASS", "changed": {"status": "report"},
                       "result": {"status": "COMPLETE"}}, entry),
    })
    assert _verdicts(got[4]) == {"g4_fail.json": "FAIL", "nested_fail.json": "FAIL", "audit_fail.json": "FAIL",
                                 "fine.json": "MATCH"}, got[4]


def test_environment_dependent_values_are_bounded(tmp_path):
    h = _harness()
    got = _compare(h, tmp_path, {
        "small.json": ({"fs2_mean": 0.88}, {"env": ENV, "metrics": {"fs2_mean": 0.90}}),
        "large.json": ({"fs2_mean": 0.50}, {"env": ENV, "metrics": {"fs2_mean": 0.90}}),
        "steps.json": ({"fs2": {"per_seed": [0.5], "sign_p": 0.73}},
                       {"env": ENV, "metrics": {"fs2.per_seed[0]": 0.9, "fs2.sign_p": 0.29}}),
        "config.json": ({"FS2-K3": {"metadata": {"seeds": [8]}}},
                        {"env": ENV, "metrics": {"FS2-K3.metadata.seeds[0]": 7}}),
        "nan.json": ({"fs2_mean": float("nan")}, {"env": ENV, "metrics": {"fs2_mean": 0.90}}),
        "stoch_small.json": ({"a": 0.80}, {"kind": "stoch", "metrics": {"a": 0.81}}),
        "stoch_large.json": ({"a": 0.50}, {"kind": "stoch", "metrics": {"a": 0.90}}),
    })
    assert _verdicts(got[4]) == {"small.json": "ENV", "large.json": "FAIL", "steps.json": "ENV", "config.json": "FAIL",
                                 "nan.json": "FAIL", "stoch_small.json": "WARN", "stoch_large.json": "FAIL"}, got[4]


def test_documented_alternatives(tmp_path, monkeypatch):
    h = _harness()
    entry = {"env": "^expert_train_unique_paths", "metrics": {"expert_train_unique_paths.df40": 703223}}
    monkeypatch.setattr(h, "ENV_DOCUMENTED", {str(tmp_path / n): {"expert_train_unique_paths.df40": (703223.0, 732349.0)}
                                              for n in ("public.json", "other.json")})
    got = _compare(h, tmp_path, {"public.json": ({"expert_train_unique_paths": {"df40": 732349}}, entry),
                                 "other.json": ({"expert_train_unique_paths": {"df40": 712345}}, entry)})
    assert _verdicts(got[4]) == {"public.json": "ENV", "other.json": "FAIL"}


def test_nothing_to_compare_fails(tmp_path):
    h = _harness()
    assert _verdicts(_compare(h, tmp_path, {"none.json": ({"a": 1}, {})})[4]) == {"none.json": "FAIL"}


def test_steps_to_redo_follow_the_reads():
    h = _harness()
    cd = "camera_ready/output/results/mitigation/content_disjoint"
    man = "camera_ready/output/results/mitigation/exclusions/x.json"
    steps = [
        h.Step("A01", "A", "a", ("python", "scripts/03.py"), ("output/results/a1.json",)),
        h.Step("A02", "A", "a", ("python", "scripts/04.py"), ("output/results/a2.json",)),
        h.Step("A03", "A", "a", ("python", "scripts/05.py"), ("output/results/a3.json",)),
        h.Step("B01", "B", "b", ("python", "x.py", "--out", man), (man,)),
        h.Step("B02", "B", "b", ("python", "y.py", "--manifest", man, "--out-dir", cd), (f"{cd}/r.json",)),
        h.Step("B03", "B", "b", ("python", "camera_ready/scripts/runs/compare_runs_v2.py", "--new-dir", cd),
               (f"{cd}/gate.json",)),
        h.Step("B04", "B", "b", ("python", "z.py", "--out-dir", cd), (f"{cd}/other.json",)),
        h.Step("C24", "C", "c", ("python", "camera_ready/scripts/authoring/generate_camera_ready_tables_v18.py"),
               ("camera_ready/output/tables/t.csv",)),
    ]
    why, stale = h.steps_to_redo(steps, {"B01": "not finished"})
    assert sorted(why) == ["B01", "B02", "B03", "C24"], why
    assert why["B03"] == "reads an output of B02" and stale == {"B02", "B03", "C24"}
    why, stale = h.steps_to_redo(steps, {"A02": "not finished"})
    assert sorted(why) == ["A02", "A03", "C24"] and stale == {"C24"}, why
    # a step that would rerun anyway (a moved checkpoint) still reruns in full when an input is recomputed
    why, stale = h.steps_to_redo(steps, {"B01": "not finished", "B02": "a checkpoint was moved aside"})
    assert why["B02"] == "a checkpoint was moved aside" and "B02" in stale


def test_block_a_gets_the_checks_the_reviewed_comparison_lacks(tmp_path, monkeypatch):
    h = _harness()
    rv = h.reviewed()
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(rv, "RESULTS", results)
    monkeypatch.setattr(rv, "REFERENCE_PATH", results / "reference_results.json")
    marker = {"skipped": True, "reason": "memory_budget", "fs1_method": "raw_feature_logreg"}
    files = {"det_nan.json": ({"a": float("nan")}, True), "stoch_far.json": ({"a": 0.5}, False),
             "stoch_near.json": ({"a": 0.89}, False), "04_ds_full21_rawfeat.json": ({**marker, "pool": "full21"}, True),
             "04_ds_arch6_rawfeat.json": ({**marker, "pool": "arch6"}, True)}
    ref = {"results": {n: {"deterministic": det, "metrics": {"a": 0.9}} for n, (_, det) in files.items()}}
    ref["results"]["broken.json"] = {"deterministic": True, "metrics": {"a": 0.9}}
    for name, (data, _) in files.items():
        (results / name).write_text(json.dumps(data), encoding="utf-8")
    (results / "broken.json").write_text("{", encoding="utf-8")
    rv.REFERENCE_PATH.write_text(json.dumps(ref), encoding="utf-8")
    fail, lines = h.guard_block_a(rv)
    failed = sorted(name for name, verdict in _verdicts(lines).items() if verdict == "FAIL")
    assert failed == ["04_ds_arch6_rawfeat.json", "broken.json", "det_nan.json", "stoch_far.json"], lines
    assert fail == 4


def _sandbox(h, monkeypatch, root):
    """Point the harness at an empty temporary repository and stub the slow or reviewed-code parts."""
    out = root / "camera_ready" / "output"
    for name, value in (("ROOT", root), ("LOGS2", out / "logs"), ("RESULTS2", out / "results"),
                        ("MIT", out / "results" / "mitigation"), ("TK", out / "results" / "analyses" / "three_class"),
                        ("BACKUP2", out / "reproducibility_backup"),
                        ("REFERENCE2", root / "camera_ready" / "reference" / "reference.json")):
        monkeypatch.setattr(h, name, value)
    monkeypatch.setattr(h, "environment_record",
                        lambda env: {"cpu": "test", "ram_gib": 1, "torch_in_run_env": {"devices": []}})
    monkeypatch.setattr(h, "mode_verify_only", lambda rv, run=None: ({"stub": True}, 0))
    monkeypatch.setattr(h, "compare_ledgers", lambda *a, **k: None)
    monkeypatch.setattr(h, "compare_posted", lambda *a, **k: None)
    for fold in ("f1", "f2"):        # the held-out generators' labels in the feature cache (the fake folds below)
        (root / "output" / "cached_features" / fold).mkdir(parents=True, exist_ok=True)
        np.save(root / "output" / "cached_features" / fold / "labels.npy", np.array([0, 1, 0, 1], dtype=np.int32))


def _stamp(h, ckpt, attempt_id, fp, array_file, key="scores_sha256"):
    """Stamp a resume file as stamp_checkpoints_v2.py does for the attempt `attempt_id` with fingerprint `fp`."""
    rec = json.loads(ckpt.read_text(encoding="utf-8"))
    rec["provenance"] = {"attempt": attempt_id, "fingerprint": h.fingerprint_digest(fp),
                         key: h._file_hash(array_file)}
    ckpt.write_text(json.dumps(rec), encoding="utf-8")


# A stand-in for the LOGO runner: same file name, checkpoint and score-file layout, and --resume rule (a fold whose
# checkpoint exists is reused); computed.txt lists every fold it computes.
FAKE_LOGO = """
import argparse, json
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser()
for flag in ("--pool", "--proj_dim", "--exclusion", "--manifest", "--out-dir"):
    ap.add_argument(flag)
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()
out = Path(a.out_dir)
excluded = {f["held_out"]: f["n_excluded"] for f in json.loads(Path(a.manifest).read_text())["folds"]}
folds = []
for fold in ("f1", "f2"):
    ckpt = out / "folds" / f"{a.pool}_d{a.proj_dim}_{fold}.json"
    if a.resume and ckpt.exists():
        folds.append(json.loads(ckpt.read_text()))
        continue
    rel = f"scores/{a.pool}_d{a.proj_dim}_{fold}.npz"
    (out / "scores").mkdir(parents=True, exist_ok=True)
    s = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    np.savez_compressed(out / rel, labels=np.array([0, 1, 0, 1], dtype=np.int8), PS_1_prob_avg=s,
                        FS_1_raw_feat_logreg=s, FS_2_mlp_seed_avg=s)
    rec = {"held_out": fold, "n_test": 4, "n_excluded_training_rows": excluded[fold], "ps_means": {"PS_1_prob_avg": 1.0},
           "raw_feat_auc": 1.0, "fs_mlp_mean": 1.0, "fs_mlp_per_seed": [1.0, 1.0], "scores_file": rel}
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text(json.dumps(rec))
    with open(out / "computed.txt", "a") as fh:
        fh.write(fold + " ")
    folds.append(rec)
(out / f"11_disjoint_{a.pool}_d{a.proj_dim}.json").write_text(json.dumps({"per_fold": folds}))
"""
WRITE = ("import sys, pathlib; p = pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True); "
         "p.write_text(sys.argv[2] if len(sys.argv) > 2 else 'x')")


def test_resume_checks_checkpoints_first_and_reruns_what_reads_them(tmp_path, monkeypatch):
    h = _harness()
    root = tmp_path / "repo"
    root.mkdir()
    _sandbox(h, monkeypatch, root)
    config = root / "configs" / "paths.yaml.example"
    config.parent.mkdir()
    config.write_text("output_dir: ./output\n", encoding="utf-8")

    def authenticate_synthetic_cache(cache, env):
        assert cache == root / "output" / "cached_features"
        assert all((cache / fold / "labels.npy").is_file() for fold in ("f1", "f2"))

    monkeypatch.setattr(h, "authenticate_cache", authenticate_synthetic_cache)
    runner = tmp_path / "runner" / "11_disjoint_data_v4.py"
    runner.parent.mkdir()
    runner.write_text(FAKE_LOGO, encoding="utf-8")
    cd, man = h.CD, f"{h.EXCL}/logo_exclusions_arch6.json"
    ref = {"results": {f"{cd}/11_disjoint_arch6_d256.json": {"metrics": {
        "per_fold[0].ps_means.PS_1_prob_avg": 0.9, "per_fold[0].fs_mlp_per_seed[0]": 0.9,
        "per_fold[0].fs_mlp_per_seed[1]": 0.9}}}}
    h.REFERENCE2.parent.mkdir(parents=True)
    h.REFERENCE2.write_text(json.dumps(ref), encoding="utf-8")
    manifest = json.dumps({"folds": [{"held_out": "f1", "n_excluded": 0}, {"held_out": "f2", "n_excluded": 7}]})
    steps = [h.Step("B01", "B", "exclusions", ("python", "-c", WRITE, man, manifest), (man,)),
             h.Step("B02", "B", "LOGO", ("python", str(runner), "--pool", "arch6", "--proj_dim", "256", "--exclusion",
                                         "content", "--manifest", man, "--out-dir", cd, "{resume}"),
                    (f"{cd}/11_disjoint_arch6_d256.json",)),
             h.Step("B03", "B", "gate", ("python", "-c", WRITE, f"{cd}/gate.json", '{"status": "PASS"}', "--new-dir",
                                         cd), (f"{cd}/gate.json",)),
             h.Step("B04", "B", "other", ("python", "-c", WRITE, "out/other.txt"), ("out/other.txt",))]
    monkeypatch.setattr(h, "selected_steps", lambda args, rv: (["B"], list(steps)))
    args = argparse.Namespace(blocks="B", gpu_name=None, df40_json=None, df40_raw=None, with_reproduction_gate=False,
                              from_step=None, only=None, run_id="t1", resume_run=None)
    assert h.mode_run(args, None) == 0
    computed = root / cd / "computed.txt"
    assert computed.read_text().split() == ["f1", "f2"]
    log = root / "camera_ready" / "output" / "logs" / "t1" / h.LOG_NAME

    # every step finished; a cut-off score file is still found, moved aside with its checkpoint, and its fold rerun
    (root / cd / "scores" / "arch6_d256_f2.npz").write_bytes(b"PK")
    args.resume_run = "t1"
    assert h.mode_resume(args, None) == 0
    rec = json.loads(log.read_text(encoding="utf-8"))
    session = rec["sessions"][-1]
    assert session["steps_planned"] == ["B02", "B03"], session
    assert session["why"] == {"B02": "a checkpoint was moved aside", "B03": "reads an output of B02"}
    assert sorted(Path(m["file"]).name for m in session["checkpoints_moved"]) == ["arch6_d256_f2.json",
                                                                                  "arch6_d256_f2.npz"]
    assert computed.read_text().split() == ["f1", "f2", "f2"]          # only the moved fold was recomputed

    # every checkpoint of the run is stamped by the attempt that wrote it (stamp_checkpoints_v2.py)
    stamp = json.loads((root / cd / "folds" / "arch6_d256_f1.json").read_text(encoding="utf-8"))["provenance"]
    attempts = {r["attempt_id"] for r in rec["steps"] if r["id"] == "B02"}
    assert stamp["attempt"] in attempts and len(stamp["scores_sha256"]) == 64
    # a checkpoint stamped by another run (here: another attempt id) is not reused
    ck = root / cd / "folds" / "arch6_d256_f1.json"
    foreign = json.loads(ck.read_text(encoding="utf-8"))
    foreign["provenance"]["attempt"] = "0" * 32
    ck.write_text(json.dumps(foreign), encoding="utf-8")
    assert h.mode_resume(args, None) == 0
    assert computed.read_text().split() == ["f1", "f2", "f2", "f1"]
    reasons = {m["reason"] for m in json.loads(log.read_text(encoding="utf-8"))["sessions"][-1]["checkpoints_moved"]}
    assert "stamped by another run or step" in reasons

    # when an input is recomputed (the exclusion list, cut off here), the LOGO step reruns in full
    (root / man).write_text("{", encoding="utf-8")
    assert h.mode_resume(args, None) == 0
    session = json.loads(log.read_text(encoding="utf-8"))["sessions"][-1]
    assert session["steps_planned"] == ["B01", "B02", "B03"] and session["rerun_in_full"] == ["B02", "B03"]
    assert computed.read_text().split() == ["f1", "f2", "f2", "f1", "f1", "f2"]


def _k3_manifest(h, sizes=((2, 1), (1, 1), (1, 2))):
    """A frozen three-class manifest of view B_arch6: per fold, (test rows, SELECT rows of that fold)."""
    rows = []
    for fold, (n, n_select) in enumerate(sizes):
        rows += [{"fold": fold, "split": "SELECT" if i < n_select else "FIT"} for i in range(n)]
    (h.TK / "K3_FROZEN_B_arch6.json").write_text(json.dumps({"classes": ["a", "b", "c"], "manifest": rows}),
                                                 encoding="utf-8")


def _k3_pair(h, stem, attempt_id, fp, meta=None, **arrays):
    np.savez_compressed(h.TK / f"{stem}.npz", **arrays)
    (h.TK / f"{stem}.json").write_text(json.dumps(meta or {"seed": 7}), encoding="utf-8")
    if attempt_id:
        _stamp(h, h.TK / f"{stem}.json", attempt_id, fp, h.TK / f"{stem}.npz", "arrays_sha256")


def test_three_class_resume_files_come_in_stamped_pairs(tmp_path, monkeypatch):
    h = _harness()
    _sandbox(h, monkeypatch, tmp_path)
    h.TK.mkdir(parents=True)
    _k3_manifest(h)
    fp = {"settings": "s"}
    record = {"steps": [{"id": "C20", "attempt_id": "a1", "fingerprint": fp}]}
    # fold 0: test = its 2 rows; select = the SELECT rows of folds 1 and 2 (1 + 1)
    probs = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    lone = h.TK / "NATIVE_K_ALIGNED_FULL_B_arch6_F0_PS3_K3_S7.npz"
    np.savez_compressed(lone, select=probs, test=probs)
    _k3_pair(h, "NATIVE_K_ALIGNED_FULL_B_arch6_F0_FS2_K3_S7", "a1", fp, select=probs, test=probs)
    _k3_pair(h, "NATIVE_K_ALIGNED_FULL_B_arch6_F0_FS2_K3_S24", None, fp, meta={"seed": 24}, select=probs, test=probs)
    _k3_pair(h, "NATIVE_K_ALIGNED_FULL_B_arch6_F0_FS2_K3_S42", "other", fp, meta={"seed": 42}, select=probs, test=probs)
    (h.TK / "NATIVE_K_ALIGNED_FULL_B_arch6_RESULT.json").write_text("{}", encoding="utf-8")
    step = h.Step("C20", "C", "aligned", ("python", "camera_ready/scripts/analyses/appI1_native_k_aligned_v2.py",
                                          "--view", "B_arch6", "{resume}"), ())
    moved, steps = h.validate_checkpoints(record, [step])
    reasons = {Path(m["file"]).name: m["reason"] for m in moved if m["reason"] != "moved with the file above"}
    assert reasons == {lone.name: "incomplete pair",
                       "NATIVE_K_ALIGNED_FULL_B_arch6_F0_FS2_K3_S24.json": "not stamped by an attempt of this run",
                       "NATIVE_K_ALIGNED_FULL_B_arch6_F0_FS2_K3_S42.json": "stamped by another run or step"}, reasons
    assert steps == {"C20"} and (h.TK / "NATIVE_K_ALIGNED_FULL_B_arch6_F0_FS2_K3_S7.npz").exists()
    assert (h.TK / "NATIVE_K_ALIGNED_FULL_B_arch6_RESULT.json").exists()


def test_three_class_arrays_and_sizes_are_checked(tmp_path, monkeypatch):
    h = _harness()
    _sandbox(h, monkeypatch, tmp_path)
    h.TK.mkdir(parents=True)
    _k3_manifest(h)
    fp = {"settings": "s"}
    record = {"steps": [{"id": "C20", "attempt_id": "a1", "fingerprint": fp}]}
    probs2, probs1 = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]), np.array([[0.2, 0.8, 0.0]])
    stems = {"good": ("NATIVE_K_ALIGNED_FULL_B_arch6_F0_PS3_K3_S7", {"select": probs2, "test": probs2}, 7),
             "nan": ("NATIVE_K_ALIGNED_FULL_B_arch6_F1_PS3_K3_S7",
                     {"select": probs1, "test": np.array([[np.nan, 0.5, 0.5]])}, 7),
             "not_probs": ("NATIVE_K_ALIGNED_FULL_B_arch6_F1_FS2_K3_S7",
                           {"select": probs1, "test": np.array([[0.5, 0.9, 0.5]])}, 7),
             "seed": ("NATIVE_K_ALIGNED_FULL_B_arch6_F0_FS2_K3_S10", {"select": probs2, "test": probs2}, 7),
             "rows": ("NATIVE_K_ALIGNED_FULL_B_arch6_F2_PS3_K3_S7", {"select": probs2, "test": probs2}, 7),
             "head_rows": ("NATIVE_K_ALIGNED_FULL_B_arch6_F1_HEADS",
                           {"fit_logits": np.zeros((3, 2, 3)), "select_logits": np.zeros((2, 2, 3)),
                            "test_logits": np.zeros((1, 2, 3))}, None)}
    for stem, arrays, seed in stems.values():
        _k3_pair(h, stem, "a1", fp, meta={"seed": seed} if seed else {"quality": {}}, **arrays)
    step = h.Step("C20", "C", "aligned", ("python", "camera_ready/scripts/analyses/appI1_native_k_aligned_v2.py",
                                          "--view", "B_arch6", "{resume}"), ())
    moved, _ = h.validate_checkpoints(record, [step])
    reasons = {Path(m["file"]).stem: m["reason"] for m in moved if m["reason"] != "moved with the file above"}
    assert set(reasons) == {stems[k][0] for k in ("nan", "not_probs", "seed", "rows", "head_rows")}, reasons
    assert "not finite" in reasons[stems["nan"][0]] and "class probabilities" in reasons[stems["not_probs"][0]]
    assert "seed" in reasons[stems["seed"][0]] and "test: 2 rows" in reasons[stems["rows"][0]]
    assert "fit_logits: 3 rows" in reasons[stems["head_rows"][0]]
    assert (h.TK / f"{stems['good'][0]}.npz").exists()


def test_block_a_values_and_statuses_are_checked_too(tmp_path, monkeypatch):
    h = _harness()
    rv = h.reviewed()
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(rv, "RESULTS", results)
    monkeypatch.setattr(rv, "REFERENCE_PATH", results / "reference_results.json")
    files = {"seed_over_one.json": ({"avg8_per_seed": [1.05]}, {"avg8_per_seed[0]": 0.97}),   # within ENV_MAX
             "status.json": ({"a": 0.9, "status": "FAIL"}, {"a": 0.9}),
             "fine.json": ({"avg8_per_seed": [0.95], "status": "PASS"}, {"avg8_per_seed[0]": 0.97})}
    ref = {"results": {n: {"deterministic": False, "metrics": m} for n, (_, m) in files.items()}}
    for name, (data, _) in files.items():
        (results / name).write_text(json.dumps(data), encoding="utf-8")
    rv.REFERENCE_PATH.write_text(json.dumps(ref), encoding="utf-8")
    fail, lines = h.guard_block_a(rv)
    failed = sorted(name for name, verdict in _verdicts(lines).items() if verdict == "FAIL")
    assert failed == ["seed_over_one.json", "status.json"] and fail == 2, lines


FOLD_REF = {"camera_ready/output/results/mitigation/content_disjoint/11_disjoint_arch6_d256.json": {"metrics": {
    "per_fold[0].ps_means.PS_1_prob_avg": 0.9, "per_fold[0].fs_mlp_per_seed[0]": 0.9,
    "per_fold[0].fs_mlp_per_seed[1]": 0.9}}}
FP = {"settings": "s", "code": "c"}


def _fold_files(h, folder, scores, labels=(0, 1, 0, 1), recorded=1.0, attempt_id="a1", fp=FP):
    """A fold checkpoint of the LOGO ladder and its score file, stamped by `attempt_id`; returns (files, name match)."""
    (folder / "folds").mkdir(parents=True, exist_ok=True)
    (folder / "scores").mkdir(exist_ok=True)
    rel_scores = "scores/arch6_d256_f1.npz"
    np.savez_compressed(folder / rel_scores, labels=np.asarray(labels, dtype=np.int8), **scores)
    rec = {"held_out": "f1", "n_test": len(labels), "n_excluded_training_rows": 0,
           "ps_means": {"PS_1_prob_avg": recorded}, "raw_feat_auc": recorded, "fs_mlp_mean": recorded,
           "fs_mlp_per_seed": [recorded, recorded], "scores_file": rel_scores}
    ckpt = folder / "folds" / "arch6_d256_f1.json"
    ckpt.write_text(json.dumps(rec), encoding="utf-8")
    if attempt_id:
        _stamp(h, ckpt, attempt_id, fp, folder / rel_scores)
    return [ckpt, folder / rel_scores], h.FAMILIES["folds"][1].match(ckpt.name)


def test_saved_scores_must_match_their_checkpoint(tmp_path, monkeypatch):
    """Saved arrays must be one-dimensional, finite and of the fold's test size, with the recorded AUROC.
    Labels must match the held-out generator's labels in the feature cache in the same order."""
    h = _harness()
    _sandbox(h, monkeypatch, tmp_path / "repo")
    attempts = {"a1": {"id": "B08", "attempt_id": "a1", "fingerprint": FP}}
    s = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    good = {"PS_1_prob_avg": s, "FS_1_raw_feat_logreg": s, "FS_2_mlp_seed_avg": s}
    flipped = {k: v[::-1].copy() for k, v in good.items()}       # rows reordered with their labels: AUROC unchanged
    cases = {"valid": (good, (0, 1, 0, 1), None),
             "nan": ({**good, "PS_1_prob_avg": np.array([np.nan, 0.9, 0.2, 0.8])}, (0, 1, 0, 1), "not finite"),
             "short": ({**good, "PS_1_prob_avg": np.array([0.5])}, (0, 1, 0, 1), "shape"),
             "zeros": ({**good, "PS_1_prob_avg": np.zeros(4)}, (0, 1, 0, 1), "AUROC"),
             "no_fs1": ({"PS_1_prob_avg": s, "FS_2_mlp_seed_avg": s}, (0, 1, 0, 1), "lacks FS_1_raw_feat_logreg"),
             "one_class": (good, (1, 1, 1, 1), "two-class"),
             "misaligned": (flipped, (1, 0, 1, 0), "labels differ from the held-out generator's labels")}
    for name, (scores, labels, expect) in cases.items():
        files, m = _fold_files(h, tmp_path / name, scores, labels)
        why = h.fold_problem("folds", files, m, attempts, FOLD_REF, {"f1": 0}, FP)
        assert (why is None) if expect is None else (why is not None and expect in why), (name, why)


def test_a_checkpoint_belongs_to_the_attempt_that_stamped_it(tmp_path, monkeypatch):
    """Reuse requires a stamp from this step and run, an unchanged attempt fingerprint and the stamped score file."""
    h = _harness()
    _sandbox(h, monkeypatch, tmp_path / "repo")
    attempts = {"a1": {"id": "B08", "attempt_id": "a1", "fingerprint": FP}}
    s = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    good = {"PS_1_prob_avg": s, "FS_1_raw_feat_logreg": s, "FS_2_mlp_seed_avg": s}

    def problem(folder, now=FP, **kw):
        files, m = _fold_files(h, tmp_path / folder, good, **kw)
        return files, h.fold_problem("folds", files, m, attempts, FOLD_REF, {"f1": 0}, now)
    assert problem("valid")[1] is None
    assert problem("unstamped", attempt_id=None)[1] == "not stamped by an attempt of this run"
    assert problem("foreign", attempt_id="b2")[1] == "stamped by another run or step"
    assert problem("forged", fp={"settings": "x"})[1] == "its fingerprint is not that of the attempt it names"
    assert problem("stale", now={"settings": "s", "code": "new"})[1] == (
        "the code or configuration changed since the attempt that wrote it")
    files, _ = problem("swapped")
    np.savez_compressed(files[1], labels=np.array([0, 1, 0, 1], dtype=np.int8), **{k: v + 0 for k, v in good.items()},
                        extra=np.zeros(1))
    m = h.FAMILIES["folds"][1].match(files[0].name)
    assert h.fold_problem("folds", files, m, attempts, FOLD_REF, {"f1": 0}, FP) == (
        "its array file is not the one written with it")


def test_the_rebuilt_ledger_must_hold_valid_numbers():
    """The generated ledger must contain finite values within their allowed ranges."""
    h = _harness()
    import csv
    import copy
    stored = {(r["table"], r["quantity"]): r for r in csv.DictReader(open(h.STORED_LEDGER, encoding="utf-8"))}
    assert h.ledger_problems(stored, stored) == []
    rebuilt = copy.deepcopy(stored)
    p_key = next(k for k, r in stored.items() if r["kind"] == "json" and h.value_rule(r["key"]) == "p")
    auc_key = next(k for k, r in stored.items() if r["kind"] == "json" and h.value_rule(r["key"]) == "prob"
                   and 0 <= float(r["value"]) <= 1)
    name_key = next(k for k, r in stored.items() if h._number(r["value"]) is None)
    rebuilt[p_key]["value"], rebuilt[auc_key]["value"], rebuilt[name_key]["value"] = "1.7", "nan", "other"
    found = {k: why for k, _, _, why in h.ledger_problems(stored, rebuilt)}
    assert found == {p_key: "p-value outside [0, 1]", auc_key: "not finite"}, found


def test_a_fingerprint_names_what_changed(tmp_path, monkeypatch):
    h = _harness()
    root = tmp_path / "repo"
    root.mkdir()
    _sandbox(h, monkeypatch, root)
    man = f"{h.EXCL}/logo_exclusions_arch6.json"
    (root / man).parent.mkdir(parents=True)
    (root / man).write_text('{"folds": []}', encoding="utf-8")
    step = h.Step("B08", "B", "LOGO", ("python", "runner.py", "--manifest", man, "--out-dir", h.CD, "{gpu_name}",
                                       "{resume}"), (f"{h.CD}/r.json",))
    args = argparse.Namespace(gpu_name=None, df40_json=None, df40_raw=None)
    fp1 = h.fingerprint(step, [step], args)
    assert h.fingerprint_change(fp1, h.fingerprint(step, [step], args)) == ""
    (root / man).write_text('{"folds": [{"held_out": "x", "n_excluded": 1}]}', encoding="utf-8")
    fp2 = h.fingerprint(step, [step], args)
    assert h.fingerprint_change(fp1, fp2) == "its input files"
    args.gpu_name = "5080"
    assert h.fingerprint_change(fp2, h.fingerprint(step, [step], args)) == "its command"
    assert h.fingerprint_change(None, fp2) == "the fingerprint (not recorded)"


