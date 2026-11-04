"""Check checkpoint stamps, FS1 score precision, the reviewed-fold FS1 refit and the comparison gates.
Tests use supplied results or synthetic data and write to temporary folders.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CR = Path(__file__).resolve().parents[1]
ROOT = CR.parent
REVIEW, RUNS = CR / "scripts" / "review", CR / "scripts" / "runs"
MIT = CR / "output" / "results" / "mitigation"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if k != "REPRO_ATTEMPT_ID"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra)
    return env


# ---------------------------------------------------------------------------------------------- stamping launcher
def test_the_launcher_stamps_three_class_caches_but_not_their_per_fold_results():
    stamp = _load(REVIEW / "stamp_checkpoints_v2.py", "stamp_v2")
    for name in ("NATIVE_K_ALIGNED_FULL_B_arch6_F0_PS3_K3_S7.json", "NATIVE_K_ALIGNED_FULL_B_arch6_F2_HEADS.json",
                 "NATIVE_K_ALIGNED_SMOKE_A_df40_F0_PS4_K3.json"):
        assert stamp.K3_RESUME.match(name), name
    for name in ("NATIVE_K_ALIGNED_FULL_B_arch6_F0_RESULT.json", "NATIVE_K_ALIGNED_FULL_B_arch6_RESULT.json",
                 "NATIVE_K_ALIGNED_FULL_B_arch6_F1_HEADS.npz"):
        assert not stamp.K3_RESUME.match(name), name


# ---------------------------------------------------------------------------------------------- score precision
def test_the_precision_check_refuses_the_stored_report_by_hand_and_writes_where_told(tmp_path):
    script = REVIEW / "check_fs1_score_precision_v2.py"
    stored = MIT / "fs1_score_precision_check.json"
    before = stored.read_bytes()
    r = subprocess.run([sys.executable, "-B", str(script), "--out", str(stored)], capture_output=True, text=True,
                       env=_env())
    assert r.returncode != 0 and "among the stored results" in r.stderr, r.stderr[-300:]
    assert stored.read_bytes() == before
    out = tmp_path / "checks" / "precision.json"
    r = subprocess.run([sys.executable, "-B", str(script), "--out", str(out)], capture_output=True, text=True,
                       env=_env())
    assert r.returncode == 0, r.stderr[-300:]
    got, ref = json.loads(out.read_text(encoding="utf-8")), json.loads(before.decode("utf-8"))
    assert got["status"] == ref["status"] == "PASS" and got["n_folds"] == ref["n_folds"] == 16
    assert got["max_pre_vs_post_cast_auc_difference"] == ref["max_pre_vs_post_cast_auc_difference"]


# ---------------------------------------------------------------------------------------------- FS1 refit
def test_the_refit_does_not_compare_its_logo_folds_with_a_run_that_excluded_rows(tmp_path, monkeypatch):
    """The refit reproduces the submitted folds and reports content-disjoint LOGO as not compared.
    Small synthetic arrays stand in for the cache."""
    sys.path.insert(0, str(ROOT))
    import src.utils.io as sio
    from src.utils.config import get_pool_experts
    from src.utils.dataset_helpers import TEST_GENERATORS
    experts = get_pool_experts("arch6")
    rng = np.random.default_rng(0)

    def fake(experts_, tag, cache):
        y = np.array([0, 1] * 20)
        return [rng.normal(size=(40, 3)) for _ in experts_], [rng.normal(size=(40, 2)) for _ in experts_], y
    monkeypatch.setattr(sio, "load_pool_data", fake)
    stored = {"exclusion": "content", "per_fold": [{"held_out": g, "raw_feat_auc": 0.5} for g in TEST_GENERATORS]}
    (tmp_path / "11_disjoint_arch6_d256.json").write_text(json.dumps(stored), encoding="utf-8")
    refit = _load(REVIEW / "refit_fs1_v3.py", "refit_v3")
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["refit_fs1_v3.py", "--codebase", str(ROOT), "--protocol", "logo",
                                      "--expect-dir", str(tmp_path), "--out", str(out)])
    assert refit.main() == 1
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["status"] == "NOT COMPARED" and len(experts) == 6
    assert rep["comparisons"]["logo"]["status"].startswith("not compared: the stored run") and \
        "exclusion 'content'" in rep["comparisons"]["logo"]["status"]
    stored["exclusion"] = "none"
    (tmp_path / "11_disjoint_arch6_d256.json").write_text(json.dumps(stored), encoding="utf-8")
    assert refit.main() == 1                       # compared now, and different from the synthetic 0.5 values
    assert json.loads(out.read_text(encoding="utf-8"))["comparisons"]["logo"]["status"] == "FAIL"


# ---------------------------------------------------------------------------------------------- comparison gates
def test_the_gates_group_methods_and_judge_each_group_at_its_tolerance():
    cmp = _load(RUNS / "compare_runs_v2.py", "compare_v2")
    assert [cmp.group(m) for m in ("PS_2_logreg", "raw_feat_auc", "disagreement_logreg", "PS_4_xgboost",
                                   "quadratic_xgb_capped", "fs_mlp_mean", "PS_3_mlp_logits", "other_method")] == \
        ["exact", "exact", "exact", "xgboost", "xgboost", "stochastic", "stochastic", "other"]
    new = {"a": {"PS_2_logreg": 0.80, "PS_4_xgboost": 0.700, "fs_mlp_mean": 0.9}, "b": {"PS_2_logreg": 0.6}}
    ref = {"a": {"PS_2_logreg": 0.80 + 2e-6, "PS_4_xgboost": 0.7005, "fs_mlp_mean": 0.8}, "c": {"PS_2_logreg": 0.6}}
    with pytest.raises(ValueError):
        cmp.diff_table(new, ref)
    new, ref = {"a": new["a"]}, {"a": ref["a"]}
    rows = cmp.diff_table(new, ref, methods=set(new["a"]), expected_folds=["a"])
    assert {(r["fold"], r["method"]) for r in rows} == {("a", "PS_2_logreg"), ("a", "PS_4_xgboost"),
                                                         ("a", "fs_mlp_mean")}
    st = cmp.summarise(rows, {"exact": 1e-6, "xgboost": 1e-3})
    assert st["PS_2_logreg"]["status"] == "FAIL" and st["PS_4_xgboost"]["status"] == "PASS"
    assert st["fs_mlp_mean"]["status"] == "report"
    with pytest.raises(ValueError):
        cmp.diff_table({}, ref)
    with pytest.raises(ValueError):
        cmp.summarise([], {"exact": 1e-6})


def test_the_gates_read_reviewed_logs_in_utf16_and_utf8(tmp_path):
    cmp = _load(RUNS / "compare_runs_v2.py", "compare_v2_logs")
    text = ("=== Fold: held-out=stargan ===\n"
            "2026-07-01 10:00:00 | INFO  | PS_2_logreg 0.7300\n"
            "2026-07-01 10:00:00 | INFO  | PS_3_mlp_logits_s7 0.7100\n"
            "2026-07-01 10:00:00 | INFO  | PS_3_mlp_logits 0.7247\n"
            "  Raw-feat LogReg: 0.8529\n"
            "    FS MLP seed-avg: 0.8800 (per-seed std 0.0100)\n")
    for enc, bom in (("utf-16-le", b"\xff\xfe"), ("utf-8", b"")):
        p = tmp_path / f"log_{enc}.log"
        p.write_bytes(bom + text.encode(enc))
        vals = cmp.log_values(p)
        assert vals == {"stargan": {"PS_2_logreg": 0.73, "PS_3_mlp_logits": 0.7247, "raw_feat_auc": 0.8529,
                                    "fs_mlp_mean": 0.88}}, (enc, vals)


def test_gate_g5_on_the_stored_runs_and_its_default_report_path(tmp_path):
    """The stored runs pass G5 with FS1 at 1e-4 and fail it at 1e-9 (the recorded history); run by hand without --out,
    the gate's default report path is the stored record, which the output guard refuses."""
    script = RUNS / "compare_runs_v2.py"
    base = [sys.executable, "-B", str(script), "g5", "--pool", "arch6", "--logo-dir", str(MIT / "content_disjoint"),
            "--mech-dir", str(MIT / "mechanism")]
    r = subprocess.run(base + ["--fs1-tol", "1e-4", "--out", str(tmp_path / "g5.json")], capture_output=True,
                       text=True, env=_env())
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]
    assert json.loads((tmp_path / "g5.json").read_text(encoding="utf-8"))["status"] == "PASS"
    r = subprocess.run(base + ["--out", str(tmp_path / "g5_strict.json")], capture_output=True, text=True, env=_env())
    assert r.returncode == 1 and json.loads((tmp_path / "g5_strict.json").read_text(encoding="utf-8"))["status"] == "FAIL"
    stored = MIT / "mechanism" / "gate_G5_arch6.json"
    before = stored.read_bytes()
    r = subprocess.run(base + ["--fs1-tol", "1e-4"], capture_output=True, text=True, env=_env())
    assert r.returncode != 0 and "among the stored results" in r.stderr, r.stderr[-400:]
    assert stored.read_bytes() == before
