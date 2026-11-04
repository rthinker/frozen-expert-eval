"""Check memory reporting, verdict exit codes, three-class resume keys and stored strict-refit statistics."""
import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

CR = Path(__file__).resolve().parents[1]
ROOT = CR.parent
AN = CR / "scripts" / "analyses"
RESULTS = CR / "output" / "results" / "analyses"


def _load(name, mod_name):
    sys.path.insert(0, str(AN))
    spec = importlib.util.spec_from_file_location(mod_name, AN / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_cost_ledger_measures_the_working_set_on_windows():
    ledger = _load("appI3_cost_ledger_v2.py", "cost_v2")
    mem = ledger._mem()
    if sys.platform == "win32":
        assert mem["working_set_gib"] > 0 and mem["peak_working_set_gib"] >= mem["working_set_gib"]
    else:
        assert mem == {"working_set_gib": None, "peak_working_set_gib": None}
    text = (AN / "appI3_cost_ledger_v2.py").read_text(encoding="utf-8")
    assert "under one hour" not in text.split('"""', 2)[2] and "extract_disjoint" not in text


def test_the_power_simulation_exits_1_when_table_a12_does_not_match(tmp_path, monkeypatch):
    power = _load("appF_power_mde_v2.py", "power_v2")
    monkeypatch.setattr(power, "PAPER", {0.05: 0.9, 0.08: 0.9, 0.12: 0.9})     # values no simulation gives
    monkeypatch.setattr(power._paths, "LOGS_DIR", tmp_path / "logs")          # its log, too, outside the repository
    out = tmp_path / "power.json"
    monkeypatch.setattr(sys, "argv", ["appF_power_mde_v2.py", "--sets", "200", "--precision-sets", "200",
                                      "--out", str(out)])
    assert power.main() == 1
    assert json.loads(out.read_text(encoding="utf-8"))["all_match_paper"] is False


@pytest.mark.parametrize("name,verdict", [
    ("appI1_k3_build_v2.py", "return 0 if ok else 1"),
    ("appI1_k3_freeze_v2.py", 'return 0 if all(v["ok"] for v in out_all.values()) else 1'),
    ("appI1_k3_exposure_audit_v2.py", 'return 0 if all(v["uniform"] for v in verdicts.values()) else 1'),
    ("appF_strict_refit_claims_v2.py", "return 0 if claim_a else 1"),
    ("appF_power_mde_v2.py", 'return 0 if res["all_match_paper"] else 1'),
])
def test_a_negative_verdict_gives_a_nonzero_exit_status(name, verdict):
    """These checks need DF40's lists or the feature cache to run; their structure is checked here: main returns the
    status from the verdict, and the script exits with it."""
    tree = ast.parse((AN / name).read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert ast.dump(main.body[-1]) == ast.dump(ast.parse(verdict.replace("return ", "def f():\n    return ", 1))
                                               .body[0].body[0]), ast.unparse(main.body[-1])
    tail = tree.body[-1]
    assert isinstance(tail, ast.If) and ast.unparse(tail.body[0]) == "raise SystemExit(main())"


def test_a_three_class_resume_reuses_only_caches_written_for_this_run(tmp_path):
    aligned = _load("appI1_native_k_aligned_v2.py", "aligned_v2")
    key = {"view": "B_arch6", "manifest_sha256": "dd13", "inner_assignment_sha256": "ab", "device": "cuda",
           "smoke": False}
    aligned.CACHE_KEY.clear()
    aligned.CACHE_KEY.update(key)
    arr, meta = tmp_path / "c.npz", tmp_path / "c.json"
    logged = []
    assert not aligned._reusable(arr, meta, logged.append)              # no files
    np.savez(arr, select=np.zeros(2), test=np.zeros(2))
    meta.write_text(json.dumps({"seed": 7}), encoding="utf-8")
    assert not aligned._reusable(arr, meta, logged.append)              # cache metadata without a key
    meta.write_text(json.dumps({"seed": 7, "cache_key": {**key, "device": "cpu"}}), encoding="utf-8")
    assert not aligned._reusable(arr, meta, logged.append)              # another device
    meta.write_text(json.dumps({"seed": 7, "cache_key": key}), encoding="utf-8")
    assert aligned._reusable(arr, meta, logged.append)
    assert len(logged) == 2 and all("not reused" in m for m in logged)
    aligned.CACHE_KEY.clear()


def test_the_appendix_f_statistics_reproduce_the_stored_strict_refit():
    """The generator bootstrap, the BCa interval and the exact sign-flip test of _common.py, recomputed from the stored
    per-generator deltas of both arms (Table A14), equal the values stored with them."""
    common = _load("_common.py", "common_stats")
    stored = json.loads((RESULTS / "run_b_path_disjoint_arch6_family.json").read_text(encoding="utf-8"))
    for arm in ("arm_full", "arm_strict"):
        c = stored[arm]["comparison"]
        d = np.array([c["per_generator_delta"][g] for g in stored["generators"]])
        boot = common.generator_bootstrap_ci(d)
        assert boot["ci_95"] == pytest.approx(c["generator_bootstrap"]["ci_95"], abs=1e-12), arm
        assert boot["p_gt_zero"] == c["generator_bootstrap"]["p_gt_zero"]
        bca = common.bca_ci(d)
        assert bca["bca_95"] == pytest.approx(c["bca"]["bca_95"], abs=1e-12)
        assert bca["z0"] == pytest.approx(c["bca"]["z0"], abs=1e-12)
        flip = common.exact_sign_flip_p(d)
        for k, v in c["sign_flip_permutation"].items():
            if isinstance(v, float):
                assert flip[k] == pytest.approx(v, abs=1e-12), (arm, k)
