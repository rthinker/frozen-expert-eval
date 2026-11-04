"""Check that split reuse compares freshly computed counts and rejects image roots with missing source files.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

CR = Path(__file__).resolve().parents[1]
SCRIPT = CR / "scripts" / "review" / "check_split_reuse_v3.py"


def _load():
    spec = importlib.util.spec_from_file_location("split_reuse_v3", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cache(root, n=4):
    m = _load()
    rng = np.random.default_rng(0)
    for split in m.SPLITS:
        d = root / split
        d.mkdir(parents=True)
        np.save(d / "labels.npy", np.array([0, 1] * (n // 2), dtype=np.int64))
        for e in [m.CONTENT_EXPERT] + m.ARCH6:
            np.save(d / f"{e}_features.npy", rng.normal(size=(n, 3)).astype(np.float16))
            np.save(d / f"{e}_logits.npy", rng.normal(size=(n, 2)).astype(np.float32))
    return m


def test_carried_over_routes_are_not_compared_with_the_expected_counts(tmp_path):
    _cache(tmp_path / "cache")
    stored = {"_inputs": {"DF40_all_json_sha256": "abc"},
              "route_a": {"per_generator": {}, "total_test_paths": 31485, "in_DF40_all": 23892, "pct": 75.9},
              "route_b": {"raw_files": {"missing_total": 0, "zero_byte_total": 17}}}
    (tmp_path / "stored.json").write_text(json.dumps(stored), encoding="utf-8")
    expect = {"route_a": {"total_test_paths": 31485, "in_DF40_all": 23892, "pct": 75.9},
              "route_b": {"raw_files": {"zero_byte_total": 17}}}
    (tmp_path / "expect.json").write_text(json.dumps(expect), encoding="utf-8")
    cmd = [sys.executable, "-B", str(SCRIPT), "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "out.json"),
           "--carry-from", str(tmp_path / "stored.json"), "--expect", str(tmp_path / "expect.json")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "0 match, 0 differ; 4 carried over from the stored result, not compared" in r.stdout, r.stdout
    assert "(carried over from the stored result, not recomputed)" in r.stdout
    res = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert res["route_a"] == stored["route_a"] and res["_carried_over"]["routes"] == ["route_a", "route_b.raw_files"]


def test_any_missing_source_file_stops_the_check():
    m = _load()
    assert m.layout_problem({"missing_total": 0}, 51163) is None
    for missing in (1, 21352, 51163):
        why = m.layout_problem({"missing_total": missing}, 51163)
        assert why and why.startswith(f"{missing:,} of 51,163 cached rows") and "Nothing was written" in why
