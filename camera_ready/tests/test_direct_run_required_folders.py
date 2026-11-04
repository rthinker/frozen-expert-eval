"""A by-hand analysis on a partial cache stops and names what is missing. No model is fitted."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CR = Path(__file__).resolve().parents[1]
ANALYSES = CR / "scripts" / "analyses"
GENERATORS = ["whichisreal", "CollabDiff", "MidJourney", "heygen", "starganv2", "deepfacelab", "stargan", "styleclip"]

# Runs the script's main() with the cache helpers replaced, so only the folders matter.
DRIVER = """
import importlib.util
import sys
from pathlib import Path
analyses, script, cache, lists = map(Path, sys.argv[1:5])
sys.argv = [script.name, *sys.argv[5:]]
sys.path.insert(0, str(analyses))
import _paths
_paths.df40_json_dir = lambda *a, **k: lists
import _verify_cache_helpers as helpers
helpers._hash_expert = lambda: "expert"
helpers._load_ids = lambda path: ["row"]
helpers._content_hashes = lambda folder, expert: ["hash"]
spec = importlib.util.spec_from_file_location(script.stem, script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.CACHE_ROOT = cache
raise SystemExit(module.main())
"""

LEDGER_DRIVER = """
import importlib.util
import sys
from pathlib import Path
analyses, cache = map(Path, sys.argv[1:3])
sys.argv = ["appI3_cost_ledger_v2.py"]
sys.path.insert(0, str(analyses))
spec = importlib.util.spec_from_file_location("cost_ledger", analyses / "appI3_cost_ledger_v2.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.measure_widths_and_storage(cache, ["x"], lambda message: None)
assert result["storage"]["pool_features_bytes"] > 0
"""


def _env(results):
    env = {k: v for k, v in os.environ.items() if k not in ("REPRO_ATTEMPT_ID", "DF40_JSON_DIR")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CR_RESULTS_DIR=str(results))
    return env


def _generator_folders(cache, names):
    for name in names:
        (cache / name).mkdir(parents=True)
        np.save(cache / name / "labels.npy", np.zeros(1, dtype=np.int32))


def _run(tmp_path, script, *args):
    lists = tmp_path / "lists"
    lists.mkdir(exist_ok=True)
    for name in ("FSAll_ff", "FRAll_ff", "EFSAll_ff"):
        (lists / f"{name}.json").write_text("{}", encoding="utf-8")
    frozen = tmp_path / "results" / "three_class" / "K3_FROZEN_A_df40.json"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps({"manifest": [], "manifest_sha256": hashlib.sha256(b"").hexdigest(), "pool": ["x"],
                                  "classes": ["a"], "n_folds": 0}), encoding="utf-8")
    return subprocess.run([sys.executable, "-B", "-c", DRIVER, str(ANALYSES), str(ANALYSES / script),
                           str(tmp_path / "cache"), str(lists), *args],
                          capture_output=True, text=True, env=_env(tmp_path / "results"), cwd=str(tmp_path))


CASES = [("appI1_k3_build_v2.py", ()), ("appI1_k3_freeze_v2.py", ()), ("appI1_k3_fit.py", ("--view", "A_df40")),
         ("appA_content_reuse.py", ())]


@pytest.mark.parametrize("script,args", CASES)
def test_a_missing_held_out_generator_folder_stops_the_run_and_is_named(tmp_path, script, args):
    _generator_folders(tmp_path / "cache", GENERATORS[:-1])
    r = _run(tmp_path, script, *args)
    assert r.returncode != 0
    assert "missing held-out generator folder" in r.stderr and "styleclip" in r.stderr, r.stderr[-500:]


@pytest.mark.parametrize("script,args", CASES)
def test_all_eight_held_out_generator_folders_pass_that_check(tmp_path, script, args):
    _generator_folders(tmp_path / "cache", GENERATORS)
    r = _run(tmp_path, script, *args)
    assert "missing held-out generator folder" not in r.stderr, r.stderr[-500:]


def test_the_cost_ledger_stops_on_a_missing_cache_file_instead_of_undercounting(tmp_path):
    cache = tmp_path / "cache"

    def write(split):
        (cache / split).mkdir(parents=True)
        np.save(cache / split / "x_features.npy", np.zeros((1, 2), dtype=np.float16))
        np.save(cache / split / "x_logits.npy", np.zeros((1, 2), dtype=np.float32))

    def run():
        return subprocess.run([sys.executable, "-B", "-c", LEDGER_DRIVER, str(ANALYSES), str(cache)], capture_output=True,
                              text=True, env=_env(tmp_path / "results"), cwd=str(tmp_path))

    write("FIT")
    r = run()
    assert r.returncode != 0 and "missing cache file" in r.stderr and "SELECT" in r.stderr, r.stderr[-500:]
    for split in ["SELECT", *GENERATORS]:
        write(split)
    r = run()
    assert r.returncode == 0, r.stderr[-500:]
