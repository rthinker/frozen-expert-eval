"""Fold arguments and no-write dry runs of the current LOGO runners."""
import importlib.util
import sys
from pathlib import Path

import pytest

CR = Path(__file__).resolve().parents[1]
ROOT = CR.parent
RUNS = CR / "scripts" / "runs"
GENS = ["whichisreal", "CollabDiff", "MidJourney", "heygen", "starganv2", "deepfacelab", "stargan", "styleclip"]


class _Stop(Exception):
    """Raised by the replaced cache loader: everything before the loading has run."""


def _load(name):
    sys.path.insert(0, str(RUNS))
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").replace("0", "r0", 1), RUNS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("script", ["11_disjoint_data_v4.py", "05_run_c_width_matched_v5.py",
                                    "08_logo_extended_rungs_v3.py"])
def test_folds_must_be_held_out_generators_each_once(script):
    mod = _load(script)
    assert mod._fold_problem(GENS, GENS) is None
    assert mod._fold_problem(["stargan", "styleclip"], GENS) is None
    why = mod._fold_problem(GENS[:7] + ["stargan"], GENS)
    assert why == "given more than once: stargan"
    why = mod._fold_problem(["StarGAN"], GENS)
    assert why.startswith("not a held-out generator: StarGAN (choose from whichisreal")


def _no_writes(monkeypatch, mod):
    """Every way the runners write (the output folder, its log) fails the test; the cache loader stops the run."""
    def refuse(*a, **k):
        raise AssertionError("a dry run asked for an output path or a log file")
    monkeypatch.setattr(mod._cr_paths, "out_path", refuse)
    monkeypatch.setattr(mod._cr_paths, "make_logger", refuse)
    import logo_common

    def stop(*a, **k):
        raise _Stop
    monkeypatch.setattr(logo_common, "load_generators", stop)


def test_a_dry_run_of_the_logo_runner_writes_nothing(tmp_path, monkeypatch):
    mod = _load("11_disjoint_data_v4.py")
    out = tmp_path / "new_folder"
    _no_writes(monkeypatch, mod)
    monkeypatch.setattr(sys, "argv", ["11_disjoint_data_v4.py", "--codebase", str(ROOT), "--pool", "arch6",
                                      "--exclusion", "none", "--out-dir", str(out), "--dry-run"])
    with pytest.raises(_Stop):
        mod.main()
    assert not out.exists()


def test_the_logo_runner_refuses_repeated_folds_before_loading_anything(tmp_path, monkeypatch):
    mod = _load("11_disjoint_data_v4.py")
    _no_writes(monkeypatch, mod)
    monkeypatch.setattr(sys, "argv", ["11_disjoint_data_v4.py", "--codebase", str(ROOT), "--pool", "arch6",
                                      "--exclusion", "none", "--out-dir", str(tmp_path / "o"), "--dry-run",
                                      "--folds", "stargan", "stargan"])
    with pytest.raises(SystemExit, match="given more than once: stargan"):
        mod.main()


def test_the_richer_input_runner_checks_its_folds_before_loading_anything(tmp_path, monkeypatch):
    mod = _load("08_logo_extended_rungs_v3.py")
    _no_writes(monkeypatch, mod)
    base = ["08_logo_extended_rungs_v3.py", "--codebase", str(ROOT), "--pool", "arch6", "--exclusion", "none",
            "--out-dir", str(tmp_path / "o"), "--dry-run", "--folds"]
    for folds, message in ((["stargan", "stargan"], "given more than once: stargan"),
                           (["nonesuch"], "not a held-out generator: nonesuch")):
        monkeypatch.setattr(sys, "argv", base + folds)
        with pytest.raises(SystemExit, match=message):
            mod.main()
    monkeypatch.setattr(sys, "argv", base + ["stargan", "styleclip"])
    with pytest.raises(_Stop):
        mod.main()


def test_a_dry_run_of_the_width_controls_writes_nothing_and_checks_its_folds(tmp_path, monkeypatch):
    mod = _load("05_run_c_width_matched_v5.py")
    out = tmp_path / "new_folder"
    _no_writes(monkeypatch, mod)
    monkeypatch.setattr(sys, "argv", ["05_run_c_width_matched_v5.py", "--codebase", str(ROOT), "--pool", "arch6",
                                      "--protocol", "logo", "--out-dir", str(out), "--dry-run"])
    with pytest.raises(_Stop):
        mod.main()
    assert not out.exists()
    import logo_common
    monkeypatch.setattr(logo_common, "load_generators", lambda *a, **k: {})
    monkeypatch.setattr(sys, "argv", ["05_run_c_width_matched_v5.py", "--codebase", str(ROOT), "--pool", "arch6",
                                      "--protocol", "logo", "--out-dir", str(out), "--dry-run",
                                      "--folds", "stargan", "nonesuch"])
    with pytest.raises(SystemExit, match="not a held-out generator: nonesuch"):
        mod.main()
