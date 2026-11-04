"""The sensitivity checks (camera_ready/scripts/sensitivity/): the summary rebuilt from the supplied outputs, the
outputs' digests, the two deduplication wrappers on small synthetic data, the reuse rules of paper_heads_logo_v2.py
(complete, consistent and current folds only) and the guard on output folders. Nothing is fitted."""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CR = Path(__file__).resolve().parents[1]
ROOT = CR.parent
SENS = CR / "scripts" / "sensitivity"
SUPPLIED = CR / "output" / "results" / "sensitivity"
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}


def _run(*args, env=None, cwd=ROOT):
    return subprocess.run([sys.executable, "-B", *map(str, args)], cwd=cwd, env=env or ENV, capture_output=True,
                          text=True)


def test_the_summary_is_rebuilt_from_the_supplied_outputs(tmp_path):
    """The current summary is reproduced from the supplied outputs without refitting."""
    copy = tmp_path / "sensitivity"
    shutil.copytree(SUPPLIED, copy)
    (copy / "SUMMARY.md").unlink()
    (copy / "summary.json").unlink()
    r = _run(SENS / "summarize_v2.py", "--dir", copy, "--out-dir", copy)
    assert r.returncode == 0, r.stderr
    assert (copy / "SUMMARY.md").read_text(encoding="utf-8").replace("the input results supplied by --dir", "the outputs in this folder") == (SUPPLIED / "SUMMARY.md").read_text(encoding="utf-8")
    assert (copy / "summary.json").read_text(encoding="utf-8") == (SUPPLIED / "summary.json").read_text(encoding="utf-8")
    s = json.loads((copy / "summary.json").read_text(encoding="utf-8"))
    arch6 = s["logo"]["arch6"]
    assert round(arch6["deduplicated"]["stats"]["fs1_full"]["mean_delta"], 4) == 0.0784
    assert round(arch6["primary"]["stats"]["fs1_full"]["mean_delta"], 4) == 0.0806
    assert round(arch6["deduplicated"]["auprc"]["delta"], 4) == 0.0648
    assert round(arch6["primary"]["auprc"]["delta"], 4) == 0.0803
    twins = s["twins"]
    folds = ("whichisreal", "CollabDiff", "MidJourney", "heygen", "starganv2", "deepfacelab", "stargan", "styleclip")
    assert [twins["logo"]["arch6"][g]["select_rows_with_fit_twin"] for g in folds] == [622, 618, 626, 616, 3, 610,
                                                                                       648, 607]
    assert twins["standard"]["arch6"]["select_rows_with_fit_twin"] == 793
    assert twins["standard"]["clip4"]["select_rows_with_fit_twin"] == 794          # the dataset card's 794
    effects = {k.split(" (")[0]: v for k, v in s["audit"].items()}
    assert round(effects["overlap distortion"]["deduplicated"], 3) == 0.087
    assert round(effects["baseline-ladder distortion, weak6"]["deduplicated"], 3) == 0.096
    assert round(effects["baseline-ladder distortion, weak6"]["paper"], 3) == 0.172


def test_the_supplied_outputs_match_their_recorded_digests():
    prov = json.loads((SUPPLIED / "provenance.json").read_text(encoding="utf-8"))
    files = {p.relative_to(SUPPLIED).as_posix() for p in SUPPLIED.rglob("*") if p.is_file()}
    assert files - set(prov["files"]) == {"PROVENANCE.md", "provenance.json", "SUMMARY.md", "summary.json"}
    for name, digest in prov["files"].items():
        assert hashlib.sha256((SUPPLIED / name).read_bytes()).hexdigest() == digest, name


def test_the_logo_wrapper_drops_training_duplicates(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "logo_common.py").write_text(
        "import hashlib\nimport numpy as np\n"
        "def row_hashes(feats):\n"
        "    block = np.concatenate([np.asarray(f, dtype=np.float32) for f in feats], axis=1)\n"
        "    return [hashlib.sha1(r.tobytes()).hexdigest() for r in block]\n"
        "def fold_view(gen_data, held_out, exclude=None):\n"
        "    f = np.array([[1.0], [2.0], [1.0], [3.0], [2.0]], dtype=np.float32)\n"
        "    y, lg = np.array([0, 1, 0, 1, 1]), np.zeros((5, 1, 2))\n"
        "    return [f], lg, y, [f[:2]], lg[:2], y[:2], 0\n", encoding="utf-8")
    (runs / "fake_runner.py").write_text(
        "import argparse, json\nfrom pathlib import Path\nimport logo_common\n"
        "ap = argparse.ArgumentParser(); ap.add_argument('--pool'); ap.add_argument('--out-dir'); a = ap.parse_args()\n"
        "tr_f, tr_lg, tr_y, te_f, te_lg, te_y, n = logo_common.fold_view({}, 'g1')\n"
        "Path(a.out_dir).mkdir(parents=True, exist_ok=True)\n"
        "(Path(a.out_dir) / 'seen.json').write_text(json.dumps({'train': tr_f[0].ravel().tolist(),\n"
        "    'labels': tr_y.tolist(), 'test': te_f[0].ravel().tolist()}))\n", encoding="utf-8")
    out = tmp_path / "out"
    r = _run(SENS / "dedup_logo_v2.py", "--wrap", runs / "fake_runner.py", "--", "--pool", "p", "--out-dir", out)
    assert r.returncode == 0, r.stderr
    seen = json.loads((out / "seen.json").read_text(encoding="utf-8"))
    assert seen == {"train": [1.0, 2.0, 3.0], "labels": [0, 1, 1], "test": [1.0, 2.0]}       # first copies; test kept
    counts = json.loads((out / "dedup_counts_fake_runner_p.json").read_text(encoding="utf-8"))
    assert counts["folds"] == {"g1": {"training_rows_before": 5, "duplicates_removed": 2, "training_rows_after": 3}}


def test_the_standard_split_wrapper_deduplicates_fit_and_select(tmp_path):
    (tmp_path / "src" / "utils").mkdir(parents=True)
    for p in (tmp_path / "src" / "__init__.py", tmp_path / "src" / "utils" / "__init__.py"):
        p.write_text("", encoding="utf-8")
    (tmp_path / "src" / "utils" / "config.py").write_text("FIT_TAG = 'FIT'\nSELECT_TAG = 'SELECT'\n", encoding="utf-8")
    ana, runs = tmp_path / "scripts" / "analyses", tmp_path / "scripts" / "runs"
    ana.mkdir(parents=True)
    runs.mkdir()
    (runs / "logo_common.py").write_text(
        "import hashlib\nimport numpy as np\n"
        "def row_hashes(feats):\n"
        "    block = np.concatenate([np.asarray(f, dtype=np.float32) for f in feats], axis=1)\n"
        "    return [hashlib.sha1(r.tobytes()).hexdigest() for r in block]\n", encoding="utf-8")
    (ana / "_paths.py").write_text("import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).resolve()"
                                   ".parents[2]))\n", encoding="utf-8")
    (ana / "_common.py").write_text(
        "import numpy as np\nDATA = {'FIT': [1, 2, 1, 3], 'SELECT': [2, 4, 4, 5], 'heygen': [1, 1]}\n"
        "def load_split(expert_names, tag, cache_root):\n"
        "    v = np.array(DATA[tag], dtype=np.float32)[:, None]\n"
        "    return [v], np.zeros((len(v), 1, 2)), np.arange(len(v)) % 2\n", encoding="utf-8")
    (ana / "fake_appB.py").write_text(
        "import argparse, json, os\nfrom pathlib import Path\nimport _paths  # noqa: F401\nimport _common\n"
        "ap = argparse.ArgumentParser(); ap.add_argument('--pool'); ap.add_argument('--xgb_mode')\n"
        "a = ap.parse_args()\n"
        "out = {tag: _common.load_split([], tag, None)[0][0].ravel().tolist() for tag in ('SELECT', 'FIT', 'heygen')}\n"
        "Path(os.environ['CR_RESULTS_DIR'], 'seen.json').write_text(json.dumps(out))\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    r = _run(SENS / "dedup_followup_v2.py", "--wrap-standard", ana / "fake_appB.py", "--", "--pool", "p", "--xgb_mode",
             "cv", env={**ENV, "CR_RESULTS_DIR": str(out)})
    assert r.returncode == 0, r.stderr
    seen = json.loads((out / "seen.json").read_text(encoding="utf-8"))
    # FIT keeps first copies; SELECT loses its row with a FIT twin (2) and its own repeat (4); test unchanged
    assert seen == {"SELECT": [4.0, 5.0], "FIT": [1.0, 2.0, 3.0], "heygen": [1.0, 1.0]}
    counts = json.loads((out / "dedup_counts_standard_p.json").read_text(encoding="utf-8"))["splits"]
    assert counts == {"FIT": {"rows_before": 4, "rows_removed": 1, "rows_after": 3},
                      "SELECT": {"rows_before": 4, "rows_removed": 2, "rows_after": 2}}


def _guard_layout(tmp_path):
    """A temporary copy of the sensitivity scripts with the folders their guard protects, so that a guard that failed
    would write only to the temporary layout."""
    root = tmp_path / "repo"
    shutil.copytree(SENS, root / "camera_ready" / "scripts" / "sensitivity",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for folder in ("camera_ready/output/results/sensitivity/dedup/standard",
                   "camera_ready/output/results/sensitivity/paper_heads",
                   "camera_ready/output/results/mitigation/content_disjoint", "camera_ready/output/results/analyses",
                   "camera_ready/inputs", "output/results", "output/cached_features"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    return root, root / "camera_ready" / "scripts" / "sensitivity"


def test_reruns_never_write_over_the_supplied_outputs(tmp_path):
    root, sens = _guard_layout(tmp_path)
    for script, folder in (("dedup_logo_v2.py", "dedup"), ("dedup_followup_v2.py", "dedup"),
                           ("paper_heads_logo_v2.py", "paper_heads")):
        r = _run(sens / script, "--out-dir", f"camera_ready/output/results/sensitivity/{folder}", cwd=root)
        assert r.returncode != 0 and "supplied outputs" in (r.stderr + r.stdout), script
    assert not any(p.is_file() for p in (root / "camera_ready" / "output").rglob("*"))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"sensitivity_{name}", SENS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_output_guard_refuses_other_spellings_of_protected_folders(tmp_path):
    g = _load("_out_guard")
    refused = ["camera_ready/output/results/sensitivity/dedup",
               "camera_ready/output/results/sensitivity/dedup/../dedup",
               "camera_ready/output/results/sensitivity/./paper_heads/",
               "camera_ready/output/results/sensitivity/dedup/standard",
               "camera_ready/output/results/sensitivity/..",               # a folder that contains the supplied outputs
               "camera_ready/output/results/mitigation/content_disjoint",   # the primary results
               "camera_ready/output/results/analyses", "camera_ready/inputs", "output/results",
               "output/cached_features/heygen", ".",
               str((CR / "output" / "results" / "sensitivity" / "dedup").resolve())]
    if os.name == "nt":
        refused.append("camera_ready\\output\\results\\sensitivity\\dedup")
    for arg in refused:
        with pytest.raises(SystemExit, match="overlaps"):
            g.output_dir(arg)
    variant = "camera_ready/output/results/SENSITIVITY/dedup"
    if (ROOT / "camera_ready/output/results/SENSITIVITY").exists():   # a case-insensitive file system (Windows)
        with pytest.raises(SystemExit, match="overlaps"):
            g.output_dir(variant)
    else:                                                             # here a different folder
        assert g.output_dir(variant) == ROOT / variant
    assert g.output_dir("camera_ready/output/results/sensitivity_rerun/dedup") == \
        ROOT / "camera_ready/output/results/sensitivity_rerun/dedup"
    assert g.output_dir(str(tmp_path / "rerun")) == tmp_path / "rerun"


def test_each_rerun_script_refuses_other_spellings_of_protected_folders(tmp_path):
    root, sens = _guard_layout(tmp_path)
    cases = [("dedup_logo_v2.py", "camera_ready/output/results/sensitivity/dedup/../dedup"),
             ("dedup_logo_v2.py", "camera_ready/output/results/mitigation/content_disjoint"),
             ("dedup_followup_v2.py", "camera_ready/output/results/sensitivity/./dedup"),
             ("paper_heads_logo_v2.py", "camera_ready/output/results/sensitivity/paper_heads/../paper_heads")]
    if os.name == "nt":
        cases.append(("paper_heads_logo_v2.py", "camera_ready/output/results/SENSITIVITY/PAPER_HEADS"))
    for script, out in cases:
        r = _run(sens / script, "--out-dir", out, cwd=root)
        assert r.returncode != 0 and "overlaps" in (r.stderr + r.stdout), (script, out, r.stdout, r.stderr)
    assert not any(p.is_file() for p in (root / "camera_ready" / "output").rglob("*"))


FP = "a" * 64


def _synthetic_fold(ph, out, pool="arch6", fold="heygen", fp=FP):
    """A complete fold, written by the script's own writer from synthetic predictions (no model is fitted)."""
    from sklearn.metrics import roc_auc_score
    y = np.array([0, 1, 0, 1, 0, 1, 1, 0], dtype=np.int8)
    rng = np.random.default_rng(0)
    scores = {name: rng.random(len(y)).astype(np.float32) for name in ph.SCORES}
    auc = {name: float(roc_auc_score(y, s)) for name, s in scores.items()}
    auc["K_cv_selected_LR"], auc["K_cv_selected_XGB"] = auc["I_quadratic_LR"], auc["I_quadratic_XGB"]
    ph.write_fold(out, pool, fold, {"pool": pool, "held_out": fold, "n_test": len(y), "auc": auc,
                                    "k_selected": "I_quadratic", "fit_fingerprint": "fit",
                                    "select_fingerprint": "select"}, scores, y, fp)
    return y


def test_paper_heads_reuses_only_a_complete_and_current_fold(tmp_path):
    ph = _load("paper_heads_logo_v2")
    y = _synthetic_fold(ph, tmp_path)
    assert not list(tmp_path.rglob("*.tmp-*"))                     # writes through renamed temporary files
    fps = {("arch6", "heygen"): FP}
    assert ph.plan(tmp_path, ["arch6"], ["heygen"], fps, truth=lambda g: y) == ([], [("arch6", "heygen")])
    run, reused = ph.plan(tmp_path, ["arch6"], ["heygen"], {("arch6", "heygen"): "b" * 64}, apply=False,
                          truth=lambda g: y)                       # another fingerprint: inputs, code or packages
    assert reused == [] and "fingerprint" in run[0][2]
    run, reused = ph.plan(tmp_path, ["arch6"], ["heygen"], fps, reuse=False, apply=False, truth=lambda g: y)
    assert reused == [] and "--no-reuse" in run[0][2]
    assert sorted(p.name for p in (tmp_path / "arch6").iterdir()) == ["arch6_heygen.json", "arch6_heygen.npz"]
    assert not (tmp_path / "moved_aside").exists()                 # a plan only (apply=False) changes nothing


@pytest.mark.parametrize("damage, reason", [
    ("unreadable record, no predictions", "record unreadable"),     # was reported as done ("0 folds to run")
    ("no predictions", "predictions file missing"),
    ("other predictions", "not the one the record was written with"),
    ("no record", "interrupted"),                                    # stopped between the two writes
    ("temporary file only", "interrupted"),                          # stopped during the first write
    ("record of the earlier version", "without a fingerprint"),
    ("other labels", "labels"),
    ("other AUROC", "do not give the recorded AUROC"),
])
def test_paper_heads_recomputes_an_interrupted_missing_or_stale_fold(tmp_path, damage, reason):
    ph = _load("paper_heads_logo_v2")
    y = _synthetic_fold(ph, tmp_path)
    d = tmp_path / "arch6"
    jp, zp = d / "arch6_heygen.json", d / "arch6_heygen.npz"
    truth = y
    if damage == "unreadable record, no predictions":
        jp.write_text("{", encoding="utf-8")
        zp.unlink()
    elif damage == "no predictions":
        zp.unlink()
    elif damage == "other predictions":
        np.savez_compressed(zp, labels=y, **{n: np.ones(len(y), np.float32) for n in ph.SCORES})
    elif damage == "no record":
        jp.unlink()
    elif damage == "temporary file only":
        jp.unlink()
        zp.rename(d / "arch6_heygen.npz.tmp-123")
    elif damage in ("record of the earlier version", "other AUROC"):
        rec = json.loads(jp.read_text(encoding="utf-8"))
        if damage == "other AUROC":
            rec["auc"]["G_disagreement_LR"] = 0.5 if rec["auc"]["G_disagreement_LR"] != 0.5 else 0.6
        else:
            rec.pop("provenance")
        jp.write_text(json.dumps(rec), encoding="utf-8")
    elif damage == "other labels":
        truth = 1 - y
    before = sorted(p.name for p in d.iterdir())
    run, reused = ph.plan(tmp_path, ["arch6"], ["heygen"], {("arch6", "heygen"): FP}, truth=lambda g: truth)
    assert reused == [] and run[0][:2] == ("arch6", "heygen") and reason in run[0][2], run
    moved = sorted(p.name for p in (tmp_path / "moved_aside").rglob("*") if p.is_file())
    assert set(before) | {"arch6_heygen.reason.txt"} == set(moved)  # moved aside with the reason, nothing deleted
    assert not list(d.iterdir())


def test_paper_heads_moves_aside_temporary_files_next_to_a_complete_fold(tmp_path):
    ph = _load("paper_heads_logo_v2")
    y = _synthetic_fold(ph, tmp_path)
    (tmp_path / "arch6" / "arch6_heygen.json.tmp-999").write_text("{", encoding="utf-8")
    assert ph.plan(tmp_path, ["arch6"], ["heygen"], {("arch6", "heygen"): FP}, truth=lambda g: y) == \
        ([], [("arch6", "heygen")])
    assert not list((tmp_path / "arch6").glob("*.tmp-*")) and list((tmp_path / "moved_aside").rglob("*.tmp-999"))


def test_paper_heads_never_reports_an_unreadable_fold_as_done(tmp_path):
    d = tmp_path / "arch6"
    d.mkdir()
    (d / "arch6_heygen.json").write_text("{", encoding="utf-8")    # an unreadable record, no predictions
    r = _run(SENS / "paper_heads_logo_v2.py", "--pools", "arch6", "--folds", "heygen", "--out-dir", tmp_path, "--plan")
    text = r.stdout + r.stderr
    assert "0 fold(s) to run" not in text
    if (ROOT / "output" / "cached_features" / "heygen" / "labels.npy").exists():
        assert r.returncode == 0 and "1 fold(s) to run" in text and "record unreadable" in text, text
    else:                                                          # a clone without the feature cache stops clearly
        assert r.returncode != 0 and "needs the feature cache" in text, text
    assert (d / "arch6_heygen.json").exists()                     # --plan changes nothing
