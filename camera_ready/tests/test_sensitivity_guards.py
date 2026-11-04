"""Check that the twin counter and summary report keep supplied outputs safe."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

CR = Path(__file__).resolve().parents[1]
SENS = CR / "scripts" / "sensitivity"


def _run(script, *args):
    return subprocess.run([sys.executable, "-B", str(SENS / script), *args], capture_output=True, text=True)


def test_count_twins_refuses_every_spelling_of_a_protected_file():
    for out in ("camera_ready/output/results/sensitivity/twins.json",
                "camera_ready/output/results/sensitivity/../sensitivity/twins.json",
                "camera_ready/output/results/sensitivity/dedup/../twins.json",
                "camera_ready/output/results/mitigation/mitigation_stats.json",
                "output/results/04_ds_arch6.json",
                str(CR / "output" / "results" / "sensitivity" / "twins.json")):
        r = _run("count_twins.py", "--out", out)
        assert r.returncode != 0 and "overlaps" in r.stderr, (out, r.stderr[-300:])


def test_count_twins_refuses_another_case_of_a_protected_folder_on_a_case_insensitive_file_system():
    """On Windows, `.../SENSITIVITY/twins.json` names the supplied counts. The
    guard compares resolved, case-folded paths and file identities, so it is refused; on a case-sensitive file
    system the spelling is another folder, which the guard does not protect, so the test does not run there (the
    script would start a rerun)."""
    variants = ["camera_ready/output/results/SENSITIVITY/twins.json",
                "Camera_Ready/Output/Results/Sensitivity/twins.json",
                str(CR / "output" / "results" / "SENSITIVITY" / "twins.json")]
    if not (CR / "output" / "results" / "SENSITIVITY").exists():
        import pytest
        pytest.skip("case-sensitive file system: these spellings name other folders")
    for out in variants:
        r = _run("count_twins.py", "--out", out)
        assert r.returncode != 0 and "overlaps" in r.stderr, (out, r.stderr[-300:])


def test_summarize_writes_only_to_the_supplied_outputs_or_an_unprotected_folder(tmp_path):
    for d in ("camera_ready/output/results/mitigation", "camera_ready/output/results/sensitivity/dedup",
              "camera_ready/inputs", "output/results"):
        r = _run("summarize_v2.py", "--out-dir", str(CR.parent / d))
        assert r.returncode != 0 and "among the stored results" in r.stderr, (d, r.stderr[-300:])
    copy = tmp_path / "sensitivity"                       # a copy of the supplied outputs elsewhere: allowed
    shutil.copytree(CR / "output" / "results" / "sensitivity", copy)
    r = _run("summarize_v2.py", "--dir", str(copy), "--out-dir", str(copy))
    assert r.returncode == 0, r.stderr[-500:]
    assert json.loads((copy / "summary.json").read_text(encoding="utf-8")) == json.loads(
        (CR / "output" / "results" / "sensitivity" / "summary.json").read_text(encoding="utf-8"))
