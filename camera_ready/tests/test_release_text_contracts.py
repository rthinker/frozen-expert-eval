"""Statements in the guides and the dataset card, checked against the release files. No model is fitted."""
import ast
import csv
import hashlib
import json
import re
import tomllib
from pathlib import Path

import pytest

CR = Path(__file__).resolve().parents[1]
ROOT = CR.parent
ANALYSES = CR / "scripts" / "analyses"
CACHE = ROOT / "output" / "cached_features"
STORED = CR / "output" / "results" / "analyses"
SPLITS = ["FIT", "SELECT", "whichisreal", "CollabDiff", "MidJourney", "heygen", "starganv2", "deepfacelab", "stargan",
          "styleclip"]


def _read(path):
    return Path(path).read_bytes().decode("utf-8").replace("\r\n", "\n")


def _flat(text):
    return " ".join(text.split())


def _ids(split):
    return _read(CR / "output" / "ids" / f"{split}.txt").splitlines()


def test_default_discovery_includes_both_test_folders():
    assert re.search(r"^testpaths = tests camera_ready/tests$", _read(ROOT / "pytest.ini"), re.M)


def test_package_metadata_matches_the_documented_python():
    project = tomllib.loads(_read(ROOT / "pyproject.toml"))["project"]
    assert project["requires-python"] == ">=3.11,<3.12"
    assert "Programming Language :: Python :: 3.12" not in project["classifiers"]
    assert "submission" not in project["description"]


@pytest.mark.parametrize("rel", ["README.md", "camera_ready/INSTALL.md", "output/cached_features/DATASET_CARD.md",
                                 "scripts/review/validate_cache_release.py", "scripts/review/reproduce_all.py"])
def test_the_download_fetches_only_the_split_folders(rel):
    text = _read(ROOT / rel)
    assert '--include "*/*"' in text and "--exclude" not in text


def test_the_three_class_protocol_section_exists_and_names_current_scripts():
    text = _read(ANALYSES / "README.md")
    section = text[text.index("\n## Three-class protocol\n"):]
    for script in ("appI1_k3_fit.py", "appI1_native_k_preflight.py", "appI1_native_k_heads.py",
                   "appI1_native_k_aligned_v2.py"):
        assert script in section and (ANALYSES / script).is_file()


def test_the_protocol_section_states_the_settings_in_the_code():
    section = _flat(_read(ANALYSES / "README.md"))
    fit, heads = _read(ANALYSES / "appI1_k3_fit.py"), _read(ANALYSES / "appI1_native_k_heads.py")
    assert 'LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")' in fit
    assert 'LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")' in section
    assert "HEAD_MAX_ITER = 500" in heads and "HEAD_TOL = 1e-3" in heads
    assert "max_iter=500" in section and "tol=1e-3" in section
    assert "N_PERM = 20" in fit and "20 permutations by default" in section


def test_scripts_cite_no_gone_script_or_unshipped_design():
    gone = ("17_k3_exposure_audit", "18_k3_freeze_manifests", "19_k3_fit", "22_native_k_class_preflight",
            "23_native_k_class_fit", "24_native_k_paper_aligned", "design doc", "design section", "rule 5")
    for path in ANALYSES.glob("appI1_*.py"):
        text = _read(path)
        assert not [g for g in gone if g in text], path.name


def test_scripts_and_the_stored_bit_identity_record_hold_no_internal_remarks():
    remarks = ("is " + "lost)", "author " + "responses", "authors' " + "workspace", "the " + "rebuttal",
               "03.camera-ready/" + "code;", "of the " + "submission")
    paths = [*ANALYSES.glob("*.py"), *(CR / "scripts" / "runs").glob("*.py"), CR / "inputs" / "verify_bit_identity.json",
             STORED / "verify_bit_identity.json"]
    for path in paths:
        text = _read(path)
        assert not [r for r in remarks if r in text], path.name


def test_direct_run_inputs_named_by_the_readme_are_supplied():
    text = _flat(_read(ANALYSES / "README.md"))
    for pattern in ("three_class/K3_folds_df40.json", "three_class/K3_FROZEN_*.json",
                    "three_class/NATIVE_K_PREFLIGHT_*.json", "run_b_*.json", "run_d_*.json"):
        assert f"`{pattern}`" in text
        assert list(STORED.glob(pattern)), pattern


def test_the_manifest_explains_which_configuration_the_ledger_hashes():
    with open(CR / "provenance" / "05_number_provenance.csv", encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["source"] == "configs/experts.yaml"]
    archived = hashlib.sha256((CR / "inputs" / "experts.yaml").read_bytes()).hexdigest()[:16]
    current = hashlib.sha256((ROOT / "configs" / "experts.yaml").read_bytes()).hexdigest()[:16]
    assert len(rows) == 49 and {r["sha256_16"] for r in rows} == {archived} and archived != current
    manifest = _read(CR / "RELEASE_MANIFEST.md")
    assert "The 49 ledger rows that name `configs/experts.yaml`" in manifest
    assert f"`{archived}`" in manifest and f"`{current}`" in manifest
    assert "standard split (Table A2) instead" not in manifest


def test_the_sensitivity_rerun_documents_name_block_b():
    assert "--blocks B" in _read(CR / "output" / "results" / "sensitivity" / "PROVENANCE.md")
    assert "--blocks B" in _read(CR / "scripts" / "sensitivity" / "README.md")
    assert "rerun Block B" in _read(CR / "scripts" / "authoring" / "generate_camera_ready_tables_v18.py")


def test_the_sensitivity_guide_keeps_the_gpu_option_optional():
    readme = _read(CR / "scripts" / "sensitivity" / "README.md")
    assert "--gpu-name 5080" not in readme and "`--gpu-name <name>`" in readme
    assert "The supplied primary runs record none, so run Block B first." in _flat(readme)


def test_the_exactness_note_states_a_threshold_not_a_result():
    guide = _flat(_read(CR / "REPRODUCTION.md"))
    assert "deterministic methods reproduce within 1e-4" not in guide
    assert "This is a comparison threshold, not a measured reproduction result" in guide
    assert "FS1 solver iterations: <n>/2000" in guide
    assert "This guide, including" in guide and "This README" not in guide
    manifest = _flat(_read(CR / "RELEASE_MANIFEST.md"))
    assert "on the tested Windows configuration" not in manifest
    assert "These are comparison thresholds, not measured reproduction results" in manifest
    assert "1e-3 for the three-class heads on fixed outputs" in manifest


@pytest.mark.parametrize("script", ["04_prediction_space", "06_bootstrap_rawfeat", "11_disjoint_data", "13_fs1_standardized",
                                    "14_merged_pool", "15_auprc_appendix"])
def test_every_block_a_fs1_fit_logs_its_solver_iterations(script):
    tree = ast.parse(_read(ROOT / "scripts" / f"{script}.py"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "FS_1"]
    assert calls
    assert all(any(keyword.arg == "logger" for keyword in call.keywords) for call in calls)


def test_the_licences_credit_the_xception_file_the_setup_downloads():
    assert "cadene/pretrainedmodels/xception-b5690688.pth" in _read(ROOT / "scripts" / "00_setup.py")
    licences = _flat(_read(CR / "LICENSES.md"))
    assert "`scripts/00_setup.py` also downloads the ImageNet Xception file" in licences
    assert "[pretrained-models](https://github.com/Cadene/pretrained-models.pytorch) (BSD-3-Clause)" in licences


def test_the_card_states_the_training_path_forms():
    rows = _ids("FIT") + _ids("SELECT")
    frames = sum(1 for r in rows if re.match(r"[^/]+/ff/frames/", r))
    numbered = sum(1 for r in rows if re.match(r"[^/]+/ff/\d+/", r))
    faceforensics = sum(1 for r in rows if r.startswith("FaceForensics++/"))
    with_class = sum(1 for r in rows if r.split("/")[1] in ("real", "fake"))
    assert (frames, numbered, faceforensics) == (14993, 5640, 719)
    assert frames + numbered + faceforensics + with_class == len(rows)
    card = _flat(_read(CACHE / "DATASET_CARD.md"))
    assert "14,993 FS and FR training rows" in card
    assert "5,640 entire-face-synthesis training rows use `<method>/ff/<n>/...`" in card
    assert "719 real training rows use `FaceForensics++/...`" in card
    assert "These 21,352 rows" in card and frames + numbered + faceforensics == 21352


def test_the_card_counts_the_youtube_style_identifiers():
    count = 0
    for split in SPLITS:
        for row in _ids(split):
            parts = row.split("/")
            if parts[0] in ("sadtalker", "wav2lip", "heygen"):
                folder = parts[-2]
                if "##" in folder or re.search(r"(^|_)[A-Za-z0-9_-]{11}_\d{5}", folder) or "Clip+" in row:
                    count += 1
    assert count == 1530
    assert "YouTube-style video identifiers (1,530 rows)" in _flat(_read(CACHE / "DATASET_CARD.md"))


def test_the_dataset_documents_name_the_identifier_sanitization():
    card = _flat(_read(CACHE / "DATASET_CARD.md"))
    generated = json.loads(_read(CACHE / "croissant.json"))["prov:wasGeneratedBy"]
    assert "sample_ids_sanitized_at" in card and "sample_ids_sanitized_at" in json.dumps(generated)
    assert "2026-04-29" in card and "2026-04-09" in card and "generated 2026-05-07" not in card
    assert "recomputes the reuse and duplicate counts" in card


def test_the_validation_record_uses_public_wording():
    text = _read(CACHE / "CROISSANT_VALIDATION.txt")
    assert "staged" not in text and "retained source audit" not in text
