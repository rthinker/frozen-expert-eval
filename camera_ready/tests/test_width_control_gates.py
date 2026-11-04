"""Gate G4 of the width-control runner, version 5 (scripts/runs/05_run_c_width_matched_v5.py): portable references
and the completeness check. No feature cache needed; nothing is fitted."""
import importlib.util
import json
from pathlib import Path

CR = Path(__file__).resolve().parents[1]
ROOT = CR.parent


def _runner():
    spec = importlib.util.spec_from_file_location("run_c_width_controls", CR / "scripts" / "runs" / "05_run_c_width_matched_v5.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(folder, name, data):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(data), encoding="utf-8")


METHODS = ["PS_4_xgboost", "FS_pca1_logreg", "FS_pca2_logreg", "FS_pca4_logreg", "FS_pca8_logreg", "FS_probe2_logreg",
           "FS_D_xgb_features", "FS_1_raw_feat_logreg"]


def test_the_shipped_references_give_the_full_gate_in_the_repository():
    """In the repository layout, the default references reproduce the stored arch6 gate: all 8 comparisons, the same
    stored values."""
    rc = _runner()
    assert rc.default_reference_dir() == CR / "inputs"
    stored = json.loads((CR / "output" / "results" / "mitigation" / "mechanism" / "run_c_v2_standard_arch6.json")
                        .read_text(encoding="utf-8"))
    gate, references, missing = rc.g4_gate("arch6", stored["macro"], [1, 2, 4, 8], [2], False, False,
                                           rc.default_reference_dir(), ROOT / "output" / "results")
    assert missing == [] and sorted(gate) == sorted(METHODS) and sorted(gate) == sorted(stored["gate_G4"])
    assert all(abs(gate[k]["stored"] - stored["gate_G4"][k]["stored"]) < 1e-12 for k in gate)
    assert all(g["status"] == "PASS" for g in gate.values()) and set(references) == {"run_c", "FS_D_xgb_features",
                                                                                     "FS_1_raw_feat_logreg"}


def test_a_missing_reference_is_reported(tmp_path):
    rc = _runner()
    macro = {k: 0.9 for k in METHODS}
    results = tmp_path / "results"
    _write(results, "12_xgb_features_arch6.json", {"avg_auc_8": 0.9})
    _write(results, "04_ds_arch6_rawfeat.json", {"avg_auc_8": 0.9})
    _write(results, "12_xgb_features_clip4.json", {"avg_auc_8": 0.9})
    _write(results, "04_ds_clip4_rawfeat.json", {"skipped": True, "reason": "memory_budget"})
    empty = tmp_path / "references"
    empty.mkdir()
    # arch6 has a reviewed run C, so the gate requires that reference.
    gate, _, missing = rc.g4_gate("arch6", macro, [1, 2, 4, 8], [2], False, False, empty, results)
    assert sorted(gate) == ["FS_1_raw_feat_logreg", "FS_D_xgb_features"] and len(missing) == 1
    assert missing[0].startswith("reviewed run C")
    # clip4 had none, so its gate is complete with FS_D and FS1; a skip marker in place of FS1's value is missing
    gate, _, missing = rc.g4_gate("clip4", macro, [1, 2, 4, 8], [2], False, False, empty, results)
    assert sorted(gate) == ["FS_D_xgb_features"] and len(missing) == 1 and missing[0].startswith("FS_1_raw_feat_logreg")


def test_a_missing_field_or_value_in_an_existing_reference_is_reported(tmp_path):
    """G4 requires every computed method to have a reference and every expected method to have a value."""
    rc = _runner()
    stored = json.loads((CR / "output" / "results" / "mitigation" / "mechanism" / "run_c_v2_standard_arch6.json")
                        .read_text(encoding="utf-8"))
    reviewed = json.loads((CR / "inputs" / "run_c_width_matched_arch6.json").read_text(encoding="utf-8"))
    del reviewed["pca_sweep"]["d2"]
    refs = tmp_path / "references"
    _write(refs, "run_c_width_matched_arch6.json", reviewed)
    gate, _, missing = rc.g4_gate("arch6", stored["macro"], [1, 2, 4, 8], [2], False, False, refs,
                                  ROOT / "output" / "results")
    assert "FS_pca2_logreg" not in gate and len(gate) == 7
    assert len(missing) == 1 and missing[0].startswith("FS_pca2_logreg (pca_sweep.d2.fs_macro")
    # the full reference, but a run that computed no FS_pca8_logreg
    macro = {k: v for k, v in stored["macro"].items() if k != "FS_pca8_logreg"}
    gate, _, missing = rc.g4_gate("arch6", macro, [1, 2, 4, 8], [2], False, False, rc.default_reference_dir(),
                                  ROOT / "output" / "results")
    assert missing == ["FS_pca8_logreg (reviewed value present, but this run computed none)"]
    # --skip-probe expects no probe field
    gate, _, missing = rc.g4_gate("arch6", stored["macro"], [1, 2, 4, 8], [2], True, False,
                                  rc.default_reference_dir(), ROOT / "output" / "results")
    assert missing == [] and "FS_probe2_logreg" not in gate
