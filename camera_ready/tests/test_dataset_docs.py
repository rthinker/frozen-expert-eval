"""Check the packaged cache documentation and its independent code-owned trust anchor."""
import hashlib
import json
import re
from pathlib import Path

CR = Path(__file__).resolve().parents[1]
TRUSTED = CR / "dataset"
CACHE = CR.parent / "output" / "cached_features"
HUB = "https://huggingface.co/datasets/anon-frozen-experts/df40-frozen-experts-anon"
REVISION = "18fb1b212f7251799fca8cb4d549be6d239ce13a"


def test_code_owned_dataset_directory_contains_only_the_trust_anchor():
    assert {path.name for path in TRUSTED.iterdir()} == {"CACHE_PROVENANCE.json"}
    assert (TRUSTED / "CACHE_PROVENANCE.json").is_file()


def test_croissant_describes_the_reviewed_files():
    path = CACHE / "croissant.json"
    data = path.read_bytes()
    doc = json.loads(data)
    checksums = dict(line.split("  ", 1)[::-1] for line in (CACHE / "CHECKSUMS.txt").read_text(encoding="utf-8").splitlines())
    distribution = doc["distribution"]
    assert len(distribution) == len(checksums) == 460
    assert checksums == {entry["@id"]: entry["sha256"] for entry in distribution}
    assert "LICENSE_NOTICE.txt" not in checksums
    assert all(entry["contentUrl"] == f"{HUB}/resolve/{REVISION}/{entry['@id']}" for entry in distribution)
    assert doc["url"] == HUB
    assert doc["license"].rstrip("/").endswith("by-nc-sa/4.0")
    assert any(key.startswith("rai:") for key in doc)
    annotations = next(record for record in doc["recordSet"] if record["@id"] == "records/source_integrity_exclusions")
    assert len(annotations["data"]) == 17

    validation = (CACHE / "CROISSANT_VALIDATION.txt").read_text(encoding="utf-8")
    assert f"File: croissant.json (SHA-256 {hashlib.sha256(data).hexdigest()})" in validation
    assert "METADATA / SCHEMA: PASSED" in validation


def test_card_points_to_the_unchanged_hub_release():
    card = (CACHE / "DATASET_CARD.md").read_text(encoding="utf-8")
    contract = json.loads((TRUSTED / "CACHE_PROVENANCE.json").read_bytes())
    croissant = json.loads((CACHE / "croissant.json").read_bytes())
    stated = re.search(r"Documentation version\s+(\d+\.\d+\.\d+);", card)
    assert stated is not None
    assert stated.group(1) == contract["documentation_version"] == croissant["version"] != "1.0.0"
    assert HUB in card and f"--revision {REVISION}" in card
    assert "(LICENSE_NOTICE.txt)" in card
    assert not card.startswith("---")
    assert "this revision" not in card and "in this repository) contains" not in card


def test_data_and_current_notice_have_separate_authenticated_contracts():
    trusted = (TRUSTED / "CACHE_PROVENANCE.json").read_bytes()
    assert trusted == (CACHE / "CACHE_PROVENANCE.json").read_bytes()
    contract = json.loads(trusted)
    assert contract["data_revision"] == REVISION
    assert contract["data_manifest"]["entries"] == 460
    for key, name in (("data_manifest", "CHECKSUMS.txt"), ("current_notice", "LICENSE_NOTICE.txt")):
        entry = contract[key]
        assert entry["path"] == name
        data = (CACHE / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert len(data) == entry["size_bytes"]
