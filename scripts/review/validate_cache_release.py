"""validate_cache_release.py: validate the derived feature cache release.

Reviewer-facing integrity + completeness check for the cache shipped per
paper's release appendix. Verifies the cache's sidecar artifacts:

    1. CHECKSUMS.txt    -- every listed file present, SHA-256 matches
    2. croissant.json   -- parses, has required Croissant 1.0 fields,
                           distribution count matches CHECKSUMS line count,
                           every distribution sha256 matches CHECKSUMS
    3. DATASET_CARD.md  -- present, has required sections (Provenance,
                           Intended use, Non-intended use, License,
                           Limitations, Verification)
    4. LICENSE_NOTICE.txt -- present, declares CC BY-NC-SA 4.0 for our contributions and names DF40
    5. CACHE_PROVENANCE.json -- binds the data manifest and current notice
                               separately to their SHA-256 hashes

Prerequisite: this script verifies SHA-256 hashes for every listed `.npy`
file in CHECKSUMS.txt. Those files are NOT in this repository (which
holds only the cache's metadata); they are hosted on Hugging Face.
Download the full cache before running this script:

    huggingface-cli download anon-frozen-experts/df40-frozen-experts-anon \\
        --repo-type dataset --revision 18fb1b212f7251799fca8cb4d549be6d239ce13a --local-dir output/cached_features/ --include "*/*"

(Or follow the equivalent instructions in the dataset card.) Running this
script against the metadata alone will fail at step 1 with
missing-file errors for every `.npy`.

Usage:
    python scripts/review/validate_cache_release.py
    python scripts/review/validate_cache_release.py --cache-root /path/to/cache

Exit code: 0 if everything passes, 1 if any check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Iterator

# This file lives at scripts/review/; codebase root is two levels up.
CODEBASE = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_ROOT = CODEBASE / "output" / "cached_features"
TRUSTED_METADATA = CODEBASE / "camera_ready" / "dataset"

# Required Croissant 1.0 top-level fields per MLCommons spec.
REQUIRED_CROISSANT_FIELDS = [
    "@context", "@type", "name", "description", "license",
    "url", "version", "citeAs", "distribution",
]

# Required dataset card sections (case-insensitive header match).
REQUIRED_CARD_SECTIONS = [
    "provenance", "intended use", "non-intended use",
    "license", "limitations", "verification",
]

# License keywords required in LICENSE_NOTICE.txt.
LICENSE_KEYWORDS = ["CC BY-NC-SA 4.0", "DF40"]


class CheckFailure(Exception):
    """Raised when a check fails; carries a short message."""


def _section(title: str):
    print(f"\n[{title}]")


def _ok(msg: str) -> None:
    print(f"  OK    {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_checksums(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (sha256, relative_path) tuples from a sha256sum-format file."""
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2:
                raise CheckFailure(f"malformed CHECKSUMS line: {line!r}")
            sha, rel = parts
            if not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise CheckFailure(f"malformed SHA-256 for {rel!r}")
            if (not rel or "\\" in rel or ":" in rel
                    or PurePosixPath(rel).is_absolute()
                    or any(part in ("", ".", "..") for part in rel.split("/"))):
                raise CheckFailure(f"unsafe CHECKSUMS path: {rel!r}")
            if rel.casefold() in seen:
                raise CheckFailure(f"duplicate CHECKSUMS path: {rel!r}")
            seen.add(rel.casefold())
            yield sha, rel


def check_provenance(cache_root: Path) -> dict:
    """Read the release contract; data and the current notice are distinct."""
    path = cache_root / "CACHE_PROVENANCE.json"
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CheckFailure(f"CACHE_PROVENANCE.json missing or invalid: {e}") from e
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise CheckFailure("unsupported CACHE_PROVENANCE.json schema")
    for key, filename in (("data_manifest", "CHECKSUMS.txt"),
                          ("current_notice", "LICENSE_NOTICE.txt")):
        item = contract.get(key)
        if (not isinstance(item, dict) or item.get("path") != filename
                or not isinstance(item.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                or type(item.get("size_bytes")) is not int or item["size_bytes"] <= 0):
            raise CheckFailure(f"invalid {key} in CACHE_PROVENANCE.json")
    count = contract["data_manifest"].get("entries")
    if type(count) is not int or count <= 0:
        raise CheckFailure("invalid data manifest entry count in CACHE_PROVENANCE.json")
    try:
        trusted = json.loads((TRUSTED_METADATA / "CACHE_PROVENANCE.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CheckFailure(f"trusted release metadata missing or invalid: {e}") from e
    for key in ("schema_version", "documentation_version", "data_revision", "data_manifest", "current_notice"):
        if contract.get(key) != trusted.get(key):
            raise CheckFailure(f"cache {key} differs from the trusted code release; restore its metadata")
    return contract


def check_bound_file(cache_root: Path, item: dict) -> None:
    path = cache_root / item["path"]
    if not path.is_file():
        raise CheckFailure(f"{item['path']} missing")
    if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
        raise CheckFailure(f"{item['path']} does not match CACHE_PROVENANCE.json; "
                           "restore the synchronized release metadata")


# ---------------------------------------------------------------------------

def check_checksums(cache_root: Path) -> tuple[int, dict[str, str]]:
    """Verify CHECKSUMS.txt: every listed file present and hashes match.

    Returns (n_files_checked, {relative_path: sha256}) for cross-checking
    against Croissant.
    """
    _section("CHECKSUMS.txt")
    contract = check_provenance(cache_root)
    check_bound_file(cache_root, contract["data_manifest"])
    checksums_path = cache_root / "CHECKSUMS.txt"
    if not checksums_path.exists():
        _fail(f"CHECKSUMS.txt missing at {checksums_path}")
        raise CheckFailure("CHECKSUMS.txt missing")

    declared: dict[str, str] = {}
    for sha, rel in parse_checksums(checksums_path):
        declared[rel] = sha
    if "LICENSE_NOTICE.txt" in declared:
        raise CheckFailure("LICENSE_NOTICE.txt must be verified separately from data")
    if len(declared) != contract["data_manifest"]["entries"]:
        raise CheckFailure("CHECKSUMS entry count differs from CACHE_PROVENANCE.json")
    _ok(f"parsed CHECKSUMS.txt: {len(declared)} entries")

    missing = []
    mismatched = []
    for rel, expected in declared.items():
        p = cache_root / rel
        if not p.resolve().is_relative_to(cache_root.resolve()):
            raise CheckFailure(f"cache path escapes cache root: {rel}")
        if not p.is_file():
            missing.append(rel)
            continue
        actual = sha256_file(p)
        if actual != expected:
            mismatched.append((rel, expected, actual))

    if missing:
        for m in missing[:5]:
            _fail(f"missing file: {m}")
        if len(missing) > 5:
            _fail(f"... and {len(missing) - 5} more missing")
        raise CheckFailure(f"{len(missing)} files in CHECKSUMS.txt not present in cache")

    if mismatched:
        for rel, exp, act in mismatched[:5]:
            _fail(f"sha256 mismatch: {rel}")
            _fail(f"  expected: {exp}")
            _fail(f"  actual:   {act}")
        if len(mismatched) > 5:
            _fail(f"... and {len(mismatched) - 5} more mismatches")
        raise CheckFailure(f"{len(mismatched)} sha256 mismatches")

    _ok(f"all {len(declared)} files present")
    _ok(f"all {len(declared)} sha256 hashes match")
    return len(declared), declared


def check_croissant(cache_root: Path, expected_checksums: dict[str, str]) -> int:
    """Verify croissant.json: parses, required fields present, distribution
    matches CHECKSUMS.

    Returns number of distribution entries verified.
    """
    _section("croissant.json")
    croissant_path = cache_root / "croissant.json"
    if not croissant_path.exists():
        _fail(f"croissant.json missing at {croissant_path}")
        raise CheckFailure("croissant.json missing")

    try:
        croissant = json.loads(croissant_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _fail(f"croissant.json is not valid JSON: {e}")
        raise CheckFailure("croissant.json invalid JSON") from e
    _ok("croissant.json parsed as JSON")

    missing_fields = [k for k in REQUIRED_CROISSANT_FIELDS if k not in croissant]
    if missing_fields:
        for k in missing_fields:
            _fail(f"missing required Croissant field: {k}")
        raise CheckFailure(f"croissant.json missing fields: {missing_fields}")
    _ok(f"all {len(REQUIRED_CROISSANT_FIELDS)} required Croissant fields present")

    # @type should be sc:Dataset
    if croissant.get("@type") != "sc:Dataset":
        _fail(f"croissant @type is {croissant.get('@type')!r}, expected 'sc:Dataset'")
        raise CheckFailure("croissant @type wrong")
    _ok("@type = sc:Dataset")

    # conformsTo should reference Croissant 1.0
    conforms = str(croissant.get("conformsTo", ""))
    if "croissant" not in conforms.lower():
        _fail(f"conformsTo does not reference Croissant: {conforms!r}")
        raise CheckFailure("conformsTo missing")
    _ok(f"conformsTo: {conforms}")

    distribution = croissant.get("distribution", [])
    if not isinstance(distribution, list):
        _fail("distribution is not a list")
        raise CheckFailure("distribution not a list")
    _ok(f"distribution has {len(distribution)} FileObject entries")

    if len(distribution) != len(expected_checksums):
        _fail(
            f"distribution count ({len(distribution)}) does not match "
            f"CHECKSUMS line count ({len(expected_checksums)})"
        )
        raise CheckFailure("distribution / CHECKSUMS count mismatch")
    _ok("distribution count matches CHECKSUMS line count")

    # Cross-check every distribution entry's sha256 against CHECKSUMS
    cross_mismatches = []
    seen = set()
    for entry in distribution:
        if not isinstance(entry, dict):
            raise CheckFailure("Croissant distribution entry is not an object")
        rel = entry.get("@id")
        sha = entry.get("sha256")
        if not isinstance(rel, str) or not isinstance(sha, str):
            cross_mismatches.append((rel, "missing @id or sha256"))
            continue
        if rel in seen:
            cross_mismatches.append((rel, "duplicate @id"))
        seen.add(rel)
        if rel not in expected_checksums:
            cross_mismatches.append((rel, "not in CHECKSUMS.txt"))
            continue
        if expected_checksums[rel] != sha:
            cross_mismatches.append(
                (rel, "sha256 differs from CHECKSUMS")
            )
        local = cache_root / rel
        if local.is_file() and str(local.stat().st_size) != str(entry.get("contentSize")):
            cross_mismatches.append((rel, "contentSize differs from local file"))
    for rel in expected_checksums.keys() - seen:
        cross_mismatches.append((rel, "missing from Croissant"))
    if cross_mismatches:
        for rel, msg in cross_mismatches[:5]:
            _fail(f"  distribution[{rel}]: {msg}")
        if len(cross_mismatches) > 5:
            _fail(f"... and {len(cross_mismatches) - 5} more")
        raise CheckFailure(f"{len(cross_mismatches)} Croissant-vs-CHECKSUMS mismatches")
    _ok(f"all {len(distribution)} sha256 entries match CHECKSUMS.txt")

    return len(distribution)


def check_dataset_card(cache_root: Path) -> None:
    """Verify DATASET_CARD.md present and has required sections."""
    _section("DATASET_CARD.md")
    card_path = cache_root / "DATASET_CARD.md"
    if not card_path.exists():
        _fail(f"DATASET_CARD.md missing at {card_path}")
        raise CheckFailure("DATASET_CARD.md missing")
    text = card_path.read_text(encoding="utf-8").lower()
    _ok(f"DATASET_CARD.md present ({card_path.stat().st_size} bytes)")

    missing_sections = []
    for section in REQUIRED_CARD_SECTIONS:
        # Match section as a header substring (## Provenance, ## Intended use, etc.)
        if f"## {section}" not in text and f"# {section}" not in text:
            missing_sections.append(section)
    if missing_sections:
        for s in missing_sections:
            _fail(f"missing required section: '{s}'")
        raise CheckFailure(f"DATASET_CARD.md missing sections: {missing_sections}")
    _ok(f"all {len(REQUIRED_CARD_SECTIONS)} required sections present")


def check_license_notice(cache_root: Path) -> None:
    """Verify the current notice's hash, size and key license terms."""
    _section("LICENSE_NOTICE.txt")
    contract = check_provenance(cache_root)
    check_bound_file(cache_root, contract["current_notice"])
    _ok("current notice size and SHA-256 match CACHE_PROVENANCE.json")
    notice_path = cache_root / "LICENSE_NOTICE.txt"
    if not notice_path.exists():
        _fail(f"LICENSE_NOTICE.txt missing at {notice_path}")
        raise CheckFailure("LICENSE_NOTICE.txt missing")
    text = notice_path.read_text(encoding="utf-8")
    _ok(f"LICENSE_NOTICE.txt present ({notice_path.stat().st_size} bytes)")

    missing_keywords = [k for k in LICENSE_KEYWORDS if k not in text]
    if missing_keywords:
        for k in missing_keywords:
            _fail(f"missing required keyword: '{k}'")
        raise CheckFailure(f"LICENSE_NOTICE.txt missing keywords: {missing_keywords}")
    _ok(f"all {len(LICENSE_KEYWORDS)} required keywords present")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT,
                    help=f"Cache root directory (default: {DEFAULT_CACHE_ROOT})")
    args = ap.parse_args()

    if not args.cache_root.exists():
        print(f"ERROR: cache root not found: {args.cache_root}", file=sys.stderr)
        return 1

    print(f"Validating cache release at: {args.cache_root}")

    try:
        n_files, checksums = check_checksums(args.cache_root)
        n_croissant = check_croissant(args.cache_root, checksums)
        check_dataset_card(args.cache_root)
        check_license_notice(args.cache_root)
    except (CheckFailure, OSError) as e:
        print(f"\nVALIDATION FAILED: {e}", file=sys.stderr)
        return 1

    print(
        f"\nVALIDATION PASSED: {n_files} files, {n_croissant} Croissant entries, "
        f"dataset card + license notice present."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
