"""Rewrite DF40's public split lists to match the feature-cache IDs (harness step C00).

    python camera_ready/scripts/analyses/prepare_df40_lists_v2.py --public DF40_LISTS --out camera_ready/output/df40_lists_cacheform

The Appendix A expert-side check and Appendix I.1 three-class manifests
match split-list paths to cache IDs. The cache IDs are relative to the DF40
image root. Map FaceForensics++ paths to FaceForensics++/<rest>, DF40_train
fake paths to <method>/ff/frames/<rest>, and use the suffix of DF40/<rest>
for the other paths.

Write the twelve required lists and PREPARED.json, which records input and
output file digests and counts of rewritten paths. Prepared lists reproduce
the stored three-class manifests and assignment digests, and the expert-side
overlap counts.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

NAMES = ("DF40_all", "FSAll_ff", "FRAll_ff", "EFSAll_ff", "whichisreal", "CollabDiff", "MidJourney", "heygen",
         "starganv2", "deepfacelab", "stargan", "styleclip")
LIST_FILES = tuple(f"{name}.json" for name in NAMES)
ROOT_DIR = "deepfakes_detection_datasets/"


def cache_form(s: str, tally: Counter) -> str:
    """Rewrite one public list path so that what follows '/DF40/' equals the cache's sample id."""
    t = s.replace("\\", "/")
    if "/DF40_train/" in t or t.startswith("DF40_train/"):
        rest = t.split("DF40_train/", 1)[1] if t.startswith("DF40_train/") else t[t.rfind("/DF40_train/") + 12:]
        parts = rest.split("/")
        if len(parts) > 2 and parts[1] == "frames":
            rest = "/".join([parts[0], "ff"] + parts[1:])
        tally["DF40_train"] += 1
        return "df40/DF40/" + rest
    if "/DF40/" in t:
        tally["DF40 (unchanged)"] += 1
        return s
    if ROOT_DIR in t:
        tally["dataset root"] += 1
        return "df40/DF40/" + t[t.rfind(ROOT_DIR) + len(ROOT_DIR):]
    return s


def rewrite(obj, tally: Counter):
    if isinstance(obj, dict):
        return {k: rewrite(v, tally) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite(v, tally) for v in obj]
    if isinstance(obj, str):
        return cache_form(obj, tally)
    return obj


def verify_prepared_lists(folder: Path) -> None:
    """Require a complete set of lists matching the preparation record."""
    marker = folder / "PREPARED.json"
    try:
        record = json.loads(marker.read_bytes())
        files = record["files"]
        if not isinstance(files, dict) or set(files) != set(LIST_FILES):
            raise ValueError("expected hashes for all 12 prepared lists")
        for name in LIST_FILES:
            with (folder / name).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != files[name]["output_sha256"]:
                raise ValueError(f"SHA-256 mismatch for {name}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"{marker} does not describe intact prepared DF40 lists ({exc}). Run "
            "camera_ready/scripts/analyses/prepare_df40_lists_v2.py "
            "--public <DF40 lists> --out <separate folder> and pass that folder."
        ) from exc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--public", type=Path, required=True, help="folder with DF40's public split list files")
    ap.add_argument("--out", type=Path, default=Path("camera_ready/output/df40_lists_cacheform"))
    a = ap.parse_args()
    source, output = a.public.resolve(), a.out.resolve()
    if source == output:
        ap.error("--out must be separate from the source lists in --public")
    missing = [name for name in LIST_FILES if not (source / name).is_file()]
    if missing:
        raise SystemExit(f"missing in {source}: {', '.join(missing)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"_written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": {}}
    with TemporaryDirectory(prefix=".df40-prepare-", dir=output.parent) as staging:
        stage = Path(staging)
        for name in LIST_FILES:
            raw = (source / name).read_bytes()
            try:
                data = json.loads(raw.decode("utf-8"))
            except ValueError as exc:
                raise SystemExit(f"Invalid DF40 list {source / name}: {exc}") from exc
            tally = Counter()
            out = json.dumps(rewrite(data, tally)).encode("utf-8")
            (stage / name).write_bytes(out)
            record["files"][name] = {"input_sha256": hashlib.sha256(raw).hexdigest(),
                                     "output_sha256": hashlib.sha256(out).hexdigest(),
                                     "paths_by_form": dict(tally)}
            print(f"{name}: {dict(tally)}")
        marker = "PREPARED.json"
        (stage / marker).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

        output.mkdir(exist_ok=True)
        # The previous record must not certify a partially replaced list set.
        (output / marker).unlink(missing_ok=True)
        for name in LIST_FILES:
            (stage / name).replace(output / name)
        (stage / marker).replace(output / marker)
    print(f"wrote {len(LIST_FILES)} files and PREPARED.json to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
