"""Attach attempt provenance to runner checkpoints.

The camera-ready harness launches resumable runners in this process:
    python camera_ready/scripts/review/stamp_checkpoints_v2.py <runner.py> [runner arguments]

Each JSON checkpoint records the harness's REPRO_ATTEMPT_ID and
REPRO_FINGERPRINT, along with the SHA-256 of its paired array file. This
covers fold checkpoints and three-class resume metadata. Three-class
per-fold result JSONs are separate outputs.

The harness reuses a checkpoint only if it belongs to an attempt of the
same step and run, the fingerprint still matches, and the paired array
matches the recorded digest.
"""
import hashlib
import json
import os
import re
import runpy
import sys
from pathlib import Path

CHECKPOINT_DIRS = frozenset({"folds", "rung_folds", "mechanism_folds"})
# Per-fold RESULT JSON files are final outputs.
K3_RESUME = re.compile(r"^NATIVE_K_ALIGNED_(FULL|SMOKE)_.+_F\d+_(?!RESULT\.json$).+\.json$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(path, record):
    """The provenance of a resume file about to be written, or None when `path` is not one."""
    base = {"attempt": os.environ.get("REPRO_ATTEMPT_ID"), "fingerprint": os.environ.get("REPRO_FINGERPRINT")}
    if path.parent.name in CHECKPOINT_DIRS and isinstance(record.get("scores_file"), str):
        scores = path.parent.parent / record["scores_file"]
        return {**base, "scores_sha256": sha256(scores) if scores.is_file() else None}
    if K3_RESUME.match(path.name):
        arrays = path.with_suffix(".npz")
        return {**base, "arrays_sha256": sha256(arrays) if arrays.is_file() else None}
    return None


def install():
    """Stamp every resume file written through pathlib (the runners' write_json and the three-class metadata)."""
    original = Path.write_text

    def write_text(self, data, *args, **kwargs):
        if self.suffix == ".json" and (self.parent.name in CHECKPOINT_DIRS or K3_RESUME.match(self.name)):
            try:
                record = json.loads(data)
            except ValueError:
                record = None
            stamp = provenance(self, record) if isinstance(record, dict) else None
            if stamp is not None:
                record["provenance"] = stamp
                data = json.dumps(record, indent=2)
        return original(self, data, *args, **kwargs)

    Path.write_text = write_text


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: stamp_checkpoints_v2.py <runner.py> [runner arguments]")
    runner = Path(sys.argv[1]).resolve()
    install()
    sys.argv = [str(runner)] + sys.argv[2:]
    sys.path.insert(0, str(runner.parent))            # as when the runner is started directly
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
