"""Build the reference file for harness blocks B and C from stored results.

The reference records the result metrics and exact fields selected by
RESULTS_SPEC in reproduce_all_v4.py, together with source digests and
comparison settings. It excludes timing and host fields and retains the
environment-dependent markers for GPU-trained results.

Use --out to write a new comparison file. The input files must match the
source digests in the shipped reference.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CR = Path(__file__).resolve().parents[2]
ROOT = CR.parent
HARNESS = CR / "scripts" / "review" / "reproduce_all_v4.py"          # version 3: the list of harness version 4
OUT = CR / "reference" / "reference_results_camera_ready_v2.json"


def load_harness():
    spec = importlib.util.spec_from_file_location("reproduce_all_v4", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.resolve() == OUT.resolve() or a.out.exists():
        raise SystemExit(f"{a.out} is not a new destination. Choose a new file with --out.")
    h = load_harness()
    expected = {spec["path"] for spec in h.RESULTS_SPEC}
    approved = h.read_reference(OUT, expected)
    results, unverified = {}, []
    for spec in h.RESULTS_SPEC:
        f = ROOT / spec["path"]
        if not f.is_file():
            unverified.append(f"{spec['path']} (missing)")
            continue
        raw = f.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != approved[spec["path"]].get("source_sha256"):
            unverified.append(f"{spec['path']} (source digest differs)")
            continue
        nums, strs = h.extract(json.loads(raw.decode("utf-8")), spec.get("strings"), spec.get("exclude"))
        results[spec["path"]] = {k: spec[k] for k in (
            "kind", "tol", "env", "strings", "key_map", "partial_ok", "step", "exclude")}
        results[spec["path"]].update({"metrics": nums, "strings_ref": strs,
                                      "source_sha256": digest})
    if unverified:
        raise SystemExit("Refusing to export a reference from unverified results: " + ", ".join(unverified))
    doc = {"_generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "_generator": "camera_ready/scripts/authoring/generate_reference_v3.py",
           "_tolerance_deterministic_auc": h.DET_TOL, "_tolerance_stochastic_auc": h.STOCH_TOL,
           "_tolerance_env_auc": h.ENV_TOL,
           "_note": "Built from the stored camera-ready results; block A uses output/results/reference_results.json.",
           "results": results}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(doc, indent=1) + "\n")
    n = sum(len(v["metrics"]) for v in results.values())
    s = sum(len(v["strings_ref"]) for v in results.values())
    print(f"wrote {a.out}: {len(results)} files, {n} numeric values, {s} exact strings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
