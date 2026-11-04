"""Verify trusted compatibility groups for stored LOGO runs.

    python -B camera_ready/scripts/review/trusted_run_pairs_v1.py --check

For stored runs without FIT and SELECT membership fingerprints, the
statistics, combined ceiling and sensitivity summary accept a pair when
both content digests belong to one recorded compatibility group. Digests
are SHA-256 with CRLF line endings (ladder_checks.content_digest). Current
runners record membership fingerprints for direct comparison.
"""
import argparse
import json
import sys
from pathlib import Path

CR = Path(__file__).resolve().parents[2]                    # camera_ready/
ROOT = CR.parent
sys.path.insert(0, str(CR / "scripts" / "runs"))
import ladder_checks  # noqa: E402

RES = "camera_ready/output/results"
POOLS = (("arch6", 256), ("clip4", 32), ("strong4", 256), ("weak6", 256))
GENS = ("whichisreal", "CollabDiff", "MidJourney", "heygen", "starganv2", "deepfacelab", "stargan", "styleclip")


def groups():
    out = []
    for pool, d in POOLS:
        files = [f"{RES}/mitigation/content_disjoint/11_disjoint_{pool}_d{d}.json",
                 f"{RES}/mitigation/content_disjoint/A3c_logo_extended_rungs_{pool}.json",
                 f"{RES}/mitigation/content_disjoint/A3c_logo_combined_ceiling_{pool}.json"]
        if pool in ("arch6", "clip4"):
            files.append(f"{RES}/mitigation/mechanism/run_c_v2_logo_{pool}.json")
        files += [f"{RES}/sensitivity/paper_heads/{pool}/{pool}_{g}.json" for g in GENS]
        out.append((f"primary {pool}: LOGO with the content exclusion, its richer inputs, combined ceiling, mechanism "
                    "and paper-heads runs", files))
    for pool, d in POOLS:
        files = [f"{RES}/sensitivity/dedup/11_disjoint_{pool}_d{d}.json",
                 f"{RES}/sensitivity/dedup/A3c_logo_extended_rungs_{pool}.json"]
        if pool in ("arch6", "clip4"):
            files.append(f"{RES}/sensitivity/dedup/run_c_v2_logo_{pool}.json")
        out.append((f"deduplicated {pool}: the sensitivity check without duplicated training rows", files))
    out.append(("submitted arch6: the reviewed LOGO run with the rebuttal's richer inputs (the statistics' self-test)",
                ["output/results/11_disjoint_arch6_d256.json", "camera_ready/inputs/11_disjoint_arch6_d256.json",
                 "camera_ready/inputs/A3c_logo_extended_rungs_arch6.json",
                 "camera_ready/inputs/A3c_logo_combined_ceiling_arch6.json"]))
    return out


def build():
    return {"note": "Groups of stored LOGO runs produced together from the same folds; content digests are SHA-256 "
                    "with CRLF line endings (camera_ready/scripts/runs/ladder_checks.py). Written by "
                    "camera_ready/scripts/review/trusted_run_pairs_v1.py.",
            "groups": [{"group": name, "files": {p: ladder_checks.content_digest(ROOT / p) for p in files}}
                       for name, files in groups()]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="compare the mapping with the stored files")
    mode.add_argument("--write", action="store_true", help="rewrite the mapping (authors only)")
    a = ap.parse_args()
    missing = [p for _, files in groups() for p in files if not (ROOT / p).is_file()]
    if missing:
        raise SystemExit(f"stored runs not found: {', '.join(missing)}")
    new = build()
    if a.write:
        ladder_checks.TRUSTED.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8", newline="\r\n")
        print(f"wrote {ladder_checks.TRUSTED.relative_to(ROOT)}: {len(new['groups'])} groups, "
              f"{sum(len(g['files']) for g in new['groups'])} files")
        return 0
    old = json.loads(ladder_checks.TRUSTED.read_text(encoding="utf-8"))
    if old != new:
        bad = [p for go, gn in zip(old["groups"], new["groups"]) for p in gn["files"]
               if go["files"].get(p) != gn["files"][p]]
        print(f"FAIL: the mapping differs from the stored files ({len(bad)} entries): {', '.join(bad[:5])}")
        return 1
    print(f"PASS: {len(new['groups'])} groups, {sum(len(g['files']) for g in new['groups'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
