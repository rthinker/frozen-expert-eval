# C04: Check expert training-list exposure using prepared DF40 lists.
"""Expert-side exposure: were the frozen experts trained on held-out test images?

The combiner-side overlap (FIT and SELECT against the held-out test
generators) is measured by other Appendix A steps. This check asks whether
the 21 frozen DF40 experts were trained on images of the eight held-out
generators, by matching each expert's training list against the held-out
test paths.

It also reports how many real frames are available from families that are
not held out, the number that bounds a generator-disjoint training set (the
`family` filter leaves 586 real rows in FIT).

Reads prepared DF40 split lists and writes one JSON report.

Usage (from the repository root)::

    python camera_ready/scripts/analyses/appA_expert_side_v3.py --json_dir <prepared DF40 lists> --out <report.json>

To print the lists' structure first::

    python camera_ready/scripts/analyses/appA_expert_side_v3.py --json_dir <prepared DF40 lists> --dump_schema
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

from prepare_df40_lists_v2 import verify_prepared_lists

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif")

# The eight held-out generators, and the JSONs that define them.
TEST_JSONS = {
    "whichisreal": "whichisreal.json",
    "CollabDiff": "CollabDiff.json",
    "MidJourney": "MidJourney.json",
    "heygen": "heygen.json",
    "starganv2": "starganv2.json",
    "deepfacelab": "deepfacelab.json",
    "stargan": "stargan.json",
    "styleclip": "styleclip.json",
}

# The JSONs that define what the frozen experts were trained on, by regime.
# Expert naming: fs-* trained on FSAll_ff, fr-* on FRAll_ff, efs-* on
# EFSAll_ff, df40-* on DF40_all.
EXPERT_TRAIN_JSONS = {
    "fs": "FSAll_ff.json",
    "fr": "FRAll_ff.json",
    "efs": "EFSAll_ff.json",
    "df40": "DF40_all.json",
}

_MARKER = "/DF40/"


def canonicalise(p: str) -> str:
    """Mirror scripts/01_extract.py::_canonicalize_sample_id."""
    s = str(p).replace("\\", "/")
    i = s.rfind(_MARKER)
    if i != -1:
        s = s[i + len(_MARKER):]
    return s


def walk(obj, keypath: Tuple[str, ...] = ()):
    """Yield (keypath, string) for every string leaf in a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, keypath + (str(k),))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from walk(v, keypath)
    elif isinstance(obj, str):
        yield keypath, obj


def collect(path: Path) -> List[Tuple[Tuple[str, ...], str]]:
    """All image-path leaves in a DF40 JSON, with their key path."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(kp, s) for kp, s in walk(data)
            if s.lower().endswith(IMG_EXT)]


def split_of(keypath: Tuple[str, ...]) -> str:
    for k in keypath:
        lk = k.lower()
        if lk in ("train", "training"):
            return "train"
        if lk in ("test", "testing"):
            return "test"
        if lk in ("val", "validation", "eval"):
            return "val"
    return "unknown"


def label_of(keypath: Tuple[str, ...]) -> str:
    for k in keypath:
        lk = k.lower()
        if lk.endswith("_real") or lk == "real":
            return "real"
        if lk.endswith("_fake") or lk == "fake":
            return "fake"
    return "unknown"


def dump_schema(path: Path, depth: int = 4) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def rec(o, d=0, prefix=""):
        pad = "  " * d
        if d >= depth:
            print(f"{pad}{prefix}... ({type(o).__name__})")
            return
        if isinstance(o, dict):
            keys = list(o.keys())
            print(f"{pad}{prefix}dict[{len(keys)}] keys={keys[:8]}"
                  f"{' ...' if len(keys) > 8 else ''}")
            for k in keys[:3]:
                rec(o[k], d + 1, f"{k}: ")
        elif isinstance(o, list):
            print(f"{pad}{prefix}list[{len(o)}]")
            if o:
                rec(o[0], d + 1, "[0]: ")
        else:
            v = str(o)
            print(f"{pad}{prefix}{type(o).__name__} = "
                  f"{v[:110]}{'...' if len(v) > 110 else ''}")

    print(f"\n=== schema: {path.name} ===")
    rec(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json_dir", required=True,
                    help="DF40 dataset_json folder (see configs/paths.yaml "
                         "-> dataset_json_folder)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump_schema", action="store_true",
                    help="Print the key structure of each JSON and exit. Run "
                         "this if the counts below look implausible.")
    args = ap.parse_args()

    jd = Path(args.json_dir)
    if not jd.is_dir():
        raise SystemExit(f"not a directory: {jd}")
    verify_prepared_lists(jd)
    # run by hand, write only where --out says (the default is the stored result the reference pins)
    if args.out is None and not args.dump_schema and not os.environ.get("REPRO_ATTEMPT_ID"):
        raise SystemExit("run by hand, this check writes only where --out says: its default output is the stored "
                         "camera_ready/output/results/analyses/gate_expert_side_check.json, which the harness rewrites "
                         "after backing it up (python camera_ready/scripts/review/reproduce_all_v4.py --only C04 "
                         "--df40-json <DF40 lists>).")

    wanted = list(TEST_JSONS.values()) + list(EXPERT_TRAIN_JSONS.values())

    if args.dump_schema:
        for w in wanted:
            dump_schema(jd / w)
        return

    # ---- test-side key sets -------------------------------------------
    print("reading held-out generator JSONs ...")
    test_keys: Dict[str, Set[str]] = {}
    test_split_counts: Dict[str, Dict[str, int]] = {}
    for gen, fn in TEST_JSONS.items():
        leaves = collect(jd / fn)
        sc: Dict[str, int] = defaultdict(int)
        keys: Set[str] = set()
        for kp, s in leaves:
            sc[split_of(kp)] += 1
            keys.add(canonicalise(s))
        test_keys[gen] = keys
        test_split_counts[gen] = dict(sc)
        print(f"  {gen:14s} {len(leaves):7d} paths, {len(keys):7d} unique, "
              f"splits={dict(sc)}")

    # ---- expert-side train sets ---------------------------------------
    print("\nreading expert-training JSONs ...")
    train_keys: Dict[str, Set[str]] = {}
    real_by_family: Dict[str, Dict[str, int]] = {}
    for regime, fn in EXPERT_TRAIN_JSONS.items():
        leaves = collect(jd / fn)
        keys: Set[str] = set()
        fam_real: Dict[str, int] = defaultdict(int)
        for kp, s in leaves:
            if split_of(kp) not in ("train", "unknown"):
                continue
            c = canonicalise(s)
            keys.add(c)
            if label_of(kp) == "real":
                fam_real[c.split("/")[0]] += 1
        train_keys[regime] = keys
        real_by_family[regime] = dict(fam_real)
        print(f"  {regime:5s} ({fn:16s}) {len(keys):7d} unique train paths")

    # ---- exposure matrix -----------------------------------------------
    print("\n" + "=" * 72)
    print("EXPOSURE: expert training lists vs held-out test generators")
    print("=" * 72)
    hdr = f"{'regime':7s} " + " ".join(f"{g[:9]:>9s}" for g in TEST_JSONS)
    print(hdr)
    print("-" * len(hdr))
    gate: Dict[str, Dict[str, int]] = {}
    worst = 0
    for regime, tk in train_keys.items():
        row = {}
        cells = []
        for gen, gk in test_keys.items():
            n = len(tk & gk)
            row[gen] = n
            worst = max(worst, n)
            cells.append(f"{n:9d}")
        gate[regime] = row
        print(f"{regime:7s} " + " ".join(cells))

    # The reading is per regime: the experts in a pool are exposed only through
    # their own training list. A regime with exposure concerns the pools that
    # use it and says nothing about pools built from other regimes (arch6 uses
    # the fs list only).
    print("\n" + "=" * 72)
    dirty = {r: sum(row.values()) for r, row in gate.items()
             if sum(row.values()) > 0}
    if not dirty:
        print("No exposure in any regime: no expert training path appears in")
        print("any held-out generator. The overlap is confined to the combiner")
        print("split (TRAIN and SELECT).")
    else:
        clean = [r for r in gate if r not in dirty]
        print("Exposure found for regime(s): "
              + ", ".join(f"{r} ({n} paths)" for r, n in dirty.items()))
        print(f"Regime(s) without exposure: {', '.join(clean) or 'none'}.")
        print("Pools whose experts train only on regimes without exposure are")
        print("unaffected; results of pools that use an exposed regime should be")
        print("read with that exposure in mind (Appendix A).")
        print("Check --dump_schema to rule out a parsing artefact.")
    print("=" * 72)

    # ---- real frames outside the held-out families ----------------------
    print("\nReal frames available from families that are not held out")
    print("The `family` filter leaves 586 real rows in FIT; a generator-disjoint")
    print("training set would need real frames from these families.")
    feas = {}
    for regime, fam in real_by_family.items():
        clean = {k: v for k, v in fam.items() if k not in TEST_JSONS}
        dirty = {k: v for k, v in fam.items() if k in TEST_JSONS}
        feas[regime] = {"real_from_clean_families": sum(clean.values()),
                        "real_from_test_families": sum(dirty.values()),
                        "top_clean_families": dict(
                            sorted(clean.items(), key=lambda x: -x[1])[:10])}
        print(f"  {regime:5s} clean-family reals={sum(clean.values()):7d}  "
              f"test-family reals={sum(dirty.values()):7d}")
        for k, v in sorted(clean.items(), key=lambda x: -x[1])[:5]:
            print(f"          {k:40s} {v:7d}")

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[2] / "output" / "results" / "analyses" /
        "gate_expert_side_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "gate": gate,
            # Global flag kept for backwards compatibility; the fields below
            # carry the per-regime reading (exposure concerns only the pools
            # whose experts train on that regime; see the printed summary).
            "gate_passed": worst == 0,
            "max_overlap": worst,
            "per_regime_overlap_totals": {
                r: sum(row.values()) for r, row in gate.items()
            },
            "clean_regimes": sorted(
                r for r, row in gate.items() if sum(row.values()) == 0
            ),
            "dirty_regimes": sorted(
                r for r, row in gate.items() if sum(row.values()) > 0
            ),
            "test_unique_paths": {g: len(k) for g, k in test_keys.items()},
            "test_split_counts": test_split_counts,
            "expert_train_unique_paths": {r: len(k) for r, k in train_keys.items()},
            "rebuild_feasibility": feas,
            "json_dir": str(jd),
            "_written_utc": datetime.now(timezone.utc).isoformat(),
            "_caveat": "Path-level intersection after canonicalisation "
                       "(strip up to last '/DF40/'). A zero here does not "
                       "prove the DF40 authors' upstream pretraining was "
                       "clean, only that these release JSONs do not share "
                       "frame paths.",
        }, f, indent=2)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
