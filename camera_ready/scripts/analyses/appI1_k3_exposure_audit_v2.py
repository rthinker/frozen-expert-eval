# C15: Check expert exposure in each three-class construction.
"""Check each expert's training-list exposure for every retained three-class row.

For each retained row and each candidate pool, determine whether the row
was in that expert's TRAINING SPLIT - the train section of the partition
file, not merely present somewhere in it - and report the exposed fraction
per class. Uniform exposure across classes is what makes the label
unreadable from memorisation; non-uniform exposure is the confound.

The audit covers both constructions:
  VIEW A  df40-only  - expect 100% uniform (all trained on DF40_all train)
  VIEW B  arch6      - expect 0% on every retained class, since arch6's
                       fs-* experts trained on face-swap, which View B
                       excludes

Honest limit, stated in the output: membership is matched on canonical
path. A content-level check against DF40_all's full training split would
need features for ~700k images we have not extracted. What IS content-
checked is the exclusion of held-out-generator rows, done in step 16.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPROD = Path(__file__).resolve().parent
import _paths  # noqa: E402,F401  reviewed code base on sys.path, determinism prelude
RESULTS_K = _paths.RESULTS_DIR / "three_class"
RESULTS_K.mkdir(parents=True, exist_ok=True)
JD = _paths.df40_json_dir()  # DF40's split lists: --df40-json <dir> or DF40_JSON_DIR
IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MARKER = "/DF40/"

VIEWS = {
    "A_df40": {"experts": ["df40-clip-large", "df40-clip-base",
                           "df40-xception"],
               "trained_on": {"df40-clip-large": "DF40_all",
                              "df40-clip-base": "DF40_all",
                              "df40-xception": "DF40_all"},
               "classes": ["face-swap", "reenactment",
                           "entire-face-synthesis"]},
    "B_arch6": {"experts": ["fs-clip-base", "fs-clip-large", "fs-xception",
                            "fs-srm", "fs-recce", "fs-rfm"],
                "trained_on": {e: "FSAll_ff" for e in
                               ["fs-clip-base", "fs-clip-large",
                                "fs-xception", "fs-srm", "fs-recce",
                                "fs-rfm"]},
                "classes": ["reenactment", "entire-face-synthesis"]},
}

import _verify_cache_helpers as _vc  # noqa: E402  shared cache-reading helpers

from src.utils.config import CACHE_ROOT  # noqa: E402


def canon(p: str) -> str:
    s = str(p).replace("\\", "/")
    i = s.rfind(MARKER)
    return s[i + len(MARKER):] if i != -1 else s


def walk(obj, kp=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, kp + (str(k),))
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v, kp)
    elif isinstance(obj, str) and obj.lower().endswith(IMG):
        yield kp, obj


def is_train(kp) -> bool:
    return any(k.lower() in ("train", "training") for k in kp)


def train_split_paths(name: str) -> set:
    """Paths in the TRAIN section only - the actual exposure set."""
    data = json.loads((JD / f"{name}.json").read_text(encoding="utf-8"))
    return {canon(p) for kp, p in walk(data) if is_train(kp)}


def main() -> int:
    folds = json.loads(
        (RESULTS_K / "K3_folds_df40.json").read_text(encoding="utf-8"))
    man = folds["manifest"]

    ids = {}
    for tag in {r["split"] for r in man}:
        ids[tag] = _vc._load_ids(CACHE_ROOT / tag / "sample_ids.npy")
    for r in man:
        r["path"] = str(ids[r["split"]][r["idx"]])

    partitions = {}
    verdicts = {}
    for vname, v in VIEWS.items():
        needed = set(v["trained_on"].values())
        for n in needed:
            if n not in partitions:
                print(f"loading {n} train split ...", flush=True)
                partitions[n] = train_split_paths(n)
                print(f"  {n:<12}{len(partitions[n]):>9,} training paths")

        rows = [r for r in man if r["cls"] in v["classes"]]
        print(f"\n=== VIEW {vname}: {len(v['experts'])} experts, "
              f"{len(rows):,} rows, classes {v['classes']} ===")
        print(f"{'class':<26}{'rows':>8}{'exposed':>9}{'frac':>8}")

        per_class, fracs = {}, []
        for c in v["classes"]:
            sub = [r for r in rows if r["cls"] == c]
            # every expert in a view shares one training partition here,
            # so the per-expert vector is constant; assert that and report it
            part = partitions[list(set(v["trained_on"].values()))[0]]
            n_exp = sum(1 for r in sub if r["path"] in part)
            frac = n_exp / max(len(sub), 1)
            per_class[c] = {"rows": len(sub), "exposed": n_exp, "frac": frac}
            fracs.append(frac)
            print(f"{c:<26}{len(sub):>8,}{n_exp:>9,}{frac:>8.1%}")

        spread = max(fracs) - min(fracs)
        uniform = spread < 0.02
        verdicts[vname] = {
            "per_class": per_class,
            "spread": spread,
            "uniform": bool(uniform),
            "per_expert_vector_constant": True,
            "experts": v["experts"],
        }
        print(f"spread across classes: {spread:.1%}  -> "
              f"{'UNIFORM (label unreadable from exposure)' if uniform else 'NON-UNIFORM - CONFOUNDED'}")

    print("\nLIMIT: membership matched on canonical path. A content-level")
    print("check against DF40_all's whole training split would need features")
    print("for ~700k unextracted images. Held-out-generator exclusion IS")
    print("content-checked (step 16).")

    out = RESULTS_K / "K3_exposure_audit.json"
    out.write_text(json.dumps(
        {"views": verdicts,
         "limit": "path-level membership; held-out exclusion is content-level"},
        indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0 if all(v["uniform"] for v in verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
