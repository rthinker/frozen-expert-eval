# C14: Build and validate construction A's three-class folds.
"""K=3 manipulation-regime attribution: dataset build + corrected pre-flight.

See camera_ready/scripts/analyses/README.md#three-class-protocol for the design:

  POOL: df40-only (df40-clip-large, df40-clip-base, df40-xception). All
  three were trained on DF40_all = FS u FR u EFS u held-out, so every fake
  class is seen by EVERY expert. Exposure is total but UNIFORM, therefore
  it carries no class information - no combiner can read the label off
  "which expert memorised this". This is the only pool with that property;
  balanced8 equalised the COUNT of exposed experts but not their IDENTITY.

  CLASSES: face-swap / reenactment / entire-face-synthesis (K=3).

  FOLDS: stratified GROUPED. Families are single-class (measured: 0 of 32
  contain >1 class), so a fold must hold out family groups from all three
  classes at once. Content-connected families are merged first by
  union-find over shared feature-row hashes.

  CHECKS (v1 got all four wrong or missing):
    1. held-out-generator removal by CONTENT, not path;
    2. per-CLASS support in BOTH train and test of every fold (v1 checked
       total held-out rows, which passed single-class folds);
    3. content-disjointness across every fold boundary after merging;
    4. exposure uniformity - NOT done here. This script only records the
       rationale in its output. The actual per-row, train-split-specific
       audit is `appI1_k3_exposure_audit_v2.py`, which must pass before fitting.
       (v2 of this file claimed check 4 was performed; it was not.)

Writes results/K3_folds_df40.json for the fitting stage.
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
MIN_PER_CLASS_TEST = 150
MIN_PER_CLASS_TRAIN = 400
N_FOLDS_TARGET = 5

REGIMES = {"face-swap": "FSAll_ff", "reenactment": "FRAll_ff",
           "entire-face-synthesis": "EFSAll_ff"}
HELD_OUT = ["whichisreal", "CollabDiff", "MidJourney", "heygen",
            "starganv2", "deepfacelab", "stargan", "styleclip"]
CLASSES = list(REGIMES)

import _verify_cache_helpers as _vc  # noqa: E402  shared cache-reading helpers

from src.utils.config import CACHE_ROOT, FIT_TAG, SELECT_TAG  # noqa: E402


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


def cls_of(kp) -> str:
    for k in kp:
        lk = k.lower()
        if lk.endswith("_real") or lk == "real":
            return "real"
        if lk.endswith("_fake") or lk == "fake":
            return "fake"
    return "unknown"


class UF:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra


def main() -> int:
    print("loading regime partitions (~40s) ...", flush=True)
    regime_fake = {}
    for r, f in REGIMES.items():
        data = json.loads((JD / f"{f}.json").read_text(encoding="utf-8"))
        regime_fake[r] = {canon(p) for kp, p in walk(data)
                          if cls_of(kp) == "fake"}

    expert = _vc._hash_expert()
    print(f"content hashing via {expert} ...", flush=True)

    # ---- CHECK 1: held-out-generator content, not just paths ----------
    heldout_hashes = set()
    for g in HELD_OUT:
        d = CACHE_ROOT / g
        if not d.is_dir():
            raise SystemExit(f"missing held-out generator folder: {d}")
        heldout_hashes |= set(_vc._content_hashes(d, expert))
    print(f"  held-out generator content hashes: {len(heldout_hashes):,}")

    rows, dropped = [], Counter()
    hashes = {}
    for tag in (FIT_TAG, SELECT_TAG):
        d = CACHE_ROOT / tag
        ids = _vc._load_ids(d / "sample_ids.npy")
        labels = np.load(d / "labels.npy")
        hashes[tag] = _vc._content_hashes(d, expert)
        for i, (sid, lab) in enumerate(zip(ids, labels)):
            if int(lab) == 0:
                dropped["real (no real class in K=3)"] += 1
                continue
            if hashes[tag][i] in heldout_hashes:
                dropped["content-matches a held-out generator row"] += 1
                continue
            s = str(sid)
            hit = [r for r, st in regime_fake.items() if s in st]
            if len(hit) != 1:
                dropped["no unique regime"] += 1
                continue
            rows.append({"split": tag, "idx": i, "family": s.split("/")[0],
                         "cls": hit[0], "h": hashes[tag][i]})

    print(f"\nrows kept: {len(rows):,}")
    for k, v in dropped.most_common():
        print(f"  dropped, {k}: {v:,}")
    bal = Counter(r["cls"] for r in rows)
    print("\nclass balance:")
    for c in CLASSES:
        print(f"  {c:<24}{bal[c]:>7,}")

    # ---- merge content-connected families (union-find) ----------------
    byhash = defaultdict(set)
    for r in rows:
        byhash[r["h"]].add(r["family"])
    uf = UF()
    for r in rows:
        uf.find(r["family"])
    merged = 0
    for fams in byhash.values():
        fams = sorted(fams)
        for f in fams[1:]:
            if uf.find(fams[0]) != uf.find(f):
                merged += 1
            uf.union(fams[0], f)
    for r in rows:
        r["group"] = uf.find(r["family"])
    groups = defaultdict(Counter)
    for r in rows:
        groups[r["group"]][r["cls"]] += 1
    print(f"\nfamilies merged into groups by shared content: {merged} merges")
    by_cls_groups = defaultdict(list)
    for g, c in groups.items():
        by_cls_groups[max(c, key=c.get)].append(g)
    for c in CLASSES:
        print(f"  {c:<24}{len(by_cls_groups[c]):>3} groups")

    n_folds = min(len(by_cls_groups[c]) for c in CLASSES)
    n_folds = min(n_folds, N_FOLDS_TARGET)
    print(f"\nfolds constructible (capped by rarest class): {n_folds}")

    # ---- stratified grouped folds: round-robin groups per class -------
    rng = np.random.default_rng(1024)
    assign = {}
    for c in CLASSES:
        gs = sorted(by_cls_groups[c], key=lambda g: -sum(groups[g].values()))
        for j, g in enumerate(gs):
            assign[g] = j % n_folds
    for r in rows:
        r["fold"] = assign[r["group"]]

    # ---- CHECK 2: per-CLASS support in BOTH train and test ------------
    print(f"\n{'fold':<6}" + "".join(f"{c[:12]:>14}" for c in CLASSES)
          + "   verdict")
    fold_ok = {}
    for f in range(n_folds):
        te = Counter(r["cls"] for r in rows if r["fold"] == f)
        tr = Counter(r["cls"] for r in rows if r["fold"] != f)
        ok = (min(te[c] for c in CLASSES) >= MIN_PER_CLASS_TEST and
              min(tr[c] for c in CLASSES) >= MIN_PER_CLASS_TRAIN)
        fold_ok[f] = ok
        print(f"{f:<6}" + "".join(f"{te[c]:>14,}" for c in CLASSES)
              + ("   OK" if ok else "   UNDER-POWERED"))

    # ---- CHECK 3: content-disjointness across fold boundaries ---------
    fold_hashes = defaultdict(set)
    for r in rows:
        fold_hashes[r["fold"]].add(r["h"])
    leaks = {}
    for f in range(n_folds):
        other = set()
        for g in range(n_folds):
            if g != f:
                other |= fold_hashes[g]
        n = len(fold_hashes[f] & other)
        if n:
            leaks[f] = n
    print(f"\ncontent shared across fold boundaries: "
          f"{leaks if leaks else 'none - clean'}")

    out = {
        "pool": ["df40-clip-large", "df40-clip-base", "df40-xception"],
        "pool_rationale": "all three trained on DF40_all, so exposure to "
                          "every fake class is identical - uniform, hence "
                          "uninformative about the label",
        "classes": CLASSES, "n_folds": n_folds,
        "class_balance": {c: bal[c] for c in CLASSES},
        "dropped": dict(dropped),
        "content_expert": expert,
        "merges": merged,
        "fold_ok": fold_ok,
        "cross_fold_content_leaks": leaks,
        "manifest": [{k: r[k] for k in ("split", "idx", "family", "group",
                                        "cls", "fold")} for r in rows],
    }
    dest = RESULTS_K / "K3_folds_df40.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")
    ok = all(fold_ok.values()) and not leaks
    print("\nPRE-FLIGHT:", "PASS" if ok else "REVIEW")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
