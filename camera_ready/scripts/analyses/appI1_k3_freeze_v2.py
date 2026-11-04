# C16: Validate and write the three-class manifests.
"""Freeze the final K=3 evaluation manifests for both constructions.

Does the three things that must happen BEFORE any model is fitted
(camera_ready/scripts/analyses/README.md#three-class-protocol):

  1. build View B  - arch6 pool, face-swap EXCLUDED, classes
     reenactment / GAN-synthesis / diffusion-synthesis, with the same
     content-level held-out removal and union-find group merging as View A;
  2. rebalance both views deterministically by the locked rule -
     within each class, sort content-groups by descending row count and
     assign each in turn to the fold currently holding the fewest rows of
     that class; ties broken by lowest fold index, then group name
     ascending. No randomness anywhere;
  3. freeze each manifest with a SHA-256 over the ordered
     (split, idx, fold, class) tuples, so the fitting stage can refuse to
     run if the splits have drifted.

Also re-runs both content checks per view (held-out exclusion, cross-fold
disjointness) and the per-CLASS support check in train AND test.

RAM-light: hashes plus set arithmetic, no feature matrices held. Safe to
run beside a LOGO job.
"""

from __future__ import annotations

import hashlib
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
MIN_TEST, MIN_TRAIN = 150, 400

REGIMES = {"face-swap": "FSAll_ff", "reenactment": "FRAll_ff",
           "entire-face-synthesis": "EFSAll_ff"}
HELD_OUT = ["whichisreal", "CollabDiff", "MidJourney", "heygen",
            "starganv2", "deepfacelab", "stargan", "styleclip"]
GAN = {"StyleGAN2", "StyleGAN3", "StyleGANXL", "VQGAN"}
DIFF = {"DiT", "SiT", "ddim", "pixart", "rddm", "sd2.1"}

VIEWS = {
    "A_df40": {"max_folds": 5,
               "keep": lambda regime, fam: regime,          # all 3 regimes
               "pool": ["df40-clip-large", "df40-clip-base", "df40-xception"]},
    "B_arch6": {"max_folds": 5,
                "keep": lambda regime, fam: (
                    None if regime == "face-swap" else
                    ("GAN-synthesis" if fam in GAN else
                     "diffusion-synthesis" if fam in DIFF else
                     "reenactment" if regime == "reenactment" else None)),
                "pool": ["fs-clip-base", "fs-clip-large", "fs-xception",
                         "fs-srm", "fs-recce", "fs-rfm"]},
}

import _verify_cache_helpers as _vc  # noqa: E402  shared cache-reading helpers

from src.utils.config import CACHE_ROOT, FIT_TAG, SELECT_TAG  # noqa: E402


def canon(p):
    s = str(p).replace("\\", "/")
    i = s.rfind(MARKER)
    return s[i + len(MARKER):] if i != -1 else s


def walk(o, kp=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, kp + (str(k),))
    elif isinstance(o, list):
        for v in o:
            yield from walk(v, kp)
    elif isinstance(o, str) and o.lower().endswith(IMG):
        yield kp, o


def cls_of(kp):
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
    print("loading regime partitions ...", flush=True)
    regime_fake = {}
    for r, f in REGIMES.items():
        d = json.loads((JD / f"{f}.json").read_text(encoding="utf-8"))
        regime_fake[r] = {canon(p) for kp, p in walk(d) if cls_of(kp) == "fake"}

    expert = _vc._hash_expert()
    print(f"content fingerprint expert: {expert} (same for both views, so "
          f"image identity is defined identically)")
    heldout_h = set()
    for g in HELD_OUT:
        dd = CACHE_ROOT / g
        if not dd.is_dir():
            raise SystemExit(f"missing held-out generator folder: {dd}")
        heldout_h |= set(_vc._content_hashes(dd, expert))

    base, hashes = [], {}
    for tag in (FIT_TAG, SELECT_TAG):
        dd = CACHE_ROOT / tag
        ids = _vc._load_ids(dd / "sample_ids.npy")
        labels = np.load(dd / "labels.npy")
        hashes[tag] = _vc._content_hashes(dd, expert)
        for i, (sid, lab) in enumerate(zip(ids, labels)):
            if int(lab) == 0 or hashes[tag][i] in heldout_h:
                continue
            s = str(sid)
            hit = [r for r, st in regime_fake.items() if s in st]
            if len(hit) != 1:
                continue
            base.append({"split": tag, "idx": i, "family": s.split("/")[0],
                         "regime": hit[0], "h": hashes[tag][i]})
    print(f"base rows after content-level held-out removal: {len(base):,}")

    out_all = {}
    for vname, v in VIEWS.items():
        rows = []
        for r in base:
            c = v["keep"](r["regime"], r["family"])
            if c:
                rows.append({**r, "cls": c})
        classes = sorted({r["cls"] for r in rows})
        print(f"\n===== VIEW {vname} =====")
        print(f"rows {len(rows):,}   classes {classes}")

        uf = UF()
        byhash = defaultdict(set)
        for r in rows:
            uf.find(r["family"]); byhash[r["h"]].add(r["family"])
        for fams in byhash.values():
            fams = sorted(fams)
            for f in fams[1:]:
                uf.union(fams[0], f)
        for r in rows:
            r["group"] = uf.find(r["family"])

        gsize = defaultdict(Counter)
        for r in rows:
            gsize[r["group"]][r["cls"]] += 1
        gcls = {g: max(c, key=c.get) for g, c in gsize.items()}
        per_cls_groups = defaultdict(list)
        for g, c in gcls.items():
            per_cls_groups[c].append(g)
        for c in classes:
            print(f"  {c:<22}{sum(1 for r in rows if r['cls']==c):>7,} rows"
                  f"  {len(per_cls_groups[c]):>3} groups")

        n_folds = min(min(len(per_cls_groups[c]) for c in classes),
                      v["max_folds"])
        print(f"  folds: {n_folds}")

        # ---- locked deterministic balancing rule ----------------------
        assign, load = {}, {c: Counter({f: 0 for f in range(n_folds)})
                            for c in classes}
        for c in classes:
            gs = sorted(per_cls_groups[c],
                        key=lambda g: (-sum(gsize[g].values()), g))
            for g in gs:
                f = min(range(n_folds), key=lambda k: (load[c][k], k))
                assign[g] = f
                load[c][f] += sum(gsize[g].values())
        for r in rows:
            r["fold"] = assign[r["group"]]

        print(f"  {'fold':<6}" + "".join(f"{c[:11]:>13}" for c in classes)
              + f"{'total':>9}   verdict")
        ok_all = True
        for f in range(n_folds):
            te = Counter(r["cls"] for r in rows if r["fold"] == f)
            tr = Counter(r["cls"] for r in rows if r["fold"] != f)
            ok = (min(te[c] for c in classes) >= MIN_TEST and
                  min(tr[c] for c in classes) >= MIN_TRAIN)
            ok_all &= ok
            print(f"  {f:<6}" + "".join(f"{te[c]:>13,}" for c in classes)
                  + f"{sum(te.values()):>9,}   {'OK' if ok else 'UNDER-POWERED'}")

        fh = defaultdict(set)
        for r in rows:
            fh[r["fold"]].add(r["h"])
        leaks = {}
        for f in range(n_folds):
            other = set()
            for g in range(n_folds):
                if g != f:
                    other |= fh[g]
            n = len(fh[f] & other)
            if n:
                leaks[f] = n
        print(f"  cross-fold content: {leaks if leaks else 'clean'}")

        # Digest covers ONLY (split, idx, fold, cls). The group field is
        # carried alongside for the group-level shuffle control and does
        # not enter the hash, so adding it cannot invalidate an existing
        # freeze - re-running this script must reproduce the same digest.
        man = sorted(((r["split"], r["idx"], r["fold"], r["cls"], r["group"])
                      for r in rows), key=lambda t: (t[0], t[1]))
        digest = hashlib.sha256(
            "\n".join(f"{a}|{b}|{c}|{d}" for a, b, c, d, _ in man)
            .encode()).hexdigest()
        sizes = {f: sum(1 for r in rows if r["fold"] == f)
                 for f in range(n_folds)}
        print(f"  fold sizes {sizes}  spread "
              f"{max(sizes.values()) - min(sizes.values()):,}")
        print(f"  SHA-256 {digest}")

        out = {"view": vname, "pool": v["pool"], "classes": classes,
               "n_folds": n_folds, "content_expert": expert,
               "class_balance": {c: sum(1 for r in rows if r["cls"] == c)
                                 for c in classes},
               "fold_sizes": sizes, "cross_fold_content_leaks": leaks,
               "all_folds_supported": bool(ok_all),
               "balancing_rule": "per class: groups by descending size -> "
                                 "fold with fewest rows of that class; ties "
                                 "by lowest fold index then group name",
               "manifest_sha256": digest,
               "manifest": [{"split": a, "idx": b, "fold": c, "cls": d,
                             "group": g} for a, b, c, d, g in man]}
        dest = RESULTS_K / f"K3_FROZEN_{vname}.json"
        dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  wrote {dest.name}")
        out_all[vname] = {"digest": digest, "ok": bool(ok_all and not leaks),
                          "folds": n_folds, "rows": len(rows)}

    print("\n===== FREEZE SUMMARY =====")
    for k, v in out_all.items():
        print(f"  {k:<10} folds={v['folds']}  rows={v['rows']:>7,}  "
              f"{'READY' if v['ok'] else 'REVIEW'}  {v['digest'][:16]}...")
    return 0 if all(v["ok"] for v in out_all.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
