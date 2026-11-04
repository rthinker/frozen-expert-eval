# C03: Count content matches beyond path matches.
"""Characterise the rows that are reused by CONTENT but not by PATH.

Tier A.4 follow-up. The extended verifier found 71.9% content-level reuse
against 66.3% path-level. This script says exactly which rows account for the
difference, whether they are real or fake, and which training split they
landed in - the detail a disclosure needs.

Runs against the reviewed code base and its feature cache (the repository root)
and uses the shared cache-reading helpers (`_verify_cache_helpers.py`).
Writes only under camera_ready/output/results/analyses/.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: E402,F401  reviewed code base on sys.path, determinism prelude

REPROD = Path(__file__).resolve().parent
OUT = _paths.RESULTS_DIR

import _verify_cache_helpers as _vc  # noqa: E402  shared cache-reading helpers

from src.utils.config import CACHE_ROOT, FIT_TAG, SELECT_TAG  # noqa: E402
from src.utils.dataset_helpers import TEST_GENERATORS  # noqa: E402

EXPERT = _vc._hash_expert()


def main() -> None:
    h2split: dict[str, str] = {}
    train_ids: list = []
    for tag in (FIT_TAG, SELECT_TAG):
        d = CACHE_ROOT / tag
        ids = _vc._load_ids(d / "sample_ids.npy")
        hs = _vc._content_hashes(d, EXPERT)
        train_ids.append(ids)
        for h in hs:
            h2split.setdefault(h, tag)
    train_id_set = set(np.concatenate(train_ids).tolist())

    report = {"content_expert": EXPERT, "per_generator": {}}
    tot = {"n": 0, "path": 0, "content": 0, "content_only": 0,
           "content_only_fake": 0}

    print(f"{'generator':<14}{'n':>6}{'path':>7}{'content':>9}"
          f"{'extra':>7}{'fake':>6}{'in FIT':>8}{'in SEL':>8}")
    for gen in TEST_GENERATORS:
        d = CACHE_ROOT / gen
        if not d.is_dir():
            raise SystemExit(f"missing held-out generator folder: {d}")
        ids = _vc._load_ids(d / "sample_ids.npy")
        labels = np.load(d / "labels.npy")
        hs = _vc._content_hashes(d, EXPERT)

        n = len(ids)
        by_path = np.array([i in train_id_set for i in ids])
        by_content = np.array([h in h2split for h in hs])
        extra = by_content & ~by_path

        n_extra = int(extra.sum())
        extra_fake = int(labels[extra].sum())
        in_fit = sum(1 for h, e in zip(hs, extra) if e and h2split[h] == FIT_TAG)
        in_sel = sum(1 for h, e in zip(hs, extra) if e and h2split[h] == SELECT_TAG)

        report["per_generator"][gen] = {
            "n": n,
            "path_reuse": int(by_path.sum()),
            "content_reuse": int(by_content.sum()),
            "content_only": n_extra,
            "content_only_fake": extra_fake,
            "content_only_real": n_extra - extra_fake,
            "content_only_in_fit": in_fit,
            "content_only_in_select": in_sel,
        }
        for k, v in (("n", n), ("path", int(by_path.sum())),
                     ("content", int(by_content.sum())),
                     ("content_only", n_extra),
                     ("content_only_fake", extra_fake)):
            tot[k] += v

        print(f"{gen:<14}{n:>6}{int(by_path.sum()):>7}{int(by_content.sum()):>9}"
              f"{n_extra:>7}{extra_fake:>6}{in_fit:>8}{in_sel:>8}")

    report["totals"] = tot
    print(f"\n{'TOTAL':<14}{tot['n']:>6}{tot['path']:>7}{tot['content']:>9}"
          f"{tot['content_only']:>7}{tot['content_only_fake']:>6}")
    print(f"path {tot['path'] / tot['n']:.1%}   "
          f"content {tot['content'] / tot['n']:.1%}")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "A4_content_only_rows.json"
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
