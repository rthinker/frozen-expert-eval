"""Gate G1: exclusion manifest for content-disjoint LOGO (camera_ready/RELEASE_MANIFEST.md, Paper items).

For one pool, hashes every held-out test row (SHA-1 of its cached feature vectors across the pool's experts) and
lists, for each LOGO fold, the training rows (rows of the other seven generators) whose features are identical to a
row of the held-out generator. The content-disjoint runners remove exactly these rows from the fold's training data.

Gate: the reuse check recorded 7 images shared by two held-out generators (CollabDiff and StyleCLIP), i.e. 14 rows
over the folds. The script exits with status 1 if the count differs (``--expect-images``, ``--expect-rows``).

Usage (from the repository root):
    python -B camera_ready/scripts/runs/logo_exclusions.py --pool arch6 --out camera_ready/output/checks/exclusions_arch6.json
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cr_paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--codebase", help="code base root (default: this repository; MITIGATION_CODEBASE overrides it)")
    ap.add_argument("--out", type=Path, required=True, help="destination for the new exclusion manifest")
    ap.add_argument("--expect-images", type=int, default=7)
    ap.add_argument("--expect-rows", type=int, default=14)
    a = ap.parse_args()
    a.out = _cr_paths.out_path(a.out)
    _cr_paths.setup(a.codebase)
    from logo_common import load_generators, row_hashes, write_json
    from src.utils.config import CACHE_ROOT, get_pool_experts
    from src.utils.dataset_helpers import TEST_GENERATORS

    experts = get_pool_experts(a.pool)
    data = load_generators(experts, CACHE_ROOT)
    hashes = {g: row_hashes(data[g][0]) for g in TEST_GENERATORS}
    owners = defaultdict(set)
    for g in TEST_GENERATORS:
        for h in hashes[g]:
            owners[h].add(g)
    shared = {h: sorted(gs) for h, gs in owners.items() if len(gs) > 1}

    folds, total = [], 0
    for held in TEST_GENERATORS:
        test = set(hashes[held])
        excl = {}
        for g in TEST_GENERATORS:
            if g == held:
                continue
            idx = [i for i, h in enumerate(hashes[g]) if h in test]
            if idx:
                excl[g] = idx
        n = sum(len(v) for v in excl.values())
        total += n
        labels = {g: {"real": int(sum(1 for i in v if data[g][2][i] == 0)),
                      "fake": int(sum(1 for i in v if data[g][2][i] == 1))} for g, v in excl.items()}
        folds.append({"held_out": held, "excluded_rows": excl, "n_excluded": n, "labels": labels})
        print(f"  fold {held:<12} excludes {n} training rows {dict((g, len(v)) for g, v in excl.items())}")

    res = {
        "pool": a.pool, "experts": experts,
        "rule": "remove from a fold's training generators every row whose cached feature vector (all experts of "
                "the pool, float32 as loaded) is identical to a row of the held-out generator's test set",
        "shared_images": len(shared),
        "shared_pairs": {"|".join(gs): sum(1 for v in shared.values() if tuple(v) == gs) for gs in
                         sorted({tuple(v) for v in shared.values()})},
        "rows_excluded_total": total,
        "expected": {"images": a.expect_images, "rows": a.expect_rows},
        "folds": folds,
        "_written_utc": datetime.now(timezone.utc).isoformat(),
    }
    ok = (len(shared) == a.expect_images and total == a.expect_rows)
    res["gate_G1"] = "PASS" if ok else "FAIL"
    out = write_json(res, a.out)
    print(f"{a.pool}: {len(shared)} shared images, {total} rows excluded over the folds -> G1 {res['gate_G1']}")
    print(f"written {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
