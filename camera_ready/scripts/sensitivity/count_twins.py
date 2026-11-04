"""Count the exact duplicates in the training data that the deduplication checks remove (no model is fitted).

For every LOGO fold of the primary content-disjoint runs (``logo_common`` merge order, each pool's exclusion list,
``split_fit_select`` with 15% SELECT and seed 1024) and for the standard split, it counts the SELECT rows that have an
exact feature twin in FIT (identical features across the pool's experts, ``logo_common.row_hashes``), the generators
those twins come from, the twins whose FIT copy has another label, and the extra copies inside FIT and inside SELECT.
StarGAN-v2's test list repeats images (4,000 rows, 1,024 distinct), so nearly every twin is a StarGAN-v2 row.

The supplied output is camera_ready/output/results/sensitivity/twins.json; summarize_v2.py reports it. A rerun writes to
--out (default camera_ready/output/results/sensitivity_rerun/twins.json). Needs the feature cache. From the
repository root:

    python camera_ready/scripts/sensitivity/count_twins.py [--pools arch6 clip4 strong4 weak6]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _out_guard import output_dir  # noqa: E402

CR = Path(__file__).resolve().parents[2]          # camera_ready/
ROOT = CR.parent
POOLS = ("arch6", "clip4", "strong4", "weak6")
RERUN = "camera_ready/output/results/sensitivity_rerun/twins.json"


def twin_counts(fit_rows, sel_rows):
    """Rows are (generator, row index, feature hash, label). Counts of SELECT rows with a FIT twin and of copies."""
    fit_h = Counter(r[2] for r in fit_rows)
    fit_labels = {}
    for r in fit_rows:
        fit_labels.setdefault(r[2], set()).add(r[3])
    twins = [r for r in sel_rows if r[2] in fit_h]
    sel_h = Counter(r[2] for r in sel_rows)
    return {"fit_rows": len(fit_rows), "select_rows": len(sel_rows), "select_rows_with_fit_twin": len(twins),
            "share": round(len(twins) / len(sel_rows), 4) if sel_rows else None,
            "twins_by_generator": dict(Counter(r[0] for r in twins)),
            "twins_with_other_label": sum(1 for r in twins if fit_labels[r[2]] != {r[3]}),
            "extra_copies_inside_fit": sum(c - 1 for c in fit_h.values() if c > 1),
            "extra_copies_inside_select": sum(c - 1 for c in sel_h.values() if c > 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pools", nargs="+", default=list(POOLS), choices=list(POOLS))
    ap.add_argument("--out", default=RERUN, help=f"relative to the repository root (default {RERUN})")
    a = ap.parse_args()
    # The folder of --out passes the same guard as the other sensitivity
    # scripts (no file in the supplied outputs, the primary results, the inputs or the feature cache, however the
    # path is spelled); it replaces a check of the path's spelling
    output_dir(str(Path(a.out).parent))
    sys.path.insert(0, str(CR / "scripts" / "runs"))
    import _cr_paths
    _cr_paths.setup(str(ROOT))
    import numpy as np
    from logo_common import load_content_manifest, load_generators, row_hashes, split_fit_select
    from src.utils.config import CACHE_ROOT, FIT_TAG, SELECT_TAG, get_pool_experts
    from src.utils.dataset_helpers import TEST_GENERATORS
    from src.utils.io import load_pool_data

    out = {"logo": {}, "standard": {}}
    for pool in a.pools:
        experts = get_pool_experts(pool)
        data = load_generators(experts, CACHE_ROOT)
        manifest_path = CR / "output/results/mitigation/exclusions" / f"logo_exclusions_{pool}.json"
        manifest = load_content_manifest(manifest_path, pool, experts, data)
        hashes = {g: row_hashes(data[g][0]) for g in TEST_GENERATORS}
        out["logo"][pool] = {}
        for held in TEST_GENERATORS:
            excl = manifest.get(held, {})
            rows = []            # the fold's training rows, in fold_view's merge order, with the exclusions applied
            for g in TEST_GENERATORS:
                if g == held:
                    continue
                drop = set(excl.get(g, []))
                rows += [(g, i, hashes[g][i], int(data[g][2][i])) for i in range(len(data[g][2])) if i not in drop]
            labels = np.array([r[3] for r in rows])
            index = [np.arange(len(rows)).reshape(-1, 1)]          # split the row indices as the runners split rows
            fit_i, _fl, _fy, sel_i, _sl, _sy = split_fit_select(index, np.zeros((len(rows), 1, 2)), labels, 0.15, 1024)
            res = twin_counts([rows[i] for i in fit_i[0].ravel()], [rows[i] for i in sel_i[0].ravel()])
            out["logo"][pool][held] = res
            print(f"{pool:<8} {held:<12} SELECT {res['select_rows']:>5}  with a FIT twin "
                  f"{res['select_rows_with_fit_twin']:>4} ({res['share']:.1%})  extra copies in FIT "
                  f"{res['extra_copies_inside_fit']}", flush=True)
        split = {}
        for tag in (FIT_TAG, SELECT_TAG):
            feats, _logits, y = load_pool_data(list(experts), tag, CACHE_ROOT)
            split[tag] = [(tag, i, h, int(y[i])) for i, h in enumerate(row_hashes(feats))]
        res = twin_counts(split[FIT_TAG], split[SELECT_TAG])
        out["standard"][pool] = res
        print(f"{pool:<8} standard     SELECT {res['select_rows']:>5}  with a FIT twin "
              f"{res['select_rows_with_fit_twin']:>4}  extra copies in FIT {res['extra_copies_inside_fit']}, "
              f"in SELECT {res['extra_copies_inside_select']}", flush=True)
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
