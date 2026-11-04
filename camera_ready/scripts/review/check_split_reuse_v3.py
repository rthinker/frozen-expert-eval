"""Check split reuse in the DF40 frozen-expert cache.

Route A uses the DF40 split JSONs to count paths from each held-out generator
that also occur in DF40_all.json, the aggregate training list. It includes GIFs.

Route B uses numeric feature arrays to count test rows with content identical
to FIT or SELECT rows. It also counts SELECT rows identical to FIT rows,
distinct test images, test rows identical to their split's first row (the
signature of the zero-byte fallback), and identical content across test splits.

Use --ids for the path and label checks (one <split>.txt per split, one ID per
line in cache order). With --raw-root, the check looks up each ID relative to
the image root and checks file sizes without opening images. It stops before
writing if any source file is missing.

The harness runs this as C01. Use --carry-from to copy Route A or image checks
when their inputs are unavailable. The report records the copied routes and
source SHA-256 under _carried_over. With --expect, only computed counts are
compared, and a mismatch exits with status 1.

    python camera_ready/scripts/review/check_split_reuse_v3.py --cache CACHE_DIR
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HELD_OUT = ["whichisreal", "CollabDiff", "MidJourney", "heygen",
            "starganv2", "deepfacelab", "stargan", "styleclip"]
FIT, SELECT = "FIT", "SELECT"
SPLITS = [FIT, SELECT] + HELD_OUT
ARCH6 = ["fs-clip-base", "fs-clip-large", "fs-xception", "fs-srm", "fs-recce", "fs-rfm"]
CONTENT_EXPERT = "df40-clip-large"
IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")
MARKER = "/DF40/"
# Retained constant for compatibility.
LAYOUT_MISSING_SHARE = 0.5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_block(cache: Path, split: str, experts: list, kind: str) -> np.ndarray:
    arrays = [np.load(cache / split / f"{e}_{kind}.npy") for e in experts]   # numeric arrays only
    if kind == "logits":
        arrays = [a.astype(np.float32) for a in arrays]
    return np.ascontiguousarray(np.concatenate(arrays, axis=1))


def row_hashes(block: np.ndarray) -> list:
    return [hashlib.sha1(row.tobytes()).hexdigest() for row in block]


# ---------------------------------------------------------------- route A

def json_paths(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from json_paths(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from json_paths(v)
    elif isinstance(obj, str) and obj.lower().endswith(IMG):
        yield obj


def canon(p: str) -> str:
    s = str(p).replace("\\", "/")
    i = s.rfind(MARKER)
    return s[i + len(MARKER):] if i != -1 else s


def route_a(json_dir: Path) -> dict:
    pooled = {canon(p) for p in json_paths(json.loads((json_dir / "DF40_all.json").read_text(encoding="utf-8")))}
    per, total, hit = {}, 0, 0
    for g in HELD_OUT:
        f = json_dir / f"{g}.json"
        if not f.exists():
            continue
        test = {canon(p) for p in json_paths(json.loads(f.read_text(encoding="utf-8")))}
        n = len(test & pooled)
        per[g] = {"test_paths": len(test), "in_DF40_all": n}
        total, hit = total + len(test), hit + n
    return {"per_generator": per, "total_test_paths": total, "in_DF40_all": hit,
            "pct": round(100 * hit / max(total, 1), 1)}


# ---------------------------------------------------------------- route B

def route_b(cache: Path) -> dict:
    labels = {s: np.load(cache / s / "labels.npy").astype(int) for s in SPLITS}
    content = {s: row_hashes(load_block(cache, s, [CONTENT_EXPERT], "features")) for s in SPLITS}
    arch6_feat = {s: row_hashes(load_block(cache, s, ARCH6, "features")) for s in SPLITS}
    arch6_logit = {s: row_hashes(load_block(cache, s, ARCH6, "logits")) for s in HELD_OUT}

    train_content = set(content[FIT]) | set(content[SELECT])
    reuse, tot_n, tot_c = {}, 0, 0
    for g in HELD_OUT:
        hit = [h in train_content for h in content[g]]
        reuse[g] = {"n": len(hit), "content_reuse": int(sum(hit))}
        tot_n, tot_c = tot_n + len(hit), tot_c + int(sum(hit))
    out = {"content_reuse": {"definition": f"test rows whose {CONTENT_EXPERT} feature row (SHA-1 of the stored "
                                           "float16 bytes) also occurs in FIT or SELECT",
                             "per_generator": reuse,
                             "totals": {"n": tot_n, "content_reuse": tot_c,
                                        "content_pct": round(100 * tot_c / tot_n, 1)}}}

    fit_c, fit_a = set(content[FIT]), set(arch6_feat[FIT])
    out["select_fit_content_twins"] = {CONTENT_EXPERT: int(sum(h in fit_c for h in content[SELECT])),
                                       "arch6_features": int(sum(h in fit_a for h in arch6_feat[SELECT]))}

    distinct = {g: len(set(arch6_feat[g])) for g in HELD_OUT}
    out["distinct_test_images"] = {"definition": "distinct arch6 feature rows per generator, summed",
                                   "per_generator": distinct, "total": sum(distinct.values())}

    first = {g: int(sum(1 for h in arch6_feat[g][1:] if h == arch6_feat[g][0])) for g in HELD_OUT}
    out["first_row_signature_test_splits"] = {
        "definition": "rows after the first whose arch6 features equal the split's first row; FIT and SELECT "
                      "were re-split after extraction, so their fallback rows cannot be located this way",
        "per_generator": {g: v for g, v in first.items() if v}, "total": sum(first.values())}

    owners = defaultdict(set)
    for g in HELD_OUT:
        for h in arch6_logit[g]:
            owners[h].add(g)
    shared, real, fake = {}, 0, 0
    for g in HELD_OUT:
        idx = [i for i, h in enumerate(arch6_logit[g]) if owners[h] - {g}]
        if idx:
            shared[g] = len(idx)
            real += int(sum(1 for i in idx if labels[g][i] == 0))
            fake += int(sum(1 for i in idx if labels[g][i] == 1))
    multi = {h: gs for h, gs in owners.items() if len(gs) > 1}
    pairs = defaultdict(int)
    for gs in multi.values():
        ordered = sorted(gs)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                pairs[f"{a}|{b}"] += 1
    out["cross_generator_matches"] = {
        "definition": "test images (12 arch6 logits, float32, bit-exact) that occur under more than one held-out "
                      "generator; rows are counted per withheld generator (LOGO fold), so one shared image "
                      "contributes a row to each fold it appears in",
        "shared_images": len(multi), "pairs": dict(pairs),
        "rows_per_fold": shared, "rows_total": sum(shared.values()), "rows_real": real, "rows_fake": fake}
    return out


def path_label(p: str):
    """0 for a path with a `real` component, 1 for `fake`, None otherwise (DF40 directory convention)."""
    for part in p.split("/"):
        if part.lower() in ("real", "fake"):
            return 0 if part.lower() == "real" else 1
    return None


def raw_checks(cache: Path, ids_dir: Path, raw_root: Path) -> dict:
    """Source-file sizes for every cached row (stat only): zero-byte and missing inputs."""
    zero, missing, zero_mismatch = {}, {}, 0
    for s in SPLITS:
        ids = (ids_dir / f"{s}.txt").read_text(encoding="utf-8").splitlines()
        labels = np.load(cache / s / "labels.npy").astype(int)
        z = m = 0
        for p, y in zip(ids, labels):
            try:
                size = (raw_root / p).stat().st_size
            except FileNotFoundError:
                m += 1
                continue
            if size == 0:
                z += 1
                t = path_label(p)
                zero_mismatch += int(t is not None and t != int(y))
        if z:
            zero[s] = z
        if m:
            missing[s] = m
    return {"zero_byte": zero, "zero_byte_total": sum(zero.values()),
            "missing": missing, "missing_total": sum(missing.values()),
            "zero_byte_rows_whose_label_disagrees_with_path": zero_mismatch}


def path_checks(cache: Path, ids_dir: Path) -> dict:
    ids = {s: (ids_dir / f"{s}.txt").read_text(encoding="utf-8").splitlines() for s in SPLITS}
    labels = {s: np.load(cache / s / "labels.npy").astype(int) for s in SPLITS}
    for s in SPLITS:
        if len(ids[s]) != len(labels[s]):
            raise SystemExit(f"{s}: {len(ids[s])} ids vs {len(labels[s])} labels")
    train = set(ids[FIT]) | set(ids[SELECT])
    per, tot, fake_gens = {}, 0, 0
    for g in HELD_OUT:
        hit = [i in train for i in ids[g]]
        fake = int(sum(1 for k, v in enumerate(hit) if v and labels[g][k] == 1))
        per[g] = {"path_reuse": int(sum(hit)), "path_reuse_fake": fake}
        tot += int(sum(hit))
        fake_gens += int(fake > 0)

    bad = {s: int(sum(1 for p, y in zip(ids[s], labels[s])
                      if path_label(p) is not None and path_label(p) != int(y))) for s in SPLITS}

    # source families of FIT's real rows (leading path component), before and after the strict
    # path-disjoint filter that drops rows whose leading component names a held-out generator
    real_fit = [p for p, y in zip(ids[FIT], labels[FIT]) if int(y) == 0]
    fam_all = defaultdict(int)
    fam_kept = defaultdict(int)
    for p in real_fit:
        lead = p.split("/")[0]
        fam_all[lead] += 1
        if lead not in HELD_OUT:
            fam_kept[lead] += 1
    families = {"real_rows": len(real_fit), "families": len(fam_all), "per_family": dict(fam_all),
                "real_rows_after_strict_filter": sum(fam_kept.values()),
                "families_after_strict_filter": len(fam_kept), "per_family_after_strict_filter": dict(fam_kept)}
    return {"path_reuse": {"per_generator": per, "total": tot,
                           "generators_with_fake_path_reuse": fake_gens},
            "fit_real_source_families": families,
            "label_path_disagreements": {"per_split": {s: v for s, v in bad.items() if v},
                                         "total": sum(bad.values())}}


# ---------------------------------------------------------------- carried-over routes

def carry_over(res: dict, stored: dict, source: str, source_sha256: str) -> list:
    """Copy the routes this run did not compute from a stored result of this check: route A (DF40's lists; with
    the digest of the DF40_all.json it read) and the image-file checks (`route_b.raw_files`). Records them under
    `_carried_over` with the stored file's name and SHA-256. Returns the carried routes."""
    carried = []
    if "route_a" not in res and "route_a" in stored:
        res["route_a"] = stored["route_a"]
        sha = stored.get("_inputs", {}).get("DF40_all_json_sha256")
        if sha:
            res["_inputs"]["DF40_all_json_sha256"] = sha
        carried.append("route_a")
    if "raw_files" not in res.get("route_b", {}) and "raw_files" in stored.get("route_b", {}):
        res["route_b"]["raw_files"] = stored["route_b"]["raw_files"]
        carried.append("route_b.raw_files")
    if carried:
        res["_carried_over"] = {"routes": carried, "from": source, "sha256": source_sha256,
                                "note": "not recomputed in this run: copied from the stored result"}
    return carried


def layout_problem(raw: dict, n_rows: int):
    """Why the image root cannot serve the image checks (None when it can): some cached rows have no source file
    under it."""
    if n_rows and raw["missing_total"] > 0:
        return (f"{raw['missing_total']:,} of {n_rows:,} cached rows have no source file under --raw-root. The "
                "image checks look up <raw root>/<cache id>, the layout of the DF40 folder the cache was "
                "extracted from (<method>/ff/frames/..., FaceForensics++/..., <generator>/...); DF40's public "
                "download is organised differently. Leave out --raw-root: with --carry-from, the stored image "
                "checks are kept. Nothing was written.")
    return None


# ---------------------------------------------------------------- expectations

def flatten(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from flatten(v, key)
        else:
            yield key, v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True, help="feature cache root")
    ap.add_argument("--df40-json", type=Path, help="directory with DF40_all.json and the held-out generator JSONs")
    ap.add_argument("--ids", type=Path, help="directory of plain-text id lists, one <split>.txt per split")
    ap.add_argument("--raw-root", type=Path, help="DF40 image root; with --ids, reports zero-byte and missing inputs")
    ap.add_argument("--out", type=Path, help="write results JSON here")
    ap.add_argument("--expect", type=Path, help="JSON of expected counts to compare against")
    ap.add_argument("--carry-from", type=Path,
                    help="stored result of this check; the routes this run does not compute are copied "
                         "from it and marked as carried over")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runs"))
    import _cr_paths
    if args.out:
        args.out = _cr_paths.out_path(args.out)

    stored = None             # read before anything is written (it may be the --out file itself)
    if args.carry_from:
        if args.carry_from.is_file():
            raw_bytes = args.carry_from.read_bytes()
            stored = (json.loads(raw_bytes.decode("utf-8")), hashlib.sha256(raw_bytes).hexdigest())
        else:
            print(f"--carry-from {args.carry_from}: not found; nothing is carried over")

    res = {"_written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "_inputs": {}}
    if args.df40_json:
        res["route_a"] = route_a(args.df40_json)
        res["_inputs"]["DF40_all_json_sha256"] = sha256(args.df40_json / "DF40_all.json")
    res["route_b"] = route_b(args.cache)
    if args.ids:
        res["route_b"].update(path_checks(args.cache, args.ids))
        if args.raw_root:
            res["route_b"]["raw_files"] = raw_checks(args.cache, args.ids, args.raw_root)
            n_rows = sum(len(np.load(args.cache / s / "labels.npy")) for s in SPLITS)
            why = layout_problem(res["route_b"]["raw_files"], n_rows)
            if why:
                raise SystemExit(why)
    if stored is not None:
        carried = carry_over(res, stored[0], args.carry_from.as_posix(), stored[1])
        if carried:
            print(f"Carried over from {args.carry_from.as_posix()} (not recomputed in this run): "
                  f"{', '.join(carried)}")

    carried = tuple(res.get("_carried_over", {}).get("routes", ()))
    if "route_a" in res:
        ra = res["route_a"]
        print(f"Route A: {ra['in_DF40_all']:,} of {ra['total_test_paths']:,} held-out test paths "
              f"({ra['pct']}%) appear in DF40_all.json"
              + (" (carried over from the stored result, not recomputed)" if "route_a" in carried else ""))
    rb = res["route_b"]
    t = rb["content_reuse"]["totals"]
    print(f"Route B: {t['content_reuse']:,} of {t['n']:,} test rows ({t['content_pct']}%) reuse FIT/SELECT feature "
          f"content; SELECT-FIT twins {rb['select_fit_content_twins'][CONTENT_EXPERT]}; distinct test images "
          f"{rb['distinct_test_images']['total']:,}; first-row signature {rb['first_row_signature_test_splits']['total']}; "
          f"images shared across held-out generators {rb['cross_generator_matches']['shared_images']} "
          f"({rb['cross_generator_matches']['rows_total']} rows across folds)")
    if args.expect:
        got = dict(flatten(res))
        rows = [(k, v, got.get(k)) for k, v in flatten(json.loads(args.expect.read_text(encoding="utf-8")))
                if not k.startswith("_") and not k.endswith(".source")]
        # a carried-over route was not computed in this run, so its values are not compared
        copied = [r for r in rows if any(r[0] == c or r[0].startswith(c + ".") for c in carried)]
        rows = [r for r in rows if r not in copied]
        diffs = [r for r in rows if r[1] != r[2]]
        for k, want, have in diffs:
            print(f"  DIFF {k}: expected {want}, got {have}")
        print(f"Expected counts: {len(rows) - len(diffs)} match, {len(diffs)} differ"
              + (f"; {len(copied)} carried over from the stored result, not compared" if copied else ""))
        if diffs:
            return 1
    if args.out:
        _cr_paths.atomic_write_text(args.out, json.dumps(res, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
