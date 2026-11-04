"""Summaries of the sensitivity checks, version 2.

Version 2 (26 October 2026, night) writes the summary of version 1, and from complete inputs the same
summary.json, after checking every input with the functions of ``camera_ready/scripts/runs/ladder_checks.py``,
before anything is computed or written:

  * completeness: every output of the checks is present (the counts of training duplicates, the deduplicated LOGO,
    richer-input and mechanism runs with the counts of removed rows, the standard-split runs, the eight folds of
    the standard-split heads per pool); every run holds the eight folds; the full prediction-space ladder is
    complete in every fold (the ten base methods, with the recorded best among them, and the twelve richer-input
    candidates, a run over the capped or the widened grid only named as such; the ten candidates of the
    standard-split heads; on the standard split the 20 methods, with the recorded ceiling the best of them). A gap
    stops the summary unless ``--allow-partial`` is given; the parts with gaps are then left out (never computed
    from partial inputs), SUMMARY.md opens with them and summary.json records them under ``input_completeness``
    (complete: false), which the table builder refuses;
  * compatibility: runs analysed together (each LOGO run with its richer-input run and its mechanism run, the
    primary LOGO run with each fold of the standard-split heads) have the same pool, exclusion and per-fold row
    counts and, when both record them, the same membership fingerprints of FIT and SELECT; runs without
    fingerprints (the supplied ones) must be listed together in ``camera_ready/provenance/trusted_run_pairs.json``.
    Each deduplicated LOGO run must be its primary run with the training duplicates removed: the same pool,
    exclusion, test sets and excluded rows, and training rows before and after the removal equal to the counts
    that the wrapper recorded. A problem here, or a missing primary result, always stops the summary.

Reruns of the checks by ``dedup_logo_v2.py``, ``dedup_followup_v2.py`` and ``paper_heads_logo_v2.py`` record the
fingerprints. The supplied primary runs record none, so the folds of a rerun of the standard-split heads can be
analysed only with primary LOGO runs rerun by the current runners (the reproducibility harness, block B).

Version 1, as before:

Summaries of the sensitivity checks, rebuilt from their outputs (no model is refitted).

Reads the outputs of count_twins.py, dedup_logo_v2.py, dedup_followup_v2.py and paper_heads_logo_v2.py in --dir (default: the
supplied outputs, camera_ready/output/results/sensitivity/) and the primary results they are compared with
(camera_ready/output/results/mitigation/ and analyses/, full_ladder_stats.json; the reviewed weak6 results in
output/results/, which the audit uses), and writes SUMMARY.md and summary.json in --out-dir (default: camera_ready/output/checks/sensitivity/):

0. The training duplicates the checks remove: SELECT rows with a FIT twin and extra copies inside FIT, per LOGO fold
   and on the standard split.
1. LOGO with exact duplicates removed from the training data, four pools: the full-ladder ceiling, FS1 and FS2
   against it (fold-bootstrap 95% CI, wins, exact sign-flip p), the folds with at least 500 test images, the rows
   removed per fold, and the macro AUPRC of FS1 and of each fold's ceiling method from the saved predictions. Each
   fold's ceiling method is the one with the best AUROC, so the AUPRC comparison uses that AUROC-selected comparator.
2. The mechanism under LOGO (Table A9, LOGO columns), arch6 and clip4.
3. The standard split, arch6, clip4 and weak6, and the audit's three effects of Table 2 with deduplicated runs.
4. The LOGO richer-input heads of the standard split (the paper's Table A3; 20-method ceiling) against those of
   the primary runs (the paper's Appendix F; 22 candidates).

The statistics come from scripts/authoring/full_ladder_stats_v4.py (logo_block, mechanism_block), the functions
behind the primary statistics. Scope: only training data are deduplicated (each LOGO fold's FIT and SELECT, the
standard split's FIT and SELECT); the test sets, with their own repeated images, are unchanged. Deduplication also
changes the training weights and mix, so a shift is not attributable to cleaner validation alone, and an interval
that crosses zero is not evidence of equivalence. From the repository root:

    python camera_ready/scripts/sensitivity/summarize_v2.py
    python camera_ready/scripts/sensitivity/summarize_v2.py --dir camera_ready/output/results/sensitivity_rerun
    python camera_ready/scripts/sensitivity/summarize_v2.py --dir <folder> --allow-partial   # the complete parts
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runs"))
import ladder_checks  # noqa: E402  version 2: completeness of the ladder, compatibility of paired runs
import _cr_paths

CR = Path(__file__).resolve().parents[2]          # camera_ready/
ROOT = CR.parent
RES = CR / "output" / "results"
POOLS = {"arch6": 256, "clip4": 32, "strong4": 256, "weak6": 256}
GENS = ["whichisreal", "CollabDiff", "MidJourney", "heygen", "starganv2", "deepfacelab", "stargan", "styleclip"]
SCOPE = ("Scope: only training data are deduplicated (each LOGO fold's FIT and SELECT; the standard split's FIT and "
         "SELECT). The test sets, with their own repeated images, are unchanged, so this is not a fully deduplicated "
         "evaluation. Deduplication also changes the training weights and mix (StarGAN-v2 keeps 1,024 of its 4,000 "
         "rows), so a shift is not attributable to cleaner validation alone; an interval that crosses zero is not "
         "evidence of equivalence. Primary results: camera_ready/output/results/mitigation/ and analyses/.")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- version 2: complete inputs, compatible runs

DEPLOYABLE_STANDARD = tuple(f"PS_{i}_{m}_{h}" for i, m in ((11, "disagreement"), (12, "uncertainty_mi"),
                                                          (13, "quadratic"), (14, "rank_ecdf"), (15, "cv_best"))
                            for h in ("logreg", "xgb"))


def _rel(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def counts_file(ddir, run, pool):
    """Version 2: the counts of removed rows that the deduplication wrapper wrote for the LOGO run `run`, named after
    the runner that wrote the run (11_disjoint_data_v2 for the supplied outputs, 11_disjoint_data_v4 for a rerun by
    dedup_logo_v2.py)."""
    return ddir / f"dedup_counts_{run.get('version') or '11_disjoint_data_v2'}_{pool}.json"


def comparison_problem(primary, dedup, counts):
    """Version 2: why the deduplicated LOGO run `dedup` is not the primary run `primary` with its training duplicates
    removed (None when it is): the same pool, exclusion, test sets and excluded rows, and in every fold the training
    rows (FIT and SELECT) before and after the removal equal to the counts that the wrapper recorded (`counts`)."""
    for k in ("pool", "exclusion"):
        if primary.get(k) != dedup.get(k):
            return f"{k} {dedup.get(k)!r} in the deduplicated run, {primary.get(k)!r} in the primary run"
    pf, df = ladder_checks.fold_records(primary), ladder_checks.fold_records(dedup)
    for g in [g for g in GENS if g in pf and g in df]:
        for k in ("n_test", "n_excluded_training_rows"):
            if pf[g].get(k) != df[g].get(k):
                return (f"fold {g}: {k} {df[g].get(k)} in the deduplicated run, {pf[g].get(k)} in the primary run "
                        "(another test set)")
        c = counts.get(g)
        if not c:
            return f"fold {g}: the wrapper recorded no count of removed rows"
        before = (pf[g].get("n_fit") or 0) + (pf[g].get("n_select") or 0)
        after = (df[g].get("n_fit") or 0) + (df[g].get("n_select") or 0)
        if (before, after) != (c.get("training_rows_before"), c.get("training_rows_after")):
            return (f"fold {g}: {before} training rows in the primary run and {after} in the deduplicated run, but the "
                    f"wrapper recorded {c.get('training_rows_before')} before and {c.get('training_rows_after')} "
                    "after the removal")
    return None


def standard_gaps(run):
    """Version 2: what a standard-split run (appB_richer_inputs.py) lacks for the summary's ceiling over the full
    ladder: the eight generators, the ten base methods and the ten deployable richer-input candidates, with the
    recorded ceiling the best of these twenty."""
    gaps = []
    if list(run.get("generators") or []) != GENS:
        gaps.append(f"generators {run.get('generators')}, not the eight held-out generators")
    base = (run.get("published_ladder") or {}).get("all_macro") or {}
    rich = run.get("extended_rungs_macro") or {}
    missing = ([m for m in ladder_checks.BASE_METHODS if m not in base]
               + [c for c in DEPLOYABLE_STANDARD if c not in rich])
    if missing:
        return gaps + [f"{len(missing)} of the 20 methods missing ({', '.join(missing)})"]
    every = {**{m: base[m] for m in ladder_checks.BASE_METHODS}, **{c: rich[c] for c in DEPLOYABLE_STANDARD}}
    best = max(every, key=every.get)
    ceiling = run.get("deployable_ceiling") or {}
    if ceiling.get("name") != best or not abs(float(ceiling.get("macro", "nan")) - every[best]) <= 1e-12:
        gaps.append(f"the recorded ceiling ({ceiling.get('name')}) is not the best of the 20 methods ({best})")
    return gaps


def score_gaps(directory, logo, rungs):
    """Validate every score array needed for the fold-wise AUPRC comparison."""
    problems = []
    base = ladder_checks.fold_records(logo)
    rich = ladder_checks.fold_records(rungs)
    for fold in GENS:
        if fold not in base or fold not in rich:
            continue  # the JSON completeness check reports this separately
        previous_labels = None
        for label, rec in (("ladder", base[fold]), ("richer inputs", rich[fold])):
            try:
                relative = rec.get("scores_file")
                if not relative:
                    raise ValueError("no scores_file")
                path = (directory / relative).resolve()
                if not path.is_relative_to(directory.resolve()):
                    raise ValueError("scores_file escapes the run directory")
                methods = (list(ladder_checks.BASE_METHODS) + ["FS_1_raw_feat_logreg"]
                           if label == "ladder" else list(rec["rungs"]))
                if label == "ladder" and rec.get("fs_mlp_mean") is not None:
                    methods.append("FS_2_mlp_seed_avg")
                with np.load(path, allow_pickle=False) as archive:
                    y = archive["labels"]
                    if y.shape != (base[fold]["n_test"],) or set(np.unique(y)) != {0, 1}:
                        raise ValueError("labels do not match the binary test fold")
                    if previous_labels is not None and not np.array_equal(y, previous_labels):
                        raise ValueError("ladder and richer-input labels are not aligned")
                    previous_labels = y.copy()
                    for method in methods:
                        scores = archive[method]
                        if scores.shape != y.shape or not np.isfinite(scores).all():
                            raise ValueError(f"invalid or unaligned scores for {method}")
            except (OSError, ValueError, KeyError, TypeError) as error:
                problems.append(f"{fold}/{label}: {error}")
    return problems


def check_inputs(out, allow_partial):
    """Version 2: every input of the summary, checked before anything is computed or written. Returns {part: gaps}
    for the parts with gaps (outputs, folds or methods missing), which stop the summary unless `allow_partial`; a
    missing primary result, or runs that cannot be analysed or compared together, always stop it."""
    cd, mech = RES / "mitigation" / "content_disjoint", RES / "mitigation" / "mechanism"
    dd, pdir = out / "dedup", out / "paper_heads"
    gaps, fatal = {}, []

    def gap(part, msg):
        gaps.setdefault(part, []).append(msg)

    def read(p, part=None):
        """The file at `p`, or None: a missing output of the checks is a gap of `part`; a missing primary result
        (`part` None) is fatal."""
        if p.is_file():
            return load(p)
        if part is None:
            fatal.append(f"{_rel(p)} not found (a primary result the summary compares with)")
        else:
            gap(part, f"{_rel(p)} not found")
        return None

    def folds_of(run, path, part):
        recs = ladder_checks.fold_records(run)
        lacking = [g for g in GENS if g not in recs]
        if lacking:
            gap(part, f"{_rel(path)}: {len(lacking)} of {len(GENS)} folds missing ({', '.join(lacking)})")
        return [g for g in GENS if g in recs]

    def pair(left, right, lp, rp, names, part):
        common = [g for g in folds_of(left, lp, part) if g in folds_of(right, rp, part)]
        why = ladder_checks.pairing_problem(left, right, lp, rp, names, folds=common) if common else None
        if why:
            fatal.append(f"{_rel(rp)} cannot be analysed with {_rel(lp)}: {why}")
        return common

    twins = read(out / "twins.json", "twins")
    if twins is not None:
        for pool in POOLS:
            lacking = [g for g in GENS if g not in ((twins.get("logo") or {}).get(pool) or {})]
            if lacking:
                gap("twins", f"LOGO counts of {pool} missing for {', '.join(lacking)}")
        lacking = [p for p in ("arch6", "clip4", "weak6") if p not in (twins.get("standard") or {})]
        if lacking:
            gap("twins", f"standard-split counts missing for {', '.join(lacking)}")
    for pool, d in POOLS.items():
        part, mpart, hpart = f"logo {pool}", f"mechanism {pool}", f"paper_heads {pool}"
        lp, rp = cd / f"11_disjoint_{pool}_d{d}.json", cd / f"A3c_logo_extended_rungs_{pool}.json"
        logo, rungs = read(lp), read(rp)
        if logo is not None and rungs is not None:
            fatal.extend(f"primary {pool}: {why}" for why in score_gaps(cd, logo, rungs))
            common = pair(logo, rungs, lp, rp, ("the primary LOGO run", "its richer-input run"), part)
            for g in ladder_checks.ladder_gaps(logo, rungs, folds=common):
                gap(part, f"primary run, {g}")
        dlp, drp = dd / f"11_disjoint_{pool}_d{d}.json", dd / f"A3c_logo_extended_rungs_{pool}.json"
        dlogo, drungs = read(dlp, part), read(drp, part)
        if dlogo is not None:
            counts = read(counts_file(dd, dlogo, pool), part)
            if drungs is not None:
                for why in score_gaps(dd, dlogo, drungs):
                    gap(part, why)
                common = pair(dlogo, drungs, dlp, drp, ("the deduplicated LOGO run", "its richer-input run"), part)
                for g in ladder_checks.ladder_gaps(dlogo, drungs, folds=common):
                    gap(part, f"deduplicated run, {g}")
            if logo is not None and counts is not None:
                why = comparison_problem(logo, dlogo, counts.get("folds") or {})
                if why:
                    fatal.append(f"{_rel(dlp)} is not {_rel(lp)} with its training duplicates removed: {why}")
        if pool in ("arch6", "clip4"):
            mp, dmp = mech / f"run_c_v2_logo_{pool}.json", dd / f"run_c_v2_logo_{pool}.json"
            m, dm = read(mp), read(dmp, mpart)
            if logo is not None and m is not None:
                pair(logo, m, lp, mp, ("the primary LOGO run", "its mechanism run"), mpart)
            if dlogo is None:
                gap(mpart, f"{_rel(dlp)} not found (the deduplicated mechanism is analysed with it)")
            elif dm is not None:
                pair(dlogo, dm, dlp, dmp, ("the deduplicated LOGO run", "its mechanism run"), mpart)
        recs = {}
        for g in GENS:
            fp = pdir / pool / f"{pool}_{g}.json"
            rec = read(fp, hpart)
            if rec is None:
                continue
            recs[g] = rec
            if logo is not None and g in ladder_checks.fold_records(logo):
                why = ladder_checks.pairing_problem(
                    logo, {"pool": rec.get("pool"), "per_fold": [rec]}, lp, fp,
                    ("the primary LOGO run", f"the {g} fold of the standard-split heads"), folds=[g])
                if why:
                    fatal.append(f"{_rel(fp)} cannot be analysed with {_rel(lp)}: {why}")
        if logo is not None and len(recs) == len(GENS):
            held = folds_of(logo, lp, hpart)
            heads = {"folds": {g: {"rungs": recs[g].get("auc") or {}} for g in GENS}}
            base = set(ladder_checks.ladder_gaps(logo, None, folds=held))
            for g in ladder_checks.ladder_gaps(logo, heads, folds=held,
                                               candidates=ladder_checks.PAPER_HEADS_CANDIDATES):
                gap(hpart, f"primary run, {g}" if g in base else f"standard-split heads, {g}")
    for pool in ("arch6", "clip4", "weak6"):
        part = f"standard {pool}"
        sp = dd / "standard" / f"run_a_extended_ladder_{pool}.json"
        run = read(sp, part)
        read(dd / "standard" / f"dedup_counts_standard_{pool}.json", part)
        if run is not None:
            for g in standard_gaps(run):
                gap(part, f"{_rel(sp)}: {g}")
        if pool in ("arch6", "clip4"):
            pp = RES / "analyses" / f"run_a_extended_ladder_{pool}.json"
            primary = read(pp)
            if primary is not None:
                fatal.extend(f"{_rel(pp)}: {g}" for g in standard_gaps(primary))
    reviewed = ROOT / "output" / "results"
    for p in (reviewed / "04_ds_weak6.json", reviewed / "04_ds_weak6_rawfeat.json",
              RES / "mitigation" / "mitigation_stats.json"):
        read(p)
    blocked = [p for p in ("logo arch6", "standard arch6", "standard weak6") if p in gaps]
    if blocked:
        gap("audit", f"left out: it needs {', '.join(blocked)}")
    gaps = {k: list(dict.fromkeys(v)) for k, v in gaps.items()}
    if fatal:
        also = [f"{k}: {g}" for k, v in gaps.items() for g in v]
        raise SystemExit("The summary cannot be written:\n  " + "\n  ".join(dict.fromkeys(fatal))
                         + ("\nAlso not complete:\n  " + "\n  ".join(also) if also else "")
                         + "\nNothing was written.")
    if gaps and not allow_partial:
        raise SystemExit("The inputs of the summary are not complete:\n  "
                         + "\n  ".join(f"{k}: {g}" for k, v in gaps.items() for g in v)
                         + "\n--allow-partial summarizes the complete parts and lists the others (the table builder "
                           "refuses such a summary). Nothing was written.")
    return gaps


def stats_module():
    spec = importlib.util.spec_from_file_location("full_ladder_stats_v4",
                                                  CR / "scripts" / "authoring" / "full_ladder_stats_v4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fmt_stat(s):
    return (f"{s['mean_delta']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}], {s['wins']} of {s['n']}, "
            f"sign-flip p {s['signflip_p']:.3f}")


def twins_section(path, summary, skip=None):
    """The training duplicates the checks remove (count_twins.py), when its output is in the folder."""
    if (skip or {}).get("twins"):                                           # version 2
        return ["## 0. Training duplicates", "", "not complete (--allow-partial): " + "; ".join(skip["twins"]),
                ""]
    if not path.exists():
        return ["## 0. Training duplicates", "", "not in this folder", ""]
    twins = load(path)
    summary["twins"] = twins
    lines = ["## 0. Training duplicates", "",
             "SELECT rows with an exact feature twin in FIT, and extra copies inside FIT, in each LOGO fold (the same "
             "in every pool) and on the standard split. Nearly every twin is a StarGAN-v2 row, whose test list "
             "repeats images.", "",
             "| LOGO fold (arch6) | SELECT rows | with a FIT twin | share | extra copies inside FIT |",
             "|---|---|---|---|---|"]
    for g, r in twins["logo"].get("arch6", {}).items():
        lines.append(f"| {g} | {r['select_rows']} | {r['select_rows_with_fit_twin']} | {r['share']:.1%} | "
                     f"{r['extra_copies_inside_fit']} |")
    same = all(twins["logo"][p] == twins["logo"].get("arch6") for p in twins["logo"])
    lines += ["", f"The other pools give the same counts: {'yes' if same else 'no'}.", "",
              "| Standard split | SELECT rows | with a FIT twin | extra copies inside FIT | "
              "extra copies inside SELECT |",
              "|---|---|---|---|---|"]
    for p, r in twins["standard"].items():
        lines.append(f"| {p} | {r['select_rows']} | {r['select_rows_with_fit_twin']} | "
                     f"{r['extra_copies_inside_fit']} | {r['extra_copies_inside_select']} |")
    lines.append("")
    return lines


def logo_section(fls, ddir, summary, skip=None):
    cd = RES / "mitigation" / "content_disjoint"
    lines = ["## 1. LOGO with exact duplicates removed from the training data", "",
             "Statistics against each fold's ceiling over the full prediction-space ladder (the ten methods and the "
             "richer inputs). AUPRC: macro over folds, from the saved predictions; the comparator of each fold is its "
             "ceiling method, chosen by AUROC.", ""]
    for pool, d in POOLS.items():
        name, rname = f"11_disjoint_{pool}_d{d}.json", f"A3c_logo_extended_rungs_{pool}.json"
        if (skip or {}).get(f"logo {pool}"):                                # version 2
            lines += [f"### {pool}", "", "not complete (--allow-partial): " + "; ".join(skip[f"logo {pool}"]),
                      ""]
            continue
        if not ((ddir / name).exists() and (ddir / rname).exists()):
            lines += [f"### {pool}", "", "not in this folder", ""]
            continue
        s_logo, d_logo = load(cd / name), load(ddir / name)
        old, new = fls.logo_block(s_logo, load(cd / rname), cd), fls.logo_block(d_logo, load(ddir / rname), ddir)
        counts = load(counts_file(ddir, d_logo, pool))["folds"]                      # version 2: by runner
        summary["logo"][pool] = {"primary": old, "deduplicated": new, "rows_removed": counts}
        so, sn = old["stats"]["fs1_full"], new["stats"]["fs1_full"]
        lines += [f"### {pool}", "", "| | primary | training duplicates removed |", "|---|---|---|",
                  f"| PS ceiling (mean over folds) | {old['ceiling_full']:.4f} | {new['ceiling_full']:.4f} |",
                  f"| FS1 (mean over folds) | {old['fs1']:.4f} | {new['fs1']:.4f} |",
                  f"| FS1 - ceiling | {fmt_stat(so)} | {fmt_stat(sn)} |"]
        if "fs2_full" in old["stats"] and "fs2_full" in new["stats"]:
            lines.append(f"| FS2 - ceiling | {fmt_stat(old['stats']['fs2_full'])} | "
                         f"{fmt_stat(new['stats']['fs2_full'])} |")
        lines.append(f"| FS1 - ceiling, folds with n >= 500 | {old['fs1_full_delta_n500']:+.4f} | "
                     f"{new['fs1_full_delta_n500']:+.4f} |")
        if old.get("auprc") and new.get("auprc"):
            oa, na = old["auprc"], new["auprc"]
            lines += [f"| AUPRC: FS1 / ceiling method | {oa['fs1']:.4f} / {oa['ceiling_full']:.4f} | "
                      f"{na['fs1']:.4f} / {na['ceiling_full']:.4f} |",
                      f"| AUPRC: FS1 - ceiling method | {oa['delta']:+.4f} | {na['delta']:+.4f} |"]
        lines += ["", "| Fold | rows removed | FIT / SELECT primary | FIT / SELECT dedup | ceiling primary (method) | "
                      "ceiling dedup (method) | FS1 - ceiling primary / dedup |", "|---|---|---|---|---|---|---|"]
        sf = {f["held_out"]: f for f in s_logo["per_fold"]}
        df = {f["held_out"]: f for f in d_logo["per_fold"]}
        for i, g in enumerate(old["folds"]):
            lines.append(f"| {g} | {counts[g]['duplicates_removed']} | {sf[g]['n_fit']} / {sf[g]['n_select']} | "
                         f"{df[g]['n_fit']} / {df[g]['n_select']} | {old['ceiling_full_per_fold'][i]:.4f} "
                         f"({old['ceiling_method_per_fold'][i]}) | {new['ceiling_full_per_fold'][i]:.4f} "
                         f"({new['ceiling_method_per_fold'][i]}) | {so['per_fold'][i]:+.4f} / "
                         f"{sn['per_fold'][i]:+.4f} |")
        lines.append("")
    return lines


def mechanism_section(fls, ddir, summary, skip=None):
    lines = ["## 2. Mechanism under LOGO (Table A9, LOGO columns)", ""]
    for pool in ("arch6", "clip4"):
        d = POOLS[pool]
        dm = ddir / f"run_c_v2_logo_{pool}.json"
        if (skip or {}).get(f"mechanism {pool}"):                           # version 2
            lines += [f"### {pool}", "", "not complete (--allow-partial): " + "; ".join(skip[f"mechanism {pool}"]),
                      ""]
            continue
        if not dm.exists():
            lines += [f"### {pool}", "", "not in this folder", ""]
            continue
        old = fls.mechanism_block(load(RES / f"mitigation/mechanism/run_c_v2_logo_{pool}.json"),
                                  load(RES / f"mitigation/content_disjoint/11_disjoint_{pool}_d{d}.json"))
        new = fls.mechanism_block(load(dm), load(ddir / f"11_disjoint_{pool}_d{d}.json"))
        summary["mechanism"][pool] = {"primary": old, "deduplicated": new}
        lines += [f"### {pool}", "", "| Method (mean AUROC) | primary | training duplicates removed |", "|---|---|---|"]
        lines += [f"| {k} | {old['mean_auroc'][k]:.4f} | {new['mean_auroc'][k]:.4f} |" for k in old["mean_auroc"]]
        lines += ["", "| Comparison | primary | training duplicates removed |", "|---|---|---|"]
        lines += [f"| {k} | {fmt_stat(old['comparisons'][k])} | {fmt_stat(new['comparisons'][k])} |"
                  for k in old["comparisons"]]
        lines.append("")
    return lines


def best13(m):
    return max(m["PS_1_prob_avg"], m["PS_2_logreg"], m["PS_3_mlp_logits"])


def standard_section(ddir, summary, skip=None):
    std = ddir / "standard"
    lines = ["## 3. Standard split and the audit (Table 2)", ""]
    new_runs = {}
    for pool in ("arch6", "clip4", "weak6"):
        dp = std / f"run_a_extended_ladder_{pool}.json"
        if (skip or {}).get(f"standard {pool}"):                            # version 2
            lines += [f"### {pool}", "", "not complete (--allow-partial): " + "; ".join(skip[f"standard {pool}"]),
                      ""]
            continue
        if not dp.exists():
            lines += [f"### {pool}", "", "not in this folder", ""]
            continue
        new = new_runs[pool] = load(dp)
        cnt = load(std / f"dedup_counts_standard_{pool}.json")["splits"]
        nm, nc = new["published_ladder"]["all_macro"], new["comparison_vs_deployable_ceiling"]
        lines += [f"### {pool}", "", f"Rows removed: FIT {cnt['FIT']['rows_removed']} of {cnt['FIT']['rows_before']}, "
                  f"SELECT {cnt['SELECT']['rows_removed']} of {cnt['SELECT']['rows_before']}.", ""]
        if pool in ("arch6", "clip4"):
            old = load(RES / f"analyses/run_a_extended_ladder_{pool}.json")
            om, oc = old["published_ladder"]["all_macro"], old["comparison_vs_deployable_ceiling"]
            summary["standard"][pool] = {"primary": old, "deduplicated": new, "rows_removed": cnt}
            def ci(c):
                lo, hi = c["generator_bootstrap"]["ci_95"]
                return f"[{lo:+.4f}, {hi:+.4f}]"
            lines += ["| | primary (same script) | training duplicates removed |", "|---|---|---|",
                      f"| PS ceiling over the full ladder (method) | {old['deployable_ceiling']['macro']:.4f} "
                      f"({old['deployable_ceiling']['name']}) | {new['deployable_ceiling']['macro']:.4f} "
                      f"({new['deployable_ceiling']['name']}) |",
                      f"| FS1 | {old['fs1_macro']:.4f} | {new['fs1_macro']:.4f} |",
                      f"| FS1 - ceiling [generator bootstrap 95% CI], wins | {oc['delta']:+.4f} {ci(oc)}, "
                      f"{oc['wins']} of {oc['n_generators']} | {nc['delta']:+.4f} {ci(nc)}, {nc['wins']} of "
                      f"{nc['n_generators']} |",
                      f"| PS4 | {om['PS_4_xgboost']:.4f} | {nm['PS_4_xgboost']:.4f} |",
                      f"| best of PS1-PS3 | {best13(om):.4f} | {best13(nm):.4f} |", ""]
        else:
            a = load(ROOT / "output/results/04_ds_weak6.json")["method_means"]
            fs1_old = load(ROOT / "output/results/04_ds_weak6_rawfeat.json")["avg_auc_8"]
            om = {k: v["mean_avg8"] for k, v in a.items()}
            summary["standard"]["weak6"] = {"primary_block_a": {"methods": om, "fs1": fs1_old}, "deduplicated": new,
                                            "rows_removed": cnt}
            lines += ["| | primary (reviewed results, Block A) | training duplicates removed |", "|---|---|---|",
                      f"| FS1 | {fs1_old:.4f} | {new['fs1_macro']:.4f} |",
                      f"| best of PS1-PS3 | {best13(om):.4f} | {best13(nm):.4f} |",
                      f"| PS4 | {om['PS_4_xgboost']:.4f} | {nm['PS_4_xgboost']:.4f} |",
                      f"| FS1 - PS4 | {fs1_old - om['PS_4_xgboost']:+.4f} | "
                      f"{new['fs1_macro'] - nm['PS_4_xgboost']:+.4f} |",
                      f"| FS1 - ceiling over the full ladder | | {nc['delta']:+.4f} "
                      f"({new['deployable_ceiling']['name']}) |",
                      ""]
    logo = summary["logo"].get("arch6")
    if logo and {"arch6", "weak6"} <= set(new_runs):
        audit = load(RES / "mitigation" / "mitigation_stats.json")["audit"]
        old_std = load(RES / "analyses/run_a_extended_ladder_arch6.json")
        a6o, a6n = old_std["published_ladder"]["all_macro"], new_runs["arch6"]["published_ladder"]["all_macro"]
        w = load(ROOT / "output/results/04_ds_weak6.json")["method_means"]
        wo, wn = {k: v["mean_avg8"] for k, v in w.items()}, new_runs["weak6"]["published_ladder"]["all_macro"]
        effects = {
            "overlap distortion (arch6: standard-split margin minus LOGO margin)": (
                audit["overlap_distortion"],
                old_std["comparison_vs_deployable_ceiling"]["delta"]
                - logo["primary"]["stats"]["fs1_full"]["mean_delta"],
                new_runs["arch6"]["comparison_vs_deployable_ceiling"]["delta"]
                - logo["deduplicated"]["stats"]["fs1_full"]["mean_delta"]),
            "baseline-ladder distortion, arch6 (full-ladder ceiling minus best of PS1-PS3)": (
                audit["baseline_distortions"]["arch6"], old_std["deployable_ceiling"]["macro"] - best13(a6o),
                new_runs["arch6"]["deployable_ceiling"]["macro"] - best13(a6n)),
            "baseline-ladder distortion, weak6 (PS4 minus best of PS1-PS3)": (
                audit["baseline_distortions"]["weak6"], wo["PS_4_xgboost"] - best13(wo),
                wn["PS_4_xgboost"] - best13(wn))}
        summary["audit"] = {k: {"paper": v[0], "primary_same_scripts": v[1], "deduplicated": v[2]}
                            for k, v in effects.items()}
        lines += ["### The audit's three effects (Table 2)", "",
                  "The standard-split margins here come from appB_richer_inputs.py, whose FS1 differs from the "
                  "canonical FS1 of Table 2 by about 3e-4; weak6 has no stored run of that script, so its primary "
                  "column is the reviewed result that Table 2 uses.", "",
                  "| Effect | paper (Table 2) | primary, same scripts | training duplicates removed |",
                  "|---|---|---|---|"]
        lines += [f"| {k} | {v[0]:+.4f} | {v[1]:+.4f} | {v[2]:+.4f} |" for k, v in effects.items()]
        lines.append("")
    return lines


def paper_heads_section(fls, pdir, summary, skip=None):
    cd = RES / "mitigation" / "content_disjoint"
    lines = ["## 4. LOGO richer-input heads of the standard split", "",
             "Ceiling over the 20 methods of the standard split (the ten ladder methods and PS_G to PS_K under a "
             "class-balanced logistic head and XGBoost with the PS4 grid and 5-fold CV) against the primary runs' 22 "
             "candidates.", ""]
    for pool, d in POOLS.items():
        if (skip or {}).get(f"paper_heads {pool}"):                         # version 2
            lines += [f"### {pool}", "", "not complete (--allow-partial): " + "; ".join(skip[f"paper_heads {pool}"]),
                      ""]
            continue
        recs = {g: load(pdir / pool / f"{pool}_{g}.json") for g in GENS if (pdir / pool / f"{pool}_{g}.json").exists()}
        if len(recs) != len(GENS):
            lines += [f"### {pool}", "", f"{len(recs)} of {len(GENS)} folds in this folder", ""]
            continue
        logo = load(cd / f"11_disjoint_{pool}_d{d}.json")
        old_rungs = load(cd / f"A3c_logo_extended_rungs_{pool}.json")
        new_rungs = {"folds": {g: {"rungs": recs[g]["auc"]} for g in GENS}}
        old, new = fls.logo_block(logo, old_rungs), fls.logo_block(logo, new_rungs)
        summary["paper_heads"][pool] = {"primary_22": old, "paper_20": new,
                                        "k_selected": {g: recs[g]["k_selected"] for g in GENS}}
        so, sn = old["stats"]["fs1_full"], new["stats"]["fs1_full"]
        lines += [f"### {pool}", "", "| | primary (22 candidates) | standard-split heads (20 methods) |",
                  "|---|---|---|",
                  f"| PS ceiling (mean over folds) | {old['ceiling_full']:.4f} | {new['ceiling_full']:.4f} |",
                  f"| FS1 - ceiling | {fmt_stat(so)} | {fmt_stat(sn)} |",
                  f"| sign / Wilcoxon p | {so['sign_p']:.3f} / {so['wilcoxon_p']:.3f} | {sn['sign_p']:.3f} / "
                  f"{sn['wilcoxon_p']:.3f} |",
                  f"| FS1 - ceiling, folds with n >= 500 | {old['fs1_full_delta_n500']:+.4f} | "
                  f"{new['fs1_full_delta_n500']:+.4f} |", "",
                  "| Fold | n | primary ceiling (method) | standard-split heads ceiling (method) |",
                  "|---|---|---|---|"]
        for i, g in enumerate(old["folds"]):
            lines.append(f"| {g} | {recs[g]['n_test']} | {old['ceiling_full_per_fold'][i]:.4f} "
                         f"({old['ceiling_method_per_fold'][i]}) | {new['ceiling_full_per_fold'][i]:.4f} "
                         f"({new['ceiling_method_per_fold'][i]}) |")
        lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=RES / "sensitivity",
                    help="folder with dedup/ and paper_heads/ (default: the supplied outputs)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="version 2: summarize the complete parts when outputs, folds or methods are missing; the "
                         "others are listed, and summary.json records input_completeness (complete: false); "
                         "without it, any gap stops the summary")
    ap.add_argument("--out-dir", type=Path, default=CR / "output" / "checks" / "sensitivity",
                    help="report destination, separate from the input results")
    a = ap.parse_args()
    out = a.dir if a.dir.is_absolute() else (ROOT / a.dir)
    report_dir = a.out_dir.resolve()
    json_path = _cr_paths.out_path(report_dir / "summary.json")
    markdown_path = _cr_paths.out_path(report_dir / "SUMMARY.md")
    skip = check_inputs(out, a.allow_partial)                          # version 2: before anything is computed
    notes = [f"{part}: {g}" for part, gaps in skip.items() for g in gaps]
    fls = stats_module()
    summary = {"scope": SCOPE, "twins": {}, "logo": {}, "mechanism": {}, "standard": {}, "audit": {}, "paper_heads": {}}
    lines = ["# Sensitivity checks: summary", "", "Written by `camera_ready/scripts/sensitivity/summarize_v2.py` from "
             "the input results supplied by --dir; no model is refitted.", ""]
    if notes:
        lines += ["**Not complete.** Written with --allow-partial: the parts below are left out, and summary.json "
                  "records them (input_completeness).", ""] + [f"- {n}" for n in notes] + [""]
    else:
        lines += ["Inputs checked: every output of the checks is present, every run holds the eight folds and the "
                  "full ladder, and the runs analysed or compared together are compatible "
                  "(`camera_ready/scripts/runs/ladder_checks.py`).", ""]
    lines += [SCOPE, ""]
    lines += twins_section(out / "twins.json", summary, skip)
    lines += logo_section(fls, out / "dedup", summary, skip)
    lines += mechanism_section(fls, out / "dedup", summary, skip)
    lines += standard_section(out / "dedup", summary, skip)
    lines += paper_heads_section(fls, out / "paper_heads", summary, skip)
    if notes:                                                          # version 2: a partial summary says so
        summary["input_completeness"] = {"complete": False, "notes": notes, "options": ["--allow-partial"]}
    serialized = json.dumps(summary, indent=2, allow_nan=False)
    _cr_paths.atomic_write_text(json_path, serialized)
    _cr_paths.atomic_write_text(markdown_path, "\n".join(lines) + "\n")
    print(f"written {markdown_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
