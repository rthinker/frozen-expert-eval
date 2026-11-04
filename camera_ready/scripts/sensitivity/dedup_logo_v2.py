"""Sensitivity check: LOGO runs with exact duplicates removed from each fold's training data, version 2.

Version 2 (the supplied outputs come from version 1)
runs the current runners, ``scripts/runs/11_disjoint_data_v4.py`` and ``08_logo_extended_rungs_v3.py``, which
record in every fold the membership fingerprints of FIT and SELECT, so that the summary (``summarize_v2.py``) can
check that the deduplicated LOGO run, its richer-input run and the mechanism run of ``dedup_followup_v2.py`` used
the same rows. The counts of removed rows are named after these runners (``dedup_counts_11_disjoint_data_v4_<pool>
.json``, which the summary finds from the LOGO run's ``version``). Everything else is as in version 1:

Sensitivity check: the LOGO runs repeated with exact duplicates removed from each fold's training data.

Inside every LOGO fold the 85/15 FIT/SELECT split is stratified by row, and StarGAN-v2's test list repeats images
(4,000 rows, 1,024 distinct), so in the seven folds that train on StarGAN-v2, 23-31% of SELECT rows have an exact
feature twin in FIT and FIT holds about 2,350 extra copies (count_twins.py; supplied counts in
camera_ready/output/results/sensitivity/twins.json). Selection on SELECT (early stopping, checkpoints) and the
inner cross-validation of PS_C, PS_4 and the richer-input XGBoost heads therefore see partly duplicated data. The
held-out test sets are kept apart from training by the exclusion lists and are not changed here: their own repeated
images remain, so this is not a fully deduplicated evaluation.

This script repeats the primary content-disjoint runs with one change: after the held-out exclusions, each fold's
training data keeps only the first copy of every exact duplicate (identical features across the pool's experts,
``logo_common.row_hashes``). FIT and SELECT are then content-disjoint and FIT holds no repeated row, so the methods'
own CV folds hold none either. This script wraps the current LOGO runners and replaces ``logo_common.fold_view``
by a version that drops the copies and records how many rows each fold lost (``dedup_counts_<runner>_<pool>.json``).
Removing the copies also changes the training mix (StarGAN-v2's repeated images no longer count about four times),
so a shift combines cleaner selection with that reweighting.

The supplied outputs are in camera_ready/output/results/sensitivity/dedup/ (PROVENANCE.md next to them). A rerun
writes to --out-dir (default camera_ready/output/results/sensitivity_rerun/dedup/), never into the supplied outputs,
the primary results, the inputs or the feature cache, however the folder is spelled (``_out_guard.py``);
``summarize_v2.py --dir`` builds the summary from either folder without refitting. Needs the feature cache. From the
repository root:

    python camera_ready/scripts/sensitivity/dedup_logo_v2.py [--pools arch6 clip4 strong4 weak6] [--gpu-name 5080]
    python camera_ready/scripts/sensitivity/dedup_logo_v2.py --pools arch6 --folds heygen     # one fold
"""
import argparse
import atexit
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _out_guard import output_dir  # noqa: E402

CR = Path(__file__).resolve().parents[2]          # camera_ready/
ROOT = CR.parent                                  # repository root
POOLS = {"arch6": 256, "clip4": 32, "strong4": 256, "weak6": 256}        # --proj_dim of the primary runs
EXCL = "camera_ready/output/results/mitigation/exclusions"
RERUN = "camera_ready/output/results/sensitivity_rerun/dedup"


def wrap(runner, runner_args):
    """Wrapper process: drop exact duplicates from every fold's training data, then run the runner unchanged."""
    sys.path.insert(0, str(Path(runner).resolve().parent))
    import numpy as np
    import logo_common
    original = logo_common.fold_view
    removed = {}

    def fold_view_dedup(gen_data, held_out, exclude=None):
        tr_f, tr_lg, tr_y, te_f, te_lg, te_y, n_excl = original(gen_data, held_out, exclude)
        seen, keep = set(), []
        for i, h in enumerate(logo_common.row_hashes(tr_f)):
            if h not in seen:
                seen.add(h)
                keep.append(i)
        keep = np.asarray(keep, dtype=int)
        removed[held_out] = {"training_rows_before": int(len(tr_y)), "duplicates_removed": int(len(tr_y) - len(keep)),
                             "training_rows_after": int(len(keep))}
        return [f[keep] for f in tr_f], tr_lg[keep], tr_y[keep], te_f, te_lg, te_y, n_excl

    logo_common.fold_view = fold_view_dedup
    out_dir = Path(runner_args[runner_args.index("--out-dir") + 1])
    pool = runner_args[runner_args.index("--pool") + 1]

    def record():
        if removed:
            p = out_dir / f"dedup_counts_{Path(runner).stem}_{pool}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"runner": Path(runner).name, "pool": pool, "folds": removed}, indent=2),
                         encoding="utf-8")
    atexit.register(record)
    sys.argv = [str(runner)] + list(runner_args)
    runpy.run_path(str(runner), run_name="__main__")


def child_env():
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}


def run_pool(pool, out, gpu_name, log, folds=None):
    runs = CR / "scripts" / "runs"
    common = ["--pool", pool, "--exclusion", "content", "--manifest", f"{EXCL}/logo_exclusions_{pool}.json",
              "--out-dir", out, "--codebase", "."] + (["--folds", *folds] if folds else [])
    gpu = ["--gpu-name", gpu_name] if gpu_name else []
    steps = [("11_disjoint_data_v4.py", common + ["--proj_dim", str(POOLS[pool])] + gpu),      # version 2
             ("08_logo_extended_rungs_v3.py", common + ["--regime", "both"])]
    for script, args in steps:
        t0 = time.time()
        cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--wrap", str(runs / script), "--", *args]
        console = ROOT / out / f"console_{pool}_{Path(script).stem}.txt"
        with open(console, "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, env=child_env(), stdout=fh, stderr=subprocess.STDOUT).returncode
        log(f"{pool:<8} {script:<30} exit {rc}  {(time.time() - t0) / 60:6.1f} min")
        if rc != 0:
            raise SystemExit(f"{pool} {script} failed (exit {rc}); see {console}")


def main() -> int:
    if "--wrap" in sys.argv:
        wrap(sys.argv[sys.argv.index("--wrap") + 1], sys.argv[sys.argv.index("--") + 1:])
        return 0
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pools", nargs="+", default=list(POOLS), choices=list(POOLS))
    ap.add_argument("--gpu-name", help="pin FS2 to the one GPU whose name contains this text (as the harness)")
    ap.add_argument("--folds", nargs="+", help="held-out generators to run (default: all 8; the summary needs all)")
    ap.add_argument("--out-dir", default=RERUN, help=f"relative to the repository root (default {RERUN})")
    a = ap.parse_args()
    output_dir(a.out_dir)                         # refuses the supplied outputs, the primary results, inputs and cache
    (ROOT / a.out_dir).mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(ROOT / a.out_dir / "PROGRESS.txt", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    for pool in a.pools:
        log(f"{pool}: start")
        run_pool(pool, a.out_dir, a.gpu_name, log, a.folds)
    log("done; summary: python camera_ready/scripts/sensitivity/summarize_v2.py --dir "
        + str(Path(a.out_dir).parent.as_posix()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
