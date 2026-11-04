"""Two follow-up sensitivity checks with exact duplicates removed from the training data, version 2.

Version 2 (the supplied outputs come from version 1) runs the mechanism under LOGO with the current runner, ``scripts/runs/05_run_c_width_matched_v5.py``,
which records in every fold the membership fingerprints of FIT and SELECT, through the wrapper of
``dedup_logo_v2.py``, so that the summary (``summarize_v2.py``) can check that it used the rows of the deduplicated
LOGO run of the same folder; it starts only when the LOGO runs of arch6 and clip4 in that folder record the
fingerprints (written by ``dedup_logo_v2.py``). The standard split runs as in version 1. Everything else is as in
version 1:

Two follow-up sensitivity checks with exact duplicates removed from the training data.

The LOGO deduplication check showed that PS4, whose inner CV saw repeated rows, rises once the copies are removed
(arch6 0.727 -> 0.743, clip4 0.730 -> 0.756). PS4 is the comparator of the mechanism analysis and enters two
standard-split figures, so these two checks repeat them the same way:

1. Mechanism under LOGO (Table A9, LOGO columns; Sec. 5.2): ``scripts/runs/05_run_c_width_matched_v5.py --protocol
   logo`` for arch6 and clip4, through the wrapper of dedup_logo_v2.py
   (each fold's training data keeps the first copy of every exact duplicate). Its statistics use the deduplicated
   LOGO ladder run of the same folder (FS1 and FS2 per fold), so dedup_logo_v2.py runs first.
2. Standard split (Table 1 standard row, Table A3, the audit's weak6 PS4 effect): ``scripts/analyses/
   appB_richer_inputs.py --xgb_mode cv`` (the script of the primary arch6 and clip4 results) for arch6, clip4 and
   weak6, with FIT keeping the first copy of every exact duplicate and SELECT dropping its rows whose features occur
   in FIT, and its own repeats (``dedup_counts_standard_<pool>.json``). The test sets are unchanged.

The supplied outputs are in camera_ready/output/results/sensitivity/dedup/ (the standard split in standard/). A
rerun writes to --out-dir (default camera_ready/output/results/sensitivity_rerun/dedup/, where dedup_logo_v2.py writes
by default), never into the supplied outputs, the primary results, the inputs or the feature cache, however the folder
is spelled (``_out_guard.py``). Needs the feature cache. From the repository root:

    python camera_ready/scripts/sensitivity/dedup_followup_v2.py [--gpu-name 5080]
"""
import argparse
import atexit
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _out_guard import output_dir  # noqa: E402

CR = Path(__file__).resolve().parents[2]          # camera_ready/
ROOT = CR.parent
WRAPPER = Path(__file__).resolve().parent / "dedup_logo_v2.py"                  # version 2
EXCL = "camera_ready/output/results/mitigation/exclusions"
RERUN = "camera_ready/output/results/sensitivity_rerun/dedup"


def wrap_standard(script, script_args):
    """Wrapper process for the standard split: FIT without repeated rows, SELECT without rows that occur in FIT."""
    analyses = Path(script).resolve().parent
    sys.path.insert(0, str(analyses))
    sys.path.insert(0, str(analyses.parent / "runs"))
    import _paths  # noqa: F401  (first, as in every analysis script)
    import numpy as np
    import _common
    from logo_common import row_hashes
    from src.utils.config import FIT_TAG, SELECT_TAG
    original = _common.load_split
    state, counts = {}, {}

    def first_copies(hashes, exclude):
        seen, keep = set(), []
        for i, h in enumerate(hashes):
            if h not in seen and h not in exclude:
                seen.add(h)
                keep.append(i)
        return np.asarray(keep, dtype=int), seen

    def load_split_dedup(expert_names, tag, cache_root):
        feats, logits3d, labels = original(expert_names, tag, cache_root)
        if tag not in (FIT_TAG, SELECT_TAG):
            return feats, logits3d, labels
        if tag == SELECT_TAG and "fit" not in state:
            load_split_dedup(expert_names, FIT_TAG, cache_root)
        keep, seen = first_copies(row_hashes(feats), state.get("fit", set()) if tag == SELECT_TAG else set())
        if tag == FIT_TAG:
            state["fit"] = seen
        counts[tag] = {"rows_before": int(len(labels)), "rows_removed": int(len(labels) - len(keep)),
                       "rows_after": int(len(keep))}
        return [f[keep] for f in feats], logits3d[keep], labels[keep]

    _common.load_split = load_split_dedup
    pool = script_args[script_args.index("--pool") + 1]

    def record():
        p = Path(os.environ["CR_RESULTS_DIR"]) / f"dedup_counts_standard_{pool}.json"
        p.write_text(json.dumps({"script": Path(script).name, "pool": pool, "splits": counts}, indent=2),
                     encoding="utf-8")
    atexit.register(record)
    sys.argv = [str(script)] + list(script_args)
    runpy.run_path(str(script), run_name="__main__")


def gpu_env(gpu_name):
    """CUDA variables that pin the one GPU whose name contains `gpu_name` ({} without a name)."""
    if not gpu_name:
        return {}
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout if shutil.which("nvidia-smi") else ""
    hits = [line.split(",")[0].strip() for line in out.splitlines() if gpu_name.lower() in line.lower()]
    if len(hits) != 1:
        raise SystemExit(f"--gpu-name {gpu_name}: {len(hits)} matching GPU(s)")
    return {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": hits[0]}


def run(out, label, cmd, env, log):
    t0 = time.time()
    console = ROOT / out / f"console_followup_{label}.txt"
    with open(console, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    log(f"{label:<22} exit {rc}  {(time.time() - t0) / 60:6.1f} min")
    if rc != 0:
        raise SystemExit(f"{label} failed (exit {rc}); see {console}")


def main() -> int:
    if "--wrap-standard" in sys.argv:
        wrap_standard(sys.argv[sys.argv.index("--wrap-standard") + 1], sys.argv[sys.argv.index("--") + 1:])
        return 0
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu-name", help="pin the standard-split neural heads to this GPU (as the primary Block C run)")
    ap.add_argument("--out-dir", default=RERUN, help=f"relative to the repository root (default {RERUN})")
    a = ap.parse_args()
    output_dir(a.out_dir)                         # refuses the supplied outputs, the primary results, inputs and cache
    for pool, d in (("arch6", 256), ("clip4", 32)):                   # version 2: LOGO runs with fingerprints
        p = ROOT / a.out_dir / f"11_disjoint_{pool}_d{d}.json"
        if not p.exists():
            raise SystemExit(f"run dedup_logo_v2.py --out-dir {a.out_dir} first (the mechanism needs its LOGO "
                             "runs)")
        per = json.loads(p.read_text(encoding="utf-8")).get("per_fold") or []
        if not per or not all(f.get("fit_fingerprint") and f.get("select_fingerprint") for f in per):
            raise SystemExit(f"{p.name} in {a.out_dir} records no membership fingerprints (written by an earlier "
                             f"runner); run dedup_logo_v2.py --out-dir {a.out_dir} first")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(ROOT / a.out_dir / "PROGRESS.txt", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    base = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    runs = CR / "scripts" / "runs"
    for pool in ("arch6", "clip4"):
        args = ["--pool", pool, "--protocol", "logo", "--exclusion", "content", "--manifest",
                f"{EXCL}/logo_exclusions_{pool}.json", "--out-dir", a.out_dir, "--codebase", "."]
        run(a.out_dir, f"mechanism_{pool}", [sys.executable, "-B", str(WRAPPER), "--wrap",
                                             str(runs / "05_run_c_width_matched_v5.py"), "--", *args], base, log)
    std = ROOT / a.out_dir / "standard"
    std.mkdir(parents=True, exist_ok=True)
    std_env = {**base, "CR_RESULTS_DIR": str(std), **gpu_env(a.gpu_name)}
    script = CR / "scripts" / "analyses" / "appB_richer_inputs.py"
    for pool in ("arch6", "clip4", "weak6"):
        run(a.out_dir, f"standard_{pool}", [sys.executable, "-B", str(Path(__file__).resolve()), "--wrap-standard",
                                            str(script), "--", "--pool", pool, "--xgb_mode", "cv"], std_env, log)
    log("follow-up checks done; summary: python camera_ready/scripts/sensitivity/summarize_v2.py --dir "
        + str(Path(a.out_dir).parent.as_posix()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
