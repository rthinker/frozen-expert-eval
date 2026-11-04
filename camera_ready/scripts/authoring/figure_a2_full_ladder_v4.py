"""Plot Figure A2 from the saved DF40 results.

For each pool, plot the larger FS1 or FS2 macro AUROC minus the strongest
tested prediction-space comparator, against the feature-to-logit
participation-ratio gap. The plot uses the full ladder for arch6 and clip4,
and the methods evaluated for the remaining pools.

The harness runs this plot as C23. For standalone use, pass --codebase,
--stats and a separate --out path without an extension. Use --fallback to
supply stored inputs when a result is a memory-budget skip marker. The
script writes a PDF and PNG.
"""
import argparse
import json
from pathlib import Path

import numpy as np

POOLS = [  # (pool, axis, best-FS source), as in the reviewed figure script
    ("arch6", "arch", "fs2"), ("clip4", "data", "fs2"), ("mixed6", "mixed", "fs2"),
    ("merged9", "mixed", "fs2"), ("strong4", "mixed", "fs2"), ("balanced8", "mixed", "fs2"),
    ("full21", "mixed", "fs2"), ("weak6", "boundary", "fs1"), ("low_het4", "boundary", "fs1"),
]
AXIS_COLORS = {"arch": "#d62728", "data": "#1f77b4", "mixed": "#2ca02c", "boundary": "#7f7f7f"}
# label position next to each point: (x multiplier, y offset, horizontal alignment)
OFFSETS = {"arch6": (1.10, 0.000, "left"), "clip4": (1.09, 0.000, "left"), "mixed6": (0.91, 0.000, "right"),
           "merged9": (1.00, 0.008, "center"), "strong4": (1.09, 0.000, "left"),
           "balanced8": (1.09, -0.003, "left"), "full21": (1.00, 0.008, "center"),
           "weak6": (1.14, 0.000, "left"), "low_het4": (1.14, 0.000, "left")}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def read_result(res, name, fallback=None):
    """Version 2: a result file, or its stored copy in `fallback` when the file is a skip marker."""
    d = load(res / name)
    if isinstance(d, dict) and d.get("skipped") is True:
        if fallback is None or not (Path(fallback) / name).is_file():
            raise SystemExit(f"{name} is a skip marker (reason: {d.get('reason', 'unspecified')}) and no stored "
                             f"copy was found; pass --fallback <folder with the stored results>")
        print(f"  {name}: skip marker in {res}; using the stored copy in {fallback}")
        d = load(Path(fallback) / name)
    return d


def read_pool(res, pool, fallback=None):
    pr_gap = float(read_result(res, f"08_pca_{pool}.json", fallback)["pr_gap"])
    if pool == "merged9":
        m = read_result(res, "14_merged9.json", fallback)
        ps, fs1 = float(m["ps6_xgboost"]["avg_auc_8"]), float(m["fs1_raw_feat_logreg"]["avg_auc_8"])
    else:
        ps = float(read_result(res, f"04_ds_{pool}.json", fallback)["best_ds_avg8"])
        fs1 = float(read_result(res, f"04_ds_{pool}_rawfeat.json", fallback)["avg_auc_8"])
    d = 32 if pool == "clip4" else 256
    fs2 = float(read_result(res, f"05_fs_{pool}_mlp_d{d}.json", fallback)["mean_avg8"])
    return pr_gap, ps, fs1, fs2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--codebase", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output path without extension")
    ap.add_argument("--fallback", type=Path,
                    help="version 2: folder of stored results, read for a result that is a skip marker")
    a = ap.parse_args()
    res = a.codebase / "output" / "results"
    full = load(a.stats)["standard"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    rows = []
    for pool, axis, src in POOLS:
        pr_gap, ps, fs1, fs2 = read_pool(res, pool, a.fallback)
        if pool in full:
            ps = full[pool]["ps_ceiling"]
        best_fs = max(fs1, fs2)                                   # version 3: as the caption says
        if (fs2 if src == "fs2" else fs1) != best_fs:
            print(f"  {pool}: the larger is {'FS1' if best_fs == fs1 else 'FS2'} here, not the reviewed script's "
                  f"{src.upper()}")
        rows.append({"pool": pool, "axis": axis, "pr_gap": pr_gap, "delta": best_fs - ps, "ps": ps})
        print(f"  {pool:10s} PR={pr_gap:5.1f}x  PS={ps:.3f}  bestFS={best_fs:.3f}  delta={best_fs - ps:+.3f}")

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    cmap = plt.cm.viridis
    vmin, vmax = 0.70, 0.96
    for r in rows:
        ax.scatter(r["pr_gap"], r["delta"], s=220, c=[cmap((r["ps"] - vmin) / (vmax - vmin))],
                   edgecolors=AXIS_COLORS[r["axis"]], linewidths=2.0, zorder=3)
    for r in rows:
        mx, dy, ha = OFFSETS.get(r["pool"], (1.08, 0.000, "left"))
        ax.annotate(r["pool"], (r["pr_gap"], r["delta"]), xytext=(r["pr_gap"] * mx, r["delta"] + dy),
                    fontsize=9, ha=ha, va="center")
    ax.set_xlabel("PR gap = PR(features) / PR(logits)")                            # version 4
    ax.set_ylabel(r"Best of FS$_1$/FS$_2$ $-$ best tested PS ($\Delta$AUROC)")    # version 4
    ax.set_xscale("log")
    ax.set_xlim(0.9, 35)
    deltas = [r["delta"] for r in rows]                          # version 3: [0, 0.15] unless a value falls outside
    ax.set_ylim(min(0.0, min(deltas) - 0.01), max(0.15, max(deltas) + 0.01))
    ax.grid(True, alpha=0.3, zorder=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Best tested PS macro AUROC")                                    # version 4
    ax.legend(handles=[Patch(edgecolor=c, facecolor="white", linewidth=2.0, label=l.capitalize())
                       for l, c in AXIS_COLORS.items()],
              title="Pool category", loc="lower left", framealpha=0.9, fontsize=8,  # version 4
              bbox_to_anchor=(0.02, 0.02))
    plt.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(a.out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"written {a.out}.pdf/.png")


if __name__ == "__main__":
    main()
