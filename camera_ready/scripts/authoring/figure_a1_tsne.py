"""Reproduce Figure A1 from the released clip4 test cache.

Run from any working directory with --cache and --out-dir. Only the output
directory is written. Sampling and t-SNE settings match Appendix D.2.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from importlib.metadata import version

import numpy as np
import sklearn
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

EXPERTS = ("fs-clip-large", "fr-clip-large", "efs-clip-large", "df40-clip-large")
GENERATORS = ("whichisreal", "CollabDiff", "MidJourney", "heygen",
              "starganv2", "deepfacelab", "stargan", "styleclip")
LABELS = ("WIR", "CDiff", "MJ6", "HeyG", "SG2", "DFL", "SG", "SCLIP")
COLORS = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a",
          "#66a61e", "#e6ab02", "#a6761d", "#666666")
EXPECTED_ROWS = (2001, 2000, 1882, 101, 4000, 149, 3968, 3824)
SETTINGS = dict(n_components=2, perplexity=30.0, early_exaggeration=12.0,
                learning_rate="auto", max_iter=1000, n_iter_without_progress=300,
                min_grad_norm=1e-7, metric="euclidean", init="pca",
                verbose=0, random_state=42, method="barnes_hut", angle=0.5,
                n_jobs=None)
STYLE = {"font.family": "serif", "font.size": 9, "axes.labelsize": 10,
         "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8,
         "ytick.labelsize": 8, "figure.dpi": 150, "savefig.dpi": 300,
         "savefig.bbox": "tight", "savefig.pad_inches": 0.05}


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_sample(cache):
    """Sample rows in the original order, casting cached values to float32."""
    rng = np.random.RandomState(42)
    features, logits, groups, row_indices = [], [], [], []
    counts = {}
    for group, (generator, expected) in enumerate(zip(GENERATORS, EXPECTED_ROWS)):
        root = Path(cache) / generator
        labels = np.load(root / "labels.npy", mmap_mode="r", allow_pickle=False)
        if labels.ndim != 1 or len(labels) != expected:
            raise ValueError(f"{generator}: expected {expected} labels, got {labels.shape}")
        idx = rng.choice(len(labels), min(625, len(labels)), replace=False)
        fs, ps = [], []
        for expert in EXPERTS:
            for suffix, width, target in (("features", 1024, fs), ("logits", 2, ps)):
                a = np.load(root / f"{expert}_{suffix}.npy", mmap_mode="r",
                            allow_pickle=False)
                if a.shape != (expected, width):
                    raise ValueError(f"{generator}/{expert}_{suffix}: unexpected shape {a.shape}")
                selected = np.asarray(a[idx], dtype=np.float32)
                if not np.isfinite(selected).all():
                    raise ValueError(f"{generator}/{expert}_{suffix}: non-finite sampled values")
                target.append(selected)
        features.append(np.concatenate(fs, axis=1))
        logits.append(np.concatenate(ps, axis=1))
        groups.extend([group] * len(idx))
        row_indices.extend(idx.tolist())
        counts[generator] = len(idx)
    return (np.concatenate(features), np.concatenate(logits),
            np.asarray(groups, dtype=np.int32),
            np.asarray(row_indices, dtype=np.int64), counts)


def draw(feature_embedding, logit_embedding, groups, out, formats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))
        for group, (label, color) in enumerate(zip(LABELS, COLORS)):
            mask = groups == group
            for ax, embedding in ((ax1, feature_embedding), (ax2, logit_embedding)):
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c=color, s=3, alpha=0.5, label=label, rasterized=True)
        ax1.set_title("(a) Concatenated features")
        ax2.set_title("(b) Concatenated logits")
        for ax in (ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, framealpha=0.9,
                   markerscale=3, bbox_to_anchor=(0.5, -0.05))
        fig.tight_layout(rect=[0, 0.08, 1, 1.0])
        for fmt in formats:
            fig.savefig(out / f"figureA1_tsne.{fmt}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True,
                        help="Root containing the released test-generator cache directories")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Destination for figures, coordinates and the reproduction record")
    parser.add_argument("--formats", choices=("png", "pdf"), nargs="+",
                        default=["pdf", "png"])
    parser.add_argument("--threads", type=int, default=1,
                        help="Numerical-library thread limit (default: 1)")
    parser.add_argument("--check-inputs", action="store_true",
                        help="Validate and fingerprint the sampled inputs without plotting or writing")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    cache, out = args.cache.resolve(), args.out_dir.resolve()
    if out == cache or cache in out.parents:
        parser.error("--out-dir must be outside the feature cache")
    formats = list(dict.fromkeys(args.formats))
    if not args.check_inputs:
        targets = [out / f"figureA1_tsne.{fmt}" for fmt in formats]
        targets += [out / "figureA1_tsne_coordinates.npz", out / "figureA1_tsne_run.json"]
        if any(p.exists() for p in targets):
            parser.error("output files already exist; choose a new --out-dir")
    features, logits, groups, row_indices, counts = load_sample(cache)
    record = {
        "figure": "A1", "pool": "clip4", "experts": EXPERTS,
        "generators": GENERATORS, "sample_counts": counts,
        "sampling": {"seed": 42, "maximum_rows_per_generator": 625,
                     "replace": False, "total_rows": len(groups)},
        "inputs": {"features_shape": features.shape, "logits_shape": logits.shape,
                   "dtype": "float32", "standardization": None,
                   "features_sha256": array_sha256(features),
                   "logits_sha256": array_sha256(logits),
                   "groups_sha256": array_sha256(groups),
                   "row_indices_sha256": array_sha256(row_indices)},
        "tsne": SETTINGS,
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                        "scipy": version("scipy"), "matplotlib": version("matplotlib"),
                        "threads": args.threads},
    }
    if args.check_inputs:
        print(json.dumps(record, indent=2))
        return 0
    print(f"Embedding {len(groups)} rows: features {features.shape}, logits {logits.shape}",
          flush=True)
    with threadpool_limits(limits=args.threads):
        tsne_f = TSNE(**SETTINGS).fit(features)
        print("Feature embedding complete", flush=True)
        tsne_p = TSNE(**SETTINGS).fit(logits)
    out.mkdir(parents=True, exist_ok=True)
    draw(tsne_f.embedding_, tsne_p.embedding_, groups, out, formats)
    np.savez_compressed(out / "figureA1_tsne_coordinates.npz",
                        features=tsne_f.embedding_, logits=tsne_p.embedding_,
                        generator_index=groups, row_index=row_indices)
    record["fits"] = {
        "features": {"kl_divergence": float(tsne_f.kl_divergence_),
                     "iterations": int(tsne_f.n_iter_)},
        "logits": {"kl_divergence": float(tsne_p.kl_divergence_),
                   "iterations": int(tsne_p.n_iter_)},
    }
    record["outputs"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(out.glob("figureA1_tsne.*"))
                         if p.suffix in (".png", ".pdf")}
    (out / "figureA1_tsne_run.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Figure, coordinates and reproduction record written to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
