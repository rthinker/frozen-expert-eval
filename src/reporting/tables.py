"""LaTeX renderers for the Block A diagnostics.

Each function takes the scan_results manifest and returns a LaTeX string
(or None if required inputs are missing). No file I/O; callers decide where
to write the result.

Tables implemented:
  - table2_baseline_ladder(pool)       -- prediction-space ladder and feature-space rows
  - table3_capacity_ablation(pool)     -- capacity ablation
  - table5_pca_bottleneck              -- effective dimensionality
  - diagnostic_scope_summary           -- per-pool Best PS vs Best FS cross-check (diagnostic)
  - table_robustness_controls          -- FS_D vs PS_4 and FS_1 raw vs standardized


All tables degrade gracefully when the required inputs are absent: they
return None and `render_all_tables` logs a warning instead of raising.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

TEST_GENERATORS = [
    "whichisreal", "CollabDiff", "MidJourney", "heygen",
    "starganv2", "deepfacelab", "stargan", "styleclip",
]


def _fmt(val, digits=3) -> str:
    if val is None:
        return "---"
    return f"{val:.{digits}f}"


# LaTeX reserved characters that must be escaped when a body cell is
# populated from a config string, pool name, method label, variant tag,
# or any other value that was not hand-written for LaTeX. Applies only
# to user data -- do not call on preamble/row-structural strings.
_TEX_ESCAPES = (
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
)


def _tex(val) -> str:
    """Escape a raw value for safe inclusion in a LaTeX table body cell."""
    if val is None:
        return "---"
    s = str(val)
    for needle, repl in _TEX_ESCAPES:
        s = s.replace(needle, repl)
    return s


# ---------------------------------------------------------------------------
# Baseline ladder
# ---------------------------------------------------------------------------

def table2_baseline_ladder(manifest: Dict[str, Any], pool: str) -> Optional[str]:
    """Render the full PS ladder plus FS rows for a pool."""
    ds = manifest.get("ds", {}).get(pool)
    if ds is None:
        log.warning(f"table2_baseline_ladder: no DS results for {pool}")
        return None

    method_means = ds.get("method_means", {})
    if not method_means:
        log.warning(
            f"table2_baseline_ladder: {pool} has no method_means; run "
            f"the updated 04_prediction_space.py"
        )
        return None

    preferred_order = [
        "PS_1_prob_avg", "PS_A_calibrated_avg", "PS_B_learned_weights",
        "PS_2_logreg", "PS_C_logreg_cv",
        "PS_3_mlp_logits", "PS_F_mlp_logits_es",
        "PS_D_mlp_probs", "PS_E_mlp_probs_es",
        "PS_4_xgboost",
    ]
    ordered = [m for m in preferred_order if m in method_means]
    ordered += [m for m in method_means if m not in preferred_order]

    lines = []
    lines.append(f"% Block A prediction-space ladder and feature-space results for pool={pool}")
    n_cols = len(TEST_GENERATORS) + 1
    lines.append("\\begin{tabular}{l" + "c" * n_cols + "}")
    lines.append("\\toprule")
    header = "Method & " + " & ".join(TEST_GENERATORS) + " & Avg(8) \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for m in ordered:
        entry = method_means[m]
        per_gen = entry.get("per_generator", {})
        row = (
            _tex(m) + " & "
            + " & ".join(_fmt(per_gen.get(g)) for g in TEST_GENERATORS)
            + " & " + _fmt(entry["mean_avg8"]) + " \\\\"
        )
        lines.append(row)

    lines.append("\\midrule")

    rawfeat = manifest.get("rawfeat", {}).get(pool)
    if rawfeat is not None:
        per_gen = rawfeat.get("per_generator", {})
        row = (
            "Raw-feat LogReg & "
            + " & ".join(_fmt(per_gen.get(g)) for g in TEST_GENERATORS)
            + " & " + _fmt(rawfeat.get("avg_auc_8")) + " \\\\"
        )
        lines.append(row)

    fs_entries = sorted(
        [(d, v) for (p, ctype, d), v in manifest.get("fs", {}).items()
         if p == pool and ctype == "mlp"],
    )
    for d, fs in fs_entries:
        # Prefer AUC on seed-averaged predictions (per_generator_seed_averaged)
        # so per-gen values and Avg(8) come from the same protocol. Fall
        # back to mean-of-per-seed per-gen AUCs for legacy JSONs.
        per_gen = fs.get("per_generator_seed_averaged")
        if not per_gen:
            per_gen = {}
            for g in TEST_GENERATORS:
                vals = [s.get("per_generator", {}).get(g)
                        for s in fs.get("per_seed", [])]
                vals = [v for v in vals if v is not None]
                if vals:
                    per_gen[g] = sum(vals) / len(vals)
        row = (
            f"FS MLP (d={d}) & "
            + " & ".join(_fmt(per_gen.get(g)) for g in TEST_GENERATORS)
            + " & \\textbf{" + _fmt(fs.get("mean_avg8")) + "} \\\\"
        )
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Capacity ablation
# ---------------------------------------------------------------------------

def table3_capacity_ablation(manifest: Dict[str, Any],
                             pool: str,
                             dim: int = 64) -> Optional[str]:
    """Rows: PCA, random-proj, linear, MLP -- all at the same d."""
    fs = manifest.get("fs", {})
    rows = []
    for ctype, label in [
        ("pca", "PCA + LogReg (unsupervised)"),
        ("random", "Random proj + MLP"),
        ("linear", "Learned proj + Linear"),
        ("mlp", "Learned proj + MLP"),
    ]:
        entry = fs.get((pool, ctype, dim))
        if entry is None:
            continue
        avg = entry.get("mean_avg8", entry.get("avg_auc_8"))
        n_params = entry.get("n_trainable_params", "---")
        rows.append((label, avg, n_params))

    if not rows:
        return None

    lines = []
    lines.append(f"% Block A capacity ablation for pool={pool}, d={dim}")
    lines.append("\\begin{tabular}{lrr}")
    lines.append("\\toprule")
    lines.append("Variant & Avg(8) AUC & Params \\\\")
    lines.append("\\midrule")
    for label, avg, n in rows:
        lines.append(f"{_tex(label)} & {_fmt(avg, 4)} & {_tex(n)} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PCA bottleneck
# ---------------------------------------------------------------------------

def table5_pca_bottleneck(manifest: Dict[str, Any]) -> Optional[str]:
    pca = manifest.get("pca", {})
    if not pca:
        return None
    lines = []
    lines.append("% Block A PCA effective dimensionality")
    lines.append("\\begin{tabular}{llrrrr}")
    lines.append("\\toprule")
    lines.append("Pool & Interface & Dim & PR & $k_{95}$ & $k_{99}$ \\\\")
    lines.append("\\midrule")
    for pool in sorted(pca.keys()):
        d = pca[pool]
        feats = d.get("features", {})
        logs = d.get("logits", {})
        lines.append(
            f"{_tex(pool)} & features & {_tex(feats.get('dim',''))} & "
            f"{_fmt(feats.get('pr'), 1)} & {_tex(feats.get('k95',''))} & {_tex(feats.get('k99',''))} \\\\"
        )
        lines.append(
            f"{_tex(pool)} & logits & {_tex(logs.get('dim',''))} & "
            f"{_fmt(logs.get('pr'), 1)} & {_tex(logs.get('k95',''))} & {_tex(logs.get('k99',''))} \\\\"
        )
        lines.append(
            f"{_tex(pool)} & gap & -- & {_fmt(d.get('pr_gap'), 1)}x & -- & -- \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scope summary
# ---------------------------------------------------------------------------

def diagnostic_scope_summary(manifest: Dict[str, Any]) -> Optional[str]:
    """Diagnostic per-pool summary: pool, Best PS, Best FS = max(FS_1, FS_2), delta, note.

    Diagnostic-only artifact for cross-checking inputs; the paper's per-pool
    table uses FS_1 (raw-feat LogReg) as the consistent anchor and is
    manually typeset in main.tex. This rendered .tex is not referenced by
    the paper.
    """
    ds = manifest.get("ds", {})
    fs = manifest.get("fs", {})
    rawfeat = manifest.get("rawfeat", {})
    disjoint = manifest.get("disjoint", {})
    rows: List[tuple] = []

    def _add(name, best_ps, best_fs, note):
        if best_ps is None or best_fs is None:
            return
        rows.append((name, best_ps, best_fs, best_fs - best_ps, note))

    for pool in sorted(ds.keys()):
        d = ds[pool]
        best_ps = d.get("best_ds_avg8")
        # Best FS = max(raw-feat LogReg, best FS MLP config) for the pool.
        # PCA / random / linear ablations are explicitly excluded so the
        # headline row cannot be replaced by an ablation variant.
        fs_candidates = [
            v.get("mean_avg8") for (p, c, _), v in fs.items()
            if p == pool and c == "mlp" and v.get("mean_avg8") is not None
        ]
        rf = rawfeat.get(pool, {}).get("avg_auc_8")
        candidates = [x for x in fs_candidates + [rf] if x is not None]
        best_fs = max(candidates) if candidates else None
        note = "main" if pool == "arch6" else "ablation"
        _add(pool, best_ps, best_fs, note)

    # LOGO overlap-control
    for pool in sorted(disjoint.keys()):
        d = disjoint[pool]
        _add(
            f"{pool} LOGO",
            d.get("avg_best_ps"),
            max(d.get("avg_fs_mlp", 0), d.get("avg_raw_feat_logreg", 0)),
            "overlap-control",
        )

    if not rows:
        return None

    lines = []
    lines.append("% Block A scope summary (positive, control, boundary)")
    lines.append("\\begin{tabular}{lrrrl}")
    lines.append("\\toprule")
    lines.append("Setting & Best PS & Best FS & Delta & Notes \\\\")
    lines.append("\\midrule")
    for name, bp, bf, delta, note in rows:
        lines.append(
            f"{_tex(name)} & {_fmt(bp)} & {_fmt(bf)} & {delta:+.3f} & {_tex(note)} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Robustness controls: FS_D (arch6 + clip4) and standardized-FS_1 deltas
# ---------------------------------------------------------------------------

def _best_ps_xgb_logit_auc(manifest: Dict[str, Any], pool: str):
    """Return the PS_4 (XGBoost on logits) avg-8 AUC for a pool, from 04_ds.

    Handles both the current schema where ``method_means[method]`` is a dict
    with ``mean_avg8`` / ``std_avg8`` / ``per_generator`` (produced by the
    updated 04_prediction_space.py) and the older ``baselines`` list-of-dicts
    fallback.
    """
    ds = manifest.get("ds", {}).get(pool)
    if ds is None:
        return None
    mm = ds.get("method_means", {})
    entry = mm.get("PS_4_xgboost")
    if isinstance(entry, dict):
        return entry.get("mean_avg8") or entry.get("avg_auc_8")
    if isinstance(entry, (int, float)):
        return float(entry)
    # Fallback: some older runs stored it under "baselines" as list of dicts.
    for b in ds.get("baselines", []):
        if isinstance(b, dict) and b.get("name") == "PS_4_xgboost":
            return b.get("auc")
    return None


def table_robustness_controls(manifest: Dict[str, Any]) -> Optional[str]:
    """Two small robustness-control sub-tables:

    (a) FS_D (XGBoost on features) vs PS_4 (XGBoost on logits), per pool.
        Tests whether the FS--PS gap is partly combiner-family mismatch.
    (b) FS_1 raw features vs FS_1 standardized features, per pool.
        Tests whether the FS_1 result depends on raw feature scale artifacts.

    Missing any input => table is rendered with whichever row(s) are available
    and the absent rows omitted. Returns None only if BOTH sub-tables have zero
    rows.
    """
    xgb_feat = manifest.get("xgb_features", {})
    std = manifest.get("fs1_standardized", {})

    # (a) FS_D vs PS_4-on-logits
    fsd_rows = []
    for pool in ("clip4", "arch6"):
        entry = xgb_feat.get(pool)
        if entry is None:
            continue
        fs_d = entry.get("avg_auc_8")
        ps_4 = _best_ps_xgb_logit_auc(manifest, pool)
        if fs_d is None:
            continue
        delta = (fs_d - ps_4) if ps_4 is not None else None
        fsd_rows.append((pool, fs_d, ps_4, delta))

    # (b) FS_1 raw vs standardized
    std_rows = []
    for pool in ("clip4", "arch6"):
        entry = std.get(pool)
        if entry is None:
            continue
        raw = entry.get("avg_auc_8_raw")
        stz = entry.get("avg_auc_8_std")
        delta = entry.get("delta_std_minus_raw")
        if raw is None and stz is None:
            continue
        std_rows.append((pool, raw, stz, delta))

    if not fsd_rows and not std_rows:
        return None

    lines = []
    lines.append("% Robustness controls: FS_D (same-family) and FS_1 standardization")

    if fsd_rows:
        lines.append("% (a) FS_D (XGBoost on features) vs PS_4 (XGBoost on logits)")
        lines.append("\\begin{tabular}{lrrr}")
        lines.append("\\toprule")
        lines.append(
            "Pool & FS\\_D: XGB on features & PS\\_4: XGB on logits & Delta \\\\"
        )
        lines.append("\\midrule")
        for pool, fs_d, ps_4, delta in fsd_rows:
            ps_4_str = _fmt(ps_4, 4)
            delta_str = f"{delta:+.4f}" if delta is not None else "---"
            lines.append(
                f"{_tex(pool)} & {_fmt(fs_d, 4)} & {ps_4_str} & {delta_str} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

    if std_rows:
        if fsd_rows:
            lines.append("")  # blank line between the two sub-tables
        lines.append("% (b) FS_1 raw features vs standardized features")
        lines.append("\\begin{tabular}{lrrr}")
        lines.append("\\toprule")
        lines.append(
            "Pool & FS\\_1 raw & FS\\_1 standardized & Delta (std$-$raw) \\\\"
        )
        lines.append("\\midrule")
        for pool, raw, stz, delta in std_rows:
            delta_str = f"{delta:+.4f}" if delta is not None else "---"
            lines.append(
                f"{_tex(pool)} & {_fmt(raw, 4)} & {_fmt(stz, 4)} & {delta_str} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def render_all_tables(manifest: Dict[str, Any],
                      tables_dir) -> List[str]:
    """Render every paper table that has inputs. Returns list of written paths."""
    from pathlib import Path

    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    written = []

    targets = [
        ("table2_ladder_clip4.tex", table2_baseline_ladder, ("clip4",)),
        ("table2a_ladder_arch6.tex", table2_baseline_ladder, ("arch6",)),
        ("table3_capacity_clip4_d64.tex", table3_capacity_ablation, ("clip4", 64)),
        ("table5_pca_bottleneck.tex", table5_pca_bottleneck, ()),
        ("diagnostic_scope_summary.tex", diagnostic_scope_summary, ()),
        ("table_robustness_controls.tex", table_robustness_controls, ()),
    ]
    for filename, fn, args in targets:
        try:
            result = fn(manifest, *args)
        except Exception as exc:
            log.warning(f"{filename}: failed to render: {exc}")
            continue
        if result is None:
            log.warning(f"{filename}: skipped (missing inputs)")
            continue
        out = tables_dir / filename
        out.write_text(result, encoding="utf-8")
        written.append(str(out))
    return written
