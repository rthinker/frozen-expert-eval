"""Build the DF40 paper tables and their source-linked numerical ledger.

Pass --inputs camera_ready/inputs and --out <folder>. See
camera_ready/RELEASE_MANIFEST.md for the harness command and outputs.
"""

import argparse
import csv
import hashlib
import itertools
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runs"))
import ladder_checks  # noqa: E402  version 13: complete ladders, runs that belong together

_HERE = Path(__file__).resolve()
CAMERA = _HERE.parents[3]                      # historical source-layout alias
CH3 = CAMERA.parent                            # historical source-layout alias
THESIS = CH3.parent
OUT_TABLES = CAMERA / "build" / "tables"
OUT_PROV = CAMERA / "answers" / "05_number_provenance.csv"
OUT_CHECK = CAMERA / "answers" / "06_posted_number_check.md"

REB = "02.rebuttal"
SUBRES = "01.initial_submission/codebase/output/results"
SOURCES = {
    "logo_arch6": f"{SUBRES}/11_disjoint_arch6_d256.json",
    "ladder_arch6": f"{REB}/results/run_a_extended_ladder_arch6.json",
    "ladder_clip4": f"{REB}/results/run_a_extended_ladder_clip4.json",
    "logo_rungs": f"{REB}/runs/reprod/results/A3c_logo_extended_rungs_arch6.json",
    "logo_combined": f"{REB}/runs/reprod/results/A3c_logo_combined_ceiling_arch6.json",
    "width": f"{REB}/results/run_c_width_matched_arch6.json",
    "k3_heads": f"{REB}/runs/reprod/results/NATIVE_K_ALIGNED_FULL_B_arch6_RESULT.json",
    "k3_view_a": f"{REB}/runs/reprod/results/K3_RESULT_A_df40.json",
    "k3_view_b": f"{REB}/runs/reprod/results/K3_RESULT_B_arch6.json",
    "cost": f"{REB}/results/run_d_cost_ledger_arch6.json",
    "diagnose": f"{REB}/results/run_b_diagnose_arch6.json",
    "content": f"{REB}/runs/reprod/results/A4_content_only_rows.json",
    "bit_identity": f"{REB}/results/verify_bit_identity.json",
    "strict": f"{REB}/results/run_b_path_disjoint_arch6_family.json",
    "strict_verify": f"{REB}/results/verify_strict_arm_claims.json",
    "appendix_h": f"{REB}/runs/reprod/results/A2_appendix_h_diff.json",
    "log_clip4": "camera_ready/inputs/A4_logo_clip4.txt",
    "log_strong4": "camera_ready/inputs/A5_logo_strong4.txt",
    "log_weak6": "camera_ready/inputs/A4_logo_weak6.txt",
    "log_logo_rerun": "camera_ready/inputs/A3_logo_arch6.txt",
    "gate": "03.camera-ready/code/output/results/analyses/gate_expert_side_check.json",
    "experts_cfg": "01.initial_submission/codebase/configs/experts.yaml",
    "pca_arch6": f"{SUBRES}/08_pca_arch6.json",
    "pca_clip4": f"{SUBRES}/08_pca_clip4.json",
    "appendix_h_published": f"{SUBRES}/16_individual_expert_auc.json",
    "merged9": f"{SUBRES}/14_merged9.json",
    # derived in 03.camera-ready by the camera-ready scripts (self-checked against the record)
    "reuse_check": "03.camera-ready/code/output/results/split_reuse_check.json",
    "appendix_h_cells": "03.camera-ready/code/output/results/16_individual_expert_auc_corrected.json",
    # statistics against the full prediction-space ladder (full_ladder_stats.py, self-checked)
    "full_ladder": "03.camera-ready/code/output/results/full_ladder_stats.json",
}
# See camera_ready/RELEASE_MANIFEST.md for mitigation runs and gates.
MIT = "03.camera-ready/code/output/results/mitigation"
# version 7: the sensitivity checks (their outputs keep their folder, next to the mitigation runs)
SENS = "03.camera-ready/code/output/results/sensitivity"
SOURCES.update({
    "cd_logo_arch6": f"{MIT}/content_disjoint/11_disjoint_arch6_d256.json",
    "cd_logo_clip4": f"{MIT}/content_disjoint/11_disjoint_clip4_d32.json",
    "cd_logo_strong4": f"{MIT}/content_disjoint/11_disjoint_strong4_d256.json",
    "cd_logo_weak6": f"{MIT}/content_disjoint/11_disjoint_weak6_d256.json",
    "cd_rungs_arch6": f"{MIT}/content_disjoint/A3c_logo_extended_rungs_arch6.json",
    "mech_logo_arch6": f"{MIT}/mechanism/run_c_v2_logo_arch6.json",
    "mech_logo_clip4": f"{MIT}/mechanism/run_c_v2_logo_clip4.json",
    "mitigation": f"{MIT}/mitigation_stats.json",
    "sens_summary": f"{SENS}/summary.json",
    "sens_twins": f"{SENS}/twins.json",
    "cr_reference": "03.camera-ready/code/reference/reference_results_camera_ready_v2.json",    # version 8
    "runner_08": "03.camera-ready/code/scripts/runs/08_logo_extended_rungs_v3.py",
    "exclusions_arch6": f"{MIT}/exclusions/logo_exclusions_arch6.json",
})
EXPERT_ORDER = ["fs-clip-base", "fs-clip-large", "fs-xception", "fs-srm", "fs-recce", "fs-rfm",
                "fr-clip-base", "fr-clip-large", "fr-xception", "fr-srm", "fr-recce", "fr-rfm",
                "efs-clip-base", "efs-clip-large", "efs-xception", "efs-srm", "efs-recce", "efs-rfm",
                "df40-xception", "df40-clip-base", "df40-clip-large"]
GENERATORS = ["whichisreal", "CollabDiff", "MidJourney", "heygen",
              "starganv2", "deepfacelab", "stargan", "styleclip"]
GEN_LABEL = {"whichisreal": "WhichIsReal", "CollabDiff": "CollabDiff",
             "MidJourney": "MidJourney", "heygen": "HeyGen",
             "starganv2": "StarGAN-v2", "deepfacelab": "DeepFaceLab",
             "stargan": "StarGAN", "styleclip": "StyleCLIP"}


# ---------------------------------------------------------------- helpers

def public_source_path(rel):
    """Describe a ledger source by its public repository location, without changing data loading."""
    candidate = rel
    for old, new in (("01.initial_submission/codebase/", ""),
                     ("03.camera-ready/code/", "camera_ready/"),
                     ("02.rebuttal/runs/reprod/results/", "camera_ready/inputs/"),
                     ("02.rebuttal/results/", "camera_ready/inputs/")):
        if rel.startswith(old):
            candidate = new + rel[len(old):]
            break
    root = Path(__file__).resolve().parents[3]
    if (root / candidate).is_file():
        return candidate
    if (root / "camera_ready" / "inputs" / Path(rel).name).is_file():
        return "camera_ready/inputs/" + Path(rel).name
    return candidate


class Ledger:
    """Collects every number with its provenance."""

    def __init__(self):
        self.rows, self.values, self._sha = [], {}, {}

    def sha(self, key: str) -> str:
        if key not in self._sha:
            self._sha[key] = hashlib.sha256(src(key).read_bytes()).hexdigest()[:16]
        return self._sha[key]

    def add(self, table, qid, value, digits, source, keypath, kind="json", note="", disp=None):
        # version 5: disp overrides the display (decimal half-up rounding of an exact ratio, Table A17)
        disp = disp if disp is not None else (fmt(value, digits) if isinstance(value, (int, float)) else str(value))
        self.values[qid] = value
        src_path = SOURCES.get(source, source)
        self.rows.append({
            "table": table, "quantity": qid,
            "value": repr(value) if isinstance(value, float) else value,
            "display": disp, "source": public_source_path(src_path), "key": keypath, "kind": kind,
            "sha256_16": self.sha(source) if kind in ("json", "log", "derived", "report") else "",
            "note": note,
        })
        return value


def fmt(v, digits, signed=False):
    if isinstance(v, int) and digits == 0:
        return f"{v:,}"
    s = f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"
    return s


def fs1_budget_fraction():
    """Version 10: FS_1_BUDGET_FRACTION as the reviewed code defines it (src/models/feature_space.py)."""
    for root in (Path(__file__).resolve().parents[3], CH3 / "01.initial_submission" / "codebase"):
        p = root / "src" / "models" / "feature_space.py"
        if p.is_file():
            m = re.search(r"^FS_1_BUDGET_FRACTION\s*=\s*([0-9.]+)", p.read_text(encoding="utf-8"), re.M)
            if m:
                return float(m[1])
    raise SystemExit("FS_1_BUDGET_FRACTION not found in the reviewed src/models/feature_space.py")


def sd(v, digits=3):
    """Signed display for tables: +0.105 or $-$0.028, as in the paper."""
    s = f"{v:+.{digits}f}"
    return s if s.startswith("+") else "$-$" + s[1:]


def sm(v, digits=3):
    """Signed display inside math mode, e.g. inside $[...]$ intervals."""
    return f"{v:+.{digits}f}"


# --inputs: a flat folder holding every source file under its own name (the release layout);
# without it, sources use historical layout aliases; public runs should supply --inputs and --out.
INPUTS = None


def src(key):
    """Resolve a source key or a relative source path."""
    rel = SOURCES.get(key, key)
    if INPUTS is not None:
        if rel.startswith(MIT + "/"):
            # release layout: the mitigation outputs keep their folders (their file names repeat those of the
            # stored runs), next to the flat inputs folder: camera_ready/output/results/mitigation/...
            return INPUTS.parent / "output" / "results" / "mitigation" / rel[len(MIT) + 1:]
        if rel.startswith(SENS + "/"):          # version 7: camera_ready/output/results/sensitivity/...
            return INPUTS.parent / "output" / "results" / "sensitivity" / rel[len(SENS) + 1:]
        if key in ("cr_reference", "runner_08", "runner_a", "install_doc"):  # version 7: files of camera_ready/ itself
            return INPUTS.parent / rel[len("03.camera-ready/code/"):]
        return INPUTS / Path(rel).name
    return CH3 / rel if key in SOURCES else THESIS / rel


def load_json(key):
    return json.loads(src(key).read_text(encoding="utf-8"))


def read_text_any(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8")
    if raw[:200].count(b"\x00") > 20:
        return raw.decode("utf-16-le", errors="replace")
    return raw.decode("utf-8", errors="replace")


def exact_signflip_p(deltas):
    n, obs = len(deltas), abs(sum(deltas) / len(deltas))
    hits = sum(1 for signs in itertools.product((1, -1), repeat=n)
               if abs(sum(s * d for s, d in zip(signs, deltas)) / n) >= obs - 1e-12)
    return hits / 2 ** n


def _fs2(fold):
    for key in ("fs_mlp_mean", "fs_mlp_auc", "fs_auc", "fs2_auc"):
        if key in fold:
            return fold[key]
    raise KeyError(f"no FS2 AUROC key in LOGO fold record: {sorted(fold)}")


def tabular(colspec, header, rows):
    out = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    out += [r + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- tables

def t01_logo_primary(L):
    d = load_json("logo_arch6")
    T = "t01"
    deltas = [f["raw_feat_auc"] - f["best_ps_auc"] for f in d["per_fold"]]
    p_exact = exact_signflip_p(deltas)
    L.add(T, "logo.ceiling", d["avg_best_ps"], 3, "logo_arch6", "avg_best_ps")
    L.add(T, "logo.fs1", d["avg_raw_feat_logreg"], 3, "logo_arch6", "avg_raw_feat_logreg")
    L.add(T, "logo.fs1.delta", d["delta_rawfeat_vs_ps"], 3, "logo_arch6", "delta_rawfeat_vs_ps")
    lo, hi = d["rawfeat_fold_bootstrap_ci"]
    L.add(T, "logo.fs1.ci_lo", lo, 3, "logo_arch6", "rawfeat_fold_bootstrap_ci[0]")
    L.add(T, "logo.fs1.ci_hi", hi, 3, "logo_arch6", "rawfeat_fold_bootstrap_ci[1]")
    L.add(T, "logo.fs1.wins", d["rawfeat_wins"], 0, "logo_arch6", "rawfeat_wins")
    L.add(T, "logo.fs1.sign_p", d["rawfeat_sign_test_p"], 3, "logo_arch6", "rawfeat_sign_test_p")
    L.add(T, "logo.fs1.wilcoxon_p", d["rawfeat_wilcoxon_p"], 3, "logo_arch6", "rawfeat_wilcoxon_p")
    L.add(T, "logo.fs1.signflip_p", p_exact, 3, "logo_arch6", "per_fold[*].raw_feat_auc - best_ps_auc",
          kind="derived", note="exact two-sided sign-flip over 2^8 sign patterns of the stored fold deltas")
    L.add(T, "logo.bootstrap_seed", d["bootstrap_seed"], 0, "logo_arch6", "bootstrap_seed")
    L.add(T, "logo.fs2", d["avg_fs_mlp"], 3, "logo_arch6", "avg_fs_mlp")
    L.add(T, "logo.fs2.delta", d["delta_fs_vs_ps"], 3, "logo_arch6", "delta_fs_vs_ps")
    flo, fhi = d["fs_fold_bootstrap_ci"]
    L.add(T, "logo.fs2.ci_lo", flo, 3, "logo_arch6", "fs_fold_bootstrap_ci[0]")
    L.add(T, "logo.fs2.ci_hi", fhi, 3, "logo_arch6", "fs_fold_bootstrap_ci[1]")
    L.add(T, "logo.fs2.wins", d["fs_wins"], 0, "logo_arch6", "fs_wins")
    L.add(T, "logo.fs2.sign_p", d["fs_sign_test_p"], 3, "logo_arch6", "fs_sign_test_p")
    L.add(T, "logo.fs2.wilcoxon_p", d["fs_wilcoxon_p"], 3, "logo_arch6", "fs_wilcoxon_p")
    # fold-delta SD under both conventions (the submission printed 0.082 without naming one)
    L.add(T, "logo.fs1.sd_pop", statistics.pstdev(deltas), 3, "logo_arch6", "pstdev(per_fold deltas)",
          kind="derived", note="population SD (ddof=0)")
    L.add(T, "logo.fs1.sd_sample", statistics.stdev(deltas), 3, "logo_arch6", "stdev(per_fold deltas)",
          kind="derived", note="sample SD (ddof=1)")
    # the July run of the LOGO folds, on another GPU and before the identical-feature exclusions (version 15: as
    # paper v12's App. J.9 describes it, not as a rerun of the reported runs): FS1 and the PS ceiling agree, FS2 moves
    rerun = list(LOG_AVG.finditer(read_text_any(src("log_logo_rerun"))))[-1]
    for col, qid in (("ps", "logo.rerun.ps"), ("raw", "logo.rerun.fs1"), ("fs", "logo.rerun.fs2")):
        L.add(T, qid, float(rerun[col]), 3, "log_logo_rerun", f"log 'Avg(8)' {col} column", kind="log",
              note="an earlier run (July 2026, on another GPU), made before the identical-feature exclusions of "
                   "App. F (App. J.9); log precision 4 dp")
    rows = [
        f"\\fsmethod{{1}} vs per-fold PS envelope & {d['avg_raw_feat_logreg']:.3f} & {sd(d['delta_rawfeat_vs_ps'])} & "
        f"$[{sm(lo)}, {sm(hi)}]$ & {d['rawfeat_wins']}/{d.get('n_folds', len(d['per_fold']))} & "
        f"{d['rawfeat_sign_test_p']:.3f} & "
        f"{d['rawfeat_wilcoxon_p']:.3f} & {p_exact:.3f}",
        f"\\fsmethod{{2}} vs per-fold PS envelope & {d['avg_fs_mlp']:.3f} & {sd(d['delta_fs_vs_ps'])} & "
        f"$[{sm(flo)}, {sm(fhi)}]$ & {d['fs_wins']}/{d.get('n_folds', len(d['per_fold']))} & "
        f"{d['fs_sign_test_p']:.3f} & "
        f"{d['fs_wilcoxon_p']:.3f} & {exact_signflip_p([_fs2(f) - f['best_ps_auc'] for f in d['per_fold']]):.3f}",
    ]
    hdr = ("\\textbf{Comparison} & \\textbf{FS} & \\textbf{$\\Delta$} & \\textbf{Fold 95\\% CI} & "
           "\\textbf{Wins} & \\textbf{Sign $p$} & \\textbf{Wilcoxon $p$} & \\textbf{Sign-flip $p$}")
    return tabular("lrrrrrrr", hdr, rows)


# version 4: the run-path FS1 differs from the canonical value only through the float32 cast of the runner
FS1_RUN_NOTE = {
    "arch6": "run A casts FS1 scores to float32 before AUROC (appB_richer_inputs.py), the effect described in "
             "mitigation/SCORE_PRECISION_NOTE.md; canonical float64 value 0.9657 (fl.std.arch6.fs1)",
    "clip4": "run A casts FS1 scores to float32 before AUROC; equal here to the canonical value 0.9654 "
             "(04_ds_clip4_rawfeat.json, fl.std.clip4.fs1)",
}


def t02_expanded_ladder(L):
    T = "t02"
    rows = []
    for pool in ("arch6", "clip4"):
        d = load_json(f"ladder_{pool}")
        src = f"ladder_{pool}"
        ext = d["extended_rungs_macro"]
        L.add(T, f"{pool}.ps4_submitted", d["published_ladder"]["best_macro"], 3, src, "published_ladder.best_macro")
        rows.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{{pool}}}}}")
        rows.append(f"\\quad submitted \\psmethod{{4}} & -- & {d['published_ladder']['best_macro']:.3f}")
        for rung, label in (("PS_11_disagreement", "pairwise disagreement"),
                            ("PS_12_uncertainty_mi", "calibrated uncertainty, dispersion, MI"),
                            ("PS_13_quadratic", "quadratic interactions"),
                            ("PS_14_rank_ecdf", "rank / ECDF"),
                            ("PS_15_cv_best", "CV-selected")):
            lr, xgb = ext[f"{rung}_logreg"], ext[f"{rung}_xgb"]
            L.add(T, f"{pool}.{rung}.lr", lr, 3, src, f"extended_rungs_macro.{rung}_logreg")
            L.add(T, f"{pool}.{rung}.xgb", xgb, 3, src, f"extended_rungs_macro.{rung}_xgb")
            rows.append(f"\\quad {label} & {lr:.3f} & {xgb:.3f}")
        orc = ext["PS_16_oracle_domain"]
        L.add(T, f"{pool}.oracle_domain", orc, 3, src, "extended_rungs_macro.PS_16_oracle_domain",
              note="non-deployable (labelled-domain) comparator")
        rows.append(f"\\quad labelled-domain comparator (non-deployable) & -- & {orc:.3f}")
        c = d["comparison_vs_deployable_ceiling"]
        L.add(T, f"{pool}.deployable_ceiling", d["deployable_ceiling"]["macro"], 3, src, "deployable_ceiling.macro")
        L.add(T, f"{pool}.deployable_ceiling.name", d["deployable_ceiling"]["name"], 0, src, "deployable_ceiling.name")
        L.add(T, f"{pool}.cv_selected_rung", d["ps_15_selected_rung"], 0, src, "ps_15_selected_rung")
        L.add(T, f"{pool}.fs1", d["fs1_macro"], 3, src, "fs1_macro", note=FS1_RUN_NOTE[pool])
        L.add(T, f"{pool}.ext.delta", c["delta"], 3, src, "comparison_vs_deployable_ceiling.delta")
        L.add(T, f"{pool}.ext.ci_lo", c["generator_bootstrap"]["ci_95"][0], 3, src,
              "comparison_vs_deployable_ceiling.generator_bootstrap.ci_95[0]")
        L.add(T, f"{pool}.ext.ci_hi", c["generator_bootstrap"]["ci_95"][1], 3, src,
              "comparison_vs_deployable_ceiling.generator_bootstrap.ci_95[1]")
        L.add(T, f"{pool}.ext.p", c["sign_flip_permutation"]["p_value"], 3, src,
              "comparison_vs_deployable_ceiling.sign_flip_permutation.p_value")
        L.add(T, f"{pool}.ext.wins", c["wins"], 0, src, "comparison_vs_deployable_ceiling.wins")
        L.add(T, f"{pool}.per_gen_max", d["ps_17_per_generator_oracle_selection"]["macro"], 3, src,
              "ps_17_per_generator_oracle_selection.macro", note="non-deployable retrospective maximum")
    lc = load_json("logo_combined")
    L.add(T, "logo_refit.ceiling_before", lc["published_ceiling"], 3, "logo_combined", "published_ceiling")
    L.add(T, "logo_refit.ceiling_after", lc["combined_ceiling"], 3, "logo_combined", "combined_ceiling")
    L.add(T, "logo_refit.delta_before", lc["delta_published_ladder"], 3, "logo_combined", "delta_published_ladder")
    L.add(T, "logo_refit.delta_after", lc["delta_ladder_plus_rungs"], 3, "logo_combined", "delta_ladder_plus_rungs")
    L.add(T, "logo_refit.extended_wins", lc["folds_where_extended_wins"], 0, "logo_combined", "folds_where_extended_wins")
    lr_ = load_json("logo_rungs")
    L.add(T, "logo_refit.extended_only_capped", lr_["extended_only_ceiling_capped"], 4, "logo_rungs",
          "extended_only_ceiling_capped", note="the extended-rungs-only LOGO ceiling, never 'the LOGO ceiling'")
    L.add(T, "logo_refit.extended_only_widened", lr_["extended_only_ceiling_widened"], 4, "logo_rungs",
          "extended_only_ceiling_widened")
    L.add(T, "logo_refit.grid_capped_n", len(lr_["grids"]["capped"]), 0, "logo_rungs", "len(grids.capped)",
          note="reduced per-fold grid; disclose against the shipped grid")
    L.add(T, "logo_refit.grid_widened_n", len(lr_["grids"]["widened"]), 0, "logo_rungs", "len(grids.widened)")
    hdr = "\\textbf{Rung} & \\textbf{LR} & \\textbf{XGBoost}"
    return tabular("lrr", hdr, rows)


def t03_width(L):
    d = load_json("width")
    T = "t03"
    rows = []
    for key in ("d1", "d2", "d4", "d8"):
        e = d["pca_sweep"][key]
        c = e["comparison_vs_ps4"]
        L.add(T, f"width.pca_{key}.macro", e["fs_macro"], 3, "width", f"pca_sweep.{key}.fs_macro")
        L.add(T, f"width.pca_{key}.delta", c["delta"], 3, "width", f"pca_sweep.{key}.comparison_vs_ps4.delta")
        L.add(T, f"width.pca_{key}.p", c["sign_flip_permutation"]["p_value"], 3, "width",
              f"pca_sweep.{key}.comparison_vs_ps4.sign_flip_permutation.p_value")
        tag = " (width-matched)" if e["width_matched"] else ""
        rows.append(f"PCA, {key[1:]} per expert{tag} & {e['coords_total']} & {e['fs_macro']:.3f} & "
                    f"{sd(c['delta'])} & {c['sign_flip_permutation']['p_value']:.3f}")
    pr = d["probe_mirror"]["d2"]
    pc = pr["comparison_vs_ps4"]
    L.add(T, "width.probe_d2.macro", pr["fs_macro"], 3, "width", "probe_mirror.d2.fs_macro",
          note="TRAIN-label-supervised directions; standard protocol, overlap-affected")
    L.add(T, "width.probe_d2.delta", pc["delta"], 3, "width", "probe_mirror.d2.comparison_vs_ps4.delta")
    L.add(T, "width.probe_d2.p", pc["sign_flip_permutation"]["p_value"], 3, "width",
          "probe_mirror.d2.comparison_vs_ps4.sign_flip_permutation.p_value")
    L.add(T, "width.ps4", d["comparator"]["macro"], 3, "width", "comparator.macro")
    for pool in ("arch6", "clip4"):
        p = load_json(f"pca_{pool}")
        L.add(T, f"pr_gap.{pool}", p["pr_gap"], 1, f"pca_{pool}", "pr_gap",
              note="feature PR / prediction PR, unrounded (the submission printed 19x from rounded PRs)")
        L.add(T, f"pr.{pool}.features", p["features"]["pr"], 1, f"pca_{pool}", "features.pr")
        L.add(T, f"pr.{pool}.logits", p["logits"]["pr"], 2, f"pca_{pool}", "logits.pr",
              note="2 dp so that the printed ratio matches pr_gap")
    rows.append(f"supervised, 2 per expert (width-matched) & {pr['coords_total']} & {pr['fs_macro']:.3f} & "
                f"{sd(pc['delta'])} & {pc['sign_flip_permutation']['p_value']:.3f}")
    hdr = ("\\textbf{Feature-side control} & \\textbf{Coords} & \\textbf{AUROC} & "
           "\\textbf{$\\Delta$ vs \\psmethod{4}} & \\textbf{Sign-flip $p$}")
    return tabular("lrrrr", hdr, rows)


def t04_k3_heads(L):
    d = load_json("k3_heads")
    T = "t04"
    rows = []
    for m in ("PS1-K3", "PS2-K3", "PS3-K3", "PS4-K3", "FS1-K3", "FS2-K3", "FSB-K3-18", "PS+-K3"):
        v = d["summary"][m]["mean_fold_macro_ovr"]
        L.add(T, f"k3.{m}", v, 3, "k3_heads", f"summary.{m}.mean_fold_macro_ovr")
        rows.append(f"{m} & {v:.3f}")
    for key, qid in (("FS1_minus_select_chosen_core_PS", "k3.fs1_minus_select"),
                     ("FS2_minus_select_chosen_core_PS", "k3.fs2_minus_select")):
        c = d["contrasts"][key]
        L.add(T, qid, c["mean"], 3, "k3_heads", f"contrasts.{key}.mean")
        L.add(T, qid + ".pos_folds", c["positive_folds"], 0, "k3_heads", f"contrasts.{key}.positive_folds")
    chosen = sorted({f["core_ps_select_chosen"]["method"] for f in d["folds"]})
    L.add(T, "k3.select_chosen", ",".join(chosen), 0, "k3_heads", "folds[*].core_ps_select_chosen.method")
    capped = set()
    for f in d["folds"]:
        for expert, q in f["head_quality"].items():
            if not q.get("all_fits_converged_before_cap", True):
                capped.add(expert)
    L.add(T, "k3.heads_at_cap", len(capped), 0, "k3_heads",
          "folds[*].head_quality[*].all_fits_converged_before_cap", note=";".join(sorted(capped)))
    return tabular("lr", "\\textbf{Method} & \\textbf{Macro OvR AUROC}", rows)


def t05_k3_views(L):
    T = "t05"
    rows = []
    for key, label in (("k3_view_a", "View A (df40 pool)"), ("k3_view_b", "View B (arch6 pool)")):
        d = load_json(key)
        pr = d["primary"]
        L.add(T, f"{key}.delta", pr["mean"], 3, key, "primary.mean")
        L.add(T, f"{key}.pos", pr["folds_positive"], 0, key, "primary.folds_positive")
        L.add(T, f"{key}.n", pr["n_folds"], 0, key, "primary.n_folds")
        floor = 2 / 2 ** pr["n_folds"]
        L.add(T, f"{key}.p_floor", floor, 4, key, "2 / 2**primary.n_folds", kind="derived",
              note="smallest two-sided exact sign-flip p attainable with n_folds")
        rows.append(f"{label} & {sd(pr['mean'])} & {pr['folds_positive']}/{pr['n_folds']} & {floor:.4f}")
    hdr = "\\textbf{Construction} & \\textbf{$\\Delta$} & \\textbf{Folds positive} & \\textbf{Exact $p$ floor}"
    return tabular("lrrr", hdr, rows)


def t06_cost(L):
    d = load_json("cost")
    T = "t06"
    w, s, c = d["widths_and_storage"]["pool_totals"], d["widths_and_storage"]["storage"], d["combiner_costs"]
    L.add(T, "cost.feature_floats", w["feature_floats"], 0, "cost", "widths_and_storage.pool_totals.feature_floats")
    L.add(T, "cost.feature_bytes", w["feature_bytes_per_sample_stored"], 0, "cost",
          "widths_and_storage.pool_totals.feature_bytes_per_sample_stored")
    L.add(T, "cost.pred_floats", w["prediction_floats"], 0, "cost", "widths_and_storage.pool_totals.prediction_floats")
    L.add(T, "cost.pred_bytes", w["prediction_bytes_per_sample_stored"], 0, "cost",
          "widths_and_storage.pool_totals.prediction_bytes_per_sample_stored")
    L.add(T, "cost.pool_gib", s["pool_features_gib"], 2, "cost", "widths_and_storage.storage.pool_features_gib")
    L.add(T, "cost.pool_mib", s["pool_logits_mib"], 1, "cost", "widths_and_storage.storage.pool_logits_mib")
    # display precision matches the table row below (2 dp GiB, 1 dp MiB)
    L.add(T, "cost.release_gib", s["release_features_gib"], 2, "cost", "widths_and_storage.storage.release_features_gib")
    L.add(T, "cost.release_mib", s["release_logits_mib"], 1, "cost", "widths_and_storage.storage.release_logits_mib")
    L.add(T, "cost.fs1_fit_s", c["FS_1_frozen_total"]["total_fit_plus_predict_sec"], 0, "cost",
          "combiner_costs.FS_1_frozen_total.total_fit_plus_predict_sec")
    L.add(T, "cost.ps4_cv_s", c["PS_4_xgboost_cv"]["total_fit_plus_predict_sec"], 0, "cost",
          "combiner_costs.PS_4_xgboost_cv.total_fit_plus_predict_sec")
    L.add(T, "cost.ps2_s", c["PS_2_logreg_logits"]["total_fit_plus_predict_sec"], 2, "cost",
          "combiner_costs.PS_2_logreg_logits.total_fit_plus_predict_sec")
    L.add(T, "cost.fs1_us", c["FS_1_config_split_timing"]["predict_us_per_sample"], 0, "cost",
          "combiner_costs.FS_1_config_split_timing.predict_us_per_sample")
    L.add(T, "cost.ps_us", c["PS_predict_only"]["predict_us_per_sample"], 1, "cost",
          "combiner_costs.PS_predict_only.predict_us_per_sample")
    n_fit = load_json("strict")["arm_full"]["n_fit"]
    est_gib = n_fit * w["feature_floats"] * 8 / 2 ** 30
    L.add(T, "cost.fs1_float64_copy_gib", est_gib, 2, "strict", "arm_full.n_fit x feature_floats x 8 B / 2^30",
          kind="derived", note="ESTIMATED, not measured: float64 copy of the TRAIN features inside sklearn lbfgs")
    rows = [
        f"values per sample (arch6) & {w['feature_floats']:,} float16 ({w['feature_bytes_per_sample_stored']:,} B) & "
        f"{w['prediction_floats']} float32 ({w['prediction_bytes_per_sample_stored']} B) & measured",
        f"cached arch6 pool & {s['pool_features_gib']:.2f} GiB & {s['pool_logits_mib']:.1f} MiB & measured",
        f"21-expert release & {s['release_features_gib']:.2f} GiB & {s['release_logits_mib']:.1f} MiB & measured",
        f"combiner fit and predict & {c['FS_1_frozen_total']['total_fit_plus_predict_sec']:.0f} s (\\fsmethod{{1}}) & "
        f"{c['PS_4_xgboost_cv']['total_fit_plus_predict_sec']:.0f} s (\\psmethod{{4}} CV); "
        f"{c['PS_2_logreg_logits']['total_fit_plus_predict_sec']:.2f} s (\\psmethod{{2}}) & measured",
        # $\mu$ from the Type 1 math font: \textmu falls back to a Type 3 font in this setup
        # the PS inference timing is for a PS_2-type logistic head (run_d: PS_predict_only.head)
        f"combiner inference & {c['FS_1_config_split_timing']['predict_us_per_sample']:.0f} $\\mu$s/sample "
        f"(\\fsmethod{{1}}) & {c['PS_predict_only']['predict_us_per_sample']:.1f} $\\mu$s/sample "
        f"(\\psmethod{{2}}) & measured",
        f"float64 training copy (\\fsmethod{{1}}) & {est_gib:.2f} GiB & -- & estimated, not measured",
        "expert forward pass & same with local checkpoints & same & shared cost",
        "feature extraction throughput & -- & -- & not measured",
        "peak working set & -- & -- & not measured",
    ]
    hdr = "\\textbf{Item} & \\textbf{Feature interface} & \\textbf{Prediction interface} & \\textbf{Status}"
    return tabular("llll", hdr, rows)


def t07_reuse(L):
    dg, ct = load_json("diagnose"), load_json("content")
    T = "t07"
    rows, fake_gens = [], 0
    for g in GENERATORS:
        ts, cr = dg["test_side"][g], ct["per_generator"][g]
        L.add(T, f"reuse.{g}.n", ts["n"], 0, "diagnose", f"test_side.{g}.n")
        L.add(T, f"reuse.{g}.path", ts["in_combiner_train"], 0, "diagnose", f"test_side.{g}.in_combiner_train")
        L.add(T, f"reuse.{g}.content", cr["content_reuse"], 0, "content", f"per_generator.{g}.content_reuse")
        L.add(T, f"reuse.{g}.fake_path", ts["fake_overlapping"], 0, "diagnose", f"test_side.{g}.fake_overlapping")
        fake_gens += int(ts["fake_overlapping"] > 0)
        rows.append(f"{GEN_LABEL[g]} & {ts['n']:,} & {ts['in_combiner_train']:,} & {cr['content_reuse']:,} & "
                    f"{ts['fake_overlapping']:,}")
    n = ct["totals"]["n"]
    L.add(T, "reuse.total_n", n, 0, "content", "totals.n")
    L.add(T, "reuse.total_path", dg["total_test_in_train"], 0, "diagnose", "total_test_in_train")
    L.add(T, "reuse.total_content", ct["totals"]["content"], 0, "content", "totals.content")
    L.add(T, "reuse.path_pct", 100 * dg["total_test_in_train"] / n, 1, "diagnose", "total_test_in_train / totals.n")
    L.add(T, "reuse.content_pct", 100 * ct["totals"]["content"] / n, 1, "content", "totals.content / totals.n")
    L.add(T, "reuse.fake_generators", fake_gens, 0, "diagnose", "count(test_side[*].fake_overlapping > 0)")
    bi = load_json("bit_identity")
    L.add(T, "reuse.bit_identical", bi["features_bit_identical_total"], 0, "bit_identity",
          "features_bit_identical_total")
    L.add(T, "reuse.bit_label_mismatch", bi["label_mismatches_total"], 0, "bit_identity", "label_mismatches_total")
    rows.append(f"\\textbf{{Total}} & {n:,} & {dg['total_test_in_train']:,} "
                f"({100 * dg['total_test_in_train'] / n:.1f}\\%) & {ct['totals']['content']:,} "
                f"({100 * ct['totals']['content'] / n:.1f}\\%) & --")
    hdr = ("\\textbf{Generator} & \\textbf{TEST rows} & \\textbf{Path reuse} & "
           "\\textbf{Content reuse} & \\textbf{Fake rows reused (path)}")
    return tabular("lrrrr", hdr, rows)


def t08_strict(L):
    b, v, lg = load_json("strict"), load_json("strict_verify"), load_json("logo_arch6")
    T = "t08"
    full, strict, matched = b["arm_full"], b["arm_strict"], b["arm_matched"]
    fc, sc = full["comparison"], strict["comparison"]
    L.add(T, "strict.full.delta", fc["delta"], 3, "strict", "arm_full.comparison.delta")
    L.add(T, "strict.full.ci_lo", fc["generator_bootstrap"]["ci_95"][0], 3, "strict",
          "arm_full.comparison.generator_bootstrap.ci_95[0]")
    L.add(T, "strict.full.ci_hi", fc["generator_bootstrap"]["ci_95"][1], 3, "strict",
          "arm_full.comparison.generator_bootstrap.ci_95[1]")
    L.add(T, "strict.full.wins", fc["wins"], 0, "strict", "arm_full.comparison.wins")
    L.add(T, "strict.full.ps4", full["best_ps_macro"], 3, "strict", "arm_full.best_ps_macro")
    L.add(T, "strict.strict.delta", sc["delta"], 3, "strict", "arm_strict.comparison.delta")
    L.add(T, "strict.strict.ci_lo", sc["generator_bootstrap"]["ci_95"][0], 3, "strict",
          "arm_strict.comparison.generator_bootstrap.ci_95[0]")
    L.add(T, "strict.strict.ci_hi", sc["generator_bootstrap"]["ci_95"][1], 3, "strict",
          "arm_strict.comparison.generator_bootstrap.ci_95[1]")
    L.add(T, "strict.strict.fs1", strict["fs1_macro"], 3, "strict", "arm_strict.fs1_macro")
    L.add(T, "strict.strict.ps", strict["best_ps_macro"], 3, "strict", "arm_strict.best_ps_macro")
    L.add(T, "strict.matched.mean", b["summary"]["delta_matched_mean"], 3, "strict", "summary.delta_matched_mean")
    L.add(T, "strict.matched.seeds", len(matched), 0, "strict", "len(arm_matched)")
    L.add(T, "strict.minus_matched", b["summary"]["leakage_effect"], 3, "strict", "summary.leakage_effect",
          note="report as 'strict minus matched', never as a leakage effect")
    L.add(T, "strict.real_rows_full", full["n_fit_real"], 0, "strict", "arm_full.n_fit_real")
    L.add(T, "strict.real_rows_strict", strict["n_fit_real"], 0, "strict", "arm_strict.n_fit_real")
    fam = load_json("reuse_check")["route_b"]["fit_real_source_families"]
    L.add(T, "strict.families_full", fam["families"], 0, "reuse_check", "route_b.fit_real_source_families.families",
          kind="derived", note="leading path component of FIT real rows")
    L.add(T, "strict.families_strict", fam["families_after_strict_filter"], 0, "reuse_check",
          "route_b.fit_real_source_families.families_after_strict_filter", kind="derived",
          note="the remaining 586 real rows are all FaceForensics++ original sequences")
    L.add(T, "strict.real_rows_strict_recount", fam["real_rows_after_strict_filter"], 0, "reuse_check",
          "route_b.fit_real_source_families.real_rows_after_strict_filter", kind="derived")
    a = v["arms"]
    L.add(T, "stargan.full", a["full (published)"]["fs1_stargan"], 3, "strict_verify",
          "arms['full (published)'].fs1_stargan")
    m = [a[k]["fs1_stargan"] for k in a if k.startswith("matched")]
    L.add(T, "stargan.matched_min", min(m), 3, "strict_verify", "min(arms[matched_*].fs1_stargan)")
    L.add(T, "stargan.matched_max", max(m), 3, "strict_verify", "max(arms[matched_*].fs1_stargan)")
    L.add(T, "stargan.strict", a["strict"]["fs1_stargan"], 3, "strict_verify", "arms.strict.fs1_stargan")
    L.add(T, "stargan.full_margin", fc["per_generator_delta"]["stargan"], 3, "strict",
          "arm_full.comparison.per_generator_delta.stargan")
    # version 4: rows 1-2 print the paper's values (already in the ledger as fl.std.arch6.fs1.* and
    # cd.logo.fs1.*); version 3 printed fc (strict runner, against PS_4) and lg (submitted LOGO run), whose ids
    # above keep their values for the posted-number check
    std = load_json("full_ladder")["standard"]["arch6"]
    cdl = load_json("mitigation")["logo_arch6"]["stats"]["fs1_full"]
    rows = [
        f"standard split (overlap-affected) & {sd(std['fs1_delta'])} & "
        f"$[{sm(std['fs1_ci95'][0])}, {sm(std['fs1_ci95'][1])}]$ & secondary",
        f"LOGO (this appendix) & {sd(cdl['mean_delta'])} & "
        f"$[{sm(cdl['ci95'][0])}, {sm(cdl['ci95'][1])}]$ & primary; {cdl['wins']}/{cdl['n']} folds",
        f"strict path-disjoint refit & {sd(sc['delta'])} & "
        f"$[{sm(sc['generator_bootstrap']['ci_95'][0])}, {sm(sc['generator_bootstrap']['ci_95'][1])}]$ & "
        f"\\fsmethod{{1}} {strict['fs1_macro']:.3f}, PS {strict['best_ps_macro']:.3f}; coverage loss",
        f"matched random drop ({len(matched)} seeds) & {sd(b['summary']['delta_matched_mean'])} & -- & same counts dropped",
    ]
    hdr = "\\textbf{Arm} & \\textbf{\\fsmethod{1} $-$ best PS} & \\textbf{95\\% CI} & \\textbf{Note}"
    return tabular("lrll", hdr, rows)


LOG_FOLD = re.compile(r"\|\s*(?P<gen>[A-Za-z0-9]+)\s+(?P<ps>\d\.\d{4})\s+(?P<name>PS_\w+)\s+"
                      r"(?P<raw>\d\.\d{4})\s+(?P<fs>\d\.\d{4})")
LOG_AVG = re.compile(r"Avg\(8\)\s+(?P<ps>\d\.\d{4})\s+(?P<raw>\d\.\d{4})\s+(?P<fs>\d\.\d{4})")
LOG_RAW = re.compile(r"Raw\s+vs best PS:\s*(?P<d>[+-]\d\.\d{4})\s+wins=(?P<w>\d)/(?P<n>\d)\s+sign p=(?P<sp>\d\.\d{4})\s+"
                     r"wilcoxon p=(?P<wp>\d\.\d{4})\s+boot 95% CI \[(?P<lo>[+-]\d\.\d{4}), (?P<hi>[+-]\d\.\d{4})\]")


def t09_boundary(L):
    T = "t09"
    sizes = {g: load_json("diagnose")["test_side"][g]["n"] for g in GENERATORS}
    large = [g for g in GENERATORS if sizes[g] >= 500]
    rows = []
    lg = load_json("logo_arch6")
    arch_d = {f["held_out"]: f["raw_feat_auc"] - f["best_ps_auc"] for f in lg["per_fold"]}
    arch_large = sum(arch_d[g] for g in large) / len(large)
    L.add(T, "pool.arch6.large_mean", arch_large, 3, "logo_arch6", "mean over folds with n_test >= 500",
          kind="derived", note="exploratory")
    rows.append(f"arch6 & {lg['avg_best_ps']:.3f} & {sd(lg['delta_rawfeat_vs_ps'])} & "
                f"{lg['rawfeat_wins']}/{lg.get('n_folds', len(lg['per_fold']))} & "
                f"{sd(arch_large)} & stored JSON")
    for pool in ("clip4", "strong4", "weak6"):
        key = f"log_{pool}"
        text = read_text_any(src(key))
        folds = {m["gen"]: m for m in LOG_FOLD.finditer(text) if m["gen"] in GENERATORS}
        avg, raw = list(LOG_AVG.finditer(text))[-1], list(LOG_RAW.finditer(text))[-1]
        if len(folds) != 8:
            raise SystemExit(f"{pool}: parsed {len(folds)} folds from the log, expected 8")
        ceiling, delta = float(avg["ps"]), float(raw["d"])
        large_mean = sum(float(folds[g]["raw"]) - float(folds[g]["ps"]) for g in large) / len(large)
        L.add(T, f"pool.{pool}.ceiling", ceiling, 3, key, "log 'Avg(8)' best-PS column", kind="log",
              note="reference precision 4 dp")
        L.add(T, f"pool.{pool}.delta", delta, 3, key, "log 'Raw vs best PS' delta", kind="log",
              note="log precision 4 dp")
        L.add(T, f"pool.{pool}.wins", int(raw["w"]), 0, key, "log 'Raw vs best PS' wins", kind="log")
        L.add(T, f"pool.{pool}.fs1", float(avg["raw"]), 3, key, "log 'Avg(8)' raw-feature column", kind="log",
              note="log precision 4 dp")
        L.add(T, f"pool.{pool}.large_mean", large_mean, 3, key, "mean over folds with n_test >= 500",
              kind="derived", note="exploratory; from 4 dp fold values")
        rows.append(f"{pool} & {ceiling:.3f} & {sd(delta)} & {raw['w']}/{raw['n']} & {sd(large_mean)} & log only")
    hdr = ("\\textbf{Pool} & \\textbf{LOGO PS envelope} & \\textbf{\\fsmethod{1} $-$ PS} & \\textbf{Wins} & "
           "\\textbf{Folds $n\\geq500$ (exploratory)} & \\textbf{Provenance}")
    return tabular("lrrrrl", hdr, rows)


def t10_appendix_h(L):
    d = load_json("appendix_h")
    T = "t10"
    rows = []
    arch6 = ["fs-clip-base", "fs-clip-large", "fs-xception", "fs-srm", "fs-recce", "fs-rfm"]
    for e in sorted(d["per_expert"]):
        r = d["per_expert"][e]
        L.add(T, f"exp.{e}.published", r["published"], 3, "appendix_h", f"per_expert.{e}.published")
        L.add(T, f"exp.{e}.corrected", r["corrected"], 3, "appendix_h", f"per_expert.{e}.corrected")
        rows.append(f"{e} & {r['published']:.3f} & {r['corrected']:.3f} & {sd(r['delta'])}")
    best_pub = max(d["per_expert"][e]["published"] for e in arch6)
    best_cor = max(d["per_expert"][e]["corrected"] for e in arch6)
    L.add(T, "exp.arch6_best.published", best_pub, 3, "appendix_h", "max over arch6 experts, published")
    L.add(T, "exp.arch6_best.corrected", best_cor, 3, "appendix_h", "max over arch6 experts, corrected")
    L.add(T, "exp.overall_best.published", d["published_range"][1], 3, "appendix_h", "published_range[1]")
    L.add(T, "exp.overall_best.corrected", d["corrected_range"][1], 3, "appendix_h", "corrected_range[1]")
    L.add(T, "exp.per_cell_values", "not stored", 0, "appendix_h", "-", kind="prose",
          note="per-generator corrected cells are available in the Appendix H output (step C12)")
    hdr = "\\textbf{Expert} & \\textbf{Published} & \\textbf{Corrected} & \\textbf{Change}"
    return tabular("lrrr", hdr, rows)


def t11_appendix_h_cells(L):
    d = load_json("appendix_h_cells")
    T = "t11"
    if not d["self_checks"]["passed"]:
        raise SystemExit("Appendix H regeneration did not pass its self-checks")
    gens = d["generators"]
    rows = []
    for e in EXPERT_ORDER:
        r = d["per_expert"][e]
        for g in gens:
            L.add(T, f"h.{e}.{g}", r["per_generator"][g], 3, "appendix_h_cells", f"per_expert.{e}.per_generator.{g}",
                  kind="derived", note="margin convention; self-checked against the published table and July record")
        L.add(T, f"h.{e}.avg", r["avg_auc_8"], 3, "appendix_h_cells", f"per_expert.{e}.avg_auc_8", kind="derived")
        rows.append(f"{e} & " + " & ".join(f"{r['per_generator'][g]:.3f}" for g in gens) + f" & {r['avg_auc_8']:.3f}")
    s = d["summary"]
    L.add(T, "h.best_arch6", s["best_arch6_avg"], 3, "appendix_h_cells", "summary.best_arch6_avg", kind="derived")
    L.add(T, "h.best_overall", s["best_overall_avg"], 3, "appendix_h_cells", "summary.best_overall_avg", kind="derived")
    avgs = [d["per_expert"][e]["avg_auc_8"] for e in EXPERT_ORDER]
    L.add(T, "h.range_low", min(avgs), 2, "appendix_h_cells", "min over experts of avg_auc_8", kind="derived",
          note="Sec. 4.1 states the range of individual expert averages")
    L.add(T, "h.range_high", max(avgs), 2, "appendix_h_cells", "max over experts of avg_auc_8", kind="derived")
    # version 8: the canonical FS1 is read from the cost ledger's full-precision refit (version 7 typed in its value,
    # 0.9657375624159219)
    fs1 = load_json("cost")["combiner_costs"]["FS_1_frozen_total"]["macro_auroc"]
    L.add(T, "h.fs1_minus_best_arch6", fs1 - s["best_arch6_avg"], 3, "appendix_h_cells",
          "cost: combiner_costs.FS_1_frozen_total.macro_auroc minus summary.best_arch6_avg", kind="derived",
          note="replaces the submitted +0.185 (0.966 - 0.780) in App. H.1")
    L.add(T, "h.cells_moving", s["cells_moving_at_3dp"], 0, "appendix_h_cells", "summary.cells_moving_at_3dp",
          kind="derived", note="cells with |margin - raw| >= 0.0005 (regenerate_appendix_h.py)")
    L.add(T, "h.cells_moving_threshold", 0.0005, 4, "appendix_h_cells", "threshold in regenerate_appendix_h.py",
          kind="derived")
    # displayed (3 dp) changes, published raw-logit cells versus corrected margin cells
    pub = load_json("appendix_h_published")["per_expert"]
    shown = sum(1 for e in EXPERT_ORDER for g in gens
                if f"{pub[e]['per_generator'][g]:.3f}" != f"{d['per_expert'][e]['per_generator'][g]:.3f}")
    shown_avg = sum(1 for e in EXPERT_ORDER if f"{pub[e]['avg_auc_8']:.3f}" != f"{d['per_expert'][e]['avg_auc_8']:.3f}")
    L.add(T, "h.cells_display_changed", shown, 0, "appendix_h_published", "3-dp display of published vs corrected cells",
          kind="derived", note="101 of 168 displayed cells change; the paper also gives the 86 with |delta| >= 0.0005")
    L.add(T, "h.avgs_display_changed", shown_avg, 0, "appendix_h_published", "3-dp display of published vs corrected averages",
          kind="derived")
    # Table A16 composition column: individual-expert range for strong4 (corrected averages)
    strong4 = ["df40-clip-large", "efs-clip-large", "fs-clip-large", "efs-clip-base"]
    s4 = [d["per_expert"][e]["avg_auc_8"] for e in strong4]
    L.add(T, "h.strong4_low", min(s4), 2, "appendix_h_cells", "min avg_auc_8 over strong4 experts", kind="derived")
    L.add(T, "h.strong4_high", max(s4), 2, "appendix_h_cells", "max avg_auc_8 over strong4 experts", kind="derived")
    hdr = ("\\textbf{Expert} & " + " & ".join(f"\\multicolumn{{1}}{{l}}{{\\textbf{{{c}}}}}" for c in
           ("WIR", "CDiff", "MJ6", "HeyG", "SG2", "DFL", "SG", "SCLIP")) +
           " & \\multicolumn{1}{l}{\\textbf{Avg.\\ AUROC}}")
    return tabular("lrrrrrrrr r", hdr, rows)


def t12_release_reuse(L):
    d = load_json("reuse_check")["route_a"]
    T = "t12"
    rows = []
    for g in GENERATORS:
        r = d["per_generator"][g]
        L.add(T, f"releaseA.{g}.test_paths", r["test_paths"], 0, "reuse_check", f"route_a.per_generator.{g}.test_paths",
              kind="derived")
        L.add(T, f"releaseA.{g}.in_DF40_all", r["in_DF40_all"], 0, "reuse_check",
              f"route_a.per_generator.{g}.in_DF40_all", kind="derived")
        pct = 100 * r["in_DF40_all"] / r["test_paths"]
        rows.append(f"{GEN_LABEL[g]} & {r['test_paths']:,} & {r['in_DF40_all']:,} & {pct:.1f}\\%")
    L.add(T, "releaseA.total_test_paths", d["total_test_paths"], 0, "reuse_check", "route_a.total_test_paths",
          kind="derived")
    L.add(T, "releaseA.in_DF40_all", d["in_DF40_all"], 0, "reuse_check", "route_a.in_DF40_all", kind="derived")
    L.add(T, "releaseA.pct", d["pct"], 1, "reuse_check", "route_a.pct", kind="derived")
    rows.append(f"\\textbf{{Total}} & {d['total_test_paths']:,} & {d['in_DF40_all']:,} & {d['pct']:.1f}\\%")
    hdr = ("\\textbf{Held-out generator} & \\textbf{Test paths in its JSON} & "
           "\\textbf{Also in \\texttt{DF40\\_all.json}} & \\textbf{Share}")
    return tabular("lrrr", hdr, rows)


def t13_integrity(L):
    rb = load_json("reuse_check")["route_b"]
    T = "t13"
    tw = rb["select_fit_content_twins"]
    dt = rb["distinct_test_images"]
    fr = rb["first_row_signature_test_splits"]["per_generator"]
    cg = rb["cross_generator_matches"]
    L.add(T, "integ.select_fit_twins", tw["df40-clip-large"], 0, "reuse_check",
          "route_b.select_fit_content_twins.df40-clip-large", kind="derived")
    L.add(T, "integ.select_fit_twins_arch6", tw["arch6_features"], 0, "reuse_check",
          "route_b.select_fit_content_twins.arch6_features", kind="derived",
          note="explains the 793 of the frozen-codebase audit (different hash source)")
    L.add(T, "integ.distinct_test", dt["total"], 0, "reuse_check", "route_b.distinct_test_images.total", kind="derived")
    for g in GENERATORS:
        L.add(T, f"integ.distinct.{g}", dt["per_generator"][g], 0, "reuse_check",
              f"route_b.distinct_test_images.per_generator.{g}", kind="derived")
    L.add(T, "integ.first_row.MidJourney", fr.get("MidJourney", 0), 0, "reuse_check",
          "route_b.first_row_signature_test_splits.per_generator.MidJourney", kind="derived",
          note="matches the recorded 5 zero-byte MidJourney inputs")
    L.add(T, "integ.shared_images", cg["shared_images"], 0, "reuse_check", "route_b.cross_generator_matches.shared_images",
          kind="derived", note="posted as '7 output-fingerprint matches'; 14 rows across the two folds, all real")
    L.add(T, "integ.shared_rows", cg["rows_total"], 0, "reuse_check", "route_b.cross_generator_matches.rows_total",
          kind="derived")
    raw, lab, pr = rb["raw_files"], rb["label_path_disagreements"], rb["path_reuse"]
    L.add(T, "integ.zero_byte_total", raw["zero_byte_total"], 0, "reuse_check", "route_b.raw_files.zero_byte_total",
          kind="derived", note="cache rows whose source-file size is zero (stat only)")
    for s, n in raw["zero_byte"].items():
        L.add(T, f"integ.zero_byte.{s}", n, 0, "reuse_check", f"route_b.raw_files.zero_byte.{s}", kind="derived")
    L.add(T, "integ.missing_sources", raw["missing_total"], 0, "reuse_check", "route_b.raw_files.missing_total",
          kind="derived")
    L.add(T, "integ.label_path_total", lab["total"], 0, "reuse_check", "route_b.label_path_disagreements.total",
          kind="derived")
    for s, n in lab["per_split"].items():
        L.add(T, f"integ.label_path.{s}", n, 0, "reuse_check", f"route_b.label_path_disagreements.per_split.{s}",
              kind="derived")
    L.add(T, "integ.zero_byte_label_mismatch", raw["zero_byte_rows_whose_label_disagrees_with_path"], 0,
          "reuse_check", "route_b.raw_files.zero_byte_rows_whose_label_disagrees_with_path", kind="derived",
          note="every label disagreement is a zero-byte input replaced by its split's first item")
    L.add(T, "integ.path_reuse_total", pr["total"], 0, "reuse_check", "route_b.path_reuse.total", kind="derived",
          note="independent recount of run_b_diagnose_arch6.json total_test_in_train")
    pair = ", ".join(f"{GEN_LABEL[a]} and {GEN_LABEL[b]}" for a, b in (k.split("|") for k in cg["pairs"]))
    # the paper calls the cache's FIT split TRAIN
    names = {"starganv2": "StarGAN-v2", "FIT": "TRAIN"}
    zb = "; ".join(f"{names.get(s, s)} {n}" for s, n in raw["zero_byte"].items())
    lb = "; ".join(f"{names.get(s, s)} {n}" for s, n in lab["per_split"].items())
    rows = [
        f"SELECT rows with feature content identical to a TRAIN row & {tw['df40-clip-large']:,} & "
        f"df40-clip-large feature rows ({tw['arch6_features']:,} with arch6 features)",
        f"distinct test images among {rb['content_reuse']['totals']['n']:,} test rows & "
        f"{dt['total']:,} & arch6 feature rows, per generator, summed",
        f"\\quad of which StarGAN-v2 & {dt['per_generator']['starganv2']:,} & of 4,000 rows",
        f"\\quad of which MidJourney & {dt['per_generator']['MidJourney']:,} & of 1,882 rows",
        f"images shared by two held-out generators & {cg['shared_images']} & {pair}; {cg['rows_total']} test rows, "
        f"all real",
        f"cache rows with zero-byte source files & {raw['zero_byte_total']} & {zb}; replaced by the split's first item at "
        f"extraction",
        f"\\quad of which the cached label disagrees with the path & {raw['zero_byte_rows_whose_label_disagrees_with_path']} "
        f"& {lb}",
        f"missing source images & {raw['missing_total']} & all cached rows resolve to a DF40 file",
    ]
    hdr = "\\textbf{Quantity} & \\textbf{Count} & \\textbf{Definition}"
    return tabular("lrl", hdr, rows)


REGIME_LABEL = {"fs": "FS (\\texttt{train\\_on\\_fs})", "fr": "FR (\\texttt{train\\_on\\_fr})",
                "efs": "EFS (\\texttt{train\\_on\\_efs})", "df40": "DF40 aggregate (\\texttt{train\\_on\\_df40})"}
POOL_ORDER = ["arch6", "clip4", "strong4", "weak6", "low_het4", "arch6_fr", "arch6_efs",
              "mixed6", "merged9", "balanced8", "full21"]


def read_pools() -> dict:
    """Minimal reader for the `pools:` block of the submitted configs/experts.yaml."""
    pools, current, inside = {}, None, False
    for raw in src("experts_cfg").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line == "pools:":
            inside = True
            continue
        if inside and not line.startswith(" "):
            break
        if inside and line.startswith("  ") and not line.startswith("    ") and line.strip().endswith(":"):
            current = line.strip()[:-1]
            pools[current] = []
        elif inside and current and line.strip().startswith("- "):
            pools[current].append(line.strip()[2:].strip())
    return pools


def t14_exposure_regimes(L):
    g = load_json("gate")
    T = "t14"
    rows = []
    for regime in ("fs", "fr", "efs", "df40"):
        cells = [g["gate"][regime][x] for x in GENERATORS]
        for x, v in zip(GENERATORS, cells):
            L.add(T, f"exposure.{regime}.{x}", v, 0, "gate", f"gate.{regime}.{x}")
        total = g["per_regime_overlap_totals"][regime]
        L.add(T, f"exposure.{regime}.total", total, 0, "gate", f"per_regime_overlap_totals.{regime}")
        rows.append(f"{REGIME_LABEL[regime]} & " + " & ".join(f"{v:,}" for v in cells) + f" & {total:,}")
    rows.append("held-out test paths in the release & " +
                " & ".join(f"{g['test_unique_paths'][x]:,}" for x in GENERATORS) +
                f" & {sum(g['test_unique_paths'].values()):,}")
    hdr = ("\\textbf{Expert training list} & " + " & ".join(f"\\multicolumn{{1}}{{l}}{{\\textbf{{{c}}}}}" for c in
           ("WIR", "CDiff", "MJ6", "HeyG", "SG2", "DFL", "SG", "SCLIP")) + " & \\textbf{Total}")
    return tabular("lrrrrrrrrr", hdr, rows)


def t15_exposure_pools(L):
    pools = read_pools()
    h = load_json("appendix_h_cells")["per_expert"]["df40-xception"]["per_generator"]
    T = "t15"
    rows = []
    for p in POOL_ORDER:
        experts = pools[p]
        regimes = sorted({e.split("-")[0] for e in experts}, key=["fs", "fr", "efs", "df40"].index)
        df40 = [e for e in experts if e.startswith("df40-")]
        L.add(T, f"pool_exposure.{p}.df40_experts", len(df40), 0, "experts_cfg", f"pools.{p}", kind="derived")
        rows.append(f"{p.replace('_', chr(92) + '_')} & {len(experts)} & "
                    f"{', '.join(r.upper() if r != 'df40' else 'DF40' for r in regimes)} & "
                    f"{', '.join(df40) if df40 else 'none'}")
    for x in ("MidJourney", "deepfacelab", "styleclip"):
        L.add(T, f"df40_xception.{x}", h[x], 3, "appendix_h_cells", f"per_expert.df40-xception.per_generator.{x}",
              kind="derived", note="a held-out set fully contained in DF40_all.json; memorised test images would "
                                   "score near 1.0")
    hdr = ("\\textbf{Pool} & \\textbf{Experts} & \\textbf{Training lists} & "
           "\\textbf{Experts from the DF40 aggregate list}")
    return tabular("lrll", hdr, rows)


POOLS_A21 = ("clip4", "arch6", "mixed6", "merged9", "strong4", "balanced8", "full21", "weak6", "low_het4")


def t17_pool_deltas(L):
    """Table A16 (per-pool, standard split): best PS, FS1, FS2 and both deltas at full precision.

    Deltas are formed before rounding; the submitted table printed three cells from rounded operands
    (arch6 FS2 +0.126, merged9 FS1 +0.031, low_het4 FS2 +0.072; correct: +0.127, +0.030, +0.073)."""
    T = "t17"
    rows = []
    for pool in POOLS_A21:
        if pool in ("arch6", "clip4"):
            # primary pools: the PS ceiling is the full ladder (richer inputs included)
            SOURCES[f"ds_{pool}_raw"] = f"{SUBRES}/04_ds_{pool}_rawfeat.json"
            ps = load_json("full_ladder")["standard"][pool]["ps_ceiling"]
            fs1 = load_json(f"ds_{pool}_raw")["avg_auc_8"]
            src_ps, src_fs1 = "full_ladder", f"ds_{pool}_raw"
        elif pool == "merged9":
            m = load_json("merged9")
            ps, fs1 = m["ps6_xgboost"]["avg_auc_8"], m["fs1_raw_feat_logreg"]["avg_auc_8"]
            src_ps = src_fs1 = "merged9"
        else:
            SOURCES[f"ds_{pool}"] = f"{SUBRES}/04_ds_{pool}.json"
            SOURCES[f"ds_{pool}_raw"] = f"{SUBRES}/04_ds_{pool}_rawfeat.json"
            ps, fs1 = load_json(f"ds_{pool}")["best_ds_avg8"], load_json(f"ds_{pool}_raw")["avg_auc_8"]
            src_ps, src_fs1 = f"ds_{pool}", f"ds_{pool}_raw"
        fs2_name = "05_fs_clip4_mlp_d32.json" if pool == "clip4" else f"05_fs_{pool}_mlp_d256.json"
        SOURCES[f"fs2_{pool}"] = f"{SUBRES}/{fs2_name}"
        fs2 = load_json(f"fs2_{pool}")["mean_avg8"]
        ps_key = ("ps6_xgboost.avg_auc_8" if pool == "merged9" else
                  f"standard.{pool}.ps_ceiling" if pool in ("arch6", "clip4") else "best_ds_avg8")
        L.add(T, f"a21.{pool}.ps", ps, 3, src_ps, ps_key)
        L.add(T, f"a21.{pool}.fs1", fs1, 3, src_fs1, "avg_auc_8" if pool != "merged9" else "fs1_raw_feat_logreg.avg_auc_8")
        L.add(T, f"a21.{pool}.fs2", fs2, 3, f"fs2_{pool}", "mean_avg8")
        L.add(T, f"a21.{pool}.d1", fs1 - ps, 3, src_fs1, "fs1 - ps (full precision)", kind="derived")
        L.add(T, f"a21.{pool}.d2", fs2 - ps, 3, f"fs2_{pool}", "fs2 - ps (full precision)", kind="derived")
        rows.append(f"{pool.replace('_', chr(92) + '_')} & {ps:.3f} & {fs1:.3f} & {sd(fs1 - ps)} & {fs2:.3f} & "
                    f"{sd(fs2 - ps)}")
    hdr = ("\\textbf{Pool} & \\textbf{Best PS} & \\textbf{\\fsmethod{1}} & \\textbf{$\\Delta_1$} & "
           "\\textbf{\\fsmethod{2}} & \\textbf{$\\Delta_2$}")
    return tabular("lrrrrr", hdr, rows)


def t18_full_ladder(L):
    """Statistics against the full prediction-space ladder, the paper's PS ceiling where the richer
    inputs were run (arch6 under both protocols, clip4 on the standard split). Values come from
    full_ladder_stats.py, which reproduces every stored ten-method statistic before computing them."""
    T = "t18"
    d = load_json("full_ladder")
    note = "full ladder; recomputed from stored per-generator or per-fold values (full_ladder_stats.py)"

    def add(qid, value, digits, key):
        L.add(T, qid, value, digits, "full_ladder", key, kind="derived", note=note)

    rows = []
    for pool in ("arch6", "clip4"):
        s, p = d["standard"][pool], f"fl.std.{pool}"
        add(f"{p}.ceiling", s["ps_ceiling"], 3, f"standard.{pool}.ps_ceiling ({s['ps_ceiling_method']})")
        add(f"{p}.fs1", s["fs1"], 3, f"standard.{pool}.fs1")
        add(f"{p}.fs2", s["fs2"], 3, f"standard.{pool}.fs2")
        add(f"{p}.fs1.delta", s["fs1_delta"], 3, f"standard.{pool}.fs1_delta")
        add(f"{p}.fs1.ci_lo", s["fs1_ci95"][0], 3, f"standard.{pool}.fs1_ci95[0]")
        add(f"{p}.fs1.ci_hi", s["fs1_ci95"][1], 3, f"standard.{pool}.fs1_ci95[1]")
        add(f"{p}.fs1.wins", s["fs1_wins"], 0, f"standard.{pool}.fs1_wins")
        add(f"{p}.fs1.signflip_p", s["fs1_signflip_p"], 3, f"standard.{pool}.fs1_signflip_p")
        add(f"{p}.fs2.delta", s["fs2_delta"], 3, f"standard.{pool}.fs2_delta")
        add(f"{p}.fs1_minus_ps4", s["fs1_minus_ps4"], 3, f"standard.{pool}.fs1_minus_ps4")
        for k in ("weighted_ps_ceiling", "weighted_fs1", "weighted_fs2", "weighted_fs1_delta", "weighted_fs2_delta"):
            add(f"{p}.{k}", s[k], 3, f"standard.{pool}.{k}")
        rows.append(f"{pool} & standard & {s['ps_ceiling']:.3f} & {s['fs1']:.3f} & {sd(s['fs1_delta'])} & "
                    f"$[{sm(s['fs1_ci95'][0])}, {sm(s['fs1_ci95'][1])}]$ & {s['fs1_wins']}/{len(GENERATORS)} & -- & -- & "
                    f"{s['fs1_signflip_p']:.3f}")
    g = d["logo_arch6"]
    add("fl.logo.ceiling", g["ps_ceiling"], 3, "logo_arch6.ps_ceiling")
    add("fl.logo.fs1.n500", g["fs1_delta_n500"], 3, "logo_arch6.fs1_delta_n500")
    add("fl.logo.weighted_ps_ceiling", g["weighted_ps_ceiling"], 3, "logo_arch6.weighted_ps_ceiling")
    add("fl.logo.weighted_fs1_delta", g["weighted_fs1"] - g["weighted_ps_ceiling"], 3,
        "logo_arch6.weighted_fs1 - weighted_ps_ceiling")
    add("fl.logo.weighted_fs2_delta", g["weighted_fs2"] - g["weighted_ps_ceiling"], 3,
        "logo_arch6.weighted_fs2 - weighted_ps_ceiling")
    for fold, c in g["changed_folds"].items():
        add(f"fl.logo.{fold}.ceiling", c["full"], 3, f"logo_arch6.changed_folds.{fold}.full ({c['best_richer_input']})")
        i = g["folds"].index(fold)
        add(f"fl.logo.{fold}.fs1.delta", g["stats"]["fs1"]["per_fold"][i], 3, f"logo_arch6.stats.fs1.per_fold[{i}]")
        add(f"fl.logo.{fold}.fs2.delta", g["stats"]["fs2"]["per_fold"][i], 3, f"logo_arch6.stats.fs2.per_fold[{i}]")
    for m in ("fs1", "fs2"):
        st = g["stats"][m]
        add(f"fl.logo.{m}.delta", st["mean_delta"], 3, f"logo_arch6.stats.{m}.mean_delta")
        add(f"fl.logo.{m}.ci_lo", st["ci95"][0], 3, f"logo_arch6.stats.{m}.ci95[0]")
        add(f"fl.logo.{m}.ci_hi", st["ci95"][1], 3, f"logo_arch6.stats.{m}.ci95[1]")
        add(f"fl.logo.{m}.wins", st["wins"], 0, f"logo_arch6.stats.{m}.wins")
        add(f"fl.logo.{m}.sign_p", st["sign_p"], 3, f"logo_arch6.stats.{m}.sign_p")
        add(f"fl.logo.{m}.wilcoxon_p", st["wilcoxon_p"], 3, f"logo_arch6.stats.{m}.wilcoxon_p")
        add(f"fl.logo.{m}.signflip_p", st["signflip_p"], 3, f"logo_arch6.stats.{m}.signflip_p")
        add(f"fl.logo.{m}.sd_pop", st["sd_population"], 3, f"logo_arch6.stats.{m}.sd_population")
        add(f"fl.logo.{m}.sd_sample", st["sd_sample"], 3, f"logo_arch6.stats.{m}.sd_sample")
        rows.append(f"arch6 ({m.upper()}) & LOGO & {g['ps_ceiling']:.3f} & {g[m]:.3f} & {sd(st['mean_delta'])} & "
                    f"$[{sm(st['ci95'][0])}, {sm(st['ci95'][1])}]$ & {st['wins']}/{len(st['per_fold'])} & "
                    f"{st['sign_p']:.3f} & "
                    f"{st['wilcoxon_p']:.3f} & {st['signflip_p']:.3f}")
    for dim, r in d["dim_sweep_clip4"].items():
        add(f"fl.dim_sweep.d{dim}.delta", r["delta_vs_full_ceiling"], 3, f"dim_sweep_clip4.{dim}.delta_vs_full_ceiling")
    # Table A4 parameter counts of the logistic heads on the richer inputs (PS_G..PS_K), from the
    # input-map definitions in scripts/analyses/appB_richer_inputs.py: E experts with 2
    # logits each; weights plus intercept, plus the E temperatures of PS_H (fitted on SELECT).
    letter = {"PS_11_disagreement": "G", "PS_12_uncertainty_mi": "H", "PS_13_quadratic": "I", "PS_14_rank_ecdf": "J"}
    for pool in ("arch6", "clip4"):
        lj = load_json(f"ladder_{pool}")
        E = len(lj["experts"])
        pairs = E * (E - 1) // 2
        dims = {"G": 2 * E + pairs, "H": E + 1 + UNCERTAINTY_SUMMARIES, "I": 2 * E + E + pairs, "J": E}
        params = {k: v + 1 for k, v in dims.items()}
        params["H"] += len(lj["calibration_temperatures_from_select"])
        params["K"] = params[letter[lj["ps_15_selected_rung"]]]
        for k, v in params.items():
            L.add(T, f"fl.params.{pool}.PS_{k}-LR", v, 0, f"ladder_{pool}",
                  f"experts ({E}), ps_15_selected_rung; input-map sizes from the runner", kind="derived",
                  note="logistic head: inputs + intercept (+ temperatures for PS_H)")
    # audit rows (Table 3): baseline-ladder and overlap distortions against the full ladder
    lad = load_json("ladder_arch6")["published_ladder"]["all_macro"]
    best13 = max(lad["PS_1_prob_avg"], lad["PS_2_logreg"], lad["PS_3_mlp_logits"])
    s = d["standard"]["arch6"]
    L.add(T, "fl.audit.simplified_delta", s["fs1"] - best13, 3, "ladder_arch6",
          "fs1 (full_ladder) - max(PS_1, PS_2, PS_3) (published_ladder.all_macro)", kind="derived")
    L.add(T, "fl.audit.baseline_distortion", (s["fs1"] - best13) - s["fs1_delta"], 3, "full_ladder",
          "(fs1 - best PS1-PS3) - standard.arch6.fs1_delta", kind="derived", note=note)
    add("fl.audit.overlap_distortion", s["fs1_delta"] - g["stats"]["fs1"]["mean_delta"],
        3, "standard.arch6.fs1_delta - logo_arch6.stats.fs1.mean_delta")
    hdr = ("\\textbf{Pool} & \\textbf{Protocol} & \\textbf{PS ceiling} & \\textbf{FS} & \\textbf{$\\Delta$} & "
           "\\textbf{95\\% CI} & \\textbf{Wins} & \\textbf{Sign $p$} & \\textbf{Wilcoxon $p$} & \\textbf{Sign-flip $p$}")
    return tabular("llrrrlrrrr", hdr, rows)


# ---------------------------------------------------------------- mitigation runs (version 2)

PS_LABEL = {
    "PS_1_prob_avg": "\\psmethod{1}: prob.\\ avg.", "PS_2_logreg": "\\psmethod{2}: LogReg",
    "PS_3_mlp_logits": "\\psmethod{3}: MLP", "PS_4_xgboost": "\\psmethod{4}: XGBoost",
    "PS_A_calibrated_avg": "\\psmethod{A}: calibrated avg.", "PS_B_learned_weights": "\\psmethod{B}: learned wts",
    "PS_C_logreg_cv": "\\psmethod{C}: LogReg-CV", "PS_D_mlp_probs": "\\psmethod{D}: MLP on probs",
    "PS_E_mlp_probs_es": "\\psmethod{E}: MLP on probs, ES", "PS_F_mlp_logits_es": "\\psmethod{F}: MLP on logits, ES",
}
RICH_LABEL = {"disagreement": ("G", "disagreement"), "uncertainty+MI": ("H", "uncertainty, MI"),
              "quadratic": ("I", "quadratic"), "rank ECDF": ("J", "rank ECDF")}


def ps_label(name):
    """Display name of a ladder method or of a richer-input head (e.g. 'uncertainty+MI_logreg' -> PS_H-LR)."""
    if name in PS_LABEL:
        return PS_LABEL[name]
    head = "LR" if name.endswith("_logreg") else "XGB"
    base = name[:-len("_logreg")] if head == "LR" else name.split("_xgb_")[0]
    letter, desc = RICH_LABEL[base]
    return f"\\psmethod{{{letter}}}-{head}: {desc}"


def ps_symbol(name):
    """Version 15: the symbol of a ladder method or richer-input head without its description (Table A10 of paper
    v12 prints \\psmethod{B} or \\psmethod{H}-LR)."""
    return ps_label(name).split(":")[0]


# version 15: the counts that paper v12 prints as words
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def word_warning(where, text):
    """Version 15: a statement of the paper about environment-dependent values (FS2-type results, which a rerun may
    change and the harness reports with their change) that does not hold for the values read here: printed, not
    fatal. Statements about the other values stop the build."""
    print(f"WARNING ({where}): {text}; these values depend on the environment, so the build goes on")


def t19_mitigation_logo(L):
    """LOGO with the training rows removed whose features are identical to a held-out test image (mitigation
    runs, gates passed), against the full ladder (the LOGO statistics of mitigation_stats.json, written by the
    statistics script): arch6 per fold (Table A10 of the camera-ready paper), the derived arch6 statistics, and the
    four pools of Table 1. Returns the per-fold body. Version 10: this docstring and the notes below named statistics
    version 2 and Table A11 (an earlier draft's numbering)."""
    T = "t19"
    st, d = load_json("mitigation"), load_json("cd_logo_arch6")
    lg = st["logo_arch6"]
    note = "LOGO, identical-feature training rows removed; full ladder (the LOGO statistics of mitigation_stats.json)"

    def add(qid, value, digits, source, key, kind="json"):
        L.add(T, qid, value, digits, source, key, kind=kind, note=note)

    ex = load_json("exclusions_arch6")
    add("cd.excluded.images", ex["shared_images"], 0, "exclusions_arch6", "shared_images")
    add("cd.excluded.rows", ex["rows_excluded_total"], 0, "exclusions_arch6", "rows_excluded_total")
    add("cd.logo.ceiling", lg["ceiling_full"], 3, "mitigation", "logo_arch6.ceiling_full")
    add("cd.logo.ceiling_ten", lg["ceiling_ten"], 3, "mitigation", "logo_arch6.ceiling_ten")
    add("cd.logo.fs1", lg["fs1"], 3, "mitigation", "logo_arch6.fs1")
    add("cd.logo.fs2", lg["fs2"], 3, "mitigation", "logo_arch6.fs2")
    for m in ("fs1", "fs2"):
        for scope, tag in (("full", ""), ("ten", ".ten")):
            s, k = lg["stats"][f"{m}_{scope}"], f"logo_arch6.stats.{m}_{scope}"
            p = f"cd.logo{tag}.{m}"
            add(f"{p}.delta", s["mean_delta"], 3, "mitigation", f"{k}.mean_delta")
            add(f"{p}.ci_lo", s["ci95"][0], 3, "mitigation", f"{k}.ci95[0]")
            add(f"{p}.ci_hi", s["ci95"][1], 3, "mitigation", f"{k}.ci95[1]")
            add(f"{p}.wins", s["wins"], 0, "mitigation", f"{k}.wins")
            add(f"{p}.sign_p", s["sign_p"], 3, "mitigation", f"{k}.sign_p")
            add(f"{p}.wilcoxon_p", s["wilcoxon_p"], 3, "mitigation", f"{k}.wilcoxon_p")
            add(f"{p}.signflip_p", s["signflip_p"], 3, "mitigation", f"{k}.signflip_p")
            add(f"{p}.sd_pop", s["sd_population"], 3, "mitigation", f"{k}.sd_population")
            add(f"{p}.sd_sample", s["sd_sample"], 3, "mitigation", f"{k}.sd_sample")
    add("cd.logo.fs1.n500", lg["fs1_full_delta_n500"], 3, "mitigation", "logo_arch6.fs1_full_delta_n500")
    w = lg["weighted"]
    add("cd.logo.weighted.ceiling", w["ceiling_full"], 3, "mitigation", "logo_arch6.weighted.ceiling_full")
    add("cd.logo.weighted.fs1", w["fs1"], 3, "mitigation", "logo_arch6.weighted.fs1")
    add("cd.logo.weighted.fs2", w["fs2"], 3, "mitigation", "logo_arch6.weighted.fs2")
    add("cd.logo.weighted.fs1_delta", w["fs1"] - w["ceiling_full"], 3, "mitigation",
        "logo_arch6.weighted.fs1 - ceiling_full", kind="derived")
    add("cd.logo.weighted.fs2_delta", w["fs2"] - w["ceiling_full"], 3, "mitigation",
        "logo_arch6.weighted.fs2 - ceiling_full", kind="derived")
    a = lg["auprc"]
    add("cd.logo.auprc.fs1", a["fs1"], 3, "mitigation", "logo_arch6.auprc.fs1")
    add("cd.logo.auprc.ceiling", a["ceiling_full"], 3, "mitigation", "logo_arch6.auprc.ceiling_full")
    add("cd.logo.auprc.delta", a["delta"], 3, "mitigation", "logo_arch6.auprc.delta")
    r = load_json("cd_rungs_arch6")
    add("cd.logo.richer_only.capped", r["extended_only_ceiling_capped"], 4, "cd_rungs_arch6",
        "extended_only_ceiling_capped")
    add("cd.logo.richer_only.widened", r["extended_only_ceiling_widened"], 4, "cd_rungs_arch6",
        "extended_only_ceiling_widened")
    add("cd.audit.overlap_distortion", st["audit"]["overlap_distortion"], 3, "mitigation",
        "audit.overlap_distortion (standard.arch6.fs1_delta - new LOGO delta)")
    rows, n_richer = [], 0
    for i, f in enumerate(d["per_fold"]):
        g, c, mname = f["held_out"], lg["ceiling_full_per_fold"][i], lg["ceiling_method_per_fold"][i]
        fs1, fs2, p = f["raw_feat_auc"], f["fs_mlp_mean"], f"cd.logo.fold.{f['held_out']}"
        n_richer += not mname.startswith("PS_")
        add(f"{p}.ceiling", c, 3, "mitigation", f"logo_arch6.ceiling_full_per_fold[{i}] ({mname})")
        add(f"{p}.fs1", fs1, 3, "cd_logo_arch6", f"per_fold[{i}].raw_feat_auc")
        add(f"{p}.fs2", fs2, 3, "cd_logo_arch6", f"per_fold[{i}].fs_mlp_mean")
        add(f"{p}.fs1.delta", fs1 - c, 3, "cd_logo_arch6", "raw_feat_auc - full-ladder ceiling", kind="derived")
        add(f"{p}.fs2.delta", fs2 - c, 3, "cd_logo_arch6", "fs_mlp_mean - full-ladder ceiling", kind="derived")
        add(f"{p}.excluded_rows", f["n_excluded_training_rows"], 0, "cd_logo_arch6",
            f"per_fold[{i}].n_excluded_training_rows")
        rows.append(f"{GEN_LABEL[g]} & {c:.3f} & {ps_label(mname)} & {fs1:.3f} & {sd(fs1 - c)} & {fs2:.3f} & "
                    f"{sd(fs2 - c)}")
    add("cd.logo.richer_folds", n_richer, 0, "mitigation", "logo_arch6.ceiling_method_per_fold (richer inputs)")
    rows.append(f"Avg.\\ AUROC & {lg['ceiling_full']:.3f} & - & {lg['fs1']:.3f} & "
                f"{sd(lg['stats']['fs1_full']['mean_delta'])} & {lg['fs2']:.3f} & "
                f"{sd(lg['stats']['fs2_full']['mean_delta'])}")
    if lg.get("ladder", "full") != "full":          # version 8: statistics version 5 records it for arch6 too
        raise SystemExit("arch6: LOGO statistics are not against the full ladder")
    for pool, b in st["logo_boundary"].items():
        if b.get("ladder") != "full":
            raise SystemExit(f"{pool}: boundary statistics are not against the full ladder")
        s, p, k = b["stats"]["fs1_full"], f"cd.pool.{pool}", f"logo_boundary.{pool}"
        add(f"{p}.ceiling", b["ceiling_full"], 3, "mitigation", f"{k}.ceiling_full")
        add(f"{p}.ceiling_ten", b["ceiling_ten"], 3, "mitigation", f"{k}.ceiling_ten")
        add(f"{p}.fs1", b["fs1"], 3, "mitigation", f"{k}.fs1")
        add(f"{p}.fs2", b["fs2"], 3, "mitigation", f"{k}.fs2")
        add(f"{p}.delta", s["mean_delta"], 3, "mitigation", f"{k}.stats.fs1_full.mean_delta")
        add(f"{p}.ci_lo", s["ci95"][0], 3, "mitigation", f"{k}.stats.fs1_full.ci95[0]")
        add(f"{p}.ci_hi", s["ci95"][1], 3, "mitigation", f"{k}.stats.fs1_full.ci95[1]")
        add(f"{p}.wins", s["wins"], 0, "mitigation", f"{k}.stats.fs1_full.wins")
        add(f"{p}.large_mean", b["fs1_full_delta_n500"], 3, "mitigation", f"{k}.fs1_full_delta_n500")
        add(f"{p}.fs2.delta", b["stats"]["fs2_full"]["mean_delta"], 3, "mitigation", f"{k}.stats.fs2_full.mean_delta")
        add(f"{p}.ten.delta", b["stats"]["fs1_ten"]["mean_delta"], 3, "mitigation", f"{k}.stats.fs1_ten.mean_delta")
        add(f"{p}.richer_folds", sum(not m.startswith("PS_") for m in b["ceiling_method_per_fold"]), 0,
            "mitigation", f"{k}.ceiling_method_per_fold (richer inputs)")
    vals = [L.values["cd.logo.fs1.delta"]] + [L.values[f"cd.pool.{p}.delta"] for p in st["logo_boundary"]]
    add("cd.pools.logo_min", min(vals), 3, "mitigation", "min over the four LOGO pools", kind="derived")
    add("cd.pools.logo_max", max(vals), 3, "mitigation", "max over the four LOGO pools", kind="derived")
    hdr = ("\\textbf{Generator} & \\multicolumn{1}{l}{\\textbf{Best PS}} & \\textbf{Best PS method} & "
           "\\multicolumn{1}{l}{\\textbf{\\fsmethod{1}}} & \\multicolumn{1}{l}{\\textbf{$\\Delta_1$}} & "
           "\\multicolumn{1}{l}{\\textbf{\\fsmethod{2}}} & \\multicolumn{1}{l}{\\textbf{$\\Delta_2$}}")
    return tabular("lrlrrrr", hdr, rows)


def t20_logo_inferential(L):
    """Table A12 body (LOGO inferential views, arch6, full ladder) from the values recorded by t19."""
    v = L.values
    rows = []
    for m in ("1", "2"):
        p = f"cd.logo.fs{m}"
        rows.append(f"\\fsmethod{{{m}}} vs Best PS & {sd(v[p + '.delta'])} & {v[p + '.sign_p']:.3f} & "
                    f"{v[p + '.wilcoxon_p']:.3f} & {v[p + '.signflip_p']:.3f} & "
                    f"$[{sm(v[p + '.ci_lo'])}, {sm(v[p + '.ci_hi'])}]$")
    hdr = ("\\textbf{Comparator} & \\multicolumn{1}{l}{\\textbf{$\\Delta$}} & "
           "\\multicolumn{1}{l}{\\textbf{Sign test $p$}} & \\multicolumn{1}{l}{\\textbf{Wilcoxon $p$}} & "
           "\\multicolumn{1}{l}{\\textbf{Sign-flip $p$}} & \\multicolumn{1}{l}{\\textbf{Boot 95\\% CI}}")
    return tabular("lrrrrr", hdr, rows)


def t21_rq2_both_protocols(L):
    """Table A8 (App. D.4) under both protocols: the standard-split columns from the version-1 ids, the LOGO
    columns (arch6, clip4) from the mechanism runs; FS1 per fold is the ladder run's.
    Also records the LOGO width sweep and every mechanism comparison (App. D.4, Sec. 5.2).
    Version 15: the combiners are named as in paper v12's Table A8 (version 14 named them as the submitted paper,
    where this table was main-text Table 2); the values are those of version 14."""
    T = "t21"
    v, st = L.values, load_json("mitigation")
    SOURCES["xgbfeat_arch6"] = f"{SUBRES}/12_xgb_features_arch6.json"
    SOURCES["xgbfeat_clip4"] = f"{SUBRES}/12_xgb_features_clip4.json"
    for pool in ("arch6", "clip4"):
        L.add(T, f"std.{pool}.fs_d", load_json(f"xgbfeat_{pool}")["avg_auc_8"], 3, f"xgbfeat_{pool}", "avg_auc_8")
    mech = {"arch6": st["mechanism_logo"], "clip4": st["mechanism_logo_clip4"]}
    fs2 = {"arch6": load_json("cd_logo_arch6")["avg_fs_mlp"], "clip4": load_json("cd_logo_clip4")["avg_fs_mlp"]}
    note = "LOGO mechanism runs (05_run_c_width_matched_v2.py), identical-feature training rows removed"
    for pool, m in mech.items():
        key = "mechanism_logo" if pool == "arch6" else "mechanism_logo_clip4"
        for k, x in m["mean_auroc"].items():
            L.add(T, f"cd.mech.{pool}.{k}", x, 3, "mitigation", f"{key}.mean_auroc.{k}", note=note)
        L.add(T, f"cd.mech.{pool}.FS_2", fs2[pool], 3, f"cd_logo_{pool}", "avg_fs_mlp", note=note)
        L.add(T, f"cd.mech.{pool}.fs1_check_max_abs_diff", m["fs1_mechanism_run_max_abs_diff"], 6, "mitigation",
              f"{key}.fs1_mechanism_run_max_abs_diff", note="FS1 per fold from the ladder run")
        for c, s in m["comparisons"].items():
            p = f"cd.mech.{pool}.cmp.{c}"
            L.add(T, f"{p}.delta", s["mean_delta"], 3, "mitigation", f"{key}.comparisons.{c}.mean_delta", note=note)
            L.add(T, f"{p}.ci_lo", s["ci95"][0], 3, "mitigation", f"{key}.comparisons.{c}.ci95[0]", note=note)
            L.add(T, f"{p}.ci_hi", s["ci95"][1], 3, "mitigation", f"{key}.comparisons.{c}.ci95[1]", note=note)
            L.add(T, f"{p}.wins", s["wins"], 0, "mitigation", f"{key}.comparisons.{c}.wins", note=note)
            L.add(T, f"{p}.signflip_p", s["signflip_p"], 3, "mitigation", f"{key}.comparisons.{c}.signflip_p",
                  note=note)
    a, c = (lambda k: v[f"cd.mech.arch6.{k}"]), (lambda k: v[f"cd.mech.clip4.{k}"])
    rows = [
        f"\\psmethod{{4}}: XGBoost on logits & 12 & - & {v['width.ps4']:.3f} & {v['clip4.ps4_submitted']:.3f} & "
        f"{a('PS_4_xgboost'):.3f} & {c('PS_4_xgboost'):.3f}",
        f"\\fsmethod{{1}}: Logistic regression on raw features & 9,984 & 10.0K & {v['main.standard.fs1']:.3f} & "
        f"{v['a21.clip4.fs1']:.3f} & {a('FS_1_raw_feat_logreg'):.3f} & {c('FS_1_raw_feat_logreg'):.3f}",
        f"\\fsmethod{{2}}: Learned projections + MLP & 9,984 & 3.5M & {v['a21.arch6.fs2']:.3f} & "
        f"{v['a21.clip4.fs2']:.3f} & {a('FS_2'):.3f} & {c('FS_2'):.3f}",
        f"\\fsmethod{{D}}: XGBoost on features & 9,984 & - & {v['std.arch6.fs_d']:.3f} & {v['std.clip4.fs_d']:.3f} & "
        f"{a('FS_D_xgb_features'):.3f} & {c('FS_D_xgb_features'):.3f}",
        f"\\fsmethod{{1}} on 12 PCA values & 12 & 13$^{{*}}$ & {v['width.pca_d2.macro']:.3f} & - & "
        f"{a('FS_pca2_logreg'):.3f} & -",
        f"\\fsmethod{{1}} on 12 label-fitted values & 12 & 13$^{{*}}$ & {v['width.probe_d2.macro']:.3f} & - & "
        f"{a('FS_probe2_logreg'):.3f} & -",
    ]
    hdr = ("\\textbf{Combiner} & \\multicolumn{1}{l}{\\textbf{Inputs (arch6)}} & "
           "\\multicolumn{1}{l}{\\textbf{Params (arch6)}} & \\multicolumn{1}{l}{\\textbf{arch6}} & "
           "\\multicolumn{1}{l}{\\textbf{clip4}} & \\multicolumn{1}{l}{\\textbf{arch6}} & "
           "\\multicolumn{1}{l}{\\textbf{clip4}}")
    return tabular("lrrrrrr", hdr, rows)


def t16_main_results(L):
    """Main-text table: FS1 vs the PS ceiling (full ladder), LOGO for four pools from the mitigation runs
    (identical-feature training rows removed; values recorded by t19) plus arch6 standard."""
    v = L.values
    fs1_canonical = load_json("cost")["combiner_costs"]["FS_1_frozen_total"]["macro_auroc"]   # version 8: read
    mst = load_json("mitigation")                    # version 10: the fold counts
    rows = []
    for pool in ("arch6", "clip4", "strong4", "weak6"):
        if pool == "arch6":
            ps, fs, d, w, big = (v["cd.logo.ceiling"], v["cd.logo.fs1"], v["cd.logo.fs1.delta"], v["cd.logo.fs1.wins"],
                                 v["cd.logo.fs1.n500"])
        else:
            ps, fs, d, w, big = (v[f"cd.pool.{pool}.ceiling"], v[f"cd.pool.{pool}.fs1"], v[f"cd.pool.{pool}.delta"],
                                 v[f"cd.pool.{pool}.wins"], v[f"cd.pool.{pool}.large_mean"])
        n = (mst["logo_arch6"] if pool == "arch6" else mst["logo_boundary"][pool])["stats"]["fs1_full"]["n"]
        rows.append(f"{pool} & LOGO & {ps:.3f} & {fs:.3f} & {sd(d)} & {w}/{n} & {sd(big)}")
    L.add("t16", "main.standard.fs1", fs1_canonical, 3, "cost", "combiner_costs.FS_1_frozen_total.macro_auroc",
          note="canonical FS1 arch6 on the standard split")
    rows.append("\\midrule")
    rows.append(f"arch6 & standard & {v['fl.std.arch6.ceiling']:.3f} & {fs1_canonical:.3f} & "
                f"{sd(v['fl.std.arch6.fs1.delta'])} & {v['fl.std.arch6.fs1.wins']}/{len(GENERATORS)} & --")
    # numeric columns: right-aligned cells under left-aligned headers, as in the submission's tables
    hdr = "\\textbf{Pool} & \\textbf{Protocol} & " + " & ".join(
        f"\\multicolumn{{1}}{{l}}{{\\textbf{{{h}}}}}"
        for h in ("PS ceiling", "\\fsmethod{1}", "$\\Delta$", "Wins", "$\\Delta$, $n\\!\\geq\\!500$"))
    body = tabular("llrrrrr", hdr, rows)
    return body.replace("\\midrule \\\\\n", "\\midrule\n")


# ---------------------------------------------------------------- posted check

POSTED = [
    # (quantity id, posted display, digits, comparison context)
    ("logo.fs1.delta", "+0.082", 3, "Recorded comparison"), ("logo.fs1.ci_lo", "+0.026", 3, "Recorded comparison"),
    ("logo.fs1.ci_hi", "+0.140", 3, "Recorded comparison"), ("logo.fs1.wins", "7", 0, "Recorded comparison"),
    ("logo.fs1.signflip_p", "0.047", 3, "Recorded comparison"), ("logo.fs1.sign_p", "0.070", 3, "Recorded comparison"),
    ("logo.fs1.wilcoxon_p", "0.055", 3, "Recorded comparison"), ("logo.ceiling", "0.816", 3, "Recorded comparison"),
    ("arch6.ps4_submitted", "0.843", 3, "Recorded comparison"), ("arch6.deployable_ceiling", "0.860", 3, "Recorded comparison"),
    ("arch6.ext.delta", "+0.105", 3, "Recorded comparison"), ("arch6.ext.ci_lo", "+0.058", 3, "Recorded comparison"),
    ("arch6.ext.ci_hi", "+0.158", 3, "Recorded comparison"), ("arch6.ext.wins", "8", 0, "Recorded comparison"),
    ("arch6.ext.p", "0.008", 3, "Recorded comparison"), ("arch6.oracle_domain", "0.868", 3, "Recorded comparison"),
    ("clip4.deployable_ceiling", "0.942", 3, "Recorded comparison"), ("clip4.ext.delta", "+0.024", 3, "Recorded comparison"),
    ("clip4.ext.ci_lo", "+0.007", 3, "Recorded comparison"), ("clip4.ext.ci_hi", "+0.043", 3, "Recorded comparison"),
    ("logo_refit.ceiling_before", "0.816", 3, "Recorded comparison"), ("logo_refit.ceiling_after", "0.818", 3, "Recorded comparison"),
    ("logo_refit.delta_before", "+0.082", 3, "Recorded comparison"), ("logo_refit.delta_after", "+0.080", 3, "Recorded comparison"),
    ("logo_refit.extended_wins", "1", 0, "Recorded comparison"),
    ("width.pca_d2.macro", "0.740", 3, "Recorded comparison"), ("width.ps4", "0.843", 3, "Recorded comparison"),
    ("width.pca_d2.delta", "-0.103", 3, "Recorded comparison"), ("width.pca_d2.p", "0.109", 3, "Recorded comparison"),
    ("width.probe_d2.macro", "0.958", 3, "Recorded comparison"),
    ("k3.PS1-K3", "0.939", 3, "Recorded comparison"), ("k3.PS2-K3", "0.942", 3, "Recorded comparison"),
    ("k3.PS3-K3", "0.954", 3, "Recorded comparison"), ("k3.PS4-K3", "0.936", 3, "Recorded comparison"),
    ("k3.FS1-K3", "0.933", 3, "Recorded comparison"), ("k3.FS2-K3", "0.945", 3, "Recorded comparison"),
    ("k3.FSB-K3-18", "0.898", 3, "Recorded comparison"), ("k3.PS+-K3", "0.941", 3, "Recorded comparison"),
    ("k3.fs2_minus_select", "-0.009", 3, "Recorded comparison"), ("k3.fs1_minus_select", "-0.021", 3, "Recorded comparison"),
    ("k3.fs1_minus_select.pos_folds", "1", 0, "Recorded comparison"), ("k3.fs2_minus_select.pos_folds", "1", 0, "Recorded comparison"),
    ("k3.select_chosen", "PS3-K3", 0, "Recorded comparison"), ("k3.heads_at_cap", "3", 0, "Recorded comparison"),
    ("k3_view_a.delta", "+0.210", 3, "Recorded comparison"), ("k3_view_a.pos", "5", 0, "Recorded comparison"),
    ("k3_view_b.delta", "+0.201", 3, "Recorded comparison"), ("k3_view_b.pos", "3", 0, "Recorded comparison"),
    ("cost.feature_floats", "9,984", 0, "Recorded comparison"), ("cost.feature_bytes", "19,968", 0, "Recorded comparison"),
    ("cost.pred_floats", "12", 0, "Recorded comparison"), ("cost.pred_bytes", "48", 0, "Recorded comparison"),
    ("cost.pool_gib", "0.95", 2, "Recorded comparison"), ("cost.pool_mib", "2.3", 1, "Recorded comparison"),
    ("cost.fs1_fit_s", "53", 0, "Recorded comparison"), ("cost.ps4_cv_s", "62", 0, "Recorded comparison"), ("cost.ps2_s", "0.16", 2, "Recorded comparison"),
    ("cost.fs1_us", "35", 0, "Recorded comparison"), ("cost.ps_us", "0.1", 1, "Recorded comparison"),
    ("strict.full.delta", "+0.123", 3, "Recorded comparison"), ("strict.full.ci_lo", "+0.078", 3, "Recorded comparison"),
    ("strict.full.ci_hi", "+0.174", 3, "Recorded comparison"), ("strict.strict.delta", "-0.228", 3, "Recorded comparison"),
    ("strict.strict.ci_lo", "-0.322", 3, "Recorded comparison"), ("strict.strict.ci_hi", "-0.130", 3, "Recorded comparison"),
    ("strict.strict.fs1", "0.536", 3, "Recorded comparison"), ("strict.strict.ps", "0.764", 3, "Recorded comparison"),
    ("strict.matched.mean", "+0.149", 3, "Recorded comparison"), ("strict.matched.seeds", "3", 0, "Recorded comparison"),
    ("strict.minus_matched", "-0.377", 3, "Recorded comparison"), ("strict.real_rows_full", "5,438", 0, "Recorded comparison"),
    ("strict.real_rows_strict", "586", 0, "Recorded comparison"),
    ("strict.families_full", "8", 0, "Recorded comparison"), ("strict.families_strict", "1", 0, "Recorded comparison"),
    ("reuse.total_n", "17,925", 0, "Recorded comparison"), ("reuse.total_path", "11,886", 0, "Recorded comparison"),
    ("reuse.path_pct", "66.3", 1, "Recorded comparison"), ("reuse.total_content", "12,893", 0, "Recorded comparison"),
    ("reuse.content_pct", "71.9", 1, "Recorded comparison"), ("reuse.fake_generators", "5", 0, "Recorded comparison"),
    ("stargan.full", "0.978", 3, "Recorded comparison"), ("stargan.matched_min", "0.963", 3, "Recorded comparison"),
    ("stargan.matched_max", "0.974", 3, "Recorded comparison"), ("stargan.strict", "0.508", 3, "Recorded comparison"),
    ("stargan.full_margin", "+0.055", 3, "Recorded comparison"),
    ("pool.clip4.delta", "-0.074", 3, "Recorded comparison"), ("pool.clip4.ceiling", "0.920", 3, "Recorded comparison"),
    ("pool.strong4.delta", "-0.059", 3, "Recorded comparison"), ("pool.strong4.ceiling", "0.909", 3, "Recorded comparison"),
    ("pool.weak6.delta", "-0.100", 3, "Recorded comparison"), ("pool.weak6.ceiling", "0.719", 3, "Recorded comparison"),
    ("pool.clip4.large_mean", "-0.009", 3, "Recorded comparison"), ("pool.strong4.large_mean", "+0.002", 3, "Recorded comparison"),
    ("pool.weak6.large_mean", "-0.055", 3, "Recorded comparison"),
    ("exp.arch6_best.published", "0.780", 3, "Recorded comparison"), ("exp.arch6_best.corrected", "0.787", 3, "Recorded comparison"),
]
LOG_TOL = 0.00051   # log values carry 4 decimals; allow the rounding ambiguity at 3


def posted_check(L):
    lines, fails = [], 0
    kinds = {r["quantity"]: r["kind"] for r in L.rows}
    for qid, posted, digits, thread in POSTED:
        v = L.values.get(qid)
        if v is None:
            status, got = "MISSING", "-"
        elif isinstance(v, str):
            got = v
            status = "PASS" if v == posted else "FAIL"
        else:
            signed = posted.startswith(("+", "-")) and digits > 0
            got = fmt(v, digits, signed=signed) if not (isinstance(v, int) and digits == 0) else f"{v:,}"
            if got == posted:
                status = "PASS"
            elif kinds.get(qid) in ("log", "derived") and abs(v - float(posted.replace(",", ""))) <= LOG_TOL:
                status = "PASS (log rounding)"
            else:
                status = "FAIL"
        fails += status in ("FAIL", "MISSING")
        lines.append(f"| `{qid}` | {thread} | {posted} | {got} | {status} |")
    return lines, fails


# ---------------------------------------------------------------- version 4: superseded rows, audit rows

SUPERSEDED = {
    "t01_logo_arch6_primary": "the submitted LOGO run (ten-method ladder, no identical-feature exclusion); "
                              "the paper prints the mitigation runs (t16, t19, t20; ids cd.*). "
                              "Kept for the posted-number check.",
    "t09_logo_boundary_pools": "the LOGO boundary pools as posted on 3 August (submitted-era logs); the paper "
                               "prints the mitigation runs (t16; ids cd.pool.*). Kept for the posted-number check.",
    "t18_full_ladder": "LOGO rows only: the full ladder before the identical-feature exclusion; the paper "
                       "prints the mitigation runs (t16, t19, t20). The standard-split rows are current.",
}
SUPERSEDED_LOGO_REFIT = ("logo_refit.ceiling_before", "logo_refit.ceiling_after", "logo_refit.delta_before",
                         "logo_refit.delta_after", "logo_refit.extended_wins",
                         "logo_refit.extended_only_capped", "logo_refit.extended_only_widened")


def mark_superseded(L):
    """Label the ledger rows whose values the paper no longer prints; values and ids are unchanged."""
    posted = {p[0] for p in POSTED}
    for r in L.rows:
        q = r["quantity"]
        if ((r["table"] in ("t01", "t09") and not q.startswith("logo.rerun.")) or q.startswith("fl.logo.")
                or q == "fl.audit.overlap_distortion" or q in SUPERSEDED_LOGO_REFIT):
            why = "superseded in the paper by the mitigation runs (ids cd.*)"
            if q in posted:
                why += "; kept for the posted-number check"
            r["note"] = why + (f"; {r['note']}" if r["note"] else "")


def audit_rows(L):
    """Provenance of the values corrected by the number audit of 19 October 2026 (camera_ready/provenance/05_number_provenance.csv)."""
    T = "audit"
    SOURCES.setdefault("xgbfeat_arch6", f"{SUBRES}/12_xgb_features_arch6.json")
    SOURCES["ds_arch6"] = f"{SUBRES}/04_ds_arch6.json"
    SOURCES["fsc_clip4"] = f"{SUBRES}/05_fs_clip4_random_d64.json"
    SOURCES["fs2_clip4_d64"] = f"{SUBRES}/05_fs_clip4_mlp_d64.json"
    fsd, ps4 = load_json("xgbfeat_arch6")["avg_auc_8"], load_json("ds_arch6")["best_ds_avg8"]
    L.add(T, "audit.ps4.arch6", ps4, 3, "ds_arch6", "best_ds_avg8",
          note="PS_4 on arch6, standard split; reference of the FS_D margin (App. B)")
    L.add(T, "audit.fsd.arch6.minus_ps4", fsd - ps4, 3, "xgbfeat_arch6", "avg_auc_8 - 04_ds_arch6.json:best_ds_avg8",
          kind="derived", note="App. B, FS_D paragraph (the submission printed +0.111 from rounded operands)")
    L.add(T, "audit.params.FS_C.clip4", load_json("fsc_clip4")["n_trainable_params"], 0, "fsc_clip4",
          "n_trainable_params", note="Table A2, printed as 264K (the submission printed 263K)")
    pools = read_pools()
    for pool in ("clip4", "arch6"):
        L.add(T, f"audit.params.PS_B.{pool}", len(pools[pool]), 0, "experts_cfg", f"len(pools.{pool})",
              kind="derived", note="Table A2: PS_B fits one weight per expert (softmax over E raw weights, "
                                   "src/models/prediction_space.py)")
    per_seed = [r["avg_auc_8"] for r in load_json("fs2_clip4_d64")["per_seed"]]
    L.add(T, "audit.a6.clip4.d64.test_mean", statistics.mean(per_seed), 3, "fs2_clip4_d64",
          "mean(per_seed[*].avg_auc_8)", kind="derived", note="Table A6 (the submission printed 0.976)")
    n_test = load_json("mitigation")["logo_arch6"]["n_test"]
    L.add(T, "audit.tsne.samples", sum(min(625, n) for n in n_test), 0, "mitigation",
          "sum over generators of min(625, logo_arch6.n_test)", kind="derived",
          note="App. D.2 and Figure A1: generate_figures.py draws at most 625 test samples per generator "
               "(the submission printed 5,000)")


# ---------------------------------------------------------------- version 5: the DF40 tables

# Sources of the paper tables below that version 4 did not read: stored results and configs of the reviewed code
for _p in ("arch6", "clip4", "mixed6", "strong4", "balanced8", "full21", "weak6", "low_het4"):
    SOURCES.setdefault(f"ds_{_p}", f"{SUBRES}/04_ds_{_p}.json")
    SOURCES.setdefault(f"ds_{_p}_raw", f"{SUBRES}/04_ds_{_p}_rawfeat.json")
for _p, _d in (("arch6", 256), ("clip4", 32), ("clip4", 64), ("clip4", 256), ("mixed6", 256), ("merged9", 256),
               ("strong4", 256), ("balanced8", 256), ("full21", 256), ("weak6", 256), ("low_het4", 256)):
    SOURCES.setdefault(f"fs2_{_p}_d{_d}", f"{SUBRES}/05_fs_{_p}_mlp_d{_d}.json")
SOURCES.update({
    "fsa_clip4": f"{SUBRES}/05_fs_clip4_linear_d64.json",
    "fsb_clip4": f"{SUBRES}/05_fs_clip4_pca_d64.json",
    "fsc_clip4": f"{SUBRES}/05_fs_clip4_random_d64.json",
    "xgbfeat_arch6": f"{SUBRES}/12_xgb_features_arch6.json",
    "xgbfeat_clip4": f"{SUBRES}/12_xgb_features_clip4.json",
    "loo_clip4": f"{SUBRES}/10_loo_clip4_d32.json",
    "loo_arch6": f"{SUBRES}/10_loo_arch6_d256.json",
    "auprc": f"{SUBRES}/15_auprc_appendix.json",
    "experiments_cfg": "01.initial_submission/codebase/configs/experiments.yaml",
    "power_mde": "03.camera-ready/code/output/results/analyses/appF_power_mde.json",
    "runner_a": "03.camera-ready/code/scripts/analyses/appB_richer_inputs.py",    # PS_K selection folds
})
# version 12 (paper v10): the two replication pools of App. G(c) and the per-fold richer-input heads of the LOGO runs
for _p in ("arch6_fr", "arch6_efs"):
    SOURCES.setdefault(f"ds_{_p}", f"{SUBRES}/04_ds_{_p}.json")
    SOURCES.setdefault(f"ds_{_p}_raw", f"{SUBRES}/04_ds_{_p}_rawfeat.json")
for _p in ("arch6", "clip4", "strong4", "weak6"):
    for _g in GENERATORS:
        SOURCES.setdefault(f"cd_rung_folds_{_p}_{_g}", f"{MIT}/content_disjoint/rung_folds/{_p}_{_g}.json")
# version 15 (paper v12, App. B): FS1 refitted on standardized features, with the unstandardized fit beside it
for _p in ("arch6", "clip4"):
    SOURCES.setdefault(f"fs1_std_{_p}", f"{SUBRES}/13_fs1_standardized_{_p}.json")
from decimal import Decimal, ROUND_HALF_UP  # noqa: E402

GEN_SHORT = ("WIR", "CDiff", "MJ6", "HeyG", "SG2", "DFL", "SG", "SCLIP")
PROTOCOL = "protocol setting (no data source)"
# label in the paper -> table name used in the ledger. Version 14: paper v12 prints the two blocks of Table A24 as two
# tables of Appendix A without a label, named here by this builder; version 15: so is the overlap table of Appendix A
PAPER_NAMES = {
    "main": "Table 1", "audit": "Table 2", "methods": "Table A1", "ladder": "Table A2", "arch6": "Table A3",
    "clip4": "Table A4", "dim_sweep": "Table A5", "fs2_stability": "Table A6", "bottleneck": "Table A7",
    "rq2": "Table A8", "loo": "Table A9", "logo": "Table A10", "logo_inferential": "Table A11", "mde": "Table A12",
    "weighting": "Table A13", "strict": "Table A14", "fs1_pools": "Table A15", "experts": "Table A16",
    "prevalence": "Table A17", "k3_heads": "Table A18", "k3_views": "Table A19", 
    "cost": "Table A21", "tiers": "Table A22", "heuristics": "Table A23", "sensitivity": "App. A",
    "sensitivity_heads": "App. A", "overlap": "App. A", "small_sets": "App. F",
}


def tab(colspec, head, rows):
    """A tabular in the paper's layout: complete head lines, then rows; '\\midrule' rows stay rules."""
    out = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *head, "\\midrule"]
    out += [r if r.startswith(("\\midrule", "\\cmidrule")) or r.endswith("\\\\") else r + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def mc(text):
    """Header cell as printed: \\multicolumn{1}{l}{\\textbf{...}}."""
    return f"\\multicolumn{{1}}{{l}}{{\\textbf{{{text}}}}}"


def head(*cells):
    return " & ".join(cells) + " \\\\"


def plain(v, digits=3):
    """+0.038 or -0.010: the plain sign style of the appendix tables."""
    return f"{v:+.{digits}f}"


def braced(v, digits=3):
    """{+}0.179 or {-}0.102, inside math, as in Table 2."""
    return ("{+}" if v >= 0 else "{-}") + f"{abs(v):.{digits}f}"


def k_count(n, one_decimal_below=20_000):
    """Parameter counts as printed: 4.1K and 10.0K below 20,000, then 329K, then 3.5M."""
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    return f"{n / 1000:.1f}K" if n < one_decimal_below else f"{round(n / 1000)}K"


def experiments_cfg():
    import yaml   # PyYAML, in requirements-lock.txt
    return yaml.safe_load(src("experiments_cfg").read_text(encoding="utf-8"))


def cfg_get(cfg, key):
    for section in cfg.values():
        if isinstance(section, dict) and key in section:
            return section[key]
    raise KeyError(key)


def ds_best13(pool):
    """Best of PS_1-PS_3 on the standard split (the 'simplified' baseline of Table 2)."""
    mm = load_json(f"ds_{pool}")["method_means"]
    name = max(("PS_1_prob_avg", "PS_2_logreg", "PS_3_mlp_logits"), key=lambda k: mm[k]["mean_avg8"])
    return mm[name]["mean_avg8"], f"method_means.{name}.mean_avg8 (best of PS_1-PS_3)"


def binom_two_sided(k, n):
    from math import comb
    pk = comb(n, k) / 2 ** n
    return min(1.0, sum(comb(n, i) / 2 ** n for i in range(n + 1) if comb(n, i) / 2 ** n <= pk + 1e-12))


def pool_standard(pool):
    """Best PS, FS1 and FS2 of a pool on the standard split, with their sources (as t17)."""
    if pool in ("arch6", "clip4"):
        ps, ps_src, ps_key = load_json("full_ladder")["standard"][pool]["ps_ceiling"], "full_ladder", \
            f"standard.{pool}.ps_ceiling"
        fs1, fs1_src, fs1_key = load_json(f"ds_{pool}_raw")["avg_auc_8"], f"ds_{pool}_raw", "avg_auc_8"
    elif pool == "merged9":
        m = load_json("merged9")
        ps, ps_src, ps_key = m["ps6_xgboost"]["avg_auc_8"], "merged9", "ps6_xgboost.avg_auc_8"
        fs1, fs1_src, fs1_key = m["fs1_raw_feat_logreg"]["avg_auc_8"], "merged9", "fs1_raw_feat_logreg.avg_auc_8"
    else:
        ps, ps_src, ps_key = load_json(f"ds_{pool}")["best_ds_avg8"], f"ds_{pool}", "best_ds_avg8"
        fs1, fs1_src, fs1_key = load_json(f"ds_{pool}_raw")["avg_auc_8"], f"ds_{pool}_raw", "avg_auc_8"
    d = 32 if pool == "clip4" else 256
    fs2 = load_json(f"fs2_{pool}_d{d}")["mean_avg8"]
    return {"ps": (ps, ps_src, ps_key), "fs1": (fs1, fs1_src, fs1_key), "fs2": (fs2, f"fs2_{pool}_d{d}", "mean_avg8")}


def logo_pool(m, pool):
    return (m["logo_arch6"], "logo_arch6") if pool == "arch6" else (m["logo_boundary"][pool], f"logo_boundary.{pool}")


# ---- main text

def p_main(L):
    T, m, s = "Table 1", load_json("mitigation"), load_json("full_ladder")["standard"]["arch6"]
    rows = []
    for pool in ("arch6", "clip4", "strong4", "weak6"):
        g, base = logo_pool(m, pool)
        st = g["stats"]["fs1_full"]
        L.add(T, f"T1.{pool}.logo.ceiling", g["ceiling_full"], 3, "mitigation", f"{base}.ceiling_full")
        L.add(T, f"T1.{pool}.logo.fs1", g["fs1"], 3, "mitigation", f"{base}.fs1")
        L.add(T, f"T1.{pool}.logo.delta", st["mean_delta"], 3, "mitigation", f"{base}.stats.fs1_full.mean_delta")
        L.add(T, f"T1.{pool}.logo.wins", st["wins"], 0, "mitigation", f"{base}.stats.fs1_full.wins")
        L.add(T, f"T1.{pool}.logo.n", st["n"], 0, "mitigation", f"{base}.stats.fs1_full.n")
        # Stable quantity id retained; this value now appears in Appendix F.
        L.add("App. F", f"T1.{pool}.logo.delta_n500", g["fs1_full_delta_n500"], 3, "mitigation", f"{base}.fs1_full_delta_n500")
        rows.append(f"{pool} & LOGO & {g['ceiling_full']:.3f} & {g['fs1']:.3f} & {sd(st['mean_delta'])} & "
                    f"{st['wins']}/{st['n']}")
    n_gen = len(load_json("ds_arch6_raw")["per_generator"])
    L.add(T, "T1.arch6.std.ceiling", s["ps_ceiling"], 3, "full_ladder", "standard.arch6.ps_ceiling")
    L.add(T, "T1.arch6.std.fs1", s["fs1"], 3, "full_ladder", "standard.arch6.fs1")
    L.add(T, "T1.arch6.std.delta", s["fs1_delta"], 3, "full_ladder", "standard.arch6.fs1_delta")
    L.add(T, "T1.arch6.std.wins", s["fs1_wins"], 0, "full_ladder", "standard.arch6.fs1_wins")
    L.add(T, "T1.arch6.std.n", n_gen, 0, "ds_arch6_raw", "len(per_generator)", kind="derived")
    rows += ["\\midrule", f"arch6 & Standard & {s['ps_ceiling']:.3f} & {s['fs1']:.3f} & {sd(s['fs1_delta'])} & "
                          f"{s['fs1_wins']}/{n_gen}"]
    return tab("llrrrr", [head("\\textbf{Pool}", "\\textbf{Protocol}", mc("PS ceiling"), mc("\\fsmethod{1}"),
                                mc("$\\Delta$"), mc("Wins"))], rows)


def p_small_sets(L):
    """Version 17: exploratory eight/six-generator comparison moved from Table 1 to Appendix F."""
    rows = []
    for pool in ("arch6", "clip4", "strong4", "weak6"):
        g, base = logo_pool(load_json("mitigation"), pool)
        delta = g["stats"]["fs1_full"]["mean_delta"]
        L.add("App. F", f"appF.small_sets.{pool}.delta_all", delta, 3, "mitigation",
              f"{base}.stats.fs1_full.mean_delta")
        # The subset values retain their T1.<pool>.logo.delta_n500 ids in p_main's ledger rows.
        rows.append(f"{pool} & {sd(delta)} & {sd(g['fs1_full_delta_n500'])}")
    return tab("lrr", [head(r"\textbf{Pool}", mc(r"$\Delta$, all eight"),
                            mc(r"$\Delta$, six with $n\!\geq\!500$"))], rows)


def p_audit(L):
    """Version 15: Table 2 as paper v12 prints it: for each evaluation choice, the protocol component, the comparison
    and the change in the reported conclusion. Every value of version 14 is kept; the operands of the differences are
    no longer printed, and the effects keep their ids `*.distortion` under the names paper v12 gives them (notes)."""
    T, m, s = "Table 2", load_json("mitigation"), load_json("full_ladder")["standard"]["arch6"]
    op = "not printed since paper v12 (Table 2 prints the differences, not their operands)"
    renamed = lambda name: f"{name} (paper v12); the id keeps 'distortion', its name until paper v11"  # noqa: E731
    b13a, k13a = ds_best13("arch6")
    fs1a = load_json("ds_arch6_raw")["avg_auc_8"]
    simp_a = fs1a - b13a
    L.add(T, "T2.arch6.best13", b13a, 3, "ds_arch6", k13a, note=op)
    L.add(T, "T2.arch6.fs1", fs1a, 3, "ds_arch6_raw", "avg_auc_8", note=op)
    L.add(T, "T2.arch6.simplified_delta", simp_a, 3, "ds_arch6", "04_ds_arch6_rawfeat.avg_auc_8 - best of PS_1-PS_3",
          kind="derived")
    L.add(T, "T2.arch6.full_ladder", s["ps_ceiling"], 3, "full_ladder", "standard.arch6.ps_ceiling", note=op)
    L.add(T, "T2.arch6.fs1_full_ladder", s["fs1"], 3, "full_ladder", "standard.arch6.fs1", note=op)
    L.add(T, "T2.arch6.protocol_delta", s["fs1_delta"], 3, "full_ladder", "standard.arch6.fs1_delta")
    dist_a = L.add(T, "T2.arch6.distortion", simp_a - s["fs1_delta"], 3, "full_ladder",
                   "simplified delta - standard.arch6.fs1_delta", kind="derived",
                   note=renamed("baseline inflation, arch6"))
    b13w, k13w = ds_best13("weak6")
    fs1w, ps4w = load_json("ds_weak6_raw")["avg_auc_8"], load_json("ds_weak6")["best_ds_avg8"]
    L.add(T, "T2.weak6.best13", b13w, 3, "ds_weak6", k13w, note=op)
    L.add(T, "T2.weak6.fs1", fs1w, 3, "ds_weak6_raw", "avg_auc_8", note=op)
    L.add(T, "T2.weak6.simplified_delta", fs1w - b13w, 3, "ds_weak6", "fs1 - best of PS_1-PS_3", kind="derived")
    L.add(T, "T2.weak6.ps4", ps4w, 3, "ds_weak6", "best_ds_avg8", note=op)
    L.add(T, "T2.weak6.protocol_delta", fs1w - ps4w, 3, "ds_weak6", "fs1 - best_ds_avg8", kind="derived")
    dist_w = L.add(T, "T2.weak6.distortion", (fs1w - b13w) - (fs1w - ps4w), 3, "ds_weak6",
                   "simplified delta - protocol delta", kind="derived", note=renamed("baseline inflation, weak6"))
    logo = m["logo_arch6"]["stats"]["fs1_full"]["mean_delta"]
    L.add(T, "T2.overlap.logo_delta", logo, 3, "mitigation", "logo_arch6.stats.fs1_full.mean_delta")
    ov = L.add(T, "T2.overlap.distortion", m["audit"]["overlap_distortion"], 3, "mitigation",
               "audit.overlap_distortion",
               note=renamed("standard-LOGO contrast of arch6, the protocol difference of Table 2"))
    # version 8: Table 2 prints these terms side by side, so they must come from one run: the audit of
    # mitigation_stats.json must have passed, against the standard-split statistics and the LOGO margin read here
    au = m["audit"]
    if au.get("status") != "PASS":
        raise SystemExit(f"Table 2: the audit in mitigation_stats.json has status {au.get('status')!r}, not PASS")
    if abs(au["standard_delta"] - s["fs1_delta"]) > 1e-12 or abs(ov - (s["fs1_delta"] - logo)) > 1e-12:
        raise SystemExit("Table 2: the audit's standard-split margin or overlap distortion differs from the "
                         "standard-split statistics and the LOGO margin read here (inputs from different runs)")
    # Row 1 uses FS1 from two files, and the baseline rows are the audit's
    if abs(fs1a - s["fs1"]) > 1e-12:
        raise SystemExit(f"Table 2: arch6 FS1 differs between 04_ds_arch6_rawfeat.json ({fs1a:.6f}) and the "
                         f"standard-split statistics ({s['fs1']:.6f}): inputs from different runs")
    base = au.get("baseline_distortions") or {}
    for pool, val in (("arch6", dist_a), ("weak6", dist_w)):
        if pool in base and abs(base[pool] - val) > 1e-12:
            raise SystemExit(f"Table 2: the {pool} baseline-ladder distortion {val:+.6f} differs from the audit's "
                             f"{base[pool]:+.6f}: inputs from different runs")
    f2w = load_json("fs2_weak6_d256")
    ps4g = load_json("ds_weak6")["method_means"]["PS_4_xgboost"]["per_generator"]
    d2w = L.add(T, "T2.reporting.fs2_delta", f2w["mean_avg8"] - ps4w, 3, "fs2_weak6_d256",
                "mean_avg8 - 04_ds_weak6.best_ds_avg8", kind="derived")
    differences = [f2w["per_generator_seed_averaged"][g] - ps4g[g] for g in GENERATORS]
    wins = sum(delta > 0 for delta in differences)
    non_ties = sum(delta != 0 for delta in differences)
    L.add(T, "T2.reporting.wins", wins, 0, "fs2_weak6_d256",
          "count of per_generator_seed_averaged > 04_ds_weak6 PS_4 per_generator", kind="derived")
    L.add(T, "T2.reporting.n", len(GENERATORS), 0, "fs2_weak6_d256", "len(per_generator_seed_averaged)", kind="derived")
    p = L.add(T, "T2.reporting.sign_p", binom_two_sided(wins, non_ties), 2, "fs2_weak6_d256",
              "two-sided exact binomial (p = 0.5) of the wins", kind="derived")
    d1 = {q: pool_standard(q)["fs1"][0] - pool_standard(q)["ps"][0] for q in POOLS_A21}
    L.add(T, "T2.pools.n", len(POOLS_A21), 0, "experts_cfg", "pools of Table A15", kind="derived",
          note=f"printed as a word ('{WORDS[len(POOLS_A21)]}') since paper v12")
    lo1 = L.add(T, "T2.pools.delta1_min", min(d1.values()), 3, "full_ladder", "min over pools of FS1 - best PS "
                f"({min(d1, key=d1.get)})", kind="derived")
    hi1 = L.add(T, "T2.pools.delta1_max", max(d1.values()), 3, "full_ladder", "max over pools of FS1 - best PS "
                f"({max(d1, key=d1.get)})", kind="derived")
    ld = {q: logo_pool(m, q)[0]["stats"]["fs1_full"]["mean_delta"] for q in ("arch6", "clip4", "strong4", "weak6")}
    L.add(T, "T2.logo.n_pools", len(ld), 0, "mitigation", "logo_arch6 and logo_boundary pools", kind="derived",
          note=f"printed as a word ('{WORDS[len(ld)]}') since paper v12")
    lol = L.add(T, "T2.logo.delta_min", min(ld.values()), 3, "mitigation", f"min over LOGO pools ({min(ld, key=ld.get)})",
                kind="derived")
    hil = L.add(T, "T2.logo.delta_max", max(ld.values()), 3, "mitigation", f"max over LOGO pools ({max(ld, key=ld.get)})",
                kind="derived")
    # version 15: the words of the last column that depend on the numbers
    if not (dist_a > 0 and dist_w > 0):
        raise SystemExit("Table 2: 'weaker baselines inflate the apparent advantage' needs both baseline inflations "
                         f"positive (arch6 {dist_a:+.3f}, weak6 {dist_w:+.3f})")
    if not (d2w > 0 and wins < len(GENERATORS)):                     # FS2: a warning
        word_warning("Table 2", "'the positive average has uneven support across generators' needs a positive mean "
                     f"and generators not won ({d2w:+.3f}, {wins}/{len(GENERATORS)})")
    if not (lo1 > 0 and lol < 0 < hil):
        raise SystemExit("Table 2: 'the magnitude, and under LOGO the sign, depends on the pool' needs positive "
                         f"standard-split margins ({lo1:+.3f} at least) and LOGO margins of both signs ({lol:+.3f} to "
                         f"{hil:+.3f})")
    rows = [
        f"Baseline strength & (i) Strong PS baseline ladder & Standard split: expanding "
        f"\\psmethod{{1}}-\\psmethod{{3}} to the tested ladder changes $\\Delta$ from ${braced(simp_a)}$ to "
        f"${braced(s['fs1_delta'])}$ on arch6, and from ${braced(fs1w - b13w)}$ to ${braced(fs1w - ps4w)}$ on weak6. "
        f"& Weaker baselines inflate the apparent advantage by $\\mathbf{{{braced(dist_a)}}}$ and "
        f"$\\mathbf{{{braced(dist_w)}}}$, respectively.",
        "\\midrule",
        f"Overlap control & (iv) Grouped evaluation controls & arch6: $\\Delta$ changes from "
        f"${braced(s['fs1_delta'])}$ on the standard split to ${braced(logo)}$ under LOGO. & A "
        f"$\\mathbf{{{braced(ov)}}}$ protocol difference; training data and PS selection also change, so this does "
        "not isolate overlap.",
        "\\midrule",
        f"Reporting granularity & (iv) Grouped evaluation controls & weak6, \\fsmethod{{2}}, standard split: mean "
        f"$\\Delta = {braced(d2w)}$, with wins on {wins}/{len(GENERATORS)} generators and sign-test "
        f"$p\\!=\\!{p:.2f}$. & The positive average has uneven support across generators.",
        "\\midrule",
        f"Pool construction & (iii) Axis-isolated pools & $\\Delta$ ranges from ${braced(lo1)}$ to ${braced(hi1)}$ "
        f"across {WORDS[len(POOLS_A21)]} standard-split pools, and from ${braced(lol)}$ to ${braced(hil)}$ across "
        f"{WORDS[len(ld)]} LOGO pools. & The magnitude, and under LOGO the sign, depends on the pool.",
    ]
    rows = [rows[0], rows[1], rows[6], rows[3], rows[2], rows[5], rows[4]]
    colspec = ">{\\raggedright\\arraybackslash}p{2.3cm}>{\\raggedright\\arraybackslash}p{2.3cm}" \
              ">{\\raggedright\\arraybackslash}p{4.4cm}>{\\raggedright\\arraybackslash}p{3.2cm}"
    return tab(colspec, [head("\\textbf{Evaluation choice}", "\\textbf{Protocol component}", "\\textbf{Comparison}",
                              "\\textbf{Change in reported conclusion}")], rows)


# ---- Appendix A and B

def p_methods(L):
    """Version 15: Table A1 as paper v12 prints it (method, description, inputs, determinism, key settings). Every
    value of version 14 is kept; notes say where the paper prints one differently."""
    T, cfg = "Table A1", experiments_cfg()

    def c(qid, key, digits=None, value=None, note=""):
        v = cfg_get(cfg, key) if value is None else value
        L.add(T, qid, v, digits if digits is not None else (1 if isinstance(v, float) else 0), "experiments_cfg", key,
              kind="config", note=note)
        return v
    as_one = lambda key: f"printed {cfg_get(cfg, key):g} since paper v12"  # noqa: E731
    as_word = lambda key: f"printed as a word ('{WORDS[cfg_get(cfg, key)]}-fold') since paper v12"  # noqa: E731
    moved = "printed in the row of PS_E since paper v12 (PS_F: the same selection procedure)"
    lr_c = c("A1.PS_2.C", "logreg_C", note=as_one("logreg_C"))
    h, drop = c("A1.PS_3.hidden", "mlp_hidden"), c("A1.PS_3.dropout", "mlp_dropout")
    ep = c("A1.PS_3.epochs", "mlp_epochs")
    xgb_cv = c("A1.PS_4.cv_folds", "xgb_cv_folds", note=as_word("xgb_cv_folds"))
    lr_cv = c("A1.PS_C.cv_folds", "logreg_cv_folds", note=as_word("logreg_cv_folds"))
    pat = c("A1.PS_F.patience", "mlp_es_patience", note=moved)
    max_ep = c("A1.PS_F.max_epochs", "mlp_es_max_epochs", note=moved)
    fs1_c = c("A1.FS_1.C", "raw_feature_logreg_C", note=as_one("raw_feature_logreg_C"))
    fs1_it = c("A1.FS_1.max_iter", "raw_feature_logreg_max_iter")
    hd, fs2_ep = cfg_get(cfg, "hidden_dims"), c("A1.FS_2.epochs", "epochs")
    L.add(T, "A1.FS_2.hidden_1", hd[0], 0, "experiments_cfg", "hidden_dims[0]", kind="config")
    L.add(T, "A1.FS_2.hidden_2", hd[1], 0, "experiments_cfg", "hidden_dims[1]", kind="config")
    k_folds = int(re.search(r"def cv_auc\(.*?n_folds: int = (\d+)\)", src("runner_a").read_text(encoding="utf-8"))[1])
    L.add(T, "A1.PS_K.cv_folds", k_folds, 0, "runner_a", "cv_auc(n_folds=...) default", kind="setting",
          note="StratifiedKFold inside TRAIN; the default is not stored in the run A output; printed as a word in the "
               f"notes of Table A1 since paper v12 ('{WORDS[k_folds]}-fold CV AUROC')")
    fd = load_json("xgbfeat_arch6")["params"]
    L.add(T, "A1.FS_D.n_estimators", fd["n_estimators"], 0, "xgbfeat_arch6", "params.n_estimators")
    L.add(T, "A1.FS_D.max_depth", fd["max_depth"], 0, "xgbfeat_arch6", "params.max_depth")
    rows = [
        "\\psmethod{1} & Probability averaging & Probabilities & Yes & Uniform average; no fitting",
        f"\\psmethod{{2}} & Logistic regression & Logits & Yes & L-BFGS, balanced class weights, $C\\!=\\!{lr_c:g}$",
        f"\\psmethod{{3}} & MLP & Logits & No & One hidden layer of width {h}; dropout {drop}; {ep} epochs",
        f"\\psmethod{{4}} & XGBoost & Logits & Yes & Hyperparameters selected by {WORDS[xgb_cv]}-fold CV within TRAIN",
        "\\midrule",
        "\\psmethod{A} & Calibrated averaging & Logits $\\rightarrow$ calibrated probabilities & Yes & One temperature "
        "per expert~\\cite{guo2017}, fitted by class-balanced NLL on SELECT",
        "\\psmethod{B} & Learned averaging weights & Probabilities & Yes & Simplex-constrained weights fitted by "
        "SELECT AUROC",
        f"\\psmethod{{C}} & Regularized logistic regression & Logits & Yes & Balanced class weights; penalty and $C$ "
        f"selected by {WORDS[lr_cv]}-fold CV within TRAIN",
        "\\psmethod{D} & MLP & Probabilities & No & \\psmethod{3} architecture and training settings",
        f"\\psmethod{{E}} & MLP with early stopping & Probabilities & No & \\psmethod{{3}} architecture; weight-decay "
        f"sweep; patience {pat}; maximum {max_ep} epochs",
        "\\psmethod{F} & MLP with early stopping & Logits & No & Same selection procedure as \\psmethod{E}",
        "\\midrule",
        "\\psmethod{G} & Pairwise disagreement & Logits and derived values & Yes & Logits plus all pairwise "
        "$|m_i - m_j|$; LR and XGBoost heads",
        "\\psmethod{H} & Uncertainty summaries & Calibrated probabilities and derived values & Yes & Per-expert "
        "probabilities, their mean, entropy summaries, and dispersion statistics; LR and XGBoost heads",
        "\\psmethod{I} & Quadratic interactions & Logits and derived values & Yes & Logits, $m_i^2$, and pairwise "
        "$m_i m_j$; LR and XGBoost heads",
        "\\psmethod{J} & Rank transformation & Logit margins & Yes & Per-expert empirical CDF fitted on TRAIN; LR and "
        "XGBoost heads",
        "\\psmethod{K} & CV-selected input map & \\psmethod{G}-\\psmethod{J} inputs & Yes & Selects the map by "
        "logistic-head CV AUROC within TRAIN; reports LR and XGBoost heads on that map",
        "\\midrule",
        f"\\fsmethod{{1}} & Logistic regression on raw features & Concatenated features & Yes & L-BFGS, balanced class "
        f"weights, $C\\!=\\!{fs1_c:g}$, maximum {fs1_it:,} iterations; no feature standardization",
        f"\\fsmethod{{2}} & Learned projections and MLP & Features & No & Per-expert Linear $\\rightarrow$ LayerNorm "
        f"$\\rightarrow$ ReLU; MLP $[{hd[0]}, {hd[1]}]$; {fs2_ep} epochs; best SELECT-AUROC checkpoint",
        "\\midrule",
        "\\fsmethod{A} & Learned projections and linear classifier & Features & No & Replaces \\fsmethod{2}'s MLP "
        "classifier with a linear classifier",
        "\\fsmethod{B} & PCA and logistic regression & Concatenated features & Yes & PCA fitted on TRAIN to $d$ total "
        "dimensions, followed by logistic regression",
        "\\fsmethod{C} & Random projections and MLP & Features & No & Fixed Gaussian projections followed by a trained "
        "MLP",
        f"\\fsmethod{{D}} & XGBoost on raw features & Concatenated features & Yes & Fixed settings; "
        f"{fd['n_estimators']} trees, maximum depth {fd['max_depth']}; no CV",
    ]
    groups = {
        r"\psmethod{1}": "Core PS baselines",
        r"\psmethod{A}": "Additional PS baselines",
        r"\psmethod{G}": "Richer PS inputs",
        r"\fsmethod{1}": "FS comparators",
        r"\fsmethod{A}": "FS controls",
    }
    grouped_rows = []
    for row in rows:
        prefix = row.split(" &", 1)[0]
        if prefix in groups:
            grouped_rows.append(r"\multicolumn{5}{l}{\textit{" + groups[prefix] + "}}")
        grouped_rows.append(row)
    rows = grouped_rows
    return tab("l>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{2.4cm}l"
               ">{\\raggedright\\arraybackslash}p{5.8cm}",
               [head("\\textbf{Method}", "\\textbf{Description}", "\\textbf{Inputs}", "\\textbf{Det.}",
                     "\\textbf{Key settings}")], rows)


# version 10: PS_H's input holds the E calibrated probabilities, their mean and these summaries (runs/richer_inputs.py,
# make_uncertainty_featmap: entropy of the mean, mean entropy, mutual information, variance, minimum, maximum, median,
# range); tests/test_builder_v10.py checks the count against the map itself
UNCERTAINTY_SUMMARIES = 8

# version 15: the blocks of Table A2 (a string opens a block under that heading) and the method names of paper v12
PS_ROWS_A2 = ("Core PS baselines",
              ("PS_1_prob_avg", "\\psmethod{1}: Probability averaging"),
              ("PS_2_logreg", "\\psmethod{2}: Logistic regression"),
              ("PS_3_mlp_logits", "\\psmethod{3}: MLP on logits"), ("PS_4_xgboost", "\\psmethod{4}: XGBoost"),
              "Additional PS baselines",
              ("PS_A_calibrated_avg", "\\psmethod{A}: Calibrated averaging"),
              ("PS_B_learned_weights", "\\psmethod{B}: Learned averaging weights"),
              ("PS_C_logreg_cv", "\\psmethod{C}: CV-tuned logistic regression"),
              ("PS_D_mlp_probs", "\\psmethod{D}: MLP on probabilities"),
              ("PS_E_mlp_probs_es", "\\psmethod{E}: MLP on probabilities + early stopping"),
              ("PS_F_mlp_logits_es", "\\psmethod{F}: MLP on logits + early stopping"))
RICH_ROWS_A2 = (("PS_11_disagreement", "G", "Disagreement"), ("PS_12_uncertainty_mi", "H", "Uncertainty summaries"),
                ("PS_13_quadratic", "I", "Quadratic interactions"), ("PS_14_rank_ecdf", "J", "Rank transformation"),
                ("PS_15_cv_best", "K", "CV-selected input"))


def ps_params(name, e):
    """Parameter count of a ladder method with e experts (2 logits each), from the reviewed implementations."""
    h = cfg_get(experiments_cfg(), "mlp_hidden")    # version 10: read from the configuration (64), not typed
    mlp = (2 * e) * h + h + h * 2 + 2                # PS_3/D/E/F: 2e inputs -> h hidden -> 2 outputs
    return {"PS_1_prob_avg": 0, "PS_2_logreg": 2 * e + 1, "PS_3_mlp_logits": mlp, "PS_4_xgboost": None,
            "PS_A_calibrated_avg": e, "PS_B_learned_weights": e, "PS_C_logreg_cv": 2 * e + 1, "PS_D_mlp_probs": mlp,
            "PS_E_mlp_probs_es": mlp, "PS_F_mlp_logits_es": mlp}[name]


def p_ladder(L):
    """Version 15: Table A2 as paper v12 prints it: blocks with headings, the methods' full names, the PS ceiling of the
    standard split in bold. The target-label diagnostic remains in the ledger and Appendix B text. The rows of version
    14 are written in the same order; the cell in bold adds one row per pool."""
    T, pools, cfg_pools = "Table A2", ("clip4", "arch6"), read_pools()
    E = {p: len(cfg_pools[p]) for p in pools}
    for p in pools:
        L.add(T, f"A2.{p}.experts", E[p], 0, "experts_cfg", f"len(pools.{p})", kind="derived")
    best = {p: load_json("full_ladder")["standard"][p]["ps_ceiling_method"] for p in pools}

    def cell(p, k, v):
        """The AUROC of method k on pool p, in bold when k is the pool's PS ceiling."""
        return f"\\textbf{{{v:.3f}}}" if k == best[p] else f"{v:.3f}"

    def block(title, first=False):
        return ([] if first else ["\\midrule"]) + [f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\"]
    rows = []
    for item in PS_ROWS_A2:
        if isinstance(item, str):
            rows += block(item, first=not rows)
            continue
        k, label = item
        cells = []
        for p in pools:
            v = L.add(T, f"A2.{p}.{k}", load_json(f"ds_{p}")["method_means"][k]["mean_avg8"], 3, f"ds_{p}",
                      f"method_means.{k}.mean_avg8")
            n = ps_params(k, E[p])
            if n is not None:
                L.add(T, f"A2.{p}.{k}.params", n, 0, "experts_cfg", f"from E = {E[p]} experts, 2 logits each",
                      kind="derived")
            cells += [cell(p, k, v), "-" if n is None else str(n)]
        rows.append(f"{label} & " + " & ".join(cells))
    rows += block("Richer PS inputs")
    for rung, letter, name in RICH_ROWS_A2:
        for head_name, suffix in (("logreg", "LR"), ("xgb", "XGB")):
            cells = []
            for p in pools:
                v = L.add(T, f"A2.{p}.{rung}_{head_name}", load_json(f"ladder_{p}")["extended_rungs_macro"][f"{rung}_{head_name}"],
                          3, f"ladder_{p}", f"extended_rungs_macro.{rung}_{head_name}")
                n = L.values.get(f"fl.params.{p}.PS_{letter}-LR") if suffix == "LR" else None
                if n is not None:
                    L.add(T, f"A2.{p}.{rung}_{head_name}.params", n, 0, f"ladder_{p}",
                          f"fl.params.{p}.PS_{letter}-LR (input-map size + intercept)", kind="derived")
                cells += [cell(p, f"{rung}_{head_name}", v), "-" if n is None else str(n)]
            what = "logistic regression" if suffix == "LR" else "XGBoost"
            rows.append(f"\\psmethod{{{letter}}}-{suffix}: {name} + {what} & " + " & ".join(cells))
    for p in pools:
        v = L.add(T, f"A2.{p}.oracle_domain", load_json(f"ladder_{p}")["extended_rungs_macro"]["PS_16_oracle_domain"], 3,
                  f"ladder_{p}", "extended_rungs_macro.PS_16_oracle_domain")
    fs1_cells, fs2_cells = [], []
    for p in pools:
        dim = load_json(f"pca_{p}")["features"]["dim"]
        v1 = L.add(T, f"A2.{p}.FS_1", load_json(f"ds_{p}_raw")["avg_auc_8"], 3, f"ds_{p}_raw", "avg_auc_8")
        L.add(T, f"A2.{p}.FS_1.params", dim + 1, 0, f"pca_{p}", "features.dim + 1", kind="derived",
              note=f"printed {k_count(dim + 1)}")
        f2 = load_json(f"fs2_{p}_d{32 if p == 'clip4' else 256}")
        v2 = L.add(T, f"A2.{p}.FS_2", f2["mean_avg8"], 3, f"fs2_{p}_d{32 if p == 'clip4' else 256}", "mean_avg8")
        L.add(T, f"A2.{p}.FS_2.params", f2["n_trainable_params"], 0, f"fs2_{p}_d{32 if p == 'clip4' else 256}",
              "n_trainable_params", note=f"printed {k_count(f2['n_trainable_params'])}")
        fs1_cells += [f"{v1:.3f}", k_count(dim + 1)]
        fs2_cells += [f"{v2:.3f}", k_count(f2["n_trainable_params"])]
    rows += block("FS comparators")
    rows += ["\\fsmethod{1}: Logistic regression on raw features & " + " & ".join(fs1_cells),
             "\\fsmethod{2}: Learned projections + MLP & " + " & ".join(fs2_cells)]
    fa, fb, fc = load_json("fsa_clip4"), load_json("fsb_clip4"), load_json("fsc_clip4")
    L.add(T, "A2.controls.d", fa["proj_dim"], 0, "fsa_clip4", "proj_dim",
          note="printed in the notes of Table A2 since paper v12")
    L.add(T, "A2.clip4.FS_A", fa["mean_avg8"], 3, "fsa_clip4", "mean_avg8")
    L.add(T, "A2.clip4.FS_A.params", fa["n_trainable_params"], 0, "fsa_clip4", "n_trainable_params",
          note=f"printed {k_count(fa['n_trainable_params'])}")
    L.add(T, "A2.clip4.FS_B", fb["avg_auc_8"], 3, "fsb_clip4", "avg_auc_8")
    L.add(T, "A2.clip4.FS_B.params", fb["proj_dim"] + 1, 0, "fsb_clip4", "proj_dim + 1 (PCA weights + intercept)",
          kind="derived")
    L.add(T, "A2.clip4.FS_C", fc["mean_avg8"], 3, "fsc_clip4", "mean_avg8")
    L.add(T, "A2.clip4.FS_C.params", fc["n_trainable_params"], 0, "fsc_clip4", "n_trainable_params",
          note=f"printed {k_count(fc['n_trainable_params'])}")
    fd = {p: L.add(T, f"A2.{p}.FS_D", load_json(f"xgbfeat_{p}")["avg_auc_8"], 3, f"xgbfeat_{p}", "avg_auc_8") for p in pools}
    rows += block("FS controls")
    rows += [f"\\fsmethod{{A}}: Learned projections + linear classifier & {fa['mean_avg8']:.3f} & "
             f"{k_count(fa['n_trainable_params'])} & - & -",
             f"\\fsmethod{{B}}: PCA + logistic regression & {fb['avg_auc_8']:.3f} & {fb['proj_dim'] + 1} & - & -",
             f"\\fsmethod{{C}}: Random projections + MLP & {fc['mean_avg8']:.3f} & {k_count(fc['n_trainable_params'])} "
             "& - & -",
             f"\\fsmethod{{D}}: XGBoost on features & {fd['clip4']:.3f} & - & {fd['arch6']:.3f} & -"]
    for p in pools:                    # version 15: the cell in bold is the PS ceiling of the full-ladder statistics
        v = L.values.get(f"A2.{p}.{best[p]}")
        ceiling = load_json("full_ladder")["standard"][p]["ps_ceiling"]
        if v is None or abs(v - ceiling) > 1e-12:
            raise SystemExit(f"Table A2: the PS ceiling of {p} ({best[p]}, {ceiling:.6f}) is not a cell of the table")
        L.add(T, f"A2.{p}.ps_ceiling_method", best[p], 0, "full_ladder", f"standard.{p}.ps_ceiling_method",
              note="Table A2: the cell in bold (the PS ceiling of the standard split), since paper v12")
    params = "\\multicolumn{1}{l}{\\textbf{Params}$^{*}$}"
    hd = [head(" ", f"\\multicolumn{{2}}{{c}}{{\\textbf{{clip4 ({E['clip4']} experts)}}}}",
               f"\\multicolumn{{2}}{{c}}{{\\textbf{{arch6 ({E['arch6']} experts)}}}}"),
          "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
          head("\\textbf{Method}", mc("Macro AUROC"), params, mc("Macro AUROC"), params)]
    return tab("lrrrr", hd, rows)


def per_generator_table(L, T, pool, d):
    ds, rf, f2 = load_json(f"ds_{pool}"), load_json(f"ds_{pool}_raw"), load_json(f"fs2_{pool}_d{d}")
    L.add(T, f"{T[6:]}.{pool}.FS_2.d", f2["proj_dim"], 0, f"fs2_{pool}_d{d}", "proj_dim")
    specs = (("\\psmethod{1}: Prob.\\ avg.", "PS_1", f"ds_{pool}", ds["method_means"]["PS_1_prob_avg"]["per_generator"],
              "method_means.PS_1_prob_avg.per_generator", ds["method_means"]["PS_1_prob_avg"]["mean_avg8"],
              "method_means.PS_1_prob_avg.mean_avg8"),
             ("\\psmethod{4}: XGBoost", "PS_4", f"ds_{pool}", ds["method_means"]["PS_4_xgboost"]["per_generator"],
              "method_means.PS_4_xgboost.per_generator", ds["method_means"]["PS_4_xgboost"]["mean_avg8"],
              "method_means.PS_4_xgboost.mean_avg8"),
             ("\\fsmethod{1}: Raw-feat LogReg", "FS_1", f"ds_{pool}_raw", rf["per_generator"], "per_generator",
              rf["avg_auc_8"], "avg_auc_8"),
             (f"\\fsmethod{{2}}: MLP ($d\\!=\\!{f2['proj_dim']}$)", "FS_2", f"fs2_{pool}_d{d}",
              f2["per_generator_seed_averaged"], "per_generator_seed_averaged", f2["mean_avg8"], "mean_avg8"))
    rows = []
    for label, short, source, per_gen, key, avg, avg_key in specs:
        for g in GENERATORS:
            L.add(T, f"{T[6:]}.{short}.{g}", per_gen[g], 3, source, f"{key}.{g}")
        L.add(T, f"{T[6:]}.{short}.avg", avg, 3, source, avg_key)
        rows.append(f"{label} & " + " & ".join(f"{per_gen[g]:.3f}" for g in GENERATORS) + f" & {avg:.3f}")
    return tab("lrrrrrrrr r", [head("\\textbf{Method}", *(mc(x) for x in GEN_SHORT), mc("Macro AUROC"))], rows)


def p_arch6(L):
    return per_generator_table(L, "Table A3", "arch6", 256)


def p_clip4(L):
    return per_generator_table(L, "Table A4", "clip4", 32)


def p_dim_sweep(L):
    """Version 15: Table A5 as paper v12 prints it; the seeds' SD is no longer printed (its rows are kept)."""
    T = "Table A5"
    ceil = load_json("full_ladder")["standard"]["clip4"]["ps_ceiling"]
    n_exp = len(read_pools()["clip4"])
    rows = []
    for d in (32, 64, 256):
        x = load_json(f"fs2_clip4_d{d}")
        src_ = f"fs2_clip4_d{d}"
        L.add(T, f"A5.d{d}.d", x["proj_dim"], 0, src_, "proj_dim")
        L.add(T, f"A5.d{d}.combined", n_exp * x["proj_dim"], 0, src_, f"{n_exp} experts x proj_dim", kind="derived")
        L.add(T, f"A5.d{d}.auroc", x["mean_avg8"], 3, src_, "mean_avg8")
        L.add(T, f"A5.d{d}.std", x["std_avg8"], 3, src_, "std_avg8", note="not printed since paper v12")
        dv = L.add(T, f"A5.d{d}.delta", x["mean_avg8"] - ceil, 3, src_, "mean_avg8 - full_ladder standard.clip4.ps_ceiling",
                   kind="derived")
        L.add(T, f"A5.d{d}.params", x["n_trainable_params"], 0, src_, "n_trainable_params",
              note=f"printed {k_count(x['n_trainable_params'])}")
        rows.append(f"{x['proj_dim']} & {n_exp * x['proj_dim']:,} & {x['mean_avg8']:.3f} & {plain(dv)} & "
                    f"{k_count(x['n_trainable_params'])}")
    return tab("rrrrr", [head(mc("Per-expert $d$"), mc("Combined dimension"), mc("Macro AUROC"),
                              mc("$\\Delta$ vs PS ceiling"), mc("Params"))], rows)


def p_fs2_stability(L):
    """Version 15: Table A6 as paper v12 prints it: the TEST minimum and maximum in one column; the width of that range
    is no longer printed (its rows are kept)."""
    T = "Table A6"
    rows = []
    for pool, d in (("clip4", 32), ("clip4", 64), ("clip4", 256), ("arch6", 256)):
        x, src_ = load_json(f"fs2_{pool}_d{d}"), f"fs2_{pool}_d{d}"
        a, v = [r["avg_auc_8"] for r in x["per_seed"]], [r["val_auc"] for r in x["per_seed"]]
        q = f"A6.{pool}.d{d}"
        L.add(T, f"{q}.d", x["proj_dim"], 0, src_, "proj_dim")
        L.add(T, f"{q}.test_mean", statistics.mean(a), 3, src_, "mean(per_seed[*].avg_auc_8)", kind="derived")
        L.add(T, f"{q}.test_std", statistics.pstdev(a), 3, src_, "pstdev(per_seed[*].avg_auc_8)", kind="derived")
        L.add(T, f"{q}.test_range", max(a) - min(a), 3, src_, "max - min of per_seed[*].avg_auc_8", kind="derived",
              note="not printed since paper v12 (the column prints the TEST minimum and maximum)")
        L.add(T, f"{q}.test_min", min(a), 3, src_, "min(per_seed[*].avg_auc_8)", kind="derived")
        L.add(T, f"{q}.test_max", max(a), 3, src_, "max(per_seed[*].avg_auc_8)", kind="derived")
        L.add(T, f"{q}.select_mean", statistics.mean(v), 3, src_, "mean(per_seed[*].val_auc)", kind="derived")
        gap = L.add(T, f"{q}.select_minus_test", statistics.mean(v) - statistics.mean(a), 3, src_,
                    "mean(val_auc) - mean(avg_auc_8)", kind="derived")
        rows.append(f"{pool} & {x['proj_dim']} & {statistics.mean(a):.3f} & {statistics.pstdev(a):.3f} & "
                    f"{min(a):.3f}-{max(a):.3f} & {statistics.mean(v):.3f} & ${plain(gap)}$")
    return tab("lrrrrrr", [head("\\textbf{Pool}", mc("$d$"), mc("TEST mean"), mc("TEST SD"), mc("TEST min-max"),
                                mc("SELECT mean"), mc("SELECT$-$TEST"))], rows)


def p_bottleneck(L):
    T = "Table A7"
    rows = []
    for pool in ("arch6", "clip4"):
        p = load_json(f"pca_{pool}")
        for rep, label, pr_digits in (("logits", "Concatenated logits", 2), ("features", "Concatenated features", 1)):
            x = p[rep]
            for k, digits in (("dim", 0), ("pr", pr_digits), ("k95", 0), ("k99", 0)):
                L.add(T, f"A7.{pool}.{rep}.{k}", x[k], digits, f"pca_{pool}", f"{rep}.{k}")
            rows.append(f"{pool} & {label} & {x['dim']:,} & {x['pr']:.{pr_digits}f} & {x['k95']:,} & {x['k99']:,}")
    return tab("llrrrr", [head("\\textbf{Pool}", "\\textbf{Representation}", mc("Input dimension"), mc("PR"),
                               mc("$k_{95}$"), mc("$k_{99}$"))], rows)


def p_rq2(L):
    """Version 15: Table A8 as paper v12 prints it: the method names of Table A2, FS_D's symbol, FS1's parameters as
    10.0K, and the parameter counts of the 12-value controls marked by a footnote."""
    T, m, w, cost = "Table A8", load_json("mitigation"), load_json("width"), load_json("cost")
    mech, mc4 = m["mechanism_logo"]["mean_auroc"], m["mechanism_logo_clip4"]["mean_auroc"]
    widths = cost["widths_and_storage"]["pool_totals"]
    n_pred = L.add(T, "A8.inputs.predictions", widths["prediction_floats"], 0, "cost",
                   "widths_and_storage.pool_totals.prediction_floats")
    dim = L.add(T, "A8.inputs.features", load_json("pca_arch6")["features"]["dim"], 0, "pca_arch6", "features.dim")
    f2 = load_json("fs2_arch6_d256")

    def a(q, v, source, key, kind="json"):
        return L.add(T, f"A8.{q}", v, 3, source, key, kind=kind)
    rows = [
        f"\\psmethod{{4}}: XGBoost on logits & {n_pred} & - & "
        f"{a('PS_4.std.arch6', load_json('ds_arch6')['method_means']['PS_4_xgboost']['mean_avg8'], 'ds_arch6', 'method_means.PS_4_xgboost.mean_avg8'):.3f} & "
        f"{a('PS_4.std.clip4', load_json('ds_clip4')['method_means']['PS_4_xgboost']['mean_avg8'], 'ds_clip4', 'method_means.PS_4_xgboost.mean_avg8'):.3f} & "
        f"{a('PS_4.logo.arch6', mech['PS_4_xgboost'], 'mitigation', 'mechanism_logo.mean_auroc.PS_4_xgboost'):.3f} & "
        f"{a('PS_4.logo.clip4', mc4['PS_4_xgboost'], 'mitigation', 'mechanism_logo_clip4.mean_auroc.PS_4_xgboost'):.3f}",
        f"\\fsmethod{{1}}: Logistic regression on raw features & {dim:,} & {k_count(dim + 1)} & "
        f"{a('FS_1.std.arch6', load_json('ds_arch6_raw')['avg_auc_8'], 'ds_arch6_raw', 'avg_auc_8'):.3f} & "
        f"{a('FS_1.std.clip4', load_json('ds_clip4_raw')['avg_auc_8'], 'ds_clip4_raw', 'avg_auc_8'):.3f} & "
        f"{a('FS_1.logo.arch6', m['logo_arch6']['fs1'], 'mitigation', 'logo_arch6.fs1'):.3f} & "
        f"{a('FS_1.logo.clip4', m['logo_boundary']['clip4']['fs1'], 'mitigation', 'logo_boundary.clip4.fs1'):.3f}",
        f"\\fsmethod{{2}}: Learned projections + MLP & {dim:,} & {k_count(f2['n_trainable_params'])} & "
        f"{a('FS_2.std.arch6', f2['mean_avg8'], 'fs2_arch6_d256', 'mean_avg8'):.3f} & "
        f"{a('FS_2.std.clip4', load_json('fs2_clip4_d32')['mean_avg8'], 'fs2_clip4_d32', 'mean_avg8'):.3f} & "
        f"{a('FS_2.logo.arch6', m['logo_arch6']['fs2'], 'mitigation', 'logo_arch6.fs2'):.3f} & "
        f"{a('FS_2.logo.clip4', m['logo_boundary']['clip4']['fs2'], 'mitigation', 'logo_boundary.clip4.fs2'):.3f}",
        f"\\fsmethod{{D}}: XGBoost on features & {dim:,} & - & "
        f"{a('FS_D.std.arch6', load_json('xgbfeat_arch6')['avg_auc_8'], 'xgbfeat_arch6', 'avg_auc_8'):.3f} & "
        f"{a('FS_D.std.clip4', load_json('xgbfeat_clip4')['avg_auc_8'], 'xgbfeat_clip4', 'avg_auc_8'):.3f} & "
        f"{a('FS_D.logo.arch6', mech['FS_D_xgb_features'], 'mitigation', 'mechanism_logo.mean_auroc.FS_D_xgb_features'):.3f} & "
        f"{a('FS_D.logo.clip4', mc4['FS_D_xgb_features'], 'mitigation', 'mechanism_logo_clip4.mean_auroc.FS_D_xgb_features'):.3f}",
    ]
    L.add(T, "A8.FS_1.params", dim + 1, 0, "pca_arch6", "features.dim + 1", kind="derived",
          note=f"printed {k_count(dim + 1)}")
    L.add(T, "A8.FS_2.params", f2["n_trainable_params"], 0, "fs2_arch6_d256", "n_trainable_params",
          note=f"printed {k_count(f2['n_trainable_params'])}")
    for key, label, mkey in (("pca_sweep", "PCA", "FS_pca2_logreg"), ("probe_mirror", "label-fitted", "FS_probe2_logreg")):
        e = w[key]["d2"]
        n = L.add(T, f"A8.{key}.inputs", e["coords_total"], 0, "width", f"{key}.d2.coords_total")
        L.add(T, f"A8.{key}.params", e["coords_total"] + 1, 0, "width", f"{key}.d2.coords_total + 1", kind="derived")
        v = a(f"{key}.std.arch6", e["fs_macro"], "width", f"{key}.d2.fs_macro")
        lv = a(f"{key}.logo.arch6", mech[mkey], "mitigation", f"mechanism_logo.mean_auroc.{mkey}")
        rows.append(f"\\fsmethod{{1}} on {n} {label} values & {n} & {n + 1}$^{{*}}$ & {v:.3f} & - & {lv:.3f} & -")
    hd = [head(" ", " ", " ", "\\multicolumn{2}{l}{\\textbf{Standard split}}", "\\multicolumn{2}{l}{\\textbf{LOGO}}"),
          "\\cmidrule(lr){4-5} \\cmidrule(lr){6-7}",
          head("\\textbf{Combiner}", mc("Inputs (arch6)"), mc("Params (arch6)"), mc("arch6"), mc("clip4"), mc("arch6"),
               mc("clip4"))]
    return tab("lrrrrrr", hd, rows)


def p_loo(L):
    """Version 15: Table A9 as paper v12 prints it: each pool opens with its full-pool reference row (a new row of the
    ledger), and the expert removed names the variant (no separate column)."""
    T = "Table A9"
    rows = []
    for pool, d in (("clip4", 32), ("arch6", 256)):
        x = load_json(f"loo_{pool}")
        L.add(T, f"A9.{pool}.d", x["proj_dim"], 0, f"loo_{pool}", "proj_dim")
        full = L.add(T, f"A9.{pool}.full.auroc", x["full_auc_mean"], 3, f"loo_{pool}", "full_auc_mean",
                     note="Table A9: the full-pool reference row, since paper v12")
        if rows:
            rows.append("\\midrule")
        rows.append(f"{pool} & None: full pool & {full:.3f} & -")
        for item in x["loo"]:
            e = item["removed"]
            v = L.add(T, f"A9.{pool}.{e}.auroc", item["avg8_mean"], 3, f"loo_{pool}", f"loo[{e}].avg8_mean")
            dm = L.add(T, f"A9.{pool}.{e}.marginal", item["delta_vs_full"], 3, f"loo_{pool}", f"loo[{e}].delta_vs_full")
            if abs((full - v) - dm) > 1e-12:
                raise SystemExit(f"Table A9: {pool} without {e}: delta_vs_full is not full_auc_mean - avg8_mean")
            shown = "0.000" if f"{abs(dm):.3f}" == "0.000" else plain(dm)
            rows.append(f"{pool} & {e} & {v:.3f} & {shown}")
    return tab("llrr", [head("\\textbf{Pool}", "\\textbf{Expert removed}", mc("Macro AUROC after retraining"),
                             mc("Full $-$ reduced AUROC"))], rows)


# ---- Appendix F

def p_logo(L):
    """Version 15: Table A10 as paper v12 prints it: the held-out generator, its PS ceiling and the symbol of the
    method selected."""
    T, m = "Table A10", load_json("mitigation")["logo_arch6"]
    per = {f["held_out"]: f for f in load_json("cd_logo_arch6")["per_fold"]}
    s1, s2 = m["stats"]["fs1_full"], m["stats"]["fs2_full"]
    rows = []
    for i, g in enumerate(m["folds"]):
        ce = L.add(T, f"A10.{g}.best_ps", m["ceiling_full_per_fold"][i], 3, "mitigation", f"logo_arch6.ceiling_full_per_fold[{i}]")
        L.add(T, f"A10.{g}.best_ps_method", m["ceiling_method_per_fold"][i], 0, "mitigation",
              f"logo_arch6.ceiling_method_per_fold[{i}]")
        f1 = L.add(T, f"A10.{g}.fs1", per[g]["raw_feat_auc"], 3, "cd_logo_arch6", f"per_fold[{g}].raw_feat_auc")
        d1 = L.add(T, f"A10.{g}.delta1", s1["per_fold"][i], 3, "mitigation", f"logo_arch6.stats.fs1_full.per_fold[{i}]")
        f2 = L.add(T, f"A10.{g}.fs2", _fs2(per[g]), 3, "cd_logo_arch6", f"per_fold[{g}].fs_mlp_mean")
        d2 = L.add(T, f"A10.{g}.delta2", s2["per_fold"][i], 3, "mitigation", f"logo_arch6.stats.fs2_full.per_fold[{i}]")
        rows.append(f"{GEN_LABEL[g]} & {ce:.3f} & {ps_symbol(m['ceiling_method_per_fold'][i])} & {f1:.3f} & {sd(d1)} & "
                    f"{f2:.3f} & {sd(d2)}")
    L.add(T, "A10.avg.best_ps", m["ceiling_full"], 3, "mitigation", "logo_arch6.ceiling_full")
    L.add(T, "A10.avg.fs1", m["fs1"], 3, "mitigation", "logo_arch6.fs1")
    L.add(T, "A10.avg.delta1", s1["mean_delta"], 3, "mitigation", "logo_arch6.stats.fs1_full.mean_delta")
    L.add(T, "A10.avg.fs2", m["fs2"], 3, "mitigation", "logo_arch6.fs2")
    L.add(T, "A10.avg.delta2", s2["mean_delta"], 3, "mitigation", "logo_arch6.stats.fs2_full.mean_delta")
    rows.append("\\midrule")
    rows.append(f"Macro AUROC & {m['ceiling_full']:.3f} & - & {m['fs1']:.3f} & {sd(s1['mean_delta'])} & "
                f"{m['fs2']:.3f} & {sd(s2['mean_delta'])}")
    return tab("lrlrrrr", [head("\\textbf{Held-out generator}", mc("PS ceiling"), "\\textbf{Selected PS method}",
                                mc("\\fsmethod{1}"), mc("$\\Delta_1$"), mc("\\fsmethod{2}"), mc("$\\Delta_2$"))], rows)


def p_logo_inferential(L):
    """Version 15: Table A11 with the names of paper v12 (comparison, sign-test p, bootstrap CI)."""
    T, st = "Table A11", load_json("mitigation")["logo_arch6"]["stats"]
    rows = []
    for k, label in (("fs1_full", "\\fsmethod{1} versus PS ceiling"), ("fs2_full", "\\fsmethod{2} versus PS ceiling")):
        s = st[k]
        for q, digits in (("mean_delta", 3), ("sign_p", 3), ("wilcoxon_p", 3), ("signflip_p", 3)):
            L.add(T, f"A11.{k}.{q}", s[q], digits, "mitigation", f"logo_arch6.stats.{k}.{q}")
        L.add(T, f"A11.{k}.ci_lo", s["ci95"][0], 3, "mitigation", f"logo_arch6.stats.{k}.ci95[0]")
        L.add(T, f"A11.{k}.ci_hi", s["ci95"][1], 3, "mitigation", f"logo_arch6.stats.{k}.ci95[1]")
        rows.append(f"{label} & {sd(s['mean_delta'])} & {s['sign_p']:.3f} & {s['wilcoxon_p']:.3f} & "
                    f"{s['signflip_p']:.3f} & $[{sm(s['ci95'][0])}, {sm(s['ci95'][1])}]$")
    return tab("lrrrrr", [head("\\textbf{Comparison}", mc("$\\Delta$"), mc("Sign-test $p$"), mc("Wilcoxon $p$"),
                               mc("Sign-flip $p$"), mc("Bootstrap 95\\% CI"))], rows)


def p_mde(L):
    """Version 15: Table A12 as paper v12 prints it (the sigmas without labels); the power target in its header gets a
    row."""
    T, x = "Table A12", load_json("power_mde")
    rows = []
    for i, r in enumerate(x["sigmas"]):
        L.add(T, f"A12.sigma{i}", r["sigma"], 2, "power_mde", f"sigmas[{i}].sigma")
        v = L.add(T, f"A12.sigma{i}.mde", r["table_a12"], 3, "power_mde", f"sigmas[{i}].table_a12",
                  note="smallest 0.005-grid effect with bootstrap-CI power >= 0.80 (precision check, 100,000 sets)")
        rows.append(f"${r['sigma']:.2f}$ & ${plain(v)}$")
    pct = 100 * x["target_power"]
    L.add(T, "A12.target_power_pct", pct, 0, "power_mde", "100 x target_power", kind="derived",
          note=f"Table A12 header: the mean effect giving at least {pct:.0f}% detection, since paper v12")
    return tab("lr", [head("\\textbf{Fold-difference SD, $\\sigma$}",
                           mc(f"Mean effect giving at least {pct:.0f}\\% detection"))], rows)


def p_weighting(L):
    """Version 15: Table A13 with the names of paper v12 (unweighted macro; pool and protocol)."""
    T, s, m = "Table A13", load_json("full_ladder")["standard"], load_json("mitigation")["logo_arch6"]
    rows = []
    for pool in ("arch6", "clip4"):
        x = s[pool]
        for fs in ("fs1", "fs2"):
            vals = []
            for q, key in ((fs, fs), ("ps", "ps_ceiling"), ("delta", f"{fs}_delta"), (f"w_{fs}", f"weighted_{fs}"),
                           ("w_ps", "weighted_ps_ceiling"), ("w_delta", f"weighted_{fs}_delta")):
                vals.append(L.add(T, f"A13.{pool}.{fs}.{q}", x[key], 3, "full_ladder", f"standard.{pool}.{key}"))
            rows.append(f"{pool}, standard & \\fsmethod{{{fs[-1]}}} & {vals[0]:.3f} & {vals[1]:.3f} & "
                        f"{plain(vals[2])} & {vals[3]:.3f} & {vals[4]:.3f} & {plain(vals[5])}")
    for fs, k in (("fs1", "fs1_full"), ("fs2", "fs2_full")):
        v = [L.add(T, f"A13.logo.{fs}.fs", m[fs], 3, "mitigation", f"logo_arch6.{fs}"),
             L.add(T, f"A13.logo.{fs}.ps", m["ceiling_full"], 3, "mitigation", "logo_arch6.ceiling_full"),
             L.add(T, f"A13.logo.{fs}.delta", m["stats"][k]["mean_delta"], 3, "mitigation", f"logo_arch6.stats.{k}.mean_delta"),
             L.add(T, f"A13.logo.{fs}.w_fs", m["weighted"][fs], 3, "mitigation", f"logo_arch6.weighted.{fs}"),
             L.add(T, f"A13.logo.{fs}.w_ps", m["weighted"]["ceiling_full"], 3, "mitigation", "logo_arch6.weighted.ceiling_full")]
        v.append(L.add(T, f"A13.logo.{fs}.w_delta", m["weighted"][fs] - m["weighted"]["ceiling_full"], 3, "mitigation",
                       f"logo_arch6.weighted.{fs} - weighted.ceiling_full", kind="derived"))
        rows.append(f"arch6, LOGO & \\fsmethod{{{fs[-1]}}} & {v[0]:.3f} & {v[1]:.3f} & {plain(v[2])} & {v[3]:.3f} & "
                    f"{v[4]:.3f} & {plain(v[5])}")
    hd = [head(" ", " ", "\\multicolumn{3}{l}{\\textbf{Unweighted macro}}",
               "\\multicolumn{3}{l}{\\textbf{Sample-weighted macro}}"),
          "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
          head("\\textbf{Pool and protocol}", "\\textbf{Variant}", mc("FS"), mc("PS"), mc("$\\Delta$"), mc("FS"),
               mc("PS"), mc("$\\Delta$"))]
    return tab("llrrrrrr", hd, rows)


def p_strict(L):
    """Version 15: Table A14 as paper v12 prints it (condition, comparator, bootstrap CI and interpretation)."""
    T = "Table A14"
    std, cdl = load_json("full_ladder")["standard"]["arch6"], load_json("mitigation")["logo_arch6"]["stats"]["fs1_full"]
    b = load_json("strict")
    sc, strict = b["arm_strict"]["comparison"], b["arm_strict"]
    words = f"printed as words ('{WORDS[cdl['wins']]} of {WORDS[cdl['n']]} folds won') since paper v12"
    L.add(T, "A14.standard.delta", std["fs1_delta"], 3, "full_ladder", "standard.arch6.fs1_delta")
    L.add(T, "A14.standard.ci_lo", std["fs1_ci95"][0], 3, "full_ladder", "standard.arch6.fs1_ci95[0]")
    L.add(T, "A14.standard.ci_hi", std["fs1_ci95"][1], 3, "full_ladder", "standard.arch6.fs1_ci95[1]")
    L.add(T, "A14.logo.delta", cdl["mean_delta"], 3, "mitigation", "logo_arch6.stats.fs1_full.mean_delta")
    L.add(T, "A14.logo.ci_lo", cdl["ci95"][0], 3, "mitigation", "logo_arch6.stats.fs1_full.ci95[0]")
    L.add(T, "A14.logo.ci_hi", cdl["ci95"][1], 3, "mitigation", "logo_arch6.stats.fs1_full.ci95[1]")
    L.add(T, "A14.logo.wins", cdl["wins"], 0, "mitigation", "logo_arch6.stats.fs1_full.wins", note=words)
    L.add(T, "A14.logo.n", cdl["n"], 0, "mitigation", "logo_arch6.stats.fs1_full.n", note=words)
    L.add(T, "A14.strict.delta", sc["delta"], 3, "strict", "arm_strict.comparison.delta")
    L.add(T, "A14.strict.ci_lo", sc["generator_bootstrap"]["ci_95"][0], 3, "strict",
          "arm_strict.comparison.generator_bootstrap.ci_95[0]")
    L.add(T, "A14.strict.ci_hi", sc["generator_bootstrap"]["ci_95"][1], 3, "strict",
          "arm_strict.comparison.generator_bootstrap.ci_95[1]")
    L.add(T, "A14.strict.fs1", strict["fs1_macro"], 3, "strict", "arm_strict.fs1_macro")
    L.add(T, "A14.strict.ps", strict["best_ps_macro"], 3, "strict", "arm_strict.best_ps_macro")
    L.add(T, "A14.matched.seeds", len(b["arm_matched"]), 0, "strict", "len(arm_matched)",
          note=f"printed as a word ('{WORDS[len(b['arm_matched'])]} removal seeds') since paper v12")
    L.add(T, "A14.matched.delta", b["summary"]["delta_matched_mean"], 3, "strict", "summary.delta_matched_mean")
    rows = [
        f"Standard split & {sd(std['fs1_delta'])} & $[{sm(std['fs1_ci95'][0])}, {sm(std['fs1_ci95'][1])}]$ & "
        "Overlap-affected; secondary",
        f"LOGO & {sd(cdl['mean_delta'])} & $[{sm(cdl['ci95'][0])}, {sm(cdl['ci95'][1])}]$ & Primary; "
        f"{WORDS[cdl['wins']]} of {WORDS[cdl['n']]} folds won",
        f"Strict path-disjoint refit & {sd(sc['delta'])} & $[{sm(sc['generator_bootstrap']['ci_95'][0])}, "
        f"{sm(sc['generator_bootstrap']['ci_95'][1])}]$ & \\fsmethod{{1}} {strict['fs1_macro']:.3f}, PS "
        f"{strict['best_ps_macro']:.3f}; substantial coverage loss",
        f"Matched random removal & {sd(b['summary']['delta_matched_mean'])} & - & Mean across "
        f"{WORDS[len(b['arm_matched'])]} removal seeds; sample and class counts matched",
    ]
    return tab("lrl>{\\raggedright\\arraybackslash}p{5cm}",
               [head("\\textbf{Condition}", mc("\\fsmethod{1} $-$ PS comparator"), "\\textbf{Bootstrap 95\\% CI}",
                     "\\textbf{Interpretation}")], rows)


# version 15: the compositions of Table A15 as paper v12 prints them; composition_problem checks each one against the
# configured pools and the individual experts' results
COMPOSITION = {
    "clip4": lambda c: f"CLIP-large, {WORDS[c['n_regimes']]} training regimes",
    "arch6": lambda c: f"{WORDS[c['n']].capitalize()} detector variants, face-swap-trained",
    "mixed6": lambda c: "Mixed architectures and training regimes",
    "merged9": lambda c: "clip4 $\\cup$ arch6; shared expert counted once",
    "strong4": lambda c: f"{WORDS[c['n']].capitalize()} high-performing CLIP-based experts",
    "balanced8": lambda c: (f"{WORDS[c['n_regimes']].capitalize()} regimes $\\times$ {WORDS[c['n_archs']]} "
                            "architectures"),
    "full21": lambda c: "All available experts in the study",
    "weak6": lambda c: f"{WORDS[c['n']].capitalize()} lowest-performing individual experts",
    "low_het4": lambda c: "Face-swap-trained Xception-family variants",
}


def composition_problem(pool, pools, h):
    """Version 15: why the composition that Table A15 prints for `pool` does not hold, or None (`pools`: the
    configured pools; `h`: the individual experts' results of Table A16)."""
    ex = pools[pool]
    arch = {e: e.split("-", 1)[1] for e in h}
    regime = {e: e.split("-", 1)[0] for e in h}
    checks = {
        "clip4": (all(arch[e] == "clip-large" for e in ex), "not every expert is CLIP-large"),
        "arch6": (all(regime[e] == "fs" for e in ex) and len({arch[e] for e in ex}) == len(ex),
                  "not distinct architectures, all trained on face swap"),
        "mixed6": (len({arch[e] for e in ex}) > 1 and len({regime[e] for e in ex}) > 1,
                   "not mixed in both architecture and training regime"),
        "merged9": (set(ex) == set(pools["clip4"]) | set(pools["arch6"])
                    and len(set(pools["clip4"]) & set(pools["arch6"])) == 1, "not clip4 and arch6 sharing one expert"),
        "strong4": (all("clip" in arch[e] for e in ex), "not every expert is CLIP-based"),
        "balanced8": (len({regime[e] for e in ex}) * len({arch[e] for e in ex}) == len(set(ex)),
                      "not every regime crossed with every architecture"),
        "full21": (set(ex) == set(h), "not every expert of the study"),
        "weak6": (set(ex) == set(sorted(h, key=lambda e: h[e]["avg_auc_8"])[:len(ex)]),
                  "not the experts with the lowest individual macro AUROC"),
        "low_het4": (all(regime[e] == "fs" and "clip" not in arch[e] for e in ex),
                     "not face-swap-trained Xception-family experts"),
    }
    ok, why = checks[pool]
    return None if ok else why


def p_fs1_pools(L):
    """Version 15: Table A15 as paper v12 prints it: the compositions in words (checked by composition_problem) and the
    PS comparator. The AUROC ranges of strong4 and weak6 are printed in App. G since paper v12 (rows kept)."""
    T, pools = "Table A15", read_pools()
    h = load_json("appendix_h_cells")["per_expert"]
    rows = []
    for pool in POOLS_A21:
        ex = pools[pool]
        why = composition_problem(pool, pools, h)
        if why:
            raise SystemExit(f"Table A15: the composition printed for {pool} does not hold: {why}")
        n = L.add(T, f"A15.{pool}.n", len(ex), 0, "experts_cfg", f"len(pools.{pool})", kind="derived")
        c = {"n": n, "n_clip_large": sum(e.endswith("clip-large") for e in ex),
             "n_regimes": len({e.split("-")[0] for e in ex}), "n_archs": len({e.split("-", 1)[1] for e in ex})}
        if pool == "clip4":
            L.add(T, "A15.clip4.n_clip_large", c["n_clip_large"], 0, "experts_cfg", "pools.clip4 (CLIP-large)",
                  kind="derived", note="not printed since paper v12 (the composition names CLIP-large and the regimes)")
            L.add(T, "A15.clip4.n_regimes", c["n_regimes"], 0, "experts_cfg", "pools.clip4 (training regimes)",
                  kind="derived",
                  note=f"printed as a word ('{WORDS[c['n_regimes']]} training regimes') since paper v12")
        if pool == "balanced8":
            L.add(T, "A15.balanced8.n_regimes", c["n_regimes"], 0, "experts_cfg", "pools.balanced8 (regimes)",
                  kind="derived", note=f"printed as a word ('{WORDS[c['n_regimes']]} regimes') since paper v12")
            L.add(T, "A15.balanced8.n_archs", c["n_archs"], 0, "experts_cfg", "pools.balanced8 (architectures)",
                  kind="derived", note=f"printed as a word ('{WORDS[c['n_archs']]} architectures') since paper v12")
        if pool in ("strong4", "weak6"):
            where = {"strong4": "App. G(d)", "weak6": "App. G(a)"}[pool]
            avgs = [h[e]["avg_auc_8"] for e in ex]
            L.add(T, f"A15.{pool}.expert_min", min(avgs), 2, "appendix_h_cells", f"min avg_auc_8 over pools.{pool}",
                  kind="derived", note=f"printed in {where} since paper v12")
            L.add(T, f"A15.{pool}.expert_max", max(avgs), 2, "appendix_h_cells", f"max avg_auc_8 over pools.{pool}",
                  kind="derived", note=f"printed in {where} since paper v12")
        v = pool_standard(pool)
        ps = L.add(T, f"A15.{pool}.best_ps", v["ps"][0], 3, v["ps"][1], v["ps"][2])
        f1 = L.add(T, f"A15.{pool}.fs1", v["fs1"][0], 3, v["fs1"][1], v["fs1"][2])
        d1 = L.add(T, f"A15.{pool}.delta1", f1 - ps, 3, v["fs1"][1], "fs1 - best PS (full precision)", kind="derived")
        f2 = L.add(T, f"A15.{pool}.fs2", v["fs2"][0], 3, v["fs2"][1], v["fs2"][2])
        d2 = L.add(T, f"A15.{pool}.delta2", f2 - ps, 3, v["fs2"][1], "fs2 - best PS (full precision)", kind="derived")
        rows.append(f"{pool.replace('_', chr(92) + '_')} & {n} & {COMPOSITION[pool](c)} & {ps:.3f} & {f1:.3f} & "
                    f"{plain(d1)} & {f2:.3f} & {plain(d2)}")
    return tab("lrlrrrrr", [head("\\textbf{Pool}", mc("N"), "\\textbf{Composition}", mc("PS comparator"),
                                 mc("\\fsmethod{1}"), mc("$\\Delta_1$"), mc("\\fsmethod{2}"), mc("$\\Delta_2$"))], rows)


# ---- Appendix H and I

def p_experts(L):
    T, d = "Table A16", load_json("appendix_h_cells")
    rows = []
    for e in EXPERT_ORDER:
        r = d["per_expert"][e]
        for g in GENERATORS:
            L.add(T, f"A16.{e}.{g}", r["per_generator"][g], 3, "appendix_h_cells", f"per_expert.{e}.per_generator.{g}",
                  kind="derived")
        L.add(T, f"A16.{e}.avg", r["avg_auc_8"], 3, "appendix_h_cells", f"per_expert.{e}.avg_auc_8", kind="derived")
        rows.append(f"{e} & " + " & ".join(f"{r['per_generator'][g]:.3f}" for g in GENERATORS) + f" & {r['avg_auc_8']:.3f}")
    return tab("lrrrrrrrr r", [head("\\textbf{Expert}", *(mc(x) for x in GEN_SHORT), mc("Macro AUROC"))], rows)


def p_prevalence(L):
    T, pv = "Table A17", load_json("auprc")["pools"]["arch6"]["prevalence"]
    rows = []
    for g in GENERATORS:
        x = pv[g]
        n = L.add(T, f"A17.{g}.n", x["n_total"], 0, "auprc", f"pools.arch6.prevalence.{g}.n_total")
        nf = L.add(T, f"A17.{g}.n_fake", x["n_fake"], 0, "auprc", f"pools.arch6.prevalence.{g}.n_fake")
        # exact ratio rounded half-up in decimal: 1,998 / 4,000 = 0.4995 prints 0.500 (binary floats give 0.499)
        shown = str((Decimal(nf) / Decimal(n)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
        L.add(T, f"A17.{g}.prevalence", nf / n, 3, "auprc", f"pools.arch6.prevalence.{g}: n_fake / n_total",
              kind="derived", disp=shown)
        rows.append(f"{GEN_LABEL[g]} & {n:,} & {nf:,} & {shown}")
    return tab("lrrr", [head("\\textbf{Generator}", mc("Test rows"), mc("Fake-labeled rows"),
                             mc("Fake-class fraction"))], rows)


def p_k3_heads(L):
    T, d = "Table A18", load_json("k3_heads")
    rows = []
    for m_ in ("PS1-K3", "PS2-K3", "PS3-K3", "PS4-K3", "FS1-K3", "FS2-K3", "FSB-K3-18", "PS+-K3"):
        v = L.add(T, f"A18.{m_}", d["summary"][m_]["mean_fold_macro_ovr"], 3, "k3_heads", f"summary.{m_}.mean_fold_macro_ovr")
        rows.append(f"{m_} & {v:.3f}")
    return tab("lr", [head("\\textbf{Method}", mc("Mean fold macro OvR AUROC"))], rows)


# version 15: the classes of the two three-class constructions, as Table A19 of paper v12 names them
K3_CLASSES = {"k3_view_a": ("entire-face-synthesis", "face-swap", "reenactment"),
              "k3_view_b": ("GAN-synthesis", "diffusion-synthesis", "reenactment")}


def mc2(top, bottom):
    """Version 15: a header cell of two lines, as Table A19 of paper v12 prints them."""
    return ("\\multicolumn{1}{l}{\\begin{tabular}[b]{@{}l@{}}\\textbf{" + top + "}\\\\ \\textbf{" + bottom +
            "}\\end{tabular}}")


def p_k3_views(L):
    """Version 15: Table A19 as paper v12 prints it; the classes of each construction are checked against the runs."""
    T = "Table A19"
    rows = []
    for key, label in (("k3_view_a", "A: entire-face synthesis, face swap, reenactment"),
                       ("k3_view_b", "B: GAN synthesis, diffusion, reenactment")):
        x = load_json(key)
        if tuple(x["classes"]) != K3_CLASSES[key]:
            raise SystemExit(f"Table A19: the classes of {key} are {x['classes']}, not those printed")
        pr = x["primary"]
        L.add(T, f"A19.{key}.delta", pr["mean"], 3, key, "primary.mean")
        L.add(T, f"A19.{key}.pos", pr["folds_positive"], 0, key, "primary.folds_positive")
        L.add(T, f"A19.{key}.n", pr["n_folds"], 0, key, "primary.n_folds")
        fl = L.add(T, f"A19.{key}.p_floor", 2 / 2 ** pr["n_folds"], 4, key, "2 / 2**primary.n_folds", kind="derived")
        rows.append(f"{label} & {sd(pr['mean'])} & {pr['folds_positive']}/{pr['n_folds']} & {fl:.4f}")
    return tab("lrrr", [head("\\textbf{Construction}", mc2("Mean FS-LR(std)-K3", "$-$ selected PS"),
                             mc("Positive folds"), mc2("Minimum attainable", "two-sided sign-flip $p$"))], rows)


def p_cost(L):
    """Version 15: Table A21 as paper v12 prints it (no row for the shared expert forward pass)."""
    T, d = "Table A21", load_json("cost")
    w, s, c = d["widths_and_storage"]["pool_totals"], d["widths_and_storage"]["storage"], d["combiner_costs"]

    def a(q, v, digits, key, kind="json", source="cost"):
        return L.add(T, f"A21.{q}", v, digits, source, key, kind=kind)
    est = load_json("strict")["arm_full"]["n_fit"] * w["feature_floats"] * 8 / 2 ** 30
    rows = [
        f"Values per sample & {a('feature_floats', w['feature_floats'], 0, 'widths_and_storage.pool_totals.feature_floats'):,} "
        f"float16; {a('feature_bytes', w['feature_bytes_per_sample_stored'], 0, 'widths_and_storage.pool_totals.feature_bytes_per_sample_stored'):,} B & "
        f"{a('pred_floats', w['prediction_floats'], 0, 'widths_and_storage.pool_totals.prediction_floats')} float32; "
        f"{a('pred_bytes', w['prediction_bytes_per_sample_stored'], 0, 'widths_and_storage.pool_totals.prediction_bytes_per_sample_stored')} B & Measured",
        f"arch6 cache & {a('pool_gib', s['pool_features_gib'], 2, 'widths_and_storage.storage.pool_features_gib'):.2f} GiB & "
        f"{a('pool_mib', s['pool_logits_mib'], 1, 'widths_and_storage.storage.pool_logits_mib'):.1f} MiB & Measured",
        f"Full {a('release_experts', s['n_release_experts'], 0, 'widths_and_storage.storage.n_release_experts')}-expert cache & "
        f"{a('release_gib', s['release_features_gib'], 2, 'widths_and_storage.storage.release_features_gib'):.2f} GiB & "
        f"{a('release_mib', s['release_logits_mib'], 1, 'widths_and_storage.storage.release_logits_mib'):.1f} MiB & Measured",
        f"Batch prediction time per sample & {a('fs1_us', c['FS_1_config_split_timing']['predict_us_per_sample'], 0, 'combiner_costs.FS_1_config_split_timing.predict_us_per_sample'):.0f}"
        f"\\,\\textmu s (\\fsmethod{{1}}) & "
        f"{a('ps_us', c['PS_predict_only']['predict_us_per_sample'], 1, 'combiner_costs.PS_predict_only.predict_us_per_sample'):.1f}"
        f"\\,\\textmu s (\\psmethod{{2}}) & Measured",
        f"Combiner fit and predict & {a('fs1_s', c['FS_1_frozen_total']['total_fit_plus_predict_sec'], 0, 'combiner_costs.FS_1_frozen_total.total_fit_plus_predict_sec'):.0f} s "
        f"(\\fsmethod{{1}}) & {a('ps4_s', c['PS_4_xgboost_cv']['total_fit_plus_predict_sec'], 0, 'combiner_costs.PS_4_xgboost_cv.total_fit_plus_predict_sec'):.0f} s "
        f"(\\psmethod{{4}}, including CV); {a('ps2_s', c['PS_2_logreg_logits']['total_fit_plus_predict_sec'], 2, 'combiner_costs.PS_2_logreg_logits.total_fit_plus_predict_sec'):.2f} s "
        f"(\\psmethod{{2}}) & Measured",
        f"Float64 TRAIN matrix for \\fsmethod{{1}} & "
        f"{a('fs1_float64_gib', est, 2, 'arm_full.n_fit x feature_floats x 8 B / 2^30', kind='derived', source='strict'):.2f} GiB "
        "& - & Estimated",
    ]
    return tab("llll", [head("\\textbf{Item}", "\\textbf{Feature interface}", "\\textbf{Prediction interface}",
                             "\\textbf{Status}")], rows)


# ---- Appendix J

def p_tiers(L):
    """Version 15: Table A22 as paper v12 prints it: the four protocol components in words. The method count and the
    bootstrap resamples of version 14 keep their rows; paper v12 prints them in the text."""
    T = "Table A22"
    n_methods = len(load_json("ds_arch6")["method_means"]) + len(RICH_ROWS_A2) * 2
    L.add(T, "A22.n_ps_methods", n_methods, 0, "ds_arch6", "len(method_means) + 2 heads x 5 richer inputs (Table A2)",
          kind="derived", note=f"not printed in Table A22 since paper v12; the text gives the {n_methods} methods")
    n_boot = load_json("ladder_arch6")["comparison_vs_deployable_ceiling"]["generator_bootstrap"]["n_boot"]
    L.add(T, "A22.n_boot", n_boot, 0, "ladder_arch6", "comparison_vs_deployable_ceiling.generator_bootstrap.n_boot",
          note=f"printed as {n_boot // 1000}K in Table A22 until paper v11; since paper v12 App. J asks for at least "
               f"{n_boot:,} bootstrap resamples")
    rows = [
        "(i) Baseline strength & Probability averaging and a tuned nonlinear stacker, with tuning confined to "
        "training data; this tier can overstate a feature gain (\\app{F}) & Add linear stacking on logits & Add neural stacking, calibration and regularization "
        "variants, and richer functions of predictions; document search budgets",
        "\\midrule",
        "(ii) Combiner capacity & A simple linear anchor, such as \\fsmethod{1} on concatenated raw features & Add a "
        "higher-capacity combiner, such as \\fsmethod{2} at one projection dimension & Sweep projection dimensions and "
        "compare the same combiner family across interfaces; add matched-input-width controls when making "
        "width-related claims",
        "\\midrule",
        "(iii) Pool construction & One pool varying a single heterogeneity axis, with the other documented axes held "
        "fixed & Include architecture-diverse and training-data-diverse pools & Add boundary pools, such as "
        "weak-expert or low-heterogeneity pools, to examine the limits of the finding",
        "\\midrule",
        "(iv) Overlap control and reporting granularity & Report overlap checks and exclusions, per-group results, and "
        "paired uncertainty at the held-out group level & Add grouped refitting where group transfer is the target, "
        "and assess the sensitivity of inference when groups are few & Examine repeated-content handling, alternative "
        "group weighting and complementary inferential summaries",
    ]
    return tab(r">{\raggedright\arraybackslash}p{2.8cm}>{\raggedright\arraybackslash}p{2.9cm}>{\raggedright\arraybackslash}p{3.0cm}>{\raggedright\arraybackslash}p{4.0cm}", [head("\\textbf{Component}", "\\textbf{Minimum}",
                                                          "\\textbf{Recommended}", "\\textbf{Stronger audit}")], rows)


HEURISTIC_THRESHOLDS = {"low": 0.80, "mid": 0.90, "high": 0.96, "dominant": 0.95, "pr_gap": 15, "pr_lo": 10, "pr_hi": 25}
ROW3_POOLS = ("clip4", "mixed6", "merged9", "balanced8", "strong4")   # as printed (finding N-04: full21 is not listed)


def p_heuristics(L):
    """Version 15: Table A23 as paper v12 prints it: observations, the evidence in this study and a practical check.
    The rows of version 14 (the thresholds and pool rows of paper v11) are kept with a note; the numbers of the new
    table are rows A23.obs*, and its words are checked against the results they describe."""
    T, m = "Table A23", load_json("mitigation")
    old = ("Table A23 of paper v11 (thresholds and pool rows); not printed since paper v12, whose Table A23 lists "
           "observations (rows A23.obs*)")
    for k, v in HEURISTIC_THRESHOLDS.items():
        L.add(T, f"A23.threshold.{k}", v, 2 if isinstance(v, float) else 0, PROTOCOL, "Table A23 threshold",
              kind="setting", note=old)
    d1 = {q: pool_standard(q)["fs1"][0] - pool_standard(q)["ps"][0] for q in POOLS_A21}
    L.add(T, "A23.row1.weak6_delta1", d1["weak6"], 2, "full_ladder", "Table A15 Delta1 of weak6", kind="derived",
          note=old)
    L.add(T, "A23.row1.low_het4_delta1", d1["low_het4"], 2, "full_ladder", "Table A15 Delta1 of low_het4",
          kind="derived", note=old)
    ld = {q: logo_pool(m, q)[0]["stats"]["fs1_full"]["mean_delta"] for q in ("arch6", "clip4", "strong4", "weak6")}
    L.add(T, "A23.row1.logo_weak6", ld["weak6"], 3, "mitigation", "logo_boundary.weak6.stats.fs1_full.mean_delta",
          note=old)
    L.add(T, "A23.row2.standard_arch6", load_json("full_ladder")["standard"]["arch6"]["fs1_delta"], 3, "full_ladder",
          "standard.arch6.fs1_delta", note=old)
    L.add(T, "A23.row2.logo_arch6", ld["arch6"], 3, "mitigation", "logo_arch6.stats.fs1_full.mean_delta", note=old)
    r3 = [d1[q] for q in ROW3_POOLS]
    L.add(T, "A23.row3.delta1_min", min(r3), 2, "full_ladder", "min Table A15 Delta1 over the row-3 pools",
          kind="derived", note=old)
    L.add(T, "A23.row3.delta1_max", max(r3), 2, "full_ladder", "max Table A15 Delta1 over the row-3 pools",
          kind="derived", note=old)
    L.add(T, "A23.row3.logo_clip4", ld["clip4"], 3, "mitigation", "logo_boundary.clip4.stats.fs1_full.mean_delta",
          note=old)
    L.add(T, "A23.row3.logo_strong4", ld["strong4"], 3, "mitigation", "logo_boundary.strong4.stats.fs1_full.mean_delta",
          note=old)
    # version 15: the table of paper v12
    lg = {q: logo_pool(m, q) for q in ("arch6", "clip4", "strong4", "weak6")}
    st = {q: g["stats"]["fs1_full"] for q, (g, _) in lg.items()}
    d_a = L.add(T, "A23.obs1.logo_arch6_delta", st["arch6"]["mean_delta"], 3, "mitigation",
                "logo_arch6.stats.fs1_full.mean_delta")
    if m["logo_arch6"]["fs2"] > m["logo_arch6"]["fs1"]:                # FS2: a warning
        word_warning("Table A23", "'the larger FS2 model provides no clear additional improvement' needs arch6 LOGO "
                     "FS2 at most FS1")
    d2 = {}
    for q in ("clip4", "strong4"):
        base = lg[q][1]
        d2[q] = L.add(T, f"A23.obs2.logo_{q}_delta", st[q]["mean_delta"], 3, "mitigation",
                      f"{base}.stats.fs1_full.mean_delta")
        words = f"printed as words ('{WORDS[st[q]['wins']]} of {WORDS[st[q]['n']]} folds in each pool') since paper v12"
        L.add(T, f"A23.obs2.{q}_wins", st[q]["wins"], 0, "mitigation", f"{base}.stats.fs1_full.wins", note=words)
        L.add(T, f"A23.obs2.{q}_n", st[q]["n"], 0, "mitigation", f"{base}.stats.fs1_full.n", note=words)
    w, n = st["clip4"]["wins"], st["clip4"]["n"]
    same_folds = (st["strong4"]["wins"], st["strong4"]["n"]) == (w, n)
    if not (same_folds and d2["clip4"] < 0 and d2["strong4"] < 0 and 2 * w > n):
        raise SystemExit("Table A23: 'despite FS1 winning five of eight folds in each pool' needs negative clip4 and "
                         "strong4 margins with the same majority of folds won")
    g6, base6 = lg["weak6"]
    ce = L.add(T, "A23.obs3.weak6_ceiling", g6["ceiling_full"], 3, "mitigation", f"{base6}.ceiling_full")
    f1 = L.add(T, "A23.obs3.weak6_fs1", g6["fs1"], 3, "mitigation", f"{base6}.fs1")
    if any(g["ceiling_full"] <= ce or g["fs1"] <= f1 for q, (g, _) in lg.items() if q != "weak6"):
        raise SystemExit("Table A23: 'limited expert transfer can constrain both interfaces' needs weak6 to have the "
                         "lowest LOGO ceiling and FS1 of the four pools")
    lower = [q for q in POOLS_A21 if pool_standard(q)["fs2"][0] < pool_standard(q)["fs1"][0]]
    if lower != ["weak6", "low_het4"]:                                   # FS2: a warning (the row names `lower`)
        word_warning("Table A23", f"FS2 is below FS1 on the standard split in {lower}, not in weak6 and low_het4 only")
    names = " and ".join(q.replace("_", "\\_") for q in lower)
    rows = [
        f"A linear feature combiner can improve on strong prediction baselines & arch6: $\\Delta\\!=\\!{plain(d_a)}$; "
        "the larger \\fsmethod{2} model provides no clear additional improvement & Include a simple feature anchor "
        "before investing in a larger combiner",
        "\\midrule",
        f"Feature access does not guarantee a better macro result & clip4: $\\Delta\\!=\\!{plain(d2['clip4'])}$; "
        f"strong4: $\\Delta\\!=\\!{plain(d2['strong4'])}$, despite \\fsmethod{{1}} winning {WORDS[w]} of {WORDS[n]} "
        "folds in each pool & Inspect the magnitudes and distribution of group differences; assess the capacity check "
        "separately",
        "\\midrule",
        f"Limited expert transfer can constrain both interfaces & weak6: PS ceiling {ce:.3f} and \\fsmethod{{1}} "
        f"{f1:.3f} macro AUROC & Evaluate absolute performance and expert transfer, rather than judging usefulness "
        "from the interface difference alone",
        "\\midrule",
        "More complex feature combination can underperform the anchor & On the overlap-affected standard split, "
        f"\\fsmethod{{2}} has lower point estimates than \\fsmethod{{1}} on {names} & Check sensitivity to model "
        "capacity, regularization and random seeds",
    ]
    colspec = ">{\\raggedright\\arraybackslash}p{3.5cm}>{\\raggedright\\arraybackslash}p{4.6cm}" \
              ">{\\raggedright\\arraybackslash}p{4.6cm}"
    return tab(colspec, [head("\\textbf{Observation}", "\\textbf{Evidence in this study}",
                              "\\textbf{Practical check}")], rows)


def sgn(v, digits=3):
    """+0.081 or $-$0.074: signed values in text mode, as in Tables 1 and A10."""
    return f"+{v:.{digits}f}" if v >= 0 else f"$-${abs(v):.{digits}f}"


def ci_math(lo, hi, digits=3):
    return f"$[{lo:+.{digits}f}, {hi:+.{digits}f}]$"


def sens_delta_ci(L, T, qid, block, prefix, wins=True):
    """Version 14 (the inner helper of version 7): a difference of the sensitivity summary with its CI, and the folds
    with a positive difference when `wins`."""
    d = L.add(T, f"{qid}.delta", block["mean_delta"], 3, "sens_summary", f"{prefix}.mean_delta")
    lo = L.add(T, f"{qid}.ci_lo", block["ci95"][0], 3, "sens_summary", f"{prefix}.ci95[0]")
    hi = L.add(T, f"{qid}.ci_hi", block["ci95"][1], 3, "sens_summary", f"{prefix}.ci95[1]")
    text = f"{sgn(d)} {ci_math(lo, hi)}"
    if wins:
        w = L.add(T, f"{qid}.wins", block["wins"], 0, "sens_summary", f"{prefix}.wins")
        n = L.add(T, f"{qid}.n", block["n"], 0, "sens_summary", f"{prefix}.n")
        text += f", {w}/{n}"
    return text


def p_sensitivity(L):
    """Version 14: the table of Appendix A on duplicated training rows (paper v12; the upper block of Table A24 in paper
    v11, version 7): the primary results against those with the training duplicates removed."""
    T, s = PAPER_NAMES["sensitivity"], load_json("sens_summary")
    rows, cells = [], []
    for side in ("primary", "deduplicated"):
        cells.append(sens_delta_ci(L, T, f"appA.logo.arch6.fs1.{side}", s["logo"]["arch6"][side]["stats"]["fs1_full"],
                                   f"logo.arch6.{side}.stats.fs1_full"))
    rows.append(f"arch6 LOGO: \\fsmethod{{1}} $-$ PS ceiling & {cells[0]} & {cells[1]}")
    au = [L.add(T, f"appA.logo.arch6.auprc.{side}", s["logo"]["arch6"][side]["auprc"]["delta"], 3, "sens_summary",
                f"logo.arch6.{side}.auprc.delta") for side in ("primary", "deduplicated")]
    rows.append(f"arch6 LOGO: AUPRC, \\fsmethod{{1}} $-$ ceiling method & {sgn(au[0])} & {sgn(au[1])}")
    other = {}
    for side in ("primary", "deduplicated"):
        other[side] = ", ".join(sgn(L.add(T, f"appA.logo.{p}.fs1.{side}.delta",
                                          s["logo"][p][side]["stats"]["fs1_full"]["mean_delta"], 3, "sens_summary",
                                          f"logo.{p}.{side}.stats.fs1_full.mean_delta"))
                                for p in ("clip4", "strong4", "weak6"))
    rows.append(f"clip4, strong4, weak6 LOGO: \\fsmethod{{1}} $-$ PS ceiling & {other['primary']} & "
                f"{other['deduplicated']}")
    for label, key, qid in (("12 PCA values $-$ \\psmethod{4}", "FS_pca2_logreg_vs_PS_4", "pca"),
                            ("12 label-fitted values $-$ \\fsmethod{1}", "FS_probe2_logreg_vs_FS_1", "probe")):
        c = [sens_delta_ci(L, T, f"appA.mech.arch6.{qid}.{side}", s["mechanism"]["arch6"][side]["comparisons"][key],
                           f"mechanism.arch6.{side}.comparisons.{key}", wins=False)
             for side in ("primary", "deduplicated")]
        rows.append(f"arch6 LOGO: {label} & {c[0]} & {c[1]}")
    for pool in ("arch6", "clip4"):
        c = []
        for side in ("primary", "deduplicated"):
            x = s["standard"][pool][side]["comparison_vs_deployable_ceiling"]
            pre = f"standard.{pool}.{side}.comparison_vs_deployable_ceiling"
            d = L.add(T, f"appA.std.{pool}.fs1.{side}.delta", x["delta"], 3, "sens_summary", f"{pre}.delta")
            lo = L.add(T, f"appA.std.{pool}.fs1.{side}.ci_lo", x["generator_bootstrap"]["ci_95"][0], 3,
                       "sens_summary", f"{pre}.generator_bootstrap.ci_95[0]")
            hi = L.add(T, f"appA.std.{pool}.fs1.{side}.ci_hi", x["generator_bootstrap"]["ci_95"][1], 3,
                       "sens_summary", f"{pre}.generator_bootstrap.ci_95[1]")
            c.append(f"{sgn(d)} {ci_math(lo, hi)}")
        rows.append(f"{pool} standard: \\fsmethod{{1}} $-$ PS ceiling & {c[0]} & {c[1]}")
    # the two effects of Table 2, under the names paper v12 gives them here (the summary keeps those of paper v7)
    eff = {k.split(" (")[0]: (k, v) for k, v in s["audit"].items()}
    k, v = eff["overlap distortion"]
    o = [L.add(T, "appA.audit.overlap.paper", v["paper"], 3, "sens_summary", f"audit.{k}.paper"),
         L.add(T, "appA.audit.overlap.deduplicated", v["deduplicated"], 3, "sens_summary", f"audit.{k}.deduplicated")]
    L.add(T, "appA.audit.overlap.same_script", v["primary_same_scripts"], 3, "sens_summary",
          f"audit.{k}.primary_same_scripts",
          note="App. A: the primary standard-LOGO contrast of the richer-input script (note of the duplicates table)")
    rows.append(f"standard--LOGO contrast, arch6 & {sgn(o[0])} & {sgn(o[1])}")
    b = {}
    for pool in ("arch6", "weak6"):
        k, v = eff[f"baseline-ladder distortion, {pool}"]
        b[pool] = [L.add(T, f"appA.audit.baseline.{pool}.{side}", v[src_key], 3, "sens_summary",
                         f"audit.{k}.{src_key}")
                   for side, src_key in (("paper", "paper"), ("deduplicated", "deduplicated"))]
    rows.append(f"standard-split baseline inflation, arch6; weak6 & {sgn(b['arch6'][0])}; "
                f"{sgn(b['weak6'][0])} & {sgn(b['arch6'][1])}; {sgn(b['weak6'][1])}")
    return tab(">{\\raggedright\\arraybackslash}p{5.6cm}ll",
               [head("\\textbf{Quantity}", "\\textbf{Reported results}", "\\textbf{Training duplicates removed}")], rows)


def p_sensitivity_heads(L):
    """Version 14: the table of Appendix A on the LOGO prediction-head settings (paper v12; the lower block of Table
    A24 in paper v11): FS1 - PS ceiling with the primary LOGO heads (22 candidates) and with the richer-input heads of
    the standard split (20 methods)."""
    T, s = PAPER_NAMES["sensitivity_heads"], load_json("sens_summary")
    c = [sens_delta_ci(L, T, f"appA.heads.arch6.fs1.{side}", s["paper_heads"]["arch6"][side]["stats"]["fs1_full"],
                       f"paper_heads.arch6.{side}.stats.fs1_full", wins=False) for side in ("primary_22", "paper_20")]
    rows = [f"arch6 & {c[0]} & {c[1]}"]
    other = {}
    for side in ("primary_22", "paper_20"):
        other[side] = ", ".join(sgn(L.add(T, f"appA.heads.{p}.fs1.{side}.delta",
                                          s["paper_heads"][p][side]["stats"]["fs1_full"]["mean_delta"], 3,
                                          "sens_summary", f"paper_heads.{p}.{side}.stats.fs1_full.mean_delta"))
                                for p in ("clip4", "strong4", "weak6"))
    rows.append(f"clip4, strong4, weak6 & {other['primary_22']} & {other['paper_20']}")
    return tab("lll", [head("\\textbf{\\fsmethod{1} $-$ PS ceiling}", "\\textbf{Primary LOGO settings}",
                            "\\textbf{Standard-split head settings}")], rows)


def p_overlap(L):
    """Version 15: the table of Appendix A on the overlap between combiner training and test data (paper v12, no
    label): per generator, the test rows, those whose path is also in TRAIN or SELECT, and the additional rows whose
    df40-clip-large features are identical to a TRAIN or SELECT row (feature matches minus same-path matches)."""
    T, r = PAPER_NAMES["overlap"], load_json("reuse_check")["route_b"]
    cr, pr = r["content_reuse"], r["path_reuse"]
    rows, n_all, extra_all = [], 0, 0
    for g in GENERATORS:
        c, p = cr["per_generator"][g], pr["per_generator"][g]
        if c["content_reuse"] < p["path_reuse"]:
            raise SystemExit(f"App. A, overlap table: {g} has fewer feature matches than same-path matches")
        n = L.add(T, f"appA.overlap.{g}.n", c["n"], 0, "reuse_check", f"route_b.content_reuse.per_generator.{g}.n")
        path = L.add(T, f"appA.overlap.{g}.path", p["path_reuse"], 0, "reuse_check",
                     f"route_b.path_reuse.per_generator.{g}.path_reuse")
        extra = L.add(T, f"appA.overlap.{g}.additional", c["content_reuse"] - p["path_reuse"], 0, "reuse_check",
                      f"route_b.content_reuse.per_generator.{g}.content_reuse - route_b.path_reuse.per_generator.{g}"
                      ".path_reuse", kind="derived")
        n_all, extra_all = n_all + n, extra_all + extra
        rows.append(f"{GEN_LABEL[g]} & {n:,} & {path:,} & {extra:,}")
    if sum(pr["per_generator"][g]["path_reuse"] for g in GENERATORS) != pr["total"]:
        raise SystemExit("App. A, overlap table: the same-path matches per generator do not add up to their total")
    n = L.add(T, "appA.overlap.total.n", n_all, 0, "reuse_check",
              "sum over generators of route_b.content_reuse.per_generator.<g>.n", kind="derived")
    path = L.add(T, "appA.overlap.total.path", pr["total"], 0, "reuse_check", "route_b.path_reuse.total")
    extra = L.add(T, "appA.overlap.total.additional", extra_all, 0, "reuse_check",
                  "sum over generators of the additional feature matches", kind="derived")
    rows += ["\\midrule", f"\\textbf{{Total}} & \\textbf{{{n:,}}} & \\textbf{{{path:,}}} & \\textbf{{{extra:,}}}"]
    return tab("lrrr", [head("\\textbf{Generator}", "\\textbf{Test rows}", "\\textbf{Same-path matches}",
                             "\\textbf{Additional feature matches}")], rows)


def text_rows(L):
    """Version 7: the other numbers the paper v7 adds to its text (table "text"), with their sources. Version 14: those
    of Appendix K are in Appendix A since paper v12 (ids text.appA.*, notes "App. A")."""
    T, s, tw = "text", load_json("sens_summary"), load_json("sens_twins")
    for pool in ("arch6", "clip4"):
        L.add(T, f"text.appA.std_twins.select.{pool}", tw["standard"][pool]["select_rows_with_fit_twin"], 0,
              "sens_twins", f"standard.{pool}.select_rows_with_fit_twin",
              note="App. A: SELECT rows with features identical to a TRAIN row (standard split)")
    L.add(T, "text.appA.std_twins.fit_copies.arch6", tw["standard"]["arch6"]["extra_copies_inside_fit"], 0,
          "sens_twins", "standard.arch6.extra_copies_inside_fit", note="App. A: further copies inside TRAIN")
    folds = {g: r for g, r in tw["logo"]["arch6"].items() if g != "starganv2"}
    shares = [r["share"] for r in folds.values()]
    L.add(T, "text.appF.logo_twins.share_min_pct", 100 * min(shares), 0, "sens_twins", "min of logo.arch6.<fold>.share "
          "(the seven folds that train on StarGAN-v2) x 100", kind="derived", note="App. A and F: 23-31%")
    L.add(T, "text.appF.logo_twins.share_max_pct", 100 * max(shares), 0, "sens_twins", "max of logo.arch6.<fold>.share "
          "x 100", kind="derived", note="App. A and F: 23-31%")
    copies = [r["extra_copies_inside_fit"] for r in folds.values()]
    L.add(T, "text.appA.logo_twins.fit_copies_min", min(copies), 0, "sens_twins",
          "min of logo.arch6.<fold>.extra_copies_inside_fit", kind="derived")
    L.add(T, "text.appA.logo_twins.fit_copies_max", max(copies), 0, "sens_twins",
          "max of logo.arch6.<fold>.extra_copies_inside_fit", kind="derived")
    L.add(T, "text.appF.starganv2.distinct", load_json("reuse_check")["route_b"]["distinct_test_images"]["per_generator"]
          ["starganv2"], 0, "reuse_check", "route_b.distinct_test_images.per_generator.starganv2",
          note="App. A and F: StarGAN-v2's 4,000 test rows hold 1,024 distinct images")
    rungs = load_json("cd_rungs_arch6")["folds"]
    n_rungs = len(next(iter(rungs.values()))["rungs"])
    L.add(T, "text.secF.logo_candidates", 10 + n_rungs, 0, "cd_rungs_arch6",
          "10 ladder methods + len(folds.<fold>.rungs)", kind="derived",
          note="Sec. 3, App. F: candidates of each fold's LOGO ceiling")
    ph = s["paper_heads"]["arch6"]["paper_20"]
    i = ph["folds"].index("heygen")
    L.add(T, "text.appF.heads.heygen.delta1", ph["stats"]["fs1_full"]["per_fold"][i], 3, "sens_summary",
          f"paper_heads.arch6.paper_20.stats.fs1_full.per_fold[{i}]", note="App. F: HeyGen row with the paper-described heads")
    L.add(T, "text.appF.heads.heygen.delta2", ph["stats"]["fs2_full"]["per_fold"][i], 3, "sens_summary",
          f"paper_heads.arch6.paper_20.stats.fs2_full.per_fold[{i}]", note="App. F: HeyGen row with the paper-described heads")
    shifts = [s["logo"][p]["deduplicated"]["stats"]["fs1_full"]["mean_delta"]
              - s["logo"][p]["primary"]["stats"]["fs1_full"]["mean_delta"] for p in ("clip4", "strong4", "weak6")]
    L.add(T, "text.appA.logo.other_pools.max_shift", max(shifts), 3, "sens_summary",
          "max over clip4, strong4, weak6 of logo.<pool>.deduplicated.stats.fs1_full.mean_delta - "
          "logo.<pool>.primary.stats.fs1_full.mean_delta", kind="derived",
          note="App. A: the other three pools move toward zero by at most 0.008")
    m = s["mechanism"]["arch6"]["deduplicated"]["mean_auroc"]
    for k in ("PS_4_xgboost", "FS_pca2_logreg"):
        L.add(T, f"text.appA.mech.dedup.{k}", m[k], 3, "sens_summary", f"mechanism.arch6.deduplicated.mean_auroc.{k}",
              note="App. A: width controls with duplicates removed")
    L.add(T, "text.appA.std.arch6.dedup.ceiling", s["standard"]["arch6"]["deduplicated"]["deployable_ceiling"]["macro"],
          3, "sens_summary", "standard.arch6.deduplicated.deployable_ceiling.macro",
          note="App. A: standard-split ceiling with duplicates removed")
    if src("cr_reference").exists():
        ref = load_json("cr_reference")["results"]
        L.add(T, "text.q5.reference.files", len(ref), 0, "cr_reference", "len(results)", kind="derived",
              note="Q5: camera-ready result files checked")
        L.add(T, "text.q5.reference.values", sum(len(e.get("metrics", {})) for e in ref.values()), 0, "cr_reference",
              "sum of len(results.<file>.metrics)", kind="derived", note="Q5: numeric values checked")
    # version 8: the numbers paper v8 adds (checklist Q8: the full21 FS1 fit's float64 input and the host size below
    # which the reviewed code skips that fit; App. F: the simulated sets of the power note and its precision check)
    import yaml   # PyYAML, in requirements-lock.txt
    cfg = yaml.safe_load(src("experts_cfg").read_text(encoding="utf-8"))
    dims = {e: cfg["architectures"][v["arch"]]["feature_dim"] for e, v in cfg["experts"].items()}
    width = sum(dims[e] for e in cfg["pools"]["full21"])
    n_fit = load_json("strict")["arm_full"]["n_fit"]
    gib = L.add(T, "text.q8.fs1_full21_input_gib", n_fit * width * 8 / 2 ** 30, 0, "experts_cfg",
                "strict: arm_full.n_fit x (sum of feature_dim over pools.full21) x 8 B / 2^30", kind="derived",
                note="Q5, Q8: the full21 FS1 fit's float64 input, about 7 GiB")
    L.add(T, "text.q8.fs1_skip_below_gib", gib / fs1_budget_fraction(), 0, "experts_cfg",      # version 10: read
          "text.q8.fs1_full21_input_gib / FS_1_BUDGET_FRACTION (0.40 in src/models/feature_space.py)", kind="derived",
          note="Q8: hosts below about 17 GiB skip the full21 FS1 fit (40% of their RAM is below its input)")
    pm = load_json("power_mde")
    L.add(T, "text.appF.mde.sets", pm["sets"], 0, "power_mde", "sets", note="App. F power note: simulated sets")
    L.add(T, "text.appF.mde.precision_sets", pm["precision_sets"], 0, "power_mde", "precision_sets",
          note="App. F power note: sets of the precision check at grid points near 80% power")
    for qid, v, what in (("grid_capped", 6, "GRID_CAPPED"), ("grid_widened", 12, "GRID_WIDENED"),
                         ("cv_folds", 3, "StratifiedKFold(n_splits=3) in cv_pick")):
        L.add(T, f"text.appF.heads.{qid}", v, 0, "runner_08", what, kind="setting",
              note="App. F, Table A1 note: the LOGO richer-input heads")
    # version 11: the numbers paper v9 adds (App. A: the zero-byte-source rows and the labels they changed; Table
    # A19: the XGBoost configurations of its PS side and the folds that pick the largest; App. I.1: the iteration
    # limit of the three-class heads and RFM's final head in the one fold where it stops below it)
    raw = load_json("reuse_check")["route_b"]["raw_files"]
    L.add(T, "text.appA.zero_byte_images", raw["zero_byte_total"], 0, "reuse_check", "route_b.raw_files.zero_byte_total",
          note="App. A: cached rows that refer to nine distinct zero-byte source-file paths in the local DF40 copy")
    L.add(T, "text.appA.zero_byte_label_changes", raw["zero_byte_rows_whose_label_disagrees_with_path"], 0,
          "reuse_check", "route_b.raw_files.zero_byte_rows_whose_label_disagrees_with_path",
          note="App. A: cached rows whose label disagrees with their path, all of them from zero-byte images")
    configs, chosen = set(), []
    for key in ("k3_view_a", "k3_view_b"):
        for fold in load_json(key)["folds"].values():
            found = (re.search(r"\|xgb_d(\d+)_n(\d+)$", k) for k in fold["ps_all"])
            configs |= {(int(m.group(1)), int(m.group(2))) for m in found if m}
            chosen.append(fold["ps_ceiling"]["rung"])
    depth, trees = max(d for d, _ in configs), max(n for _, n in configs)
    L.add(T, "text.appI1.k3_views.ps_xgb_configs", len(configs), 0, "k3_view_a",
          "distinct xgb_d<depth>_n<trees> keys of folds.<k>.ps_all, in k3_view_a and k3_view_b", kind="derived",
          note="Table A19: the XGBoost configurations of the PS side, on each input map")
    L.add(T, "text.appI1.k3_views.ps_xgb_max_trees", trees, 0, "k3_view_a",
          "largest <trees> of those keys", kind="derived", note="Table A19: up to 200 trees")
    L.add(T, "text.appI1.k3_views.ps_xgb_max_depth", depth, 0, "k3_view_a",
          "largest <depth> of those keys", kind="derived", note="Table A19: depth up to 4")
    largest = sum(r.endswith(f"|xgb_d{depth}_n{trees}") for r in chosen)
    L.add(T, "text.appI1.k3_views.largest_chosen_folds", largest, 0, "k3_view_a",
          "folds of k3_view_a and k3_view_b whose ps_ceiling.rung is the largest configuration", kind="derived",
          note=f"Table A19: the PS side picks its largest configuration in {largest} of {len(chosen)} folds")
    hq = [f["head_quality"]["fs-rfm"] for f in load_json("k3_heads")["folds"]]
    L.add(T, "text.appI1.k3.rfm_final_n_iter_min", min(q["final_n_iter"] for q in hq), 0, "k3_heads",
          "min over folds of head_quality.fs-rfm.final_n_iter", kind="derived",
          note="App. I.1: RFM's final head stops below the iteration limit in one fold")
    L.add(T, "text.appI1.k3.head_max_iter", hq[0]["head_max_iter"], 0, "k3_heads",
          "folds[0].head_quality.fs-rfm.head_max_iter", note="App. I.1: the iteration limit of the three-class heads")
    # version 12: the numbers paper v10 adds. App. F, reading of Table A12: with no true effect, how often the
    # percentile interval excludes zero, and the effect that the exact sign-flip and t-tests need, at sigma = 0.08
    i08 = next(i for i, x in enumerate(pm["sigmas"]) if x["sigma"] == 0.08)
    x08 = pm["sigmas"][i08]
    z = x08["delta"].index(0.0)
    L.add(T, "text.appF.mde.boot_null_pct", 100 * x08["bootstrap_ci_power"][z], 0, "power_mde",
          f"sigmas[{i08}].bootstrap_ci_power[{z}] (delta 0) x 100", kind="derived",
          note="App. F, reading of Table A12: the percentile interval excludes zero in about 11% of the null sets "
               "(11.2 to 11.6% over the three sigmas)")
    for test in ("signflip", "t_test"):
        L.add(T, f"text.appF.mde.sigma08.{test}", x08["mde"][test], 3, "power_mde", f"sigmas[{i08}].mde.{test}",
              note="App. F, reading of Table A12: the effect the size-correct tests need at sigma = 0.08")
    # App. F, choosing the ceiling per fold: the single PS method with the best mean over the eight folds, among the
    # candidates of each fold's ceiling (the ten ladder methods of the LOGO run and the richer-input heads)
    mit = load_json("mitigation")
    single = {}
    for pool in ("arch6", "clip4", "strong4", "weak6"):
        blk, base = logo_pool(mit, pool)
        run = {f["held_out"]: f for f in load_json(f"cd_logo_{pool}")["per_fold"]}
        cand = {}
        for g in blk["folds"]:
            for k, v in run[g]["ps_means"].items():
                cand.setdefault(k, {})[g] = v
            for k, v in load_json(f"cd_rung_folds_{pool}_{g}")["rungs"].items():
                cand.setdefault(f"rung:{k}", {})[g] = v
        full = {k: v for k, v in cand.items() if len(v) == len(blk["folds"])}
        tops = [max(v[g] for v in full.values()) for g in blk["folds"]]
        if max(abs(a - b) for a, b in zip(tops, blk["ceiling_full_per_fold"])) > 1e-12:
            raise SystemExit(f"{pool}: the fold maxima of the stored candidates are not the ceilings of the statistics")
        means = {k: sum(v.values()) / len(v) for k, v in full.items()}
        best = max(means, key=means.get)
        single[pool] = blk["fs1"] - means[best]
        L.add(T, f"text.appF.single_best.{pool}.method", best.replace("rung:", ""), 0, f"cd_logo_{pool}",
              "the candidate with the largest mean over the folds of per_fold[*].ps_means and of "
              f"rung_folds/{pool}_<fold>.json rungs", kind="derived",
              note="App. F: the single PS method with the best mean over the eight LOGO folds")
        L.add(T, f"text.appF.single_best.{pool}.margin", single[pool], 3, f"cd_logo_{pool}",
              f"{base}.fs1 of mitigation_stats.json minus that candidate's mean", kind="derived",
              note="App. F: FS1 minus the single best PS method")
        if pool == "arch6":
            L.add(T, "text.appF.single_best.arch6.ps", means[best], 3, "cd_logo_arch6", "that candidate's mean",
                  kind="derived", note="App. F and Sec. 6: the single best PS method under LOGO")
        if pool in ("clip4", "strong4"):
            L.add(T, f"text.appF.ps1.{pool}", means["PS_1_prob_avg"], 3, f"cd_logo_{pool}",
                  "mean over the folds of per_fold[*].ps_means.PS_1_prob_avg", kind="derived",
                  note="App. F: probability averaging under LOGO")
    std = load_json("full_ladder")["standard"]["arch6"]
    per_gen = load_json("ladder_arch6")["ps_17_per_generator_oracle_selection"]["macro"]
    std_pg = L.add(T, "text.appF.std.per_generator_margin", std["fs1"] - per_gen, 3, "ladder_arch6",
                   "full_ladder_stats.json standard.arch6.fs1 minus ps_17_per_generator_oracle_selection.macro",
                   kind="derived", note="App. F: the standard-split margin against the best method per generator")
    L.add(T, "text.appF.same_rule.single", std["fs1_delta"] - single["arch6"], 3, "full_ladder",
          "standard.arch6.fs1_delta minus text.appF.single_best.arch6.margin", kind="derived",
          note="App. F: Table 2's overlap contrast with the single best method on both protocols")
    L.add(T, "text.appF.same_rule.per_generator", std_pg - mit["logo_arch6"]["stats"]["fs1_full"]["mean_delta"], 3,
          "ladder_arch6", "text.appF.std.per_generator_margin minus mitigation_stats.json "
          "logo_arch6.stats.fs1_full.mean_delta", kind="derived",
          note="App. F: Table 2's overlap contrast with the best method per generator (per fold) on both protocols")
    # App. G(c): the two pools that keep arch6's architectures with experts trained on FR or EFS
    for pool in ("arch6_fr", "arch6_efs"):
        ds = load_json(f"ds_{pool}")
        if ds["best_ds"] != "PS_4_xgboost":
            raise SystemExit(f"{pool}: the best of the first ten PS methods is {ds['best_ds']}, not PS_4 (App. G(c))")
        L.add(T, f"text.appG.{pool}.ps4", ds["best_ds_avg8"], 3, f"ds_{pool}", "best_ds_avg8",
              note="App. G(c): PS4 on the replication pool")
        L.add(T, f"text.appG.{pool}.delta", load_json(f"ds_{pool}_raw")["avg_auc_8"] - ds["best_ds_avg8"], 3,
              f"ds_{pool}_raw", f"avg_auc_8 minus 04_ds_{pool}.json best_ds_avg8", kind="derived",
              note="App. G(c): FS1 minus PS4 on the replication pool")


def text_rows_v15(L):
    """Version 15: the numbers that paper v12 adds to its text (table "text"), with their sources: the refit of FS1 on
    standardized features (App. B), the small improvement from removing fs-rfm (Table A9's note), the margins over
    the best single expert and in AUPRC (App. H), the rows of the three-class constructions (App. I.1),
    and the timing batch, TRAIN size and Python version of the cost table (App. I.3)."""
    T = "text"
    # App. B, raw-feature scaling check: FS1 refitted on features standardized with TRAIN statistics
    note = "App. B: raw-feature scaling check, macro AUROC (4 decimals)"
    macro = {}
    for pool in ("arch6", "clip4"):
        key, x = f"fs1_std_{pool}", load_json(f"fs1_std_{pool}")
        if abs(x["avg_auc_8_raw"] - load_json(f"ds_{pool}_raw")["avg_auc_8"]) > 1e-12:
            raise SystemExit(f"App. B: the unstandardized fit of {key} is not that of 04_ds_{pool}_rawfeat.json")
        if abs(x["delta_std_minus_raw"] - (x["avg_auc_8_std"] - x["avg_auc_8_raw"])) > 1e-12:
            raise SystemExit(f"App. B: delta_std_minus_raw of {key} is not avg_auc_8_std - avg_auc_8_raw")
        L.add(T, f"text.appB.fs1_std.{pool}.raw", x["avg_auc_8_raw"], 4, key, "avg_auc_8_raw", note=note)
        L.add(T, f"text.appB.fs1_std.{pool}.std", x["avg_auc_8_std"], 4, key, "avg_auc_8_std", note=note)
        macro[pool] = L.add(T, f"text.appB.fs1_std.{pool}.delta", x["delta_std_minus_raw"], 4, key,
                            "delta_std_minus_raw", note=note)
        per_gen = max(abs(x["per_generator_std"][g] - x["per_generator_raw"][g]) for g in GENERATORS)
        if per_gen <= abs(macro[pool]):
            raise SystemExit(f"App. B: 'individual-generator changes are larger' does not hold on {pool}")
    bound = L.add(T, "text.appB.fs1_std.bound", 0.0005, 4, PROTOCOL, "the bound the text states", kind="setting",
                  note="App. B: the macro average changes by less than 0.0005 in either pool")
    if max(abs(v) for v in macro.values()) >= bound:
        raise SystemExit("App. B: a macro change of the standardized refit is not below 0.0005")
    x = load_json("fs1_std_clip4")
    L.add(T, "text.appB.fs1_std.clip4.heygen.raw", x["per_generator_raw"]["heygen"], 4, "fs1_std_clip4",
          "per_generator_raw.heygen", note="App. B: clip4 HeyGen before standardization")
    L.add(T, "text.appB.fs1_std.clip4.heygen.std", x["per_generator_std"]["heygen"], 4, "fs1_std_clip4",
          "per_generator_std.heygen", note="App. B: clip4 HeyGen after standardization")
    # Table A9's note (App. E): removing fs-rfm from arch6 improves the macro AUROC slightly
    item = next(it for it in load_json("loo_arch6")["loo"] if it["removed"] == "fs-rfm")
    gain = -item["delta_vs_full"]
    if not (gain > 0 and f"{gain:.3f}" == "0.000"):                       # the leave-one-out runs train FS2
        word_warning("Table A9's note", "removing fs-rfm no longer gives a small improvement that rounds to 0.000")
    L.add(T, "text.appE.loo.arch6.fs-rfm.improvement", gain, 6, "loo_arch6", "-loo[fs-rfm].delta_vs_full",
          kind="derived",
          note=f"Table A9's note: removing fs-rfm improves AUROC by about {gain:.6f}, an effect of 0.000")
    # App. H.1: the best single expert of arch6 and FS1's margins over it
    hc = load_json("appendix_h_cells")
    e = L.add(T, "text.appH1.arch6.best_expert", hc["summary"]["best_arch6_expert"], 0, "appendix_h_cells",
              "summary.best_arch6_expert", note="App. H.1: the best single expert of arch6")
    best = L.add(T, "text.appH1.arch6.best_expert_avg", hc["summary"]["best_arch6_avg"], 3, "appendix_h_cells",
                 "summary.best_arch6_avg", note="App. H.1: its macro AUROC")
    arch6 = read_pools()["arch6"]
    if e != max(arch6, key=lambda q: hc["per_expert"][q]["avg_auc_8"]) or best != hc["per_expert"][e]["avg_auc_8"]:
        raise SystemExit("App. H.1: summary.best_arch6_expert is not the arch6 expert with the best macro AUROC")
    L.add(T, "text.appH1.arch6.fs1_std_margin", load_json("ds_arch6_raw")["avg_auc_8"] - best, 3, "ds_arch6_raw",
          "avg_auc_8 - 16_individual_expert_auc_corrected.json summary.best_arch6_avg", kind="derived",
          note="App. H.1: FS1 on the standard split minus the best single expert")
    L.add(T, "text.appH1.arch6.fs1_logo_margin", load_json("mitigation")["logo_arch6"]["fs1"] - best, 3, "mitigation",
          "logo_arch6.fs1 - 16_individual_expert_auc_corrected.json summary.best_arch6_avg", kind="derived",
          note="App. H.1: FS1 under LOGO minus the best single expert")
    # App. H.2: FS1 minus PS4 on the standard split, in macro AUPRC and AUROC
    au = load_json("auprc")["pools"]
    for pool in ("arch6", "clip4"):
        f, p = au[pool]["methods"]["FS_1_raw_feat_logreg"], au[pool]["methods"]["PS_4_xgboost"]
        pre = f"pools.{pool}.methods"
        ap = L.add(T, f"text.appH2.{pool}.auprc_margin", f["avg_auprc"] - p["avg_auprc"], 3, "auprc",
                   f"{pre}.FS_1_raw_feat_logreg.avg_auprc - {pre}.PS_4_xgboost.avg_auprc", kind="derived",
                   note="App. H.2: FS1 minus PS4 in macro AUPRC (standard split)")
        ar = L.add(T, f"text.appH2.{pool}.auroc_margin", f["avg_auc"] - p["avg_auc"], 3, "auprc",
                   f"{pre}.FS_1_raw_feat_logreg.avg_auc - {pre}.PS_4_xgboost.avg_auc", kind="derived",
                   note="App. H.2: the corresponding AUROC margin")
        if not (ap > 0 and ar > 0):
            raise SystemExit(f"App. H.2: 'the positive point estimates' does not hold on {pool}")
    # App. I.1: the rows of the two three-class constructions
    for key, label in (("k3_view_a", "A"), ("k3_view_b", "B")):
        x = load_json(key)
        n = sum(f["n_test"] for f in x["folds"].values())
        oof = (x.get("pooled_oof") or {}).get("n_rows")
        if len(x["folds"]) != x["primary"]["n_folds"] or oof not in (None, n):
            raise SystemExit(f"App. I.1: the folds of {key} do not match its primary result or pooled rows")
        L.add(T, f"text.appI1.{key}.rows", n, 0, key, "sum over folds of folds.<k>.n_test", kind="derived",
              note=f"App. I.1: construction {label}, rows in {WORDS[len(x['folds'])]} folds")
    # App. I.3: the batch the prediction times divide by, the TRAIN size of the fits and the run record's platform
    cost = load_json("cost")
    cc = cost["combiner_costs"]
    batch = [cc[k]["predict_sec_total"] / cc[k]["predict_us_per_sample"] * 1e6
             for k in ("FS_1_config_split_timing", "PS_predict_only")]
    r = load_json("reuse_check")["route_b"]["content_reuse"]["per_generator"]
    n_test = sum(r[g]["n"] for g in GENERATORS)
    if any(abs(b - n_test) > 1e-6 for b in batch):
        raise SystemExit("App. I.3: the prediction times are not per sample of the full test batch")
    L.add(T, "text.appI3.cost.batch", round(batch[0]), 0, "cost", "combiner_costs.FS_1_config_split_timing."
          "predict_sec_total / predict_us_per_sample x 10^6 (PS_predict_only gives the same)", kind="derived",
          note="App. I.3: the batch of test samples that the prediction times divide by")
    L.add(T, "text.appI3.cost.fit_samples", load_json("strict")["arm_full"]["n_fit"], 0, "strict", "arm_full.n_fit",
          note="App. I.3: the TRAIN samples of the fit-and-predict timings")
    L.add(T, "text.appI3.cost.python", cost["host"]["python"], 0, "cost", "host.python",
          note=f"App. I.3: the run record's platform (host.platform {cost['host']['platform']}, printed Windows)")


ALL_TABLES = (("main", p_main), ("audit", p_audit), ("methods", p_methods), ("ladder", p_ladder),
                ("arch6", p_arch6), ("clip4", p_clip4), ("dim_sweep", p_dim_sweep), ("fs2_stability", p_fs2_stability),
                ("bottleneck", p_bottleneck), ("rq2", p_rq2), ("loo", p_loo), ("logo", p_logo),
                ("logo_inferential", p_logo_inferential), ("mde", p_mde), ("weighting", p_weighting),
                ("strict", p_strict), ("fs1_pools", p_fs1_pools), ("experts", p_experts),
                ("prevalence", p_prevalence), ("k3_heads", p_k3_heads), ("k3_views", p_k3_views),
                 ("cost", p_cost), ("tiers", p_tiers), ("heuristics", p_heuristics),
                ("sensitivity", p_sensitivity), ("sensitivity_heads", p_sensitivity_heads), ("overlap", p_overlap), ("small_sets", p_small_sets))


# Historical ledger table identifiers are intentionally stable. Current numbering is a separate map.
ADDITIONAL_LABELS = frozenset({"mde", "heuristics", "sensitivity_heads"})
PAPER_TABLES = tuple((label, fn) for label, fn in ALL_TABLES if label not in ADDITIONAL_LABELS)
CURRENT_PAPER_NAMES = dict(PAPER_NAMES)
for _label, _name in PAPER_NAMES.items():
    if _name.startswith("Table A") and int(_name[7:]) >= 13 and _label not in ADDITIONAL_LABELS:
        CURRENT_PAPER_NAMES[_label] = "Table A" + str(int(_name[7:]) - 1)
for _label in ADDITIONAL_LABELS:
    CURRENT_PAPER_NAMES[_label] = "Additional analysis (not a printed table)"


# The numbered sensitivity table precedes the method specifications.
for _label, _name in list(CURRENT_PAPER_NAMES.items()):
    if _name.startswith("Table A"):
        CURRENT_PAPER_NAMES[_label] = "Table A" + str(int(_name[7:]) + 1)
CURRENT_PAPER_NAMES["sensitivity"] = "Table A1"

# ---- the check against the paper (authoring only: the paper's LaTeX source is not part of the release)

_NUM = re.compile(r"(?<![A-Za-z_\d.])-?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)|\d+\.\d+|\d+)(?:[KM](?![A-Za-z]))?")


def table_numbers(text):
    """The numbers of a LaTeX tabular in reading order, as printed (signs '+' dropped, '-' kept, ',' removed)."""
    s = re.sub(r"\\begin\{tabular\}\{(?:[^{}]|\{[^{}]*\})*\}", " ", text).replace("\\end{tabular}", " ")
    s = re.sub(r"\\c(?:mid)?rule(?:\([^)]*\))?\{[^}]*\}|\\cline\{[^}]*\}", " ", s)
    s = re.sub(r"\\(?:ref|cite|label|eqref)\{[^}]*\}", " ", s)
    s = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", " ", s)
    s = re.sub(r"\\(?:psmethod|fsmethod)\{[^}]*\}", " M ", s)
    s = s.replace("{,}", "").replace("$-$", "-").replace("{+}", "+").replace("{-}", "-").replace("\u2212", "-")
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"\\.", " ", s).replace("$", " ").replace("{", " ").replace("}", " ")
    s = re.sub(r"(?<=\d)-(?=\d)", " ", s)
    return [t.replace(",", "") for t in _NUM.findall(s)]


# version 14: the two sensitivity tables of Appendix A (paper v12) have no label; each is the first tabular after the
# paragraph that opens it. Version 15: so has the overlap table of Appendix A (paper v12)
UNLABELED = {"overlap": "Overlap between combiner training and test data.",
             "small_sets": "Sensitivity to small test sets."}


def paper_ref(label):
    """Version 14: how the paper's source names a table: its label, or the paragraph before a table without one."""
    return f"tab:{label}" if label not in UNLABELED else f"after \\paragraph{{{UNLABELED[label]}}}"


def outer_tabular(s, start=0):
    """Version 15: the first tabular of `s` from `start`, tabulars nested in its cells included (the two-line header
    cells of Table A19 in paper v12); None when there is none or it does not close."""
    i = s.find("\\begin{tabular}", start)
    if i < 0:
        return None
    depth = 0
    for m in re.finditer(r"\\(begin|end)\{tabular\}", s[i:]):
        depth += 1 if m.group(1) == "begin" else -1
        if depth == 0:
            return s[i:i + m.end()]
    return None


def printed_tables(tex_path):
    tex = Path(tex_path).read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", tex, re.S):
        lab = re.search(r"\\label\{tab:([^}]*)\}", m.group(0))
        tb = outer_tabular(m.group(0))                                   # version 15: nested tabulars
        if lab and tb:
            out[lab.group(1)] = tb
    for label, title in UNLABELED.items():                              # version 14: the tables without a label
        i = tex.find("\\paragraph{" + title + "}")
        tb = outer_tabular(tex, i) if i >= 0 else None
        if tb:
            out[label] = tb
    return out


def check_against_paper(bodies, tex_path):
    """Per paper table: are the generated numbers, in reading order, the printed ones? Version 10: and are the words
    the printed ones (whitespace normalised)?"""
    printed = printed_tables(tex_path)
    lines, bad = [], 0
    for label, body in bodies.items():
        if label not in printed:
            lines.append(f"| {CURRENT_PAPER_NAMES[label]} | `{paper_ref(label)}` | - | - | NOT IN PAPER |")
            bad += 1
            continue
        g, p = table_numbers(body), table_numbers(printed[label])
        gt, pt = " ".join(body.split()), " ".join(printed[label].split())          # version 10: the words too
        if g == p and gt == pt:
            lines.append(f"| {CURRENT_PAPER_NAMES[label]} | `{paper_ref(label)}` | {len(g)} | {len(p)} | identical |")
            continue
        if g == p:
            bad += 1
            i = next((k for k, (a, b) in enumerate(zip(gt, pt)) if a != b), min(len(gt), len(pt)))
            cut = lambda s: s[max(0, i - 20):i + 40].replace("|", "/").replace("`", "'")  # noqa: E731
            lines.append(f"| {CURRENT_PAPER_NAMES[label]} | `{paper_ref(label)}` | {len(g)} | {len(p)} | numbers identical; "
                         f"TEXT DIFFERS at character {i + 1}: generated `{cut(gt)}`; printed `{cut(pt)}` |")
            continue
        bad += 1
        i = next((k for k, (a, b) in enumerate(zip(g, p)) if a != b), min(len(g), len(p)))
        lines.append(f"| {CURRENT_PAPER_NAMES[label]} | `{paper_ref(label)}` | {len(g)} | {len(p)} | DIFFERENT from number "
                     f"{i + 1}: generated {' '.join(g[i:i + 6])}; printed {' '.join(p[i:i + 6])} |")
    missing = sorted(set(printed) - set(bodies))
    for label in missing:
        lines.append(f"| ? | `tab:{label}` | - | - | NOT GENERATED |")
        bad += 1
    return lines, bad


# ---------------------------------------------------------------- main

def statistics_problem(st):
    """Explain why the LOGO statistics `st` (mitigation_stats.json) cannot feed the paper's
    tables, or None when they can."""
    ic = st.get("input_completeness")
    if ic is not None and ic.get("complete") is not True:
        return "they are marked as not complete (" + "; ".join(ic.get("notes") or ["no notes"]) + ")"
    boundary = st.get("logo_boundary") or {}
    for pool, b in [("arch6", st.get("logo_arch6") or {}),
                    *((pool, boundary.get(pool) or {}) for pool in ("clip4", "strong4", "weak6"))]:
        if b.get("folds_missing"):
            return f"{pool}: folds missing ({', '.join(b['folds_missing'])})"
        folds = b.get("folds") or []
        if len(folds) != len(GEN_LABEL) or set(folds) != set(GEN_LABEL):
            return f"{pool}: expected all {len(GEN_LABEL)} held-out generators exactly once"
        n = ((b.get("stats") or {}).get("fs1_full") or {}).get("n")
        if n != len(folds):
            return f"{pool}: FS1 summary covers {n} folds, expected {len(folds)}"
        if b.get("ladder", "full" if pool == "arch6" and ic is None else None) != "full":
            return f"{pool}: expected full-ladder statistics"
        if any(len(b.get(key) or []) != len(folds)
               for key in ("n_test", "ceiling_full_per_fold", "ceiling_method_per_fold")):
            return f"{pool}: incomplete fold-level statistics"
    exclusion = (st.get("logo_arch6") or {}).get("exclusion")
    if exclusion != "content":
        return f"arch6: exclusion {exclusion!r}, not 'content' (the primary LOGO runs exclude shared training rows)"
    return None


def summary_problem(s):
    """Explain why the sensitivity summary cannot feed the Appendix A tables and text, or return None."""
    ic = s.get("input_completeness")
    if ic is not None and ic.get("complete") is not True:
        return "it is marked as not complete (" + "; ".join(ic.get("notes") or ["no notes"]) + ")"
    four = ("arch6", "clip4", "strong4", "weak6")
    need = ([("logo", p) for p in four] + [("paper_heads", p) for p in four] + [("mechanism", "arch6")]
            + [("standard", p) for p in ("arch6", "clip4")])
    lacking = [f"{part}.{pool}" for part, pool in need if pool not in (s.get(part) or {})]
    effects = {k.split(" (")[0] for k in (s.get("audit") or {})}
    lacking += [f"audit: {e}" for e in ("overlap distortion", "baseline-ladder distortion, arch6",
                                        "baseline-ladder distortion, weak6") if e not in effects]
    return f"it lacks {', '.join(lacking)}" if lacking else None


RUN_PAIRS = (("logo_arch6", "logo_rungs", "the reviewed LOGO run", "the rebuttal's richer-input run"),
             ("logo_arch6", "logo_combined", "the reviewed LOGO run", "the rebuttal's combined ceiling"),
             ("cd_logo_arch6", "cd_rungs_arch6", "the arch6 LOGO run", "its richer-input run"),
             ("cd_logo_arch6", "mech_logo_arch6", "the arch6 LOGO run", "its mechanism run"),
             ("cd_logo_clip4", "mech_logo_clip4", "the clip4 LOGO run", "its mechanism run"))


def runs_problem():
    """Explain why runs that the tables use together do not belong together, or
    why the full ladder of a run whose ceiling they print is not complete (None when neither)."""
    for a, b, na, nb in RUN_PAIRS:
        why = ladder_checks.pairing_problem(load_json(a), load_json(b), src(a), src(b), (na, nb))
        if why:
            return f"{SOURCES[b]} with {SOURCES[a]}: {why}"
    for a, b in (("logo_arch6", "logo_rungs"), ("cd_logo_arch6", "cd_rungs_arch6")):
        gaps = ladder_checks.ladder_gaps(load_json(a), load_json(b))
        if gaps:
            return f"{SOURCES[a]} with {SOURCES[b]}: the full ladder is not complete ({'; '.join(gaps)})"
    ic = load_json("logo_combined").get("input_completeness")
    if ic is not None and ic.get("complete") is not True:
        return f"{SOURCES['logo_combined']} is marked as not complete ({'; '.join(ic.get('notes') or [])})"
    return None


def pairing_problem():
    """Explain why files that the tables print side by side disagree, or None. Table A10 prints
    FS1 and FS2 of the arch6 LOGO run beside the ceiling and the differences of the statistics (mitigation_stats.json);
    Table 1 prints the standard-split ceiling of full_ladder_stats.json, Table A2 the richer-input values of the run A
    files. Each pair must come from the same run."""
    m = load_json("mitigation")
    la = m["logo_arch6"]
    per = {f["held_out"]: f for f in load_json("cd_logo_arch6")["per_fold"]}
    if sorted(per) != sorted(la["folds"]):
        return "Table A10: the folds of the arch6 LOGO run are not those of the statistics"
    for i, g in enumerate(la["folds"]):
        for what, fs, key in (("FS1", per[g]["raw_feat_auc"], "fs1_full"), ("FS2", _fs2(per[g]), "fs2_full")):
            if abs(fs - la["ceiling_full_per_fold"][i] - la["stats"][key]["per_fold"][i]) > 1e-12:
                return (f"Table A10, {GEN_LABEL[g]}: {what} of the LOGO run minus the ceiling is not the difference of "
                        "the statistics; recompute the statistics (harness step B32) after the run")
    for what, values, mean in (("FS1", [per[g]["raw_feat_auc"] for g in la["folds"]], la["fs1"]),
                               ("FS2", [_fs2(per[g]) for g in la["folds"]], la["fs2"])):
        if abs(sum(values) / len(values) - mean) > 1e-12:
            return f"Table A10: the mean {what} of the LOGO run is not the statistics' average"
    standard = load_json("full_ladder")["standard"]
    for pool in ("arch6", "clip4"):
        if abs(standard[pool]["ps_ceiling"] - load_json(f"ladder_{pool}")["deployable_ceiling"]["macro"]) > 1e-12:
            return (f"Tables 1 and A2, {pool}: the standard-split ceiling of full_ladder_stats.json is not that of the "
                    "run A file; recompute the standard-split statistics (harness step C08) after the run")
    return None


def summary_warnings():
    """Version 12: the cells of the primary columns of the sensitivity tables (Appendix A since paper v12, Table A24
    before; from the sensitivity summary) that differ, at the printed precision, from the statistics that Tables 1
    and A10 print. The harness does not rebuild the summary (camera_ready/scripts/sensitivity/summarize_v2.py does),
    so a difference is reported, not fatal."""
    m, s, out = load_json("mitigation"), load_json("sens_summary"), []
    for pool in ("arch6", "clip4", "strong4", "weak6"):
        mine = logo_pool(m, pool)[0]["stats"]["fs1_full"]
        for side, block in (("primary", s["logo"][pool]["primary"]["stats"]["fs1_full"]),
                            ("primary_22", s["paper_heads"][pool]["primary_22"]["stats"]["fs1_full"])):
            pairs = [("mean_delta", block["mean_delta"], mine["mean_delta"])]
            pairs += [(f"ci95[{i}]", block["ci95"][i], mine["ci95"][i]) for i in (0, 1)]
            for k, a, b in pairs:
                if f"{a:.3f}" != f"{b:.3f}":
                    out.append(f"{pool} {side} {k}: summary {a:.3f}, statistics {b:.3f}")
    ov = next(v for k, v in s["audit"].items() if k.startswith("overlap distortion"))
    if f"{ov['paper']:.3f}" != f"{m['audit']['overlap_distortion']:.3f}":
        out.append(f"overlap distortion: summary {ov['paper']:.3f}, statistics {m['audit']['overlap_distortion']:.3f}")
    return out


import math

# Version 16: approved v13 review edits. Historical ledger rows keep their identifiers and values.
SOURCES.update({
    "mech_standard_clip4": f"{MIT}/mechanism/run_c_v2_standard_clip4.json",
    "install_doc": "03.camera-ready/code/INSTALL.md",
})


def text_rows_v16(L):
    """Trace the added v13 numerical statements to stored results; no fitting or resampling."""
    def add(q, value, source, key, digits=3, kind="json", note=""):
        return L.add("text", "text.v16." + q, value, digits, source, key, kind=kind, note=note)

    mit = load_json("mitigation")
    inflation = {}
    for pool in ("arch6", "clip4", "strong4", "weak6"):
        branch = "logo_arch6" if pool == "arch6" else f"logo_boundary.{pool}"
        stats = mit["logo_arch6"] if pool == "arch6" else mit["logo_boundary"][pool]
        run_key = f"cd_logo_{pool}"
        folds = load_json(run_key)["per_fold"]
        by_gen = {f["held_out"]: f for f in folds}
        if len(folds) != 8 or len(by_gen) != 8 or set(by_gen) != set(stats["folds"]):
            raise SystemExit(f"v13 baseline comparison: {pool} does not have the same eight unique folds")
        ordered = [by_gen[g] for g in stats["folds"]]
        for f in ordered:
            values = [f["ps_means"][k] for k in ("PS_1_prob_avg", "PS_2_logreg", "PS_3_mlp_logits", "PS_4_xgboost")]
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise SystemExit(f"v13 baseline comparison: non-finite or invalid AUROC in {pool}")
        truncated = sum(max(f["ps_means"][k] for k in ("PS_1_prob_avg", "PS_2_logreg", "PS_3_mlp_logits"))
                        for f in ordered) / 8
        inflation[pool] = stats["ceiling_full"] - truncated
        add(f"baseline.{pool}.truncated_inflation", inflation[pool], run_key,
            f"mitigation_stats.json {branch}.ceiling_full - mean of per_fold max(PS_1, PS_2, PS_3)", kind="derived")
        add(f"baseline.{pool}.ps4_mean", sum(f["ps_means"]["PS_4_xgboost"] for f in ordered) / 8,
            run_key, "mean(per_fold.ps_means.PS_4_xgboost)", kind="derived")
        if pool == "arch6":
            minimum = sum(max(f["ps_means"][k] for k in ("PS_1_prob_avg", "PS_4_xgboost")) for f in ordered) / 8
            add("baseline.arch6.minimum_margin", stats["fs1"] - minimum, run_key,
                "mitigation_stats.json logo_arch6.fs1 - mean of per_fold max(PS_1, PS_4)", kind="derived")
        else:
            add(f"boundary.{pool}.fs2_minus_fs1", stats["fs2"] - stats["fs1"], "mitigation",
                f"{branch}.fs2 - {branch}.fs1", kind="derived", note="Point estimate, not a significance claim")
            add(f"boundary.{pool}.fs2_minus_ceiling", stats["stats"]["fs2_full"]["mean_delta"],
                "mitigation", f"{branch}.stats.fs2_full.mean_delta")
            for i, label in enumerate(("lo", "hi")):
                add(f"boundary.{pool}.fs1_ci_{label}", stats["stats"]["fs1_full"]["ci95"][i], "mitigation",
                    f"{branch}.stats.fs1_full.ci95[{i}]", note="Nominal percentile-bootstrap interval")
    add("baseline.max_truncated_inflation", max(inflation.values()), "mitigation",
        "max of text.v16.baseline.<pool>.truncated_inflation (four content-disjoint LOGO run sources)", kind="derived")
    summary = load_json("sens_summary")
    audit_key = "baseline-ladder distortion, arch6 (full-ladder ceiling minus best of PS1-PS3)"
    for arm in ("paper", "deduplicated"):
        add(f"sensitivity.arch6.baseline_inflation.{arm}", summary["audit"][audit_key][arm], "sens_summary",
            f"audit.{audit_key}.{arm}")
    methods = ("FS_probe2_logreg", "FS_1_raw_feat_logreg", "FS_pca2_logreg", "PS_4_xgboost")
    for protocol, source, field in (("standard", "mech_standard_clip4", "macro"),
                                     ("logo", "mitigation", "mechanism_logo_clip4.mean_auroc")):
        values = load_json(source)
        for part in field.split("."):
            values = values[part]
        for method in methods:
            add(f"width.clip4.{protocol}.{method}", values[method], source, f"{field}.{method}")
    comp = mit["mechanism_logo_clip4"]["comparisons"]["FS_probe2_logreg_vs_FS_1"]
    path = "mechanism_logo_clip4.comparisons.FS_probe2_logreg_vs_FS_1"
    add("width.clip4.logo.probe_minus_fs1", comp["mean_delta"], "mitigation", path + ".mean_delta")
    for i, label in enumerate(("lo", "hi")):
        add(f"width.clip4.logo.probe_minus_fs1_ci_{label}", comp["ci95"][i], "mitigation", path + f".ci95[{i}]")
    # The platform example is a documented installation result, not a new refit.
    doc = src("install_doc").read_text(encoding="utf-8")
    match = re.search(r"Avg\(8\) = ([0-9.]+)`; ([0-9.]+) on Linux, within the tolerance of ([0-9.]+)\)", doc)
    if not match:
        raise SystemExit("v13 platform disclosure: documented FS1 smoke result not found in INSTALL.md")
    for label, value, digits in zip(("stored", "linux", "smoke_tolerance"), match.groups(), (4, 4, 3)):
        add("platform.fs1." + label, float(value), "install_doc", "FS1 smoke example: " + label,
            digits=digits, kind="report", note="Documented installation result; not a full-run acceptance tolerance")


def main() -> int:
    global INPUTS, OUT_TABLES, OUT_PROV, OUT_CHECK
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-fail", action="store_true", help="exit 0 even if a posted number disagrees")
    ap.add_argument("--inputs", type=Path, help="flat folder with every source file under its own name "
                    "(release layout; supply this for a public build)")
    ap.add_argument("--out", type=Path, help="write tables, provenance and the posted check here "
                    "(required for a public build)")
    ap.add_argument("--paper-tex", type=Path, help="compare every generated paper table with the printed one in "
                    "this LaTeX source and write 07_paper_tables_check.md (authoring check; the paper is not "
                    "part of the release)")
    args = ap.parse_args()
    if args.inputs is None and not (CH3 / "01.initial_submission").is_dir():          # version 8: a clone
        ap.error("pass --inputs camera_ready/inputs and --out <folder>; implicit paths "
                 "are not supported for public builds")
    if args.inputs is not None and args.out is None and not (CH3 / "01.initial_submission").is_dir():
        ap.error("pass --out <folder> to choose the public build destination")
    if args.inputs is not None:
        INPUTS = args.inputs.resolve()
    if args.out is not None:
        out = args.out.resolve()
        OUT_TABLES, OUT_PROV, OUT_CHECK = out / "tables", out / "05_number_provenance.csv", out / "06_posted_number_check.md"

    why = statistics_problem(load_json("mitigation"))                 # version 9: before anything is written
    if why:
        raise SystemExit(f"mitigation_stats.json cannot feed the paper's tables: {why}. Recompute the statistics from "
                         "the complete runs (camera_ready/scripts/authoring/full_ladder_stats_v13.py without the "
                         "options of a partial analysis).")
    why = pairing_problem()                                           # version 12: before anything is written
    if why:
        raise SystemExit(f"The paper's tables would combine files that disagree: {why}. Nothing was written.")
    why = runs_problem()                                              # version 13: before anything is written
    if why:
        raise SystemExit(f"The paper's tables would combine runs that do not belong together: {why}. Nothing was "
                         "written.")
    why = summary_problem(load_json("sens_summary"))                  # version 13
    if why:
        raise SystemExit(f"summary.json of the sensitivity checks cannot feed the sensitivity tables of Appendix A and "
                         f"the text: {why}. Rebuild "
                         "it from complete outputs (camera_ready/scripts/sensitivity/summarize_v2.py without "
                         "--allow-partial). Nothing was written.")
    for w in summary_warnings():                                      # version 12
        print(f"WARNING (App. A, primary column of the sensitivity tables): {w}; rerun Block B and the sensitivity "
              "refits in the same checkout, then rebuild the sensitivity summary "
              "(camera_ready/scripts/sensitivity/summarize_v2.py) from those outputs")
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_PROV.parent.mkdir(parents=True, exist_ok=True)
    L = Ledger()
    tables = {
        "t01_logo_arch6_primary": t01_logo_primary(L),
        "t02_expanded_ladder": t02_expanded_ladder(L),
        "t03_width_controls": t03_width(L),
        "t04_k3_heads": t04_k3_heads(L),
        "t05_k3_fixed_output_views": t05_k3_views(L),
        "t06_cost_access_ledger": t06_cost(L),
        "t07_reuse_per_generator": t07_reuse(L),
        "t08_strict_refit_arms": t08_strict(L),
    }
    tables["t18_full_ladder"] = t18_full_ladder(L)
    tables["t09_logo_boundary_pools"] = t09_boundary(L)
    tables["t10_appendix_h_expert_means"] = t10_appendix_h(L)
    tables["t11_appendix_h_individual_experts"] = t11_appendix_h_cells(L)
    tables["t12_release_level_reuse"] = t12_release_reuse(L)
    tables["t13_integrity_counts"] = t13_integrity(L)
    tables["t14_expert_exposure_by_training_list"] = t14_exposure_regimes(L)
    tables["t15_expert_exposure_by_pool"] = t15_exposure_pools(L)
    tables["t19_logo_arch6_folds"] = t19_mitigation_logo(L)
    tables["t20_logo_inferential"] = t20_logo_inferential(L)
    tables["t16_main_results"] = t16_main_results(L)
    tables["t17_pool_deltas"] = t17_pool_deltas(L)
    tables["t21_rq2_both_protocols"] = t21_rq2_both_protocols(L)
    audit_rows(L)
    mark_superseded(L)
    all_bodies, locations = {}, {}
    for label, fn in ALL_TABLES:
        before = len(L.rows)
        all_bodies[label] = fn(L)
        locations[label] = {
            "status": "additional" if label in ADDITIONAL_LABELS else "printed",
            "current_name": CURRENT_PAPER_NAMES[label], "historical_ledger_name": PAPER_NAMES[label],
            "quantity_ids": [row["quantity"] for row in L.rows[before:]],
        }
    # p_main emits these stable subset ids, which moved to App. F in v17.
    moved = [q for q in locations["main"]["quantity_ids"] if q.endswith(".logo.delta_n500")]
    locations["main"]["quantity_ids"] = [q for q in locations["main"]["quantity_ids"] if q not in moved]
    locations["small_sets"]["quantity_ids"].extend(moved)
    diagnostic_ids = [q for q in locations["ladder"]["quantity_ids"] if q.endswith(".oracle_domain")]
    locations["ladder"]["quantity_ids"] = [q for q in locations["ladder"]["quantity_ids"] if q not in diagnostic_ids]
    locations["ladder"]["text_quantity_ids"] = diagnostic_ids
    locations["ladder"]["text_location"] = "Appendix B: Target-label diagnostic"
    paper = {label: body for label, body in all_bodies.items() if label not in ADDITIONAL_LABELS}
    text_rows(L)                                                     # versions 7, 8, 11, 12: the text's numbers
    for name, body in tables.items():
        header = (f"% {name}: generated by camera_ready/scripts/authoring/generate_camera_ready_tables_v18.py\n"
                  f"% from stored artefacts; do not edit by hand (provenance: 05_number_provenance.csv, written with "
                  f"these tables)\n")
        if name in SUPERSEDED:
            header += f"% SUPERSEDED: {SUPERSEDED[name]}\n"
        (OUT_TABLES / f"{name}.tex").write_text(header + body, encoding="utf-8")
    (OUT_TABLES / "paper").mkdir(parents=True, exist_ok=True)
    (OUT_TABLES / "additional").mkdir(parents=True, exist_ok=True)
    for label, body in all_bodies.items():
        folder = "additional" if label in ADDITIONAL_LABELS else "paper"
        if folder == "additional":
            stale = OUT_TABLES / "paper" / f"{label}.tex"
            if stale.exists():
                old_text = stale.read_text(encoding="utf-8")
                if "generated by camera_ready/scripts/authoring/generate_camera_ready_tables" not in old_text[:400]:
                    raise SystemExit(f"Refusing to move a non-generated file: {stale}")
                stale.replace(OUT_TABLES / "additional" / f"{label}.tex")
        header = (f"% {CURRENT_PAPER_NAMES[label]} ({paper_ref(label)}): generated by camera_ready/scripts/authoring/"
                  f"generate_camera_ready_tables_v18.py\n% from stored artefacts; do not edit by hand. "
                  f"Historical ledger group: '{PAPER_NAMES[label]}'; current locations: paper_table_locations.json\n")
        (OUT_TABLES / folder / f"{label}.tex").write_text(header + body, encoding="utf-8")
    (OUT_PROV.parent / "paper_table_locations.json").write_text(
        json.dumps({"paper_version": "v13 shortened appendix", "printed_tables": len(paper),
                    "ledger_table_field": "historical identifiers retained unchanged", "tables": locations},
                   indent=2) + "\n", encoding="utf-8")

    text_rows_v15(L)                                                 # version 15: paper v12 (Apps. B, E, H and I)
    text_rows_v16(L)                                                 # approved v13 review edits
    with open(OUT_PROV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(L.rows[0]))
        w.writeheader()
        w.writerows(L.rows)

    lines, fails = posted_check(L)
    n_pass = sum(1 for ln in lines if "| PASS" in ln)
    doc = [
        "# 06 Posted-number check",
        "",
        "Generated by `camera_ready/scripts/authoring/generate_camera_ready_tables_v18.py`; do not edit.",
        "Compares published discussion values with the supplied reference results at their display precision.",
        "Sources and current paper locations are listed in the number ledger and paper table map.",
        "",
        f"**{n_pass} of {len(lines)} posted numbers agree; {fails} disagree or are missing.**",
        "",
        "| Quantity | Thread | Posted | From artefact | Status |",
        "|---|---|---|---|---|",
        *lines,
        "",
    ]
    OUT_CHECK.write_text("\n".join(doc), encoding="utf-8")
    def shown(p):
        return p.relative_to(CAMERA) if p.is_relative_to(CAMERA) else p

    print(f"tables: {len(tables)} -> {shown(OUT_TABLES)}")
    print(f"provenance rows: {len(L.rows)} -> {shown(OUT_PROV)}")
    print(f"posted numbers: {n_pass}/{len(lines)} agree, {fails} disagree or missing -> {shown(OUT_CHECK)}")
    n_rows = sum(len(locations[label]["quantity_ids"]) for label in paper)
    print(f"paper tables: {len(paper)} -> {shown(OUT_TABLES / 'paper')}; {n_rows} associated historical ledger rows")
    check_fails = 0
    if args.paper_tex is not None:
        rows, check_fails = check_against_paper(paper, args.paper_tex)
        out_check = OUT_PROV.parent / "07_paper_tables_check.md"
        out_check.write_text("\n".join([
            "# 07 Paper tables: generated against printed", "",
            "Generated by `camera_ready/scripts/authoring/generate_camera_ready_tables_v18.py --paper-tex`; do not "
            "edit.",
            f"Each table generated from stored artefacts against the table printed in `{args.paper_tex.name}`: "
            "the numbers of the tabular in reading order (signs '+' ignored, '-' kept, thousands separators removed).",
            "", f"**{len(paper) - check_fails} of {len(paper)} tables identical.**", "",
            "| Table | Label | Numbers generated | Numbers printed | Result |", "|---|---|---|---|---|", *rows, ""]),
            encoding="utf-8")
        print(f"paper check: {len(paper) - check_fails}/{len(paper)} tables identical -> {shown(out_check)}")
    return 0 if ((fails == 0 and check_fails == 0) or args.no_fail) else 1


if __name__ == "__main__":
    sys.exit(main())
