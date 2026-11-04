# Sensitivity checks: outputs and provenance

These are the outputs of the sensitivity checks of `camera_ready/scripts/sensitivity/`, kept apart from the primary
results (`camera_ready/output/results/mitigation/` and `analyses/`). `SUMMARY.md` and `summary.json` are rebuilt from
them by `summarize_v2.py`, without refitting any model, after it checks that every input is complete (every output,
the eight folds, the full ladder in every fold) and that the runs analysed or compared together belong together:

    python camera_ready/scripts/sensitivity/summarize_v2.py

| Folder | Check | Script | What it holds |
|---|---|---|---|
| `twins.json` | the training duplicates the checks remove (counting only), four pools | `count_twins.py` | per LOGO fold and for the standard split: SELECT rows with an exact feature twin in FIT, the generators of the twins, twins with another label, extra copies inside FIT and inside SELECT |
| `dedup/` | LOGO with exact duplicates removed from each fold's training data, four pools | `dedup_logo_v2.py` | the ladder and richer-input results (`11_disjoint_<pool>_d<d>.json`, `A3c_logo_extended_rungs_<pool>.json`), their fold checkpoints (`folds/`, `rung_folds/`) and per-sample predictions (`scores/`), and the rows removed in each fold (`dedup_counts_*.json`) |
| `dedup/` | the mechanism under LOGO with the same removal, arch6 and clip4 | `dedup_followup_v2.py` | `run_c_v2_logo_<pool>.json`, `mechanism_folds/`, their predictions in `scores/`, `dedup_counts_05_*.json` |
| `dedup/standard/` | the standard split with FIT and SELECT deduplicated, arch6, clip4 and weak6 | `dedup_followup_v2.py` | `run_a_extended_ladder_<pool>.json`, `dedup_counts_standard_<pool>.json` |
| `paper_heads/` | the LOGO richer-input heads of the standard split, four pools | `paper_heads_logo_v2.py` | per fold: the heads' AUROCs, PS_K's selection and settings (`<pool>_<fold>.json`) and predictions (`.npz`) |

Scope: only training data are deduplicated (each LOGO fold's FIT and SELECT; the standard split's FIT and SELECT);
the test sets, with their own repeated images, are unchanged. Deduplication also changes the training weights and
mix, so a shift is not attributable to cleaner validation alone, and an interval that crosses zero is not evidence of
equivalence. The AUPRC of `SUMMARY.md` compares FS1 with each fold's ceiling method, which is chosen by AUROC.

## Reproducing the checks

The commands below use the locked dependencies. The LOGO runners record fold membership fingerprints.
The supplied outputs are paired using `camera_ready/provenance/trusted_run_pairs.json`. File digests are in
`provenance.json`.

Before refitting, rerun Block B in the same checkout with
`python camera_ready/scripts/review/reproduce_all_v4.py --blocks B`. The supplied primary LOGO runs record no
membership fingerprints, so a summary of rerun outputs stops unless it has primary runs from the current runners.

    python camera_ready/scripts/sensitivity/dedup_logo_v2.py
    python camera_ready/scripts/sensitivity/dedup_followup_v2.py
    python camera_ready/scripts/sensitivity/paper_heads_logo_v2.py
    python camera_ready/scripts/sensitivity/count_twins.py

A rerun writes to `camera_ready/output/results/sensitivity_rerun/` (each script's `--out-dir`; `count_twins.py` takes `--out`), never over these
outputs. From the repository root, summarize it with:

    python camera_ready/scripts/sensitivity/summarize_v2.py --dir camera_ready/output/results/sensitivity_rerun
