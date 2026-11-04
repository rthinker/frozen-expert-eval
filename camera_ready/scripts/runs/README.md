# LOGO and width-control runners

These commands form [Block B of the harness](../review/README.md).
The [release manifest](../../RELEASE_MANIFEST.md) lists exact steps and inputs.

| Task | Current script |
|---|---|
| Generate per-pool content exclusion lists | `logo_exclusions.py` |
| Run the base prediction ladder, FS1 and FS2 | `11_disjoint_data_v4.py` |
| Fit richer prediction-space maps and heads | `08_logo_extended_rungs_v3.py` |
| Combine the two compatible ladders | `09_logo_combined_ceiling_v3.py` |
| Run the standard-split and LOGO width controls | `05_run_c_width_matched_v5.py` |
| Compare runs through the reproduction and parity gates | `compare_runs_v2.py` |

Use `--help` on a script to see its arguments. For a direct run, supply a
separate `--out-dir` or `--out`. The runners default to the repository's
packaged exclusion lists when `--exclusion content` is selected.
The harness backs up supplied outputs before recomputing them. Only the
harness enables checkpoint reuse after validating the run and its artifacts.

`logo_common.py` constructs the folds, applies training exclusions and
makes the stratified 85/15 FIT/SELECT split. `richer_inputs.py` implements the
four maps used by the LOGO richer ladder. `ladder_checks.py` verifies
the candidate sets and compatibility of runs using recorded counts and
partition fingerprints. The supplied runs written before fingerprints are
paired by the digests in [trusted_run_pairs.json](../../provenance/trusted_run_pairs.json).
The [score-precision note](../../output/results/mitigation/SCORE_PRECISION_NOTE.md)
explains why identical saved FS1 prediction arrays can correspond to slightly
different recorded AUROCs in the ladder and width-control runs.
