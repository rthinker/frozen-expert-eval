# Additional analyses

This document provides the power calculation and secondary training-regime comparisons
that accompany the main results. Harness step C24 writes the additional tables under
`camera_ready/output/tables/tables/additional/`: `mde.tex` (power thresholds),
`sensitivity_heads.tex` (head-setting comparison) and `heuristics.tex` (DF40 examples).
Paths are relative to the repository root. The [table-location map](paper_table_locations.json)
identifies the 25 DF40 tables and distinguishes these additional analyses.

## Power at eight folds

The original power-simulation script and outputs were not retained. The released
`camera_ready/scripts/analyses/appF_power_mde_v2.py` reconstructs the calculation below;
step C10 checks the three retained threshold values.

The simulation draws 10,000 sets of eight independent Normal fold differences at each
standard deviation, with 100,000-set precision checks near the bootstrap's 80% detection
threshold. The threshold is the smallest mean effect on a 0.005 AUROC grid for which the
nominal 95% percentile interval excludes zero in at least 80% of simulated sets.
These simulations do not model dependence between LOGO folds and therefore do not
estimate the actual study's power.

| Fold-difference SD | Mean effect giving at least 80% detection |
|---|---|
| 0.05 | +0.050 |
| 0.08 | +0.080 |
| 0.12 | +0.115 |

The observed FS1 fold differences have SD 0.085 using the population convention,
or 0.091 using the sample convention. At SD 0.08, the bootstrap threshold is close
to the observed mean advantage. However, the nominal 95% interval excludes zero in
approximately 11% of simulations with no true effect. Its lower detection threshold
therefore partly reflects inflated false-positive rates. The exact sign-flip test
remains near the nominal 5% rate in this simulation and requires an effect of
approximately +0.095 for 80% power. The Wilcoxon and sign tests are conservative
under this simulation; with eight folds, the two-sided sign test reaches p < 0.05
only with eight differences of the same sign. Read the bootstrap interval alongside
the test results, rather than as stand-alone evidence of statistical significance.

## Additional training-regime pools

These comparisons are secondary, overlap-affected standard-split evidence using the
original ten-method prediction ladder. Two pools keep arch6's six detector variants
but replace face-swap training with face-reenactment or entire-face-synthesis training
(arch6_fr and arch6_efs).

| Pool | Expert training regime | PS4 macro AUROC | FS1 minus PS4 |
|---|---|---|---|
| arch6 | Face swapping | 0.843 | +0.123 |
| arch6_fr | Face reenactment | 0.894 | +0.061 |
| arch6_efs | Entire-face synthesis | 0.901 | +0.059 |

With the detector variants held fixed, the interface gap also varies with the
experts' training regime. Both additional pools were evaluated with the original
ten-method ladder; these comparisons do not use the primary LOGO protocol or the
full prediction-space ceiling.
