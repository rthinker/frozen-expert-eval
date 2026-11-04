# Sensitivity checks: summary

Written by `camera_ready/scripts/sensitivity/summarize_v2.py` from the outputs in this folder; no model is refitted.

Inputs checked: every output of the checks is present, every run holds the eight folds and the full ladder, and the runs analysed or compared together are compatible (`camera_ready/scripts/runs/ladder_checks.py`).

Scope: only training data are deduplicated (each LOGO fold's FIT and SELECT; the standard split's FIT and SELECT). The test sets, with their own repeated images, are unchanged, so this is not a fully deduplicated evaluation. Deduplication also changes the training weights and mix (StarGAN-v2 keeps 1,024 of its 4,000 rows), so a shift is not attributable to cleaner validation alone; an interval that crosses zero is not evidence of equivalence. Primary results: camera_ready/output/results/mitigation/ and analyses/.

## 0. Training duplicates

SELECT rows with an exact feature twin in FIT, and extra copies inside FIT, in each LOGO fold (the same in every pool) and on the standard split. Nearly every twin is a StarGAN-v2 row, whose test list repeats images.

| LOGO fold (arch6) | SELECT rows | with a FIT twin | share | extra copies inside FIT |
|---|---|---|---|---|
| whichisreal | 2389 | 622 | 26.0% | 2366 |
| CollabDiff | 2388 | 618 | 25.9% | 2360 |
| MidJourney | 2407 | 626 | 26.0% | 2352 |
| heygen | 2674 | 616 | 23.0% | 2369 |
| starganv2 | 2089 | 3 | 0.1% | 9 |
| deepfacelab | 2667 | 610 | 22.9% | 2377 |
| stargan | 2094 | 648 | 30.9% | 2335 |
| styleclip | 2115 | 607 | 28.7% | 2371 |

The other pools give the same counts: yes.

| Standard split | SELECT rows | with a FIT twin | extra copies inside FIT | extra copies inside SELECT |
|---|---|---|---|---|
| arch6 | 6647 | 793 | 1880 | 167 |
| clip4 | 6647 | 794 | 1882 | 167 |
| strong4 | 6647 | 793 | 1880 | 167 |
| weak6 | 6647 | 793 | 1880 | 167 |

## 1. LOGO with exact duplicates removed from the training data

Statistics against each fold's ceiling over the full prediction-space ladder (the ten methods and the richer inputs). AUPRC: macro over folds, from the saved predictions; the comparator of each fold is its ceiling method, chosen by AUROC.

### arch6

| | primary | training duplicates removed |
|---|---|---|
| PS ceiling (mean over folds) | 0.8174 | 0.8176 |
| FS1 (mean over folds) | 0.8979 | 0.8960 |
| FS1 - ceiling | +0.0806 [+0.0240, +0.1401], 7 of 8, sign-flip p 0.055 | +0.0784 [+0.0245, +0.1353], 6 of 8, sign-flip p 0.055 |
| FS2 - ceiling | +0.0788 [+0.0216, +0.1412], 6 of 8, sign-flip p 0.055 | +0.0687 [+0.0189, +0.1205], 6 of 8, sign-flip p 0.055 |
| FS1 - ceiling, folds with n >= 500 | +0.1147 | +0.1118 |
| AUPRC: FS1 / ceiling method | 0.8980 / 0.8178 | 0.8946 / 0.8297 |
| AUPRC: FS1 - ceiling method | +0.0803 | +0.0648 |

| Fold | rows removed | FIT / SELECT primary | FIT / SELECT dedup | ceiling primary (method) | ceiling dedup (method) | FS1 - ceiling primary / dedup |
|---|---|---|---|---|---|---|
| whichisreal | 2988 | 13535 / 2389 | 10995 / 1941 | 0.5873 (PS_B_learned_weights) | 0.5910 (PS_B_learned_weights) | +0.2267 / +0.2170 |
| CollabDiff | 2981 | 13530 / 2388 | 10996 / 1941 | 0.8844 (PS_C_logreg_cv) | 0.8891 (PS_C_logreg_cv) | +0.1143 / +0.1097 |
| MidJourney | 2983 | 13636 / 2407 | 11101 / 1959 | 0.7128 (PS_C_logreg_cv) | 0.7058 (PS_B_learned_weights) | +0.1550 / +0.1558 |
| heygen | 2988 | 15150 / 2674 | 12610 / 2226 | 0.7694 (uncertainty+MI_logreg) | 0.7608 (PS_D_mlp_probs) | +0.0012 / -0.0067 |
| starganv2 | 12 | 11836 / 2089 | 11826 / 2087 | 0.8146 (PS_B_learned_weights) | 0.8146 (PS_B_learned_weights) | +0.0383 / +0.0434 |
| deepfacelab | 2988 | 15109 / 2667 | 12569 / 2219 | 0.9601 (PS_B_learned_weights) | 0.9601 (PS_B_learned_weights) | -0.0446 / -0.0368 |
| stargan | 2988 | 11863 / 2094 | 9323 / 1646 | 0.9509 (PS_B_learned_weights) | 0.9505 (PS_B_learned_weights) | +0.0195 / +0.0200 |
| styleclip | 2981 | 11979 / 2115 | 9446 / 1667 | 0.8593 (PS_C_logreg_cv) | 0.8687 (PS_C_logreg_cv) | +0.1341 / +0.1250 |

### clip4

| | primary | training duplicates removed |
|---|---|---|
| PS ceiling (mean over folds) | 0.9206 | 0.9178 |
| FS1 (mean over folds) | 0.8462 | 0.8453 |
| FS1 - ceiling | -0.0744 [-0.1817, -0.0019], 5 of 8, sign-flip p 0.250 | -0.0726 [-0.1864, +0.0039], 5 of 8, sign-flip p 0.250 |
| FS2 - ceiling | -0.0280 [-0.1045, +0.0255], 6 of 8, sign-flip p 0.750 | -0.0258 [-0.1246, +0.0460], 6 of 8, sign-flip p 0.750 |
| FS1 - ceiling, folds with n >= 500 | -0.0102 | -0.0002 |
| AUPRC: FS1 / ceiling method | 0.8695 / 0.9182 | 0.8662 / 0.9125 |
| AUPRC: FS1 - ceiling method | -0.0487 | -0.0462 |

| Fold | rows removed | FIT / SELECT primary | FIT / SELECT dedup | ceiling primary (method) | ceiling dedup (method) | FS1 - ceiling primary / dedup |
|---|---|---|---|---|---|---|
| whichisreal | 2988 | 13535 / 2389 | 10995 / 1941 | 0.9275 (rank ECDF_logreg) | 0.9286 (rank ECDF_logreg) | +0.0164 / +0.0099 |
| CollabDiff | 2981 | 13530 / 2388 | 10996 / 1941 | 0.9993 (rank ECDF_logreg) | 0.9992 (rank ECDF_logreg) | +0.0007 / +0.0008 |
| MidJourney | 2983 | 13636 / 2407 | 11101 / 1959 | 0.7314 (PS_3_mlp_logits) | 0.7104 (PS_3_mlp_logits) | -0.0985 / -0.0330 |
| heygen | 2988 | 15150 / 2674 | 12610 / 2226 | 0.7910 (PS_1_prob_avg) | 0.7910 (PS_1_prob_avg) | -0.4125 / -0.4325 |
| starganv2 | 12 | 11836 / 2089 | 11826 / 2087 | 0.9500 (rank ECDF_logreg) | 0.9497 (rank ECDF_logreg) | +0.0157 / +0.0170 |
| deepfacelab | 2988 | 15109 / 2667 | 12569 / 2219 | 0.9716 (PS_A_calibrated_avg) | 0.9700 (PS_A_calibrated_avg) | -0.1216 / -0.1469 |
| stargan | 2988 | 11863 / 2094 | 9323 / 1646 | 0.9972 (PS_F_mlp_logits_es) | 0.9972 (PS_F_mlp_logits_es) | +0.0016 / +0.0015 |
| styleclip | 2981 | 11979 / 2115 | 9446 / 1667 | 0.9966 (PS_B_learned_weights) | 0.9966 (PS_B_learned_weights) | +0.0029 / +0.0029 |

### strong4

| | primary | training duplicates removed |
|---|---|---|
| PS ceiling (mean over folds) | 0.9096 | 0.9093 |
| FS1 (mean over folds) | 0.8501 | 0.8563 |
| FS1 - ceiling | -0.0595 [-0.1545, +0.0075], 5 of 8, sign-flip p 0.312 | -0.0531 [-0.1372, +0.0070], 5 of 8, sign-flip p 0.258 |
| FS2 - ceiling | -0.0282 [-0.0787, +0.0104], 5 of 8, sign-flip p 0.344 | -0.0134 [-0.0619, +0.0225], 6 of 8, sign-flip p 0.773 |
| FS1 - ceiling, folds with n >= 500 | +0.0013 | +0.0001 |
| AUPRC: FS1 / ceiling method | 0.8789 / 0.8983 | 0.8802 / 0.8978 |
| AUPRC: FS1 - ceiling method | -0.0193 | -0.0176 |

| Fold | rows removed | FIT / SELECT primary | FIT / SELECT dedup | ceiling primary (method) | ceiling dedup (method) | FS1 - ceiling primary / dedup |
|---|---|---|---|---|---|---|
| whichisreal | 2988 | 13535 / 2389 | 10995 / 1941 | 0.9224 (rank ECDF_logreg) | 0.9249 (PS_B_learned_weights) | +0.0251 / +0.0220 |
| CollabDiff | 2981 | 13530 / 2388 | 10996 / 1941 | 0.9990 (PS_B_learned_weights) | 0.9990 (PS_B_learned_weights) | +0.0009 / +0.0010 |
| MidJourney | 2983 | 13636 / 2407 | 11101 / 1959 | 0.7249 (PS_2_logreg) | 0.7336 (PS_2_logreg) | -0.0485 / -0.0556 |
| heygen | 2988 | 15150 / 2674 | 12610 / 2226 | 0.7341 (PS_1_prob_avg) | 0.7341 (PS_1_prob_avg) | -0.3486 / -0.3086 |
| starganv2 | 12 | 11836 / 2089 | 11826 / 2087 | 0.9454 (PS_B_learned_weights) | 0.9429 (rank ECDF_logreg) | +0.0252 / +0.0281 |
| deepfacelab | 2988 | 15109 / 2667 | 12569 / 2219 | 0.9586 (PS_A_calibrated_avg) | 0.9476 (PS_A_calibrated_avg) | -0.1355 / -0.1161 |
| stargan | 2988 | 11863 / 2094 | 9323 / 1646 | 0.9956 (PS_3_mlp_logits) | 0.9957 (PS_3_mlp_logits) | +0.0027 / +0.0022 |
| styleclip | 2981 | 11979 / 2115 | 9446 / 1667 | 0.9966 (PS_B_learned_weights) | 0.9968 (PS_B_learned_weights) | +0.0025 / +0.0026 |

### weak6

| | primary | training duplicates removed |
|---|---|---|
| PS ceiling (mean over folds) | 0.7213 | 0.7217 |
| FS1 (mean over folds) | 0.6191 | 0.6270 |
| FS1 - ceiling | -0.1022 [-0.2117, -0.0135], 3 of 8, sign-flip p 0.086 | -0.0947 [-0.2258, +0.0100], 4 of 8, sign-flip p 0.250 |
| FS2 - ceiling | -0.0886 [-0.1976, -0.0038], 2 of 8, sign-flip p 0.148 | -0.0813 [-0.2075, +0.0161], 3 of 8, sign-flip p 0.305 |
| FS1 - ceiling, folds with n >= 500 | -0.0590 | -0.0579 |
| AUPRC: FS1 / ceiling method | 0.6509 / 0.7158 | 0.6592 / 0.7122 |
| AUPRC: FS1 - ceiling method | -0.0649 | -0.0530 |

| Fold | rows removed | FIT / SELECT primary | FIT / SELECT dedup | ceiling primary (method) | ceiling dedup (method) | FS1 - ceiling primary / dedup |
|---|---|---|---|---|---|---|
| whichisreal | 2988 | 13535 / 2389 | 10995 / 1941 | 0.7374 (PS_C_logreg_cv) | 0.7367 (PS_C_logreg_cv) | +0.0131 / +0.0004 |
| CollabDiff | 2981 | 13530 / 2388 | 10996 / 1941 | 0.9546 (PS_E_mlp_probs_es) | 0.9545 (PS_E_mlp_probs_es) | +0.0216 / +0.0191 |
| MidJourney | 2983 | 13636 / 2407 | 11101 / 1959 | 0.4989 (quadratic_logreg) | 0.4782 (quadratic_logreg) | -0.2718 / -0.2730 |
| heygen | 2988 | 15150 / 2674 | 12610 / 2226 | 0.5016 (PS_1_prob_avg) | 0.5231 (PS_A_calibrated_avg) | -0.0541 / +0.0675 |
| starganv2 | 12 | 11836 / 2089 | 11826 / 2087 | 0.6694 (PS_C_logreg_cv) | 0.6709 (PS_C_logreg_cv) | -0.0234 / -0.0265 |
| deepfacelab | 2988 | 15109 / 2667 | 12569 / 2219 | 0.8653 (PS_B_learned_weights) | 0.8835 (PS_B_learned_weights) | -0.4094 / -0.4776 |
| stargan | 2988 | 11863 / 2094 | 9323 / 1646 | 0.5765 (PS_B_learned_weights) | 0.5569 (rank ECDF_logreg) | -0.0986 / -0.0747 |
| styleclip | 2981 | 11979 / 2115 | 9446 / 1667 | 0.9664 (rank ECDF_logreg) | 0.9694 (rank ECDF_logreg) | +0.0052 / +0.0074 |

## 2. Mechanism under LOGO (Table A9, LOGO columns)

### arch6

| Method (mean AUROC) | primary | training duplicates removed |
|---|---|---|
| PS_4_xgboost | 0.7275 | 0.7426 |
| FS_1_raw_feat_logreg | 0.8979 | 0.8960 |
| FS_D_xgb_features | 0.8691 | 0.8702 |
| FS_pca1_logreg | 0.7719 | 0.7672 |
| FS_pca2_logreg | 0.7300 | 0.7103 |
| FS_pca4_logreg | 0.7953 | 0.8081 |
| FS_pca8_logreg | 0.8280 | 0.8322 |
| FS_probe2_logreg | 0.8801 | 0.8728 |

| Comparison | primary | training duplicates removed |
|---|---|---|
| FS_D_xgb_features_vs_PS_4 | +0.1416 [+0.0932, +0.1915], 8 of 8, sign-flip p 0.008 | +0.1276 [+0.0789, +0.1792], 8 of 8, sign-flip p 0.008 |
| FS_pca1_logreg_vs_PS_4 | +0.0444 [+0.0192, +0.0714], 7 of 8, sign-flip p 0.016 | +0.0246 [-0.0073, +0.0595], 6 of 8, sign-flip p 0.258 |
| FS_pca2_logreg_vs_PS_4 | +0.0025 [-0.0261, +0.0327], 4 of 8, sign-flip p 0.883 | -0.0323 [-0.0619, +0.0010], 1 of 8, sign-flip p 0.117 |
| FS_pca4_logreg_vs_PS_4 | +0.0678 [+0.0322, +0.0942], 7 of 8, sign-flip p 0.016 | +0.0655 [+0.0219, +0.1085], 7 of 8, sign-flip p 0.039 |
| FS_pca8_logreg_vs_PS_4 | +0.1005 [+0.0312, +0.1610], 7 of 8, sign-flip p 0.039 | +0.0896 [+0.0221, +0.1453], 7 of 8, sign-flip p 0.047 |
| FS_probe2_logreg_vs_PS_4 | +0.1526 [+0.0935, +0.2115], 8 of 8, sign-flip p 0.008 | +0.1302 [+0.0691, +0.1894], 8 of 8, sign-flip p 0.008 |
| FS_probe2_logreg_vs_FS_1 | -0.0178 [-0.0429, -0.0013], 2 of 8, sign-flip p 0.039 | -0.0232 [-0.0565, +0.0020], 3 of 8, sign-flip p 0.383 |
| FS_2_vs_FS_1 | -0.0018 [-0.0182, +0.0122], 6 of 8, sign-flip p 0.812 | -0.0097 [-0.0270, +0.0052], 4 of 8, sign-flip p 0.367 |

### clip4

| Method (mean AUROC) | primary | training duplicates removed |
|---|---|---|
| PS_4_xgboost | 0.7305 | 0.7561 |
| FS_1_raw_feat_logreg | 0.8462 | 0.8453 |
| FS_D_xgb_features | 0.8522 | 0.8497 |
| FS_pca1_logreg | 0.8277 | 0.8094 |
| FS_pca2_logreg | 0.8312 | 0.8487 |
| FS_pca4_logreg | 0.8434 | 0.8478 |
| FS_pca8_logreg | 0.8694 | 0.8681 |
| FS_probe2_logreg | 0.8725 | 0.8753 |

| Comparison | primary | training duplicates removed |
|---|---|---|
| FS_D_xgb_features_vs_PS_4 | +0.1218 [+0.0022, +0.2663], 7 of 8, sign-flip p 0.133 | +0.0937 [-0.0113, +0.2191], 7 of 8, sign-flip p 0.258 |
| FS_pca1_logreg_vs_PS_4 | +0.0973 [+0.0191, +0.1985], 8 of 8, sign-flip p 0.008 | +0.0533 [+0.0053, +0.1131], 7 of 8, sign-flip p 0.031 |
| FS_pca2_logreg_vs_PS_4 | +0.1008 [-0.0088, +0.2452], 6 of 8, sign-flip p 0.289 | +0.0926 [-0.0185, +0.2380], 5 of 8, sign-flip p 0.414 |
| FS_pca4_logreg_vs_PS_4 | +0.1129 [-0.0136, +0.2738], 6 of 8, sign-flip p 0.383 | +0.0917 [-0.0239, +0.2452], 4 of 8, sign-flip p 0.453 |
| FS_pca8_logreg_vs_PS_4 | +0.1389 [-0.0352, +0.3443], 6 of 8, sign-flip p 0.266 | +0.1120 [-0.0436, +0.2990], 6 of 8, sign-flip p 0.266 |
| FS_probe2_logreg_vs_PS_4 | +0.1420 [+0.0324, +0.2795], 8 of 8, sign-flip p 0.008 | +0.1192 [+0.0212, +0.2491], 8 of 8, sign-flip p 0.008 |
| FS_probe2_logreg_vs_FS_1 | +0.0263 [-0.0035, +0.0756], 5 of 8, sign-flip p 0.352 | +0.0300 [+0.0019, +0.0786], 6 of 8, sign-flip p 0.102 |
| FS_2_vs_FS_1 | +0.0464 [+0.0043, +0.0991], 6 of 8, sign-flip p 0.094 | +0.0468 [+0.0116, +0.0924], 5 of 8, sign-flip p 0.094 |

## 3. Standard split and the audit (Table 2)

### arch6

Rows removed: FIT 1880 of 26591, SELECT 848 of 6647.

| | primary (same script) | training duplicates removed |
|---|---|---|
| PS ceiling over the full ladder (method) | 0.8603 (PS_13_quadratic_xgb) | 0.7995 (PS_4_xgboost) |
| FS1 | 0.9654 | 0.9653 |
| FS1 - ceiling [generator bootstrap 95% CI], wins | +0.1050 [+0.0579, +0.1580], 8 of 8 | +0.1658 [+0.1133, +0.2179], 8 of 8 |
| PS4 | 0.8426 | 0.7995 |
| best of PS1-PS3 | 0.7865 | 0.7878 |

### clip4

Rows removed: FIT 1882 of 26591, SELECT 849 of 6647.

| | primary (same script) | training duplicates removed |
|---|---|---|
| PS ceiling over the full ladder (method) | 0.9417 (PS_11_disagreement_xgb) | 0.9372 (PS_11_disagreement_xgb) |
| FS1 | 0.9654 | 0.9649 |
| FS1 - ceiling [generator bootstrap 95% CI], wins | +0.0238 [+0.0074, +0.0427], 8 of 8 | +0.0278 [+0.0127, +0.0447], 8 of 8 |
| PS4 | 0.9390 | 0.9329 |
| best of PS1-PS3 | 0.9014 | 0.9058 |

### weak6

Rows removed: FIT 1880 of 26591, SELECT 848 of 6647.

| | primary (reviewed results, Block A) | training duplicates removed |
|---|---|---|
| FS1 | 0.8222 | 0.8251 |
| best of PS1-PS3 | 0.5902 | 0.6080 |
| PS4 | 0.7624 | 0.7038 |
| FS1 - PS4 | +0.0598 | +0.1213 |
| FS1 - ceiling over the full ladder | | +0.1078 (PS_11_disagreement_xgb) |

### The audit's three effects (Table 2)

The standard-split margins here come from appB_richer_inputs.py, whose FS1 differs from the canonical FS1 of Table 2 by about 3e-4; weak6 has no stored run of that script, so its primary column is the reviewed result that Table 2 uses.

| Effect | paper (Table 2) | primary, same scripts | training duplicates removed |
|---|---|---|---|
| overlap distortion (arch6: standard-split margin minus LOGO margin) | +0.0248 | +0.0245 | +0.0874 |
| baseline-ladder distortion, arch6 (full-ladder ceiling minus best of PS1-PS3) | +0.0739 | +0.0739 | +0.0118 |
| baseline-ladder distortion, weak6 (PS4 minus best of PS1-PS3) | +0.1721 | +0.1721 | +0.0958 |

## 4. LOGO richer-input heads of the standard split

Ceiling over the 20 methods of the standard split (the ten ladder methods and PS_G to PS_K under a class-balanced logistic head and XGBoost with the PS4 grid and 5-fold CV) against the primary runs' 22 candidates.

### arch6

| | primary (22 candidates) | standard-split heads (20 methods) |
|---|---|---|
| PS ceiling (mean over folds) | 0.8174 | 0.8173 |
| FS1 - ceiling | +0.0806 [+0.0240, +0.1401], 7 of 8, sign-flip p 0.055 | +0.0806 [+0.0240, +0.1402], 7 of 8, sign-flip p 0.055 |
| sign / Wilcoxon p | 0.070 / 0.055 | 0.070 / 0.055 |
| FS1 - ceiling, folds with n >= 500 | +0.1147 | +0.1147 |

| Fold | n | primary ceiling (method) | standard-split heads ceiling (method) |
|---|---|---|---|
| whichisreal | 2001 | 0.5873 (PS_B_learned_weights) | 0.5873 (PS_B_learned_weights) |
| CollabDiff | 2000 | 0.8844 (PS_C_logreg_cv) | 0.8844 (PS_C_logreg_cv) |
| MidJourney | 1882 | 0.7128 (PS_C_logreg_cv) | 0.7128 (PS_C_logreg_cv) |
| heygen | 101 | 0.7694 (uncertainty+MI_logreg) | 0.7690 (H_uncertainty_mi_LR) |
| starganv2 | 4000 | 0.8146 (PS_B_learned_weights) | 0.8146 (PS_B_learned_weights) |
| deepfacelab | 149 | 0.9601 (PS_B_learned_weights) | 0.9601 (PS_B_learned_weights) |
| stargan | 3968 | 0.9509 (PS_B_learned_weights) | 0.9509 (PS_B_learned_weights) |
| styleclip | 3824 | 0.8593 (PS_C_logreg_cv) | 0.8593 (PS_C_logreg_cv) |

### clip4

| | primary (22 candidates) | standard-split heads (20 methods) |
|---|---|---|
| PS ceiling (mean over folds) | 0.9206 | 0.9207 |
| FS1 - ceiling | -0.0744 [-0.1817, -0.0019], 5 of 8, sign-flip p 0.250 | -0.0745 [-0.1817, -0.0020], 5 of 8, sign-flip p 0.250 |
| sign / Wilcoxon p | 0.727 / 0.742 | 0.727 / 0.742 |
| FS1 - ceiling, folds with n >= 500 | -0.0102 | -0.0103 |

| Fold | n | primary ceiling (method) | standard-split heads ceiling (method) |
|---|---|---|---|
| whichisreal | 2001 | 0.9275 (rank ECDF_logreg) | 0.9280 (J_rank_ecdf_LR) |
| CollabDiff | 2000 | 0.9993 (rank ECDF_logreg) | 0.9993 (J_rank_ecdf_LR) |
| MidJourney | 1882 | 0.7314 (PS_3_mlp_logits) | 0.7314 (PS_3_mlp_logits) |
| heygen | 101 | 0.7910 (PS_1_prob_avg) | 0.7910 (PS_1_prob_avg) |
| starganv2 | 4000 | 0.9500 (rank ECDF_logreg) | 0.9502 (J_rank_ecdf_LR) |
| deepfacelab | 149 | 0.9716 (PS_A_calibrated_avg) | 0.9716 (PS_A_calibrated_avg) |
| stargan | 3968 | 0.9972 (PS_F_mlp_logits_es) | 0.9972 (PS_F_mlp_logits_es) |
| styleclip | 3824 | 0.9966 (PS_B_learned_weights) | 0.9966 (PS_B_learned_weights) |

### strong4

| | primary (22 candidates) | standard-split heads (20 methods) |
|---|---|---|
| PS ceiling (mean over folds) | 0.9096 | 0.9096 |
| FS1 - ceiling | -0.0595 [-0.1545, +0.0075], 5 of 8, sign-flip p 0.312 | -0.0595 [-0.1545, +0.0075], 5 of 8, sign-flip p 0.312 |
| sign / Wilcoxon p | 0.727 / 0.742 | 0.727 / 0.742 |
| FS1 - ceiling, folds with n >= 500 | +0.0013 | +0.0013 |

| Fold | n | primary ceiling (method) | standard-split heads ceiling (method) |
|---|---|---|---|
| whichisreal | 2001 | 0.9224 (rank ECDF_logreg) | 0.9225 (J_rank_ecdf_LR) |
| CollabDiff | 2000 | 0.9990 (PS_B_learned_weights) | 0.9990 (PS_B_learned_weights) |
| MidJourney | 1882 | 0.7249 (PS_2_logreg) | 0.7249 (PS_2_logreg) |
| heygen | 101 | 0.7341 (PS_1_prob_avg) | 0.7341 (PS_1_prob_avg) |
| starganv2 | 4000 | 0.9454 (PS_B_learned_weights) | 0.9454 (PS_B_learned_weights) |
| deepfacelab | 149 | 0.9586 (PS_A_calibrated_avg) | 0.9586 (PS_A_calibrated_avg) |
| stargan | 3968 | 0.9956 (PS_3_mlp_logits) | 0.9956 (PS_3_mlp_logits) |
| styleclip | 3824 | 0.9966 (PS_B_learned_weights) | 0.9966 (PS_B_learned_weights) |

### weak6

| | primary (22 candidates) | standard-split heads (20 methods) |
|---|---|---|
| PS ceiling (mean over folds) | 0.7213 | 0.7212 |
| FS1 - ceiling | -0.1022 [-0.2117, -0.0135], 3 of 8, sign-flip p 0.086 | -0.1021 [-0.2116, -0.0135], 3 of 8, sign-flip p 0.086 |
| sign / Wilcoxon p | 0.727 / 0.109 | 0.727 / 0.109 |
| FS1 - ceiling, folds with n >= 500 | -0.0590 | -0.0589 |

| Fold | n | primary ceiling (method) | standard-split heads ceiling (method) |
|---|---|---|---|
| whichisreal | 2001 | 0.7374 (PS_C_logreg_cv) | 0.7374 (PS_C_logreg_cv) |
| CollabDiff | 2000 | 0.9546 (PS_E_mlp_probs_es) | 0.9546 (PS_E_mlp_probs_es) |
| MidJourney | 1882 | 0.4989 (quadratic_logreg) | 0.4983 (I_quadratic_LR) |
| heygen | 101 | 0.5016 (PS_1_prob_avg) | 0.5016 (PS_1_prob_avg) |
| starganv2 | 4000 | 0.6694 (PS_C_logreg_cv) | 0.6694 (PS_C_logreg_cv) |
| deepfacelab | 149 | 0.8653 (PS_B_learned_weights) | 0.8653 (PS_B_learned_weights) |
| stargan | 3968 | 0.5765 (PS_B_learned_weights) | 0.5765 (PS_B_learned_weights) |
| styleclip | 3824 | 0.9664 (rank ECDF_logreg) | 0.9664 (J_rank_ecdf_LR) |

