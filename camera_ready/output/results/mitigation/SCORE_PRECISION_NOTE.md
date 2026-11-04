# FS1 score precision under LOGO

All 16 paired arch6 and clip4 folds have identical labels and float32 FS1
predictions in the ladder and width-control archives. AUROC calculated from
these arrays agrees with the width-control results to within 1e-12.

The ladder scores FS1 before converting predictions to float32. The
width-control runner converts the predictions before scoring. The conversion
can introduce ties. For arch6 WhichIsReal, the ladder records 0.8139970029970031
and the float32 scores give 0.8139895104895104, a difference of about 7.5e-6.

The paper uses the ladder AUROC for FS1. The comparison gate uses an FS1
tolerance of 1e-4 and a PS4 tolerance of 1e-9. See
[fs1_score_precision_check.json](fs1_score_precision_check.json) for the
recorded check. From the repository root, check the saved predictions with:

```text
python -B camera_ready/scripts/review/check_fs1_score_precision_v2.py --out camera_ready/output/checks/score_precision_check.json
```
