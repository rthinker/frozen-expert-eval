# Sensitivity checks

These checks support Appendix A's training-duplicate and LOGO
prediction-head-setting analyses. Supplied outputs are in
[`camera_ready/output/results/sensitivity/`](../../output/results/sensitivity/).

| Script | Task |
|---|---|
| `count_twins.py` | Count duplicate training rows and SELECT rows with exact feature twins in FIT. |
| `dedup_logo_v2.py` | Rerun the four LOGO pools after retaining the first exact copy of each training row before FIT/SELECT splitting. |
| `dedup_followup_v2.py` | Rerun LOGO width controls and standard-split sensitivity checks. Run the deduplicated LOGO ladder first. |
| `paper_heads_logo_v2.py` | Rerun LOGO richer-input heads with the standard-split settings. |
| `summarize_v2.py` | Summarize supplied or rerun evidence without fitting. |

Before refitting the sensitivity analyses, run Block B in the same checkout
with `python camera_ready/scripts/review/reproduce_all_v4.py --blocks B`.
The summary needs the membership fingerprints from those primary LOGO reruns.

From the repository root:

```
python camera_ready/scripts/sensitivity/summarize_v2.py
python camera_ready/scripts/sensitivity/count_twins.py
python camera_ready/scripts/sensitivity/dedup_logo_v2.py
python camera_ready/scripts/sensitivity/dedup_followup_v2.py
python camera_ready/scripts/sensitivity/paper_heads_logo_v2.py --workers 4 --threads 4
python camera_ready/scripts/sensitivity/summarize_v2.py --dir camera_ready/output/results/sensitivity_rerun
```

`dedup_logo_v2.py` and `dedup_followup_v2.py` accept an optional
`--gpu-name <name>`, which requires one visible GPU with that name and stops
otherwise. The refits require the feature cache. Their default output root is
`camera_ready/output/results/sensitivity_rerun/`. The summary writes
`summary.json` and `SUMMARY.md` to `camera_ready/output/checks/sensitivity/`
by default. Use `--out-dir` for another report folder. The summary verifies
folds, methods, compatible runs and score arrays before writing. It refuses
incomplete inputs unless `--allow-partial` is given, in which case it omits
affected sections and marks the report incomplete. Missing primary evidence
stops the summary.

A new standard-head run needs primary LOGO runs with matching membership
fingerprints. The supplied primary runs record none, so run Block B first. To see which folds would be reused without writing anything,
pass `--plan` to `paper_heads_logo_v2.py`. It checks each fold's input
fingerprint, held-out labels and score archive before reuse.

Only training rows are deduplicated. Test rows and their multiplicities are
unchanged. Removing copies changes training weights and composition, so the
comparison combines the effects of deduplication and reweighting. The AUPRC
summary uses each fold's AUROC-selected ceiling method.
