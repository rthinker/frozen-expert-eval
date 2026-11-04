# Reproducing the camera-ready results

This guide accompanies *When Are Predictions Enough? An Evaluation Protocol for Frozen
Expert Composition* (NeurIPS 2026, Evaluations & Datasets track), Appendix J.9.

The repository root contains the standard-split pipeline and results (Block A).
`camera_ready/` contains the LOGO runs (Block B), further analyses (Block C), the shared
reproduction harness and the supplied reference values. Paths beginning `output/` refer
to the repository root unless prefixed with `camera_ready/`.

## Documentation index

| Topic | Authoritative documentation |
|---|---|
| Dataset overlap, repeated content, expert-training exposure, intended uses and limitations | [Dataset card on Hugging Face](https://huggingface.co/datasets/anon-frozen-experts/df40-frozen-experts-anon) |
| Reproduction coverage, retained outputs, hardware and execution limitations | This guide, including the requirements, coverage and known-limitations sections; [installation](INSTALL.md) |
| Code, derived-data and upstream licence scope | [Licensing and attribution](LICENSES.md) |
| Paper-to-command/output mapping | [Release manifest](RELEASE_MANIFEST.md) |

The standard split is overlap-affected. LOGO is the primary protocol; it removes identified
combiner train/test reuse, but does not eliminate repeated content within training/selection
data or undo frozen-expert training exposure. See the dataset card for the complete caveats.

## Current entry points

| Task | Entry point |
|---|---|
| Rerun and verify the DF40 experiments in blocks A, B and C | `scripts/review/reproduce_all_v4.py` (`--verify-only`, `--check`, a full run, `--resume-run <run>`; see "Reproducing the results") |
| Reference values of blocks B and C | `reference/reference_results_camera_ready_v2.json` (`scripts/authoring/generate_reference_v3.py`) |
| Statistics of the LOGO runs and the audit of Table 2 | `scripts/authoring/full_ladder_stats_v13.py` (steps B31, B32, C08a; run by hand, it does not write among the stored results; run on its own, it stops on incomplete inputs unless a partial analysis is asked for, `--allow-ten-methods`, `--allow-partial-folds` or `--allow-incomplete-ladder`, and the result then says so; it also stops on runs that do not belong together (`scripts/runs/ladder_checks.py`); its self-test reads the stored standard-split statistics of `inputs/`) |
| Standard-split statistics; Appendix H table | `scripts/authoring/full_ladder_stats_standard_v3.py` (step C08); `scripts/authoring/regenerate_appendix_h_v3.py` (step C12); run by hand, neither writes among the stored results (give them `--out`) |
| The DF40 tables and the number ledger | `scripts/authoring/generate_camera_ready_tables_v18.py` (step C24; it refuses statistics or a sensitivity summary marked as a partial analysis, runs that do not belong together, an incomplete full ladder, and Table 2 rows, Table A11's values and the standard-split ceiling of Tables 1 and A3 from different runs) |
| Figure A1, outside the harness | `scripts/authoring/figure_a1_tsne.py`; see "Reproducing Figure A1" |
| Figure A2 | `scripts/authoring/figure_a2_full_ladder_v4.py` (step C23; the larger of FS1 and FS2 per pool, as the caption says) |
| Appendix A reuse counts; FS1 refit | `scripts/review/check_split_reuse_v3.py` (step C01); `scripts/review/refit_fs1_v3.py` (the submitted folds; its report goes to `camera_ready/output/checks/`) |
| Appendix A expert-side check; Appendix H comparison | `scripts/analyses/appA_expert_side_v3.py` (step C04); `scripts/analyses/appH_logit_convention_diff_v2.py` (step C13); started by hand, both write only where `--out` says |
| LOGO runs (the ten methods, FS1 and FS2 per fold) | `scripts/runs/11_disjoint_data_v4.py` (the LOGO steps of Block B; `--folds` takes each held-out generator once, and a dry run writes nothing; every fold records the membership fingerprints of FIT and SELECT, and `--resume` reuses only checkpoints that hold them) |
| Richer prediction-space inputs and the combined ceiling under LOGO | `scripts/runs/08_logo_extended_rungs_v3.py` (fingerprints and resume as for the LOGO runs; a dry run writes nothing); `scripts/runs/09_logo_combined_ceiling_v3.py` (stops on runs that do not belong together, and on an incomplete ladder unless `--allow-incomplete-ladder`) |
| Width controls under LOGO and gate G4 | `scripts/runs/05_run_c_width_matched_v5.py` (steps B12, B13, B27, B28; `--folds`, `--dry-run`, fingerprints and resume as for the LOGO runs) |
| Runs that belong together; a complete ladder | `scripts/runs/ladder_checks.py` (the shared checks); `provenance/trusted_run_pairs.json`, the groups of stored runs, which record no fingerprints, produced together from the same folds (`scripts/review/trusted_run_pairs_v1.py --check` compares it with the stored files) |
| Sensitivity checks of Appendix A, outside the harness | `scripts/sensitivity/`: `summarize_v2.py` rebuilds the summary from the supplied outputs, after checking that every input is complete and that the runs analysed or compared together belong together; `dedup_logo_v2.py`, `dedup_followup_v2.py` and `paper_heads_logo_v2.py` rerun the checks with fingerprints (see the README there) |

## Requirements

* Python 3.11 and the packages pinned in `requirements-lock.txt`; installation routes for Windows, Linux and
  macOS are in `INSTALL.md`.
* The derived feature cache (about 3.2 GiB) in `output/cached_features/`, downloaded from the Hugging Face Hub
  (https://huggingface.co/datasets/anon-frozen-experts/df40-frozen-experts-anon; the data are unchanged since review).
  Hugging Face is the canonical source for the dataset documentation. The checkout supplies the local
  verifier files in `output/cached_features/` and the trust anchor in `camera_ready/dataset/CACHE_PROVENANCE.json`.
  `--verify-only` works without it.
* Optional, for the steps that read DF40's own split lists (the Appendix A expert-side check and release-level
  reuse counts, and the Appendix I.1 three-class manifests): the DF40 split files `DF40_all.json`,
  `FSAll_ff.json`, `FRAll_ff.json`, `EFSAll_ff.json` and one JSON per held-out generator, from the DF40
  release (linked from the DF40 repository README). Step C00 (`scripts/analyses/prepare_df40_lists_v2.py`)
  first rewrites their paths into the form of the cache's sample ids: the public files name training fakes
  `DF40_train/<method>/frames/...` and real images `deepfakes_detection_datasets/FaceForensics++/...`, while the
  cache, extracted from a DF40 folder laid out differently, names them `<method>/ff/frames/...` and
  `FaceForensics++/...`. With the rewritten lists the stored three-class manifests are reproduced exactly (the same
  content and manifest digests; the files' line endings follow the platform).
  The harness adds C00 when a selected step needs the rewritten lists. C00 stages all twelve lists before replacing
  them and writes `PREPARED.json` last. Steps that use the rewritten lists verify the file hashes in that record.
  Step C01 uses the original lists for the release-level reuse count.
  Without the lists those steps are skipped and the stored manifests are used. The DF40 image root (file sizes
  only) completes the Appendix A reuse counts when it follows the cache's layout ("Reproducing the results").
* Memory: as in the reviewed code, the full21 FS1 fit is skipped with a marker when 40% of the host's RAM is below
  its float64 input (about 6.7 GiB), that is on hosts below about 16.7 GiB; the table step and Figure A2 then use
  the stored result. Hosts with about 17 to 24 GiB attempt the fit, which peaks at about 12 to 15 GiB and may run
  out of memory: set `FS1_BUDGET_GIB` to a value below 6.7 to skip it, or `ALLOW_FS1_OOM=1` to force it.
  Everything else runs on a 16 GiB host.
* GPU: optional. FS2 and the neural three-class heads use a CUDA GPU when one is visible, otherwise the CPU.
  FS2 values depend on the GPU and software environment, so they are checked at 0.005 and reported separately.
  The stored three-class run of Table A18 used an RTX 2070 Super. On an RTX 5080, FS2-K3 gives
  0.944 for the printed 0.945.

## Reproducing the results

From the repository root:

```
python camera_ready/scripts/review/reproduce_all_v4.py --verify-only      # supplied metrics and archived evidence digests (no cache)
python camera_ready/scripts/review/reproduce_all_v4.py --check            # smoke tests + verification + cache validation
python camera_ready/scripts/review/reproduce_all_v4.py --dry-run          # every step and its exact command
python camera_ready/scripts/review/reproduce_all_v4.py                    # full rerun of blocks A, B and C
python camera_ready/scripts/review/reproduce_all_v4.py --gpu-name 5080 --df40-json <DF40 lists> --df40-raw <DF40 images>
python camera_ready/scripts/review/reproduce_all_v4.py --only C11         # one step (the strict refit of Table A14)
python camera_ready/scripts/review/reproduce_all_v4.py --resume-run <run> # continue after any interruption
```

`INSTALL.md`, "Expected outcomes", gives what the checks and the tests print when everything agrees.
Verification compares the supplied result summaries with references derived from those summaries and checks archived
scores, checkpoints and the provenance ledger against `reference/stored_evidence.sha256`. It detects changes in the
checked artifacts. Reproduction requires running the workflow from the cache. A run restricted by `--only`,
`--from` or `--blocks` reports its partial scope, and its log lists the steps that were recomputed or skipped.

A cache-reading plan authenticates the feature cache before moving or clearing outputs. If a prerequisite
step fails, dependent steps are blocked. The harness uses its own checkout and output paths even when a
standalone analysis override is set in the shell.

A full rerun takes about 36 hours with an Intel i9-10980HK and RTX 5080. This is an estimate from the
blocks timed separately: A about 14 hours, B about 14 hours, C about 8 hours. Gate G2 (step B11) compares the
content-disjoint arch6 run with the run without exclusion, which must come from one environment: it runs when
`--with-reproduction-gate` recomputes both (about 4 hours more) and is otherwise recorded as SKIPPED, with its
stored verdict kept. Without DF40's lists and image root, step C01 copies the counts it cannot compute (route A, the
image-file checks) from the stored result and marks them as carried over. If it is interrupted (reboot, crash,
power loss, closed terminal),
`--resume-run <run>` continues it with its own options. It checks each step's declared outputs against the
recorded SHA-256 digests before skipping it, using the latest successful producer for shared outputs. It passes `--resume`
to the runners that keep checkpoints, so an interrupted LOGO run loses at most its current fold. Every attempt of a
step records a random id and a fingerprint of what it depends on (its command, the code and configuration, the
Python packages, the feature cache and its input files, DF40's lists when given and, with `--df40-raw`, the
existence and size of every image file step C01 looks up); steps that keep checkpoints run through
`scripts/review/stamp_checkpoints_v2.py`, which writes the attempt id, the fingerprint and the digest of the score file
into every checkpoint. A resumed run first checks every checkpoint: stamped by an attempt of its own step in this
run whose fingerprint is unchanged, its score file the one stamped, the excluded rows of this run's exclusion list,
the held-out generator's labels from the feature cache in the same order, and every saved score array of the fold's
test size, finite, with the AUROC the checkpoint records. Any checkpoint that fails is moved aside, so its fold is
recomputed. A step whose fingerprint changed runs again, and a step that reads an output of a step that runs again
reruns in full. It keeps every earlier record of the run. `RUNBOOK.md` in the run's log folder gives the exact
command. A resumed run refuses to start, before it moves anything, when the feature cache is not available, when
DF40's lists or image root given to the run (`--df40-json`, `--df40-raw`) are missing, or, with `--gpu-name`, when
that GPU is not visible.
Long runs are best started from a terminal that stays open, on a machine kept awake and with system updates
paused. `--from <step>` and `--only <step>` start a new run at one step: they back up first and recompute the
selected steps fully, reusing no checkpoint.

`--df40-raw` names an image root in which `<root>/<sample id>` is each cached image, as in the DF40 folder the cache
was extracted from (the ids are in `camera_ready/output/ids/`, for example
`FaceForensics++/original_sequences/youtube/c23/frames/651/134.png` and `one_shot_free/ff/frames/038_125/118.png`).
DF40's public download is laid out differently (its lists name training fakes `DF40_train/<method>/frames/...`), so
with it step C01 stops and reports how many files it could not find. Independent steps can continue, but C24
is blocked, so that run exits with status 1 without rebuilding the DF40 tables or ledger. Check an image root first with `--only C01 --df40-json <DF40 lists>
--df40-raw <DF40 images>` (a few minutes). A resumed run keeps the options it started with, so to drop `--df40-raw`,
start a new run (`--only C01`, or `--from <step>`).

Rerun a stored result through the harness (`--only <step>`; `--dry-run` lists every step with its command): it backs
the stored results up first. Standalone write protections apply to the current camera-ready entry points listed
here. The root `scripts/04_*` through `scripts/17_*` and `scripts/authoring/full_ladder_stats_v4.py` under
`camera_ready/` can overwrite files in release output directories when invoked directly. The analyses of `scripts/analyses/` stop before any work unless `CR_RESULTS_DIR` names another folder for their
results, the runners and gates of `scripts/runs/` refuse such an output path (give them `--out-dir` or `--out`), and
the expert-side check and the Appendix H comparison write only where `--out` says, and the statistics
(`full_ladder_stats_v13.py`, `full_ladder_stats_standard_v3.py`) and the Appendix H table (`regenerate_appendix_h_v3.py`)
refuse such a path (give them `--out-dir` or `--out`). The harness marks the steps it runs with the environment
variable `REPRO_ATTEMPT_ID`, which these checks read.

The harness runs three blocks:

* **Block A**: the reviewed pipeline, `scripts/03` to `17`, with the reviewed commands, checked against the
  reviewed reference `output/results/reference_results.json` by the reviewed harness's own functions, plus the
  checks of version 3 that they lack (finite values valid for their kind, recorded statuses, only the two
  documented full21 memory-budget skips). The supplied files retain outputs produced at different stages of
  the study with multiple revisions of the pipeline. Their reference documents the metrics selected for comparison.
  Exact writer revisions were not recorded for every file.
* **Block B**: the LOGO analyses with the camera-ready runners in `scripts/runs/`: training rows identical to
  a held-out test image excluded, the full prediction-space ladder in every fold, the width controls under
  LOGO, the comparison gates and the statistics.
* **Block C**: the other post-submission analyses in `scripts/analyses/`, in the order of the appendices
  (A, B, D, F, H, I), then Figure A2 and the DF40 table bodies with their number ledger.

Blocks B and C are checked against `reference/reference_results_camera_ready_v2.json`, with the tolerances and checks
of `RELEASE_MANIFEST.md` (every value finite and valid for its kind: p-values and AUROCs within [0, 1], counts
non-negative integers; every recorded status PASS, report or COMPLETE; an environment-dependent value more than
0.1 from its reference fails; a gate file the run did not rewrite is compared value by value). Step C08a
recomputes the audit of Table 2 after C08 has regenerated the standard-split statistics it reads, and checks the
identities between its inputs; in a full run, B32 defers an audit whose inputs disagree to it. The last step (C24,
table builder v18, which refuses statistics or a sensitivity summary marked as a partial analysis, runs that do
not belong together, an incomplete full ladder, and Table 2 rows, Table A11's values and the standard-split ceiling
of Tables 1 and A3 from different runs)
rebuilds the 25 DF40 tables (in `camera_ready/output/tables/tables/paper/`) and the number ledger. It compares each number, at its printed precision, with the
stored ledger `provenance/05_number_provenance.csv` (1,937 public entries, including additional analyses), checks that every rebuilt number is valid
(finite; p-values, counts and values read from a result within their range), and checks the 96 numbers posted in
the review discussion. A changed number that depends on the GPU (FS2 and the statistics derived from it, the
three-class heads trained on a GPU) is reported as environment-dependent, and a changed timing (Table A21) as
machine-dependent, both with their change. The public ledger
numbers are judged only when the table step ran in the run.

Each step's console output goes to `camera_ready/output/logs/<run>/<step>.log`. The run log
(`reproducibility_log_v4.json`) records the environment, the start, end and wall-clock time of every step and
the verification; `TIMINGS.md` says what the run covered (complete, or partial with the steps not run, skipped or
not finished, whose stored outputs were kept) and compares a time the paper states only with a run that covers it. A full rerun backs up
`camera_ready/output/results/` to `camera_ready/output/reproducibility_backup/<time>/` first (Block A backs up the
reviewed outputs to `output/reproducibility_backup/`, as the reviewed harness does).

## Results and the steps that produce them

`RELEASE_MANIFEST.md` lists every step with its exact command, output and tolerance. The main results:

| Paper item or diagnostic | Selected results and reference checks | Steps |
|---|---|---|
| Table 1, arch6 LOGO (primary estimate) | FS1 0.898, prediction-space ceiling 0.817, +0.081 [+0.024, +0.140], 7 of 8; sign 0.070, Wilcoxon 0.055, sign-flip 0.055 | B01, B08-B10, B32 |
| Table 1, clip4, strong4, weak6 LOGO | -0.074 (5/8), -0.060 (5/8), -0.102 (3/8) | B02-B04, B15-B26, B32 |
| Table 1 standard row, Tables A3 and A15 | arch6 0.860 (full ladder) vs 0.966, +0.105 [+0.058, +0.158]; +0.123 against PS4 | A02-A09, A23-A27, C06-C08 |
| Table 2 (sensitivity audit) | standard-LOGO contrast +0.025 (arch6); baseline inflation +0.074 (arch6) and +0.172 (weak6) | A02-A09, B32, C06-C08, C08a |
| Table A9, D.4 (interface vs capacity) | standard split: PCA 0.740, label-fitted 0.958; LOGO arch6: PS4 0.727, FS1 0.898, label-fitted 0.880 | C09, B12-B14, B27-B29, B32 |
| the retained power-threshold analysis (simulated detection thresholds, 8 folds) | +0.050, +0.080, +0.115 for fold-delta SD 0.05, 0.08, 0.12 | C10 |
| Table A14, Sec. 5.1 (strict refit) | -0.228 [-0.322, -0.130]; matched +0.149; 5,438 to 586 real rows | C11, C22 |
| Table A16 (individual experts) | best arch6 expert 0.787, best overall 0.884 | C12, C13 |
| Table A18 (three classes, paper-aligned) | PS3-K3 0.954, FS2-K3 0.945, FS1-K3 0.933 | C00, C14-C16 (DF40 lists), C19, C20 |
| Table A19 (three classes, fixed outputs) | +0.210 (5/5), +0.201 (3/3) | C17, C18 |
| Table A21 (cost and access) | sizes exact; timings depend on the machine | C21 |
| Appendix A (train/test reuse) | 11,886 of 17,925 by path, 12,893 by df40-clip-large feature content; fake reuse in 5 of 8 generators; held-out test paths only in the DF40 aggregate list (23,892; 0 in FS, FR, EFS) | C00-C05 |
| Figure A2; all table bodies | | C23, C24 |

## Sensitivity checks

`scripts/sensitivity/` holds two checks of the results against choices of the implementation, outside the harness:
the LOGO runs, the mechanism under LOGO and the standard split with exact duplicates removed from the training data
(FIT and SELECT then share no row), and the LOGO richer-input heads of the standard split. The paper reports both in
its Appendix A (the duplicate-removal table and the prediction-head-setting paragraph; the table builder reads `summary.json`). Their outputs are kept apart from
the primary results, in `camera_ready/output/results/sensitivity/` with their provenance; `python
camera_ready/scripts/sensitivity/summarize_v2.py` rebuilds their summary (`SUMMARY.md`, with AUPRC) from those outputs,
without refitting, after checking that every input is complete and that the runs it compares belong together. Only training data are deduplicated; the test sets are unchanged (`scripts/sensitivity/README.md`
gives the scope and the commands of a rerun).

## Supported scope

The supported rerun route uses the released feature cache for the DF40 binary and Appendix I.1 three-class
evaluations after the locked installation of `INSTALL.md`.

* **From the released cache.** The extraction and split scripts (`scripts/01_extract.py` and
  `scripts/02_resplit_fit_select.py`) document how the arrays were produced. The packaged cache uses portable
  relative sample IDs and includes split provenance and checksums. The cache integrity check authenticates
  this pinned release, so use the supplied cache for supported reruns.
* **Two-class labels.** The metric helpers of the reviewed code (`src/utils/metrics.py`) are used here for test
  sets that hold both classes, as every result of the paper does; outside that case they return placeholder values
  (below). For reuse beyond the paper, `maintained/metrics_v2.py` treats those cases explicitly and agrees with the
  reviewed helpers on two-class test sets (`tests/test_metrics_v2.py`). It is not installed as a package: put
  `camera_ready/maintained` on `sys.path` and `import metrics_v2`.
* **Installation.** Only the tested route of `INSTALL.md` is supported: a clone installed in editable mode with
  `requirements-lock.txt`. A built wheel and the `df40` extra of `pyproject.toml` are not (below).

## Reproducing Figure A1

Figure A1 has a separate plotting command; it is not a step of the main harness.
From the repository root, after installing the pinned dependencies and downloading the cache:

```bash
python camera_ready/scripts/authoring/figure_a1_tsne.py --cache output/cached_features --out-dir camera_ready/output/figures/a1
```

The command writes PDF and PNG figures, sampled row indices and embedding coordinates
(`figureA1_tsne_coordinates.npz`), and a JSON record of input fingerprints,
settings and software versions. Use `--formats png` for PNG only or
`--check-inputs` to validate the sampled inputs without fitting or writing.
Existing output files are not overwritten; choose a new output directory for a rerun.

The fixed clip4 sampling uses seed 42 and up to 625 rows per generator: 4,000 rows
in total. Features have 4,096 values and logits eight values per row. Both t-SNE
fits use perplexity 30, 1,000 iterations and seed 42 without feature-wise
standardization; all remaining settings are explicit in the script. The default
numerical-library thread limit is one. The reference input fingerprints and
validation environment are in `provenance/figure_a1_reproduction.json`.
Embedding coordinates and rendering can vary with the numerical libraries or
platform; the figure is a qualitative diagnostic.

## Reproduction limitations

* **Feature extraction from DF40 images.** The released cache is the starting point (Supported scope, above).
  Extraction needs the DF40 images, the 21 detector checkpoints and DF40's code (`camera_ready/INSTALL.md`;
  `scripts/01_extract.py` in the reviewed code), and its throughput was not measured.
* **Power thresholds.** Step C10 (`scripts/analyses/appF_power_mde_v2.py`) implements the simulation
  in `provenance/APPENDIX_ADDITIONAL_DETAILS.md`. With 10,000 sets the 0.12 value is on a grid
  point whose power is within simulation error of 0.80. A 100,000-set check resolves the grid point.
* **Exactness.** The harness compares deterministic Block A values with the reference at 1e-4. This is a comparison
  threshold, not a measured reproduction result: the reference collects stored outputs of several pipeline revisions
  (see Block A above), and this release holds no recorded Block A rerun of its final code. The stored FS1 results do
  not record solver iterations; a rerun logs `FS1 solver iterations: <n>/2000`, as a warning when a fit reaches the
  cap. The L-BFGS fits can move by more than the threshold between platforms: on Linux (Ubuntu 22.04) the FS1 smoke
  refit gives 0.9667 for the stored 0.9657, within the smoke test's 0.005 tolerance, so a full rerun
  may flag FS1-based values at 1e-4, and so may the gates that compare FS1 with the supplied references: G0log (the
  deterministic methods against the four-decimal reference values), G4 (the standard split's width-control values
  against the reviewed ones at 1e-3) and, with `--with-reproduction-gate`, G0 (1e-6). The observed
  3.5e-4 standard-split FS1 gap in the strict-refit comparison comes from scoring float32-rounded predictions
  instead of the canonical uncast predictions. G5 compares two runners of the same rerun, so it can show a platform
  difference only when a partial run recomputes one of its two inputs.
  Block A files marked stochastic in the reference use a 0.005 comparison tolerance. The
  prediction-space result files (`04_ds_*.json`) use 1e-4 throughout, including the MLP heads.
  Valid changes in the reviewed LOGO report's win counts and paired p-values are reported without the
  continuous-metric 0.1 cutoff.
  FS2 on a GPU is reported separately (above).

## Known limitations of the reviewed code

The following limitations apply to the Block A implementation and standalone package.

* `src/utils/metrics.py`: `compute_auc` returns 0.5 and `compute_auprc` returns 0.0 when the labels hold one
  class, and `bootstrap_paired_auc` also counts exact ties in `n_degenerate`. No result in the paper is
  affected: every held-out generator's test set holds both classes (`camera_ready/output/ids/`), as does every stored LOGO
  score file, and the paper does not use `n_degenerate`. `maintained/metrics_v2.py` corrects these cases (AUROC
  undefined, average precision 1.0 or undefined, degenerate resamples counted apart from ties).
* `scripts/01_extract.py`: the released cache was extracted with the default `--train_samples 0`, which keeps
  every entry of the split list. When requesting a smaller subset, the script groups candidates by label
  and source folder. The collection loops count the rows they receive, so
  the ids would shift if DF40's loader dropped an item: in the released cache, the 9 rows whose label disagrees
  with their path are all rows whose source image is a zero-byte file
  (`camera_ready/output/results/split_reuse_check.json`),
  so no shift is visible. An expert that fails to load is logged and skipped, while the dataset's closing line
  still counts every expert; the file counts per dataset in the final summary show it.
* `composition_eval/` (the reviewed facade over `src/`): `evaluate(pool).headline()` reports FS1 minus the best
  of PS1, PS2 and PS4 on the standard split (the submitted headline, +0.123 on arch6), not the camera-ready
  primary estimate (LOGO against the full prediction-space ladder, +0.081); its docstring points to Appendix I of
  the submission for the protocol card, which is Appendix J of the camera-ready paper.
* `pyproject.toml`: the `df40` extra asks for `albumentations>=1.4.0`, while `requirements-lock.txt` pins
  1.1.0, and a built wheel does not contain `configs/`. `INSTALL.md` installs from a clone in editable mode,
  with the lock only.

## Layout

| Folder | What it holds |
|---|---|
| `scripts/review/` | The reproduction harness, checkpoint launcher, split-reuse and FS1 refit checks |
| `scripts/runs/` | LOGO and richer-input runners, exclusion lists and comparison gates (Block B) |
| `scripts/analyses/` | Paper analyses (Block C) |
| `scripts/sensitivity/` | The sensitivity checks and their summary (see its README) |
| `scripts/authoring/` | Statistics, figures, table builder and reference generator |
| `maintained/` | `metrics_v2.py`: the reviewed metric helpers with their edge cases treated explicitly, for reuse beyond the paper (not called by the pipeline) |
| `reference/` | Pinned reference values for blocks B and C |
| `inputs/` | Supplied numerical inputs for statistics, table generation and comparison gates |
| `output/` | Plain-text sample ids (`ids/`) and the stored results (`results/`): the LOGO reruns with their fold checkpoints and per-sample predictions (`results/mitigation/`), the Block C results (`results/analyses/`), the sensitivity checks with their provenance and summary (`results/sensitivity/`) |
| `provenance/` | Number ledger, paper table locations, paired-run digests and figure settings |
| `dataset/CACHE_PROVENANCE.json` | Trusted cache contract. The local verifier files ship in `output/cached_features/` at the repository root, and the canonical documentation and data are on Hugging Face |
| `tests/` | Expected counts for the reuse check; synthetic tests |

See [Licensing and attribution](LICENSES.md) for the component terms.

### Numerical references

C24 writes `tables/additional/` for the power thresholds, head-setting sensitivity and protocol examples.
Use `provenance/paper_table_locations.json` to find ledger quantities by table. See
`provenance/APPENDIX_ADDITIONAL_DETAILS.md` for the analysis details.

Current report and resume safeguards: `check_split_reuse_v3.py` and `refit_fs1_v3.py` also guard explicit
output paths. A failed split-reuse `--expect` comparison writes no report. The sensitivity summary reads
`--dir` and writes to a separate `--out-dir` (default `camera_ready/output/checks/sensitivity/`).
Only a run whose backup setup committed can resume; standalone LOGO resume is refused.
An empty `--only` selection, or one with no runnable/completed requested step, exits unsuccessfully.
