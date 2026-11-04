# DF40 Frozen-Expert Feature Cache

Frozen-expert feature cache derived from the DF40 deepfake benchmark
(Yan et al., NeurIPS 2024), released with _When Are Predictions Enough? An
Evaluation Protocol for Frozen Expert Composition_ (Pereira, Ransom and
Mitchell, NeurIPS 2026, Evaluations & Datasets track). Documentation version
1.0.8; the data are those of version 1.0.0, unchanged (FIT and SELECT created
2026-04-09, sample ids sanitized 2026-04-29, metadata revision 2026-05-07).

[Dataset and metadata](https://huggingface.co/datasets/anon-frozen-experts/df40-frozen-experts-anon) · [Code and verification instructions](https://github.com/rthinker/frozen-expert-eval)

## Summary

Penultimate features and binary classifier logits from 21 frozen deepfake
detectors (DF40) for the FIT and SELECT training splits and 8 held-out test
generators. Total: 460 data/provenance files, about 3.2 GiB (3,308 MiB), stored per split
and expert.

Each split folder holds, per expert, `{expert}_features.npy` (float16,
n x 768 for CLIP-base, 1,024 for CLIP-large and 2,048 for Xception, SRM,
RECCE and RFM) and `{expert}_logits.npy` (float32, n x 2), plus
`labels.npy` (int32) and `sample_ids.npy`, whose rows align with the
arrays, and two small records: `provenance.json` (how the split and its ids
were made) and `fingerprint.txt` (its row and class counts with two hashes).
Logit columns follow the binary label order. Column 0 is real and column 1 is fake.
Numeric arrays are read with `numpy.load(..., allow_pickle=False)`.
`sample_ids.npy` contains object-dtype path strings; the release never
unpickles it. `src.utils.cache_ids.load_sample_ids` authenticates its raw
bytes against the trusted code-owned manifest and uses the corresponding
verified plain-text rows in `camera_ready/output/ids/`. Before reading the
cache, the integrity checker authenticates its data contract against
`camera_ready/dataset/` and verifies all data digests. A checksum file
provided by the downloaded cache alone is not a trust anchor.

The cache is a reusable artifact for auditing and reproducing frozen-expert
composition methods on DF40 without re-extracting features from the
upstream images.

## Provenance

- Source models: 21 frozen DF40 detectors trained on the FS, FR, EFS, and
  combined-DF40 training lists across CLIP-base, CLIP-large, Xception, SRM,
  RECCE, and RFM architectures. Detectors trained on the combined-DF40 list
  (`df40-*`) were trained on a list that contains many of the held-out test
  paths. Aggregate held-out scores do not establish the absence of expert-training
  exposure or memorization (see the paper, Appendix A).
- Source data: DF40 release (Yan et al., NeurIPS 2024), read from the split
  files of a local copy of the release without the e4e subset (see
  Limitations). Every other entry of those files was used without further
  subsampling; for video sources the DeepfakeBench loader keeps one frame
  per video, so the cache holds far fewer HeyGen and DeepFaceLab rows (101
  and 149) than the release lists frame paths.
- Generation: the DF40 loader uses c23 inputs at 256 x 256, no augmentation,
  and a mean and standard deviation of 0.5 per channel. CLIP inputs are
  resized bilinearly to 224 x 224 with `align_corners=False`. Spatial feature
  maps are averaged over their spatial dimensions. Each frozen detector runs
  in PyTorch evaluation mode with gradients disabled; outputs saved as `{expert}_features.npy` and
  `{expert}_logits.npy` per split, with `labels.npy` and `sample_ids.npy`.
  Sample ids are relative DF40 paths (`<generator>/<class>/.../<file>`).
  For 14,993 FS and FR training rows they name the per-method folder
  (`<method>/ff/frames/...`), where `DF40_all.json` has
  `DF40_train/<method>/frames/...` with the same video and frame names.
  A further 5,640 entire-face-synthesis training rows use
  `<method>/ff/<n>/...`, and 719 real training rows use `FaceForensics++/...`.
  These 21,352 rows have no `real` or `fake` path component, so read their
  class from `labels.npy`. The extraction scripts write Unicode identifier
  arrays; the published `sample_ids.npy` files are object arrays from a later
  sanitization that keeps the path after the last `/DF40/` with forward
  slashes. Each split's `provenance.json` records it as
  `sample_ids_sanitized_at` (2026-04-29) and `sample_ids_normalization`; the
  code repository has no script for that step.
  No raw DF40 media and no detector checkpoint weights are included.
- Splits: FIT/SELECT is a stratified 80/20 split (seed 1024, stratified by
  binary label) of the DF40 training pool after deduplication by sample id.

## Splits

| Split         | Samples | Real  | Fake   | Fake share |
| ------------- | ------- | ----- | ------ | ---------- |
| `FIT`         | 26,591  | 5,438 | 21,153 | 79.5%      |
| `SELECT`      | 6,647   | 1,359 | 5,288  | 79.6%      |
| `whichisreal` | 2,001   | 1,000 | 1,001  | 50.0%      |
| `CollabDiff`  | 2,000   | 1,000 | 1,000  | 50.0%      |
| `MidJourney`  | 1,882   | 1,005 | 877    | 46.6%      |
| `heygen`      | 101     | 50    | 51     | 50.5%      |
| `starganv2`   | 4,000   | 2,002 | 1,998  | 50.0%      |
| `deepfacelab` | 149     | 49    | 100    | 67.1%      |
| `stargan`     | 3,968   | 1,984 | 1,984  | 50.0%      |
| `styleclip`   | 3,824   | 1,000 | 2,824  | 73.8%      |

`FIT` is the paper's TRAIN split.

## Train/test overlap

Although these generators define the designated held-out evaluation sets, the DF40
training aggregate (`DF40_all.json`), from which FIT and SELECT are drawn,
contains many of their test images. In this cache, 11,886 of the 17,925 test
samples (66.3%) also appear in FIT or SELECT under the same path, and 12,893
(71.9%) have df40-clip-large features identical to a FIT or SELECT sample; fake images are
reused for 5 of the 8 generators. In the DF40 release files, 23,892 of the
31,485 held-out test paths (75.9%) appear in `DF40_all.json`. FIT and SELECT
share no sample id, but 794 SELECT rows have df40-clip-large features identical
to a FIT row (793 with the arch6 experts' features; the same image under two
paths).

| Generator   | Test rows | Same path in FIT/SELECT | Same features in FIT/SELECT | Fake rows reused (path) |
| ----------- | --------- | ----------------------- | --------------------------- | ----------------------- |
| WhichIsReal | 2,001     | 988                     | 988                         | 0                       |
| CollabDiff  | 2,000     | 1,993                   | 2,000                       | 1,000                   |
| MidJourney  | 1,882     | 1,882                   | 1,882                       | 877                     |
| HeyGen      | 101       | 50                      | 50                          | 0                       |
| StarGAN-v2  | 4,000     | 3,000                   | 4,000                       | 999                     |
| DeepFaceLab | 149       | 149                     | 149                         | 100                     |
| StarGAN     | 3,968     | 0                       | 0                           | 0                       |
| StyleCLIP   | 3,824     | 3,824                   | 3,824                       | 2,824                   |
| **Total**   | 17,925    | 11,886 (66.3%)          | 12,893 (71.9%)              | -                       |

Results computed with the standard FIT/SELECT/TEST protocol on this cache are
therefore affected by train/test overlap. The accompanying paper uses a
leave-one-generator-out control, which fits combiners on the other test
generators only, as its primary estimate (Appendix F of the paper).

### Residual limitations under LOGO

The primary leave-one-generator-out (LOGO) analysis excludes identified training rows
whose features match a held-out test image. Identical-feature checks cannot rule out
re-encoded copies, and combiner retraining cannot undo exposure during frozen-expert training.
Repeated content remains within each fold's training data, between FIT and SELECT, and
across inner cross-validation folds. Repeated test rows also affect image weighting.
Appendix A reports sensitivity checks that remove exact duplicate training rows before
splitting; this also changes the relative weight of repeated source images. The code and
stored outputs are documented in the repository's
[sensitivity README](https://github.com/rthinker/frozen-expert-eval/blob/main/camera_ready/scripts/sensitivity/README.md).

## Known integrity properties

- Seventeen cache rows refer to nine distinct zero-byte source-file paths
  in our local copy of the DF40 image folders (FIT 7, SELECT 1,
  MidJourney 5, StarGAN-v2 4). The loader substituted the first item of
  the corresponding extraction split, including its features and label.
  Nine cached labels disagree with their requested source paths
  (FIT 2, MidJourney 5, StarGAN-v2 2); eight affected rows have matching
  classes. The arrays were not relabelled or filtered. All 17 row
  exclusions are listed at the end of this card and embedded directly in
  `croissant.json` under `records/source_integrity_exclusions`.
- Repeated images within a split: StarGAN-v2 holds 1,024 distinct images
  among its 4,000 rows; MidJourney 1,877 among 1,882.
- Seven images, all real, appear in both the CollabDiff and StyleCLIP test
  splits (14 rows).

The reuse check of the code repository (`camera_ready/scripts/review/`, step
C01 of its harness) recomputes the reuse and duplicate counts in this card
from this cache. The counts in DF40's split lists need those lists, and the
zero-byte image counts need the DF40 image folders in the layout this cache was
extracted from (`camera_ready/REPRODUCTION.md` of the code repository); DF40's
public download is laid out differently.

## Intended use

- Auditing and reproducing frozen-expert composition methods on the DF40
  benchmark with the paired code release (CC BY-NC-SA 4.0).
- Methodological extensions: varying combiner choices, pool compositions, or
  evaluation controls. Report results with an overlap-aware protocol, or
  label standard-protocol results as overlap-affected.
- Educational and methodological reuse under CC BY-NC-SA 4.0, subject
  to the upstream terms described below.

## Non-intended use

- Training or fine-tuning deepfake detection systems intended for
  deployment, distribution, or commercial use.
- Building commercial deepfake detection or generation products.
- Surveillance, attribution, or forensic applications without independent
  validation, calibration, and human oversight.
- Any use incompatible with the upstream DF40 CC BY-NC 4.0 license terms.
- Attempting to re-identify individuals. Some sample ids keep upstream
  file names that contain YouTube-style video identifiers (1,530 rows) or
  MidJourney prompt text with demographic words describing synthetic
  faces (1,764 rows); they are kept because the reuse check matches
  paths. Numerical embeddings and these source identifiers should not be
  assumed anonymous; absence of raw images does not establish absence of
  identity information.

## License and attribution

Our derived-cache contributions are distributed under CC BY-NC-SA 4.0
(https://creativecommons.org/licenses/by-nc-sa/4.0/).

See the [licence and attribution notice](LICENSE_NOTICE.txt) for upstream terms and citations.

Rights holders of an upstream source who have concerns about this cache can
open an issue at https://github.com/rthinker/frozen-expert-eval/issues; we
will review each request and remove affected data where appropriate.

## Limitations

- Train/test overlap (see above): standard-protocol results on this cache
  are overlap-affected.
- One benchmark (DF40); numerical effect sizes should not be generalized
  beyond this setting without further study.
- Binary labels only (K = 2).
- FE-trained detector variants and the e4e subset (its 29,126 images in
  `DF40_all.json` and its test set) are excluded because, in our setup, their
  results did not match the DF40 authors' reported numbers within tolerance.
- No demographic, fairness, or calibration metadata.

## Verification

`CHECKSUMS.txt` contains the 460 unchanged data/provenance digests.
`croissant.json` lists the same 460 files at their pinned Hub addresses.
`CACHE_PROVENANCE.json` binds the manifest and the current
`LICENSE_NOTICE.txt` separately by size and SHA-256. The current package includes
that notice. The pinned review revision retains its own notice. The download
below keeps the current notice while fetching the pinned data:

```
huggingface-cli download anon-frozen-experts/df40-frozen-experts-anon --repo-type dataset --revision 18fb1b212f7251799fca8cb4d549be6d239ce13a --local-dir output/cached_features --include "*/*"
python scripts/review/validate_cache_release.py --cache-root output/cached_features
```

Run these commands from the code repository root. The `--include` filter
fetches only the split folders, which preserves the synchronized metadata
shipped in that folder. The validator checks
every data hash, the manifest and current notice independently, the
Croissant inventory, and this card as `DATASET_CARD.md`. For another cache
location, first copy the sidecar files from `output/cached_features/`, then
change both paths. `sha256sum -c CHECKSUMS.txt` checks only the data;
use the validator for the full release contract.

Metadata/schema validation is distinct from inventory, label and record-
loading checks. `mlcroissant` 1.1.0 can read the embedded exclusion list,
but has no decoder for `application/x-numpy`. Use NumPy and the paired
release code for the original arrays. See the [Croissant validation record](CROISSANT_VALIDATION.txt).

## Citation

```bibtex
@inproceedings{pereira2026predictions,
  title     = {When Are Predictions Enough? An Evaluation Protocol for Frozen Expert Composition},
  author    = {Pereira, Alessandro and Ransom, Keith J and Mitchell, Lewis},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2026), Evaluations and Datasets Track},
  year      = {2026}
}
```

Please also cite DF40 (Yan et al., NeurIPS 2024), from which the cache is
derived, and DeepfakeBench (Yan et al., NeurIPS 2023), on which DF40's
codebase is based. DF40 explicitly requests both citations; see its
[attribution instructions](https://github.com/YZY-stack/DF40#citations)
and the [licence and attribution notice](LICENSE_NOTICE.txt).

## Source-integrity exclusions

This is the complete list of **17 affected cache rows**, grouped by their
**nine distinct source-file paths** to avoid repeating long identifiers.
Each number is a **zero-based row index** in the named original cache
split. A dash means that the path has no affected row in that split.

| Source identifier (exact relative path)                                                                                    |    FIT | SELECT | MidJourney | starganv2 |
| -------------------------------------------------------------------------------------------------------------------------- | -----: | -----: | ---------: | --------: |
| `MidJourney/fake/13592447708_Clean_East_Asian_male_face_only_face_shown_close-up_3ad609fd-9d87-4242-9d5b-d2776a3c2c7e.png` |   3678 |      — |       652† |         — |
| `starganv2/real/547.jpg`                                                                                                   |  4842† |      — |          — |      1957 |
| `starganv2/real/1132.jpg`                                                                                                  | 10932† |      — |          — |      2836 |
| `MidJourney/fake/13592447708_Clean_East_Asian_female_face_face_only_shown_face_c_ba354e74-3cb4-4f63-9ad6-f40a641b558a.png` |  12646 |      — |      1189† |         — |
| `MidJourney/fake/13592447708_Single_person_East_Asian_male_looking_at_the_camera_75e759ff-c23a-4679-bd68-76170370d3d1.png` |  15495 |      — |        28† |         — |
| `MidJourney/fake/13592447708_Single_person_white_female_looking_at_the_camera_fa_d28fc222-bd3f-4ee6-ad9f-7af441607d82.png` |  20936 |      — |       496† |         — |
| `starganv2/fake/1096.jpg`                                                                                                  |  22368 |      — |          — |     2752† |
| `MidJourney/fake/13592447708_Single_person_white_female_looking_at_the_camera_fa_72edc876-2d1f-41b6-b7cb-caaa907da5ab.png` |      — |   3161 |       588† |         — |
| `starganv2/fake/779.jpg`                                                                                                   |      — |      — |          — |     1610† |

Totals: **FIT 7; SELECT 1; MidJourney 5; starganv2 4**. No exclusions
were found in the other six splits. A dagger (†) marks the nine rows whose
cached labels disagree with the requested path class. The two marked FIT
rows store label 1 for real paths; the seven marked test rows store label 0
for fake paths. **All 17 rows are affected**, including the eight without a
dagger. The source-file size was zero for every entry.

The identical list is embedded as a Croissant RecordSet with id
`records/source_integrity_exclusions`, using `data` for the actual records
(not examples). Each record contains `split`, `row_index`, `sample_id`,
`cached_label`, `path_label`, `label_path_disagreement`, `source_size_bytes`,
`reason`, and a unique `exclusion_id`. No separate exclusion file is needed.

The released cache and reported results retain all 17 rows. For optional
filtering, remove each listed row from all aligned features, logits, labels
and identifiers together. For LOGO, apply the MidJourney and starganv2
indices to their source arrays before constructing folds.
