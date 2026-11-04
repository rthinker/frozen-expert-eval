# Installation

Python 3.11. Commands run from the repository root.

The GPU route was tested in a clean environment on Windows (Python 3.11.3) and the CPU route on Linux (Ubuntu
22.04 under WSL, Python 3.11.15); the macOS route has not been tested. Every route installs from a clone of the
repository in editable mode (`pip install -e .`), with the packages of `requirements-lock.txt`: the code reads
`configs/` from the source tree, which a built wheel does not contain. This is the only supported installation; a
built wheel and the `df40` extra are not supported (see "Optional extras" below and [supported scope](REPRODUCTION.md#supported-scope)).

## GPU (CUDA 12.8; Windows or Linux with an NVIDIA GPU), as used for the paper

```
python -m venv .venv
.venv\Scripts\activate                 # Windows
source .venv/bin/activate              # Linux
pip install -r requirements-lock.txt   # pins torch==2.10.0+cu128 via an extra index
pip install -e ".[dev]"                # the dev extra provides pytest
python -m pytest tests camera_ready/tests
```

In Windows cmd.exe, leave out the `#` comments (PowerShell and bash accept them).

If the 2.9 GB torch download stops, fetch the wheel first with a download that resumes after a stall, then install
it in the activated environment and run the remaining commands. For example, on Windows with Python 3.11 (the wheel
is saved next to the repository, not in it):

```
curl -L -C - -o ../torch-2.10.0+cu128-cp311-cp311-win_amd64.whl https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp311-cp311-win_amd64.whl
pip install --no-deps ../torch-2.10.0+cu128-cp311-cp311-win_amd64.whl
```

On other platforms, pick the matching wheel from https://download.pytorch.org/whl/cu128/torch/ and save it with `+`
in place of `%2B` in its name.

## CPU only (Windows or Linux)

The lock file pins the CUDA build of torch, so installing it after a CPU build would replace that build.
Install the CPU build first, then the rest of the lock without its two torch lines and the CUDA index line:

```
python -m venv .venv
(activate as above)
python -m pip install --upgrade "pip>=23.3.2"
pip install torch==2.10.0+cpu torchvision==0.25.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -c "import re; L=open('requirements-lock.txt').read().splitlines(); open('requirements-cpu.txt','w').write('\n'.join(l for l in L if not re.match(r'(torch|torchvision)==|--extra-index-url', l)) + '\n')"
pip install -r requirements-cpu.txt
pip install -e ".[dev]"
python -m pytest tests camera_ready/tests
```

If the pinned CPU torch installation fails, resolve that error before installing the remaining dependencies.

## macOS on Apple Silicon (CPU; untested)

Requires macOS 12 or later and native ARM64 Python 3.11. The pinned PyTorch/torchvision wheels
do not provide this route for Intel Macs or Python running under Rosetta. XGBoost 3.2.0
requires macOS 12+ for its ARM64 wheel.

PyTorch publishes no CUDA build for macOS; the standard PyPI build of the same version runs on the CPU (the
reviewed code uses CUDA when it is available and otherwise the CPU). Install it first, then the rest of the lock
as in the CPU route:

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.10.0 torchvision==0.25.0
python -c "import re; L=open('requirements-lock.txt').read().splitlines(); open('requirements-cpu.txt','w').write('\n'.join(l for l in L if not re.match(r'(torch|torchvision)==|--extra-index-url', l)) + '\n')"
pip install -r requirements-cpu.txt
pip install -e ".[dev]"
python -m pytest tests camera_ready/tests
```

If `import xgboost` fails on macOS, install the OpenMP runtime (`brew install libomp`). FS2 then runs on the
CPU and its values differ from the GPU runs within the tolerance stated in the [reproduction guide](REPRODUCTION.md#reproduction-limitations).

## Feature cache

All scripts run on the CPU once the feature cache (about 3.2 GiB) is downloaded into `output/cached_features/` from
the Hugging Face Hub. See the [dataset card on Hugging Face](https://huggingface.co/datasets/anon-frozen-experts/df40-frozen-experts-anon). `--verify-only` does not need the cache. From the
repository root:

```
huggingface-cli download anon-frozen-experts/df40-frozen-experts-anon --repo-type dataset --revision 18fb1b212f7251799fca8cb4d549be6d239ce13a --local-dir output/cached_features --include "*/*"
```

The `--include` filter fetches only the split folders, which preserves the verifier files already shipped in
`output/cached_features/`, including the local dataset card, Croissant metadata, CC BY-NC-SA 4.0 notice,
checksum manifest and cache contract. The download retrieves the 460 data and provenance files from the
reviewed revision. Validate the populated cache
without fitting any models:

```
python scripts/review/validate_cache_release.py --cache-root output/cached_features
```

The validator compares the local `output/cached_features/CACHE_PROVENANCE.json` with the trusted
`camera_ready/dataset/CACHE_PROVENANCE.json`. It checks the data manifest and current licence notice
against that contract, verifies every data digest and the Croissant inventory, and checks the local card
for the required sections.
The camera-ready harness authenticates the cache before changing any outputs on cache-reading runs,
including partial runs and resumes. `--check` authenticates it before its smoke fit. The harness uses
the checkout's `output/` directory so its children write to the locations it backs up and verifies.
Object-dtype identifiers are never unpickled: their raw bytes must match the code-owned ID manifest,
and the loader uses the corresponding verified plain-text IDs in `camera_ready/output/ids/`.
Custom identifier arrays must use a non-pickled NumPy string dtype. Do not replace the shipped manifest with another Hub revision.

## Expected outcomes

With the supplied results and matching references:

* the tests (`python -m pytest tests camera_ready/tests`, before the cache is downloaded) complete without
  failures. The test for alternate case spellings is skipped on case-sensitive file systems;
* `python camera_ready/scripts/review/reproduce_all_v4.py --verify-only` (no cache needed) ends with
  `Verification: block A 64 MATCH, 0 WARN, 0 FAIL, 0 FAIL in the version 3 checks; blocks B and C 52 MATCH, 0
  WARN/ENV, 0 FAIL, 0 SKIP; stored evidence 0 FAIL.`, and exits 0;
* `--check` (with the cache, about 2 minutes) also prints `[smoke] PASS` for the reviewed FS1 refit (arch6
  `Avg(8) = 0.9657`; 0.9667 on Linux, within the tolerance of 0.005), `score precision: PASS; 16 fold pairs` for the
  saved LOGO score arrays, and `VALIDATION PASSED: 460 files, 460 Croissant entries, dataset card + license notice
  present.` for the cache; exit status 0.

## Optional extras

These extras are outside the supported route ([supported scope](REPRODUCTION.md#supported-scope)).

* **Re-extracting features from DF40 images: not supported.** The release starts from the released feature
  cache; rebuilding the cache with the reviewed `scripts/01_extract.py` is not a supported route ([supported scope](REPRODUCTION.md#supported-scope) and [reproduction limitations](REPRODUCTION.md#reproduction-limitations)), and the release does not test it. Notes for an
  attempt all the same: the lock file already pins every package of the `df40` extra (transformers, Pillow,
  albumentations 1.1.0, opencv-python, lmdb), so install nothing more. Do not add the extra itself: its floor
  `albumentations>=1.4.0`, in the reviewed `pyproject.toml` (kept unchanged), would replace the pinned 1.1.0.
  Extraction also needs the DF40 images and checkpoints, set in `configs/paths.yaml`, and DF40's code: the
  `DeepfakeBench_DF40` folder of the DF40 repository at commit `ce80816`, copied to
  `third_party/DeepfakeBench_DF40/`. The cache was extracted from DF40's split lists without the e4e entries of
  `DF40_all.json` (dataset card, "Limitations"); the ids of FIT and SELECT are in `camera_ready/output/ids/`.
  The known limitations of `scripts/01_extract.py` are listed in the [reproduction guide](REPRODUCTION.md#known-limitations-of-the-reviewed-code).
* `pip install -e ".[authoring]"` (optional; not needed to reproduce the results): `mlcroissant`, to validate the
  dataset's Croissant metadata (`mlcroissant validate --jsonld output/cached_features/croissant.json`), and `pathspec`.

## Memory

FS1 fits need about N x D x 8 bytes for the FIT matrix inside L-BFGS (arch6: about 2 GiB; full21: about
6.7 GiB, skipped automatically by the reviewed scripts when 40% of the RAM is below it, i.e. on hosts below
about 16.7 GiB; on hosts with 17 to 24 GiB set `FS1_BUDGET_GIB` below 6.7 to skip it, or `ALLOW_FS1_OOM=1` to
force it). `camera_ready/scripts/review/refit_fs1_v3.py` refits FS1 and scores it in batches, which lowers only the
memory of scoring (the fit needs the FIT matrix above); it writes a report to `camera_ready/output/checks/`, not
result files. Byte-identical outputs are not expected across machines (BLAS thread counts change L-BFGS iterates slightly); the harness checks the
tolerances of `camera_ready/RELEASE_MANIFEST.md`.
