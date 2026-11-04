# Frozen-expert evaluation

Code for *When Are Predictions Enough? An Evaluation Protocol for Frozen Expert Composition*
by Alessandro Pereira, Keith J Ransom and Lewis Mitchell (NeurIPS 2026, Evaluations & Datasets track).

The [dataset card](https://huggingface.co/datasets/anon-frozen-experts/df40-frozen-experts-anon)
documents overlap, repeated content and expert-training exposure.

## Reproduce the paper

Run the commands from the repository root.

### 1. Clone and install

```bash
git clone https://github.com/rthinker/frozen-expert-eval.git
cd frozen-expert-eval
```

Use Python 3.11 and follow the [installation guide](camera_ready/INSTALL.md) for the pinned CPU or GPU dependencies.

### 2. Verify the supplied results

This check compares the supplied results against their references and checks the SHA-256 digests of archived score
arrays, fold checkpoints and the provenance ledger. It requires neither the feature cache nor model fitting.

```bash
python camera_ready/scripts/review/reproduce_all_v4.py --verify-only
```

A successful run reports 64 matches for Block A and 52 for Blocks B and C.

### 3. Download and validate the pinned cache

For experiments that read features, download the approximately 3.2 GiB cache. The `--include` filter
fetches only the split folders, which preserves the verifier files that ship in the repository.

```bash
huggingface-cli download anon-frozen-experts/df40-frozen-experts-anon --repo-type dataset --revision 18fb1b212f7251799fca8cb4d549be6d239ce13a --local-dir output/cached_features --include "*/*"
python scripts/review/validate_cache_release.py --cache-root output/cached_features
```

The validator checks all 460 files.

### 4. Choose a run

| Goal | Command |
|---|---|
| Preview the experiment plan | `python camera_ready/scripts/review/reproduce_all_v4.py --dry-run` |
| Validate the cache, verify the supplied results and run a smoke fit | `python camera_ready/scripts/review/reproduce_all_v4.py --check` |
| Rerun the DF40 workflow | `python camera_ready/scripts/review/reproduce_all_v4.py` |

Budget about 36 hours for a full rerun on the configuration described in the
[reproduction guide](camera_ready/REPRODUCTION.md). Some checks also need the public DF40 split lists.
The guide covers those inputs, partial runs, resuming, figures and sensitivity checks.
The [release manifest](camera_ready/RELEASE_MANIFEST.md) maps paper results to steps, outputs and tolerances.

## Repository layout

| Path | Contents |
|---|---|
| `src/`, `composition_eval/` and `scripts/` | Evaluation protocol and standard-split pipeline (Block A) |
| `camera_ready/` | LOGO pipeline (Block B), further analyses (Block C), reproduction harness and verification inputs |
| `camera_ready/dataset/CACHE_PROVENANCE.json` | Trusted cache contract for local verification |
| `output/cached_features/` | Local verifier files and download location for the data |

Open an issue for questions or problems with the release.

## Licensing and attribution

See [LICENSE](LICENSE) for the release terms and [Licensing and attribution](camera_ready/LICENSES.md) for the cache and upstream components.

## Citation

```bibtex
@inproceedings{pereira2026predictions,
  title     = {When Are Predictions Enough? An Evaluation Protocol for Frozen Expert Composition},
  author    = {Pereira, Alessandro and Ransom, Keith J and Mitchell, Lewis},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2026), Evaluations and Datasets Track},
  year      = {2026}
}
```
