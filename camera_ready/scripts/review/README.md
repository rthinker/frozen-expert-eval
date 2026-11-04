# Reproduction and verification

Run the current harness, `reproduce_all_v4.py`, from the repository root. It
runs the reviewed pipeline as Block A and the camera-ready LOGO experiments
and analyses as Blocks B and C. The [release manifest](../../RELEASE_MANIFEST.md)
maps each step to its inputs and outputs. The [installation guide](../../INSTALL.md)
describes the cache and environment.

| Script | Purpose |
|---|---|
| `reproduce_all_v4.py` | Verify supplied results, run a smoke check, plan or run selected steps, and resume a recorded run. |
| `stamp_checkpoints_v2.py` | Harness launcher for checkpointed steps. It binds checkpoints and their paired arrays to the step attempt. |
| `check_split_reuse_v3.py` | Appendix A reuse counts (C01). Reports which routes were carried over when the optional DF40 lists or image root are absent. |
| `check_fs1_score_precision_v2.py` | Compare saved ladder and mechanism FS1 predictions without fitting (B30). |
| `refit_fs1_v3.py` | Standalone batched FS1 refit. LOGO mode uses the reviewed folds. |
| `trusted_run_pairs_v1.py --check` | Verify the digests of stored runs paired by the release's provenance mapping. |

Use `--verify-only` to compare supplied results and check the stored-evidence digests without the feature cache.
`--check` uses the cache and fits the arch6 FS1 smoke model. Resume through
`reproduce_all_v4.py --resume-run <run>`, which checks the run record and
checkpoints before reuse. For a standalone report, use a separate output path
such as `camera_ready/output/checks/`.
