# Reference values of Blocks B and C

[`reference_results_camera_ready_v2.json`](reference_results_camera_ready_v2.json)
is the reference used by the [current harness](../scripts/review/README.md).
Block A uses the reviewed reference in
[`output/results/reference_results.json`](../../output/results/reference_results.json).

Each reference entry identifies a result file, its source digest, the
expected numeric values and exact strings, and its comparison class.
The [release manifest](../RELEASE_MANIFEST.md) describes the tolerances,
including environment-dependent metrics and rewritten gate evidence.

The maintainer script `generate_reference_v3.py --out <new file>` exports a
reference only when the inputs match the source digests in the shipped
reference. It requires a new destination.

[`stored_evidence.sha256`](stored_evidence.sha256) separately records the
shipped score archives, fold checkpoints and provenance ledger. The
stored-results check verifies their exact bytes. Reruns compare the resulting
metrics using the documented tolerances.
