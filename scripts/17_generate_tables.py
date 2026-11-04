"""Render Block A diagnostic tables from output/results/*.json."""

import _bootstrap  # noqa: F401  # MUST be first: binds CUBLAS_WORKSPACE_CONFIG before torch.

import argparse

from src.utils.config import (
    DEFAULT_TRAIN_TAG, RESULTS_ROOT, TABLES_ROOT, FIGURES_ROOT,
)
from src.utils.logging_setup import setup_logging

from src.reporting.manifest import scan_results
from src.reporting.tables import render_all_tables


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_tag", type=str, default=DEFAULT_TRAIN_TAG)
    args = p.parse_args()

    log = setup_logging("17_generate_tables")

    log.info("Scanning result JSONs...")
    manifest = scan_results(RESULTS_ROOT, train_tag=args.train_tag)
    log.info(
        f"  ds={len(manifest['ds'])}, fs={len(manifest['fs'])}, "
        f"pca={len(manifest['pca'])}, "
        f"disjoint={len(manifest['disjoint'])}, "
        f"rawfeat={len(manifest['rawfeat'])}, "
        f"bootstrap_rawfeat={len(manifest['sample_bootstrap'])}, "
        f"bootstrap_mlp={len(manifest['sample_bootstrap_mlp'])}"
    )

    TABLES_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    written = render_all_tables(manifest, TABLES_ROOT)
    log.info(f"\nWrote {len(written)} tables to {TABLES_ROOT}:")
    for path in written:
        log.info(f"  {path}")


if __name__ == "__main__":
    main()
