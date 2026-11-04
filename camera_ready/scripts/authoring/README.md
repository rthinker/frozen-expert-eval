# Figures, statistics and tables

The [release manifest](../../RELEASE_MANIFEST.md) gives the inputs and
outputs of the current authoring steps.

| Script | Purpose |
|---|---|
| `full_ladder_stats_v13.py` | LOGO statistics, inference and the Table 2 audit (B31, B32, C08a). |
| `full_ladder_stats_standard_v3.py` | Statistics against the full ladder on the standard split (C08). |
| `generate_camera_ready_tables_v18.py` | DF40 tables, additional analyses, number ledger and paper-location mapping (C24). |
| `figure_a1_tsne.py` | Standalone Figure A1 command using the feature cache. |
| `figure_a2_full_ladder_v4.py` | Exploratory feature-width and performance plot (C23). |
| `regenerate_appendix_h_v3.py` | Appendix H individual-expert results with the fake-minus-real logit convention (C12). |
| `generate_reference_v3.py` | Maintainer command for rebuilding the pinned Blocks B and C reference. |

`full_ladder_stats_v4.py` supplies the LOGO and width-control summary
functions imported by [the sensitivity summary](../sensitivity/README.md).
Use version 13 for new standalone LOGO statistics. It checks fold and
ladder completeness and run compatibility. Explicit partial modes record
their scope, and the table builder refuses partial inputs.

For current table locations use
[`paper_table_locations.json`](../../provenance/paper_table_locations.json).
The [number ledger](../../provenance/05_number_provenance.csv) retains stable
quantity identifiers for the supplied DF40 results.
