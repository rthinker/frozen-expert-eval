"""Reporting module: manifest scanning, LaTeX table generation, figure rendering.

Scripts should import from here instead of re-implementing classification /
rendering logic. Keeps reporting code testable without side-effecting
script-level path manipulations (import _bootstrap, etc.).
"""

from src.reporting.manifest import scan_results  # noqa: F401
from src.reporting.tables import render_all_tables  # noqa: F401
