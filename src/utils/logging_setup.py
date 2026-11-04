"""Logging setup: configure Python logger to console + file.

Every script calls setup_logging() at the top. Logger output is sent to
both console and a timestamped log file under output/logs/. Note: this
captures log.info() etc., not raw print() or sys.stderr. Scripts should
use the returned logger for all output.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path


def setup_logging(
    script_name: str,
    output_dir: Path = None,
) -> logging.Logger:
    """Configure logging to console + file.

    Args:
        script_name: Name of the calling script (e.g., "02_train").
        output_dir: Directory for log files. Defaults to LOGS_ROOT from config.

    Returns:
        Configured logger.
    """
    if output_dir is None:
        from src.utils.config import LOGS_ROOT
        output_dir = LOGS_ROOT
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    log_file = output_dir / f"{script_name}_{ts}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Log the invocation
    logger.info(f"Script: {script_name}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"CLI: {' '.join(sys.argv)}")

    return logger
