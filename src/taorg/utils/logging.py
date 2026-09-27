"""Structured logging setup for TAORG.

Provides consistent logging configuration across the project.
Library code uses the standard logging module; this configures handlers.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure project-wide logging.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path to write logs to.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
    ]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
