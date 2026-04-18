"""Project-local logging helpers."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger.

    Keep configuration responsibility at the application entry; this helper
    only standardizes logger naming and retrieval.
    """
    return logging.getLogger(name)
