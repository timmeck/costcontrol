"""Logging utility for CostControl."""

import logging

from src.config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """Get a named logger with consistent formatting."""
    logger = logging.getLogger(f"costcontrol.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger
