"""
Logging utilities for healthdq-ai framework
Author: Agate Jarmakoviča
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger
import os


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "text",
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Configure loguru logger for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, only console output)
        log_format: Format type ('text' or 'json')
        rotation: When to rotate log file
        retention: How long to keep old logs
    """
    # Remove default logger
    logger.remove()

    # Determine format
    if log_format == "json":
        format_string = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
        serialize = True
    else:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        serialize = False

    # Add console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=log_level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=format_string,
            level=log_level,
            rotation=rotation,
            retention=retention,
            serialize=serialize,
            enqueue=True,  # Thread-safe
        )

    logger.info(f"Logger initialized with level: {log_level}")


def get_logger(name: str):
    """Get a logger instance with the specified name."""
    return logger.bind(name=name)


# Initialize logger from environment variables
def init_from_env():
    """Initialize logger from environment variables."""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE")
    log_format = os.getenv("LOG_FORMAT", "text")

    setup_logger(
        log_level=log_level,
        log_file=log_file,
        log_format=log_format,
    )


# Export logger instance
__all__ = ["logger", "setup_logger", "get_logger", "init_from_env"]
