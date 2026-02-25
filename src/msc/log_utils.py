"""Logging utilities."""

import logging
import sys
from datetime import datetime
from logging import StreamHandler
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

from .config import Params

# Centralized log format (plain text for files)
LOG_FORMAT = "<cyan>[{time:YYYY-MM-DD HH:mm:ss}]</cyan><blue>[{name}]</blue>[<level>{level}</level>] {message}"  # noqa: E501


def setup_logging(
    cfg: Params, output_dir: str = "logs", log_level: str = "INFO"
) -> Path:
    """Setup logging with file output.

    Args:
        cfg: Configuration object
        output_dir: Base output directory (default: "logs")
        log_level: Log level (default: "INFO")

    Returns:
        Path to the run directory
    """
    # Update console log level
    logger.remove()
    logger.add(
        sink=StreamHandler(sys.stdout),
        format=LOG_FORMAT,
        colorize=True,
        level=logging.getLevelNamesMapping()[log_level.upper()],
    )

    # Create output directory with date/time subdirectories
    now = datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    time_dir = now.strftime("%H-%M-%S")
    run_dir = Path(output_dir, date_dir, time_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = run_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)

    # Add file handler to loguru (plain text, no colors)
    log_file = run_dir / "main.log"
    logger.add(log_file, format=LOG_FORMAT, level="DEBUG", colorize=False)

    # Log initial info
    logger.info(f"Logging to: {log_file.absolute()}")
    logger.info(f"Config saved to: {config_path.absolute()}")

    return run_dir
