"""Logging utilities."""

import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

from .config import Args

LOG_FORMAT = "<cyan>[{time:YYYY-MM-DD HH:mm:ss}]</cyan><blue>[{name}]</blue>[<level>{level}</level>] {message}"  # noqa: E501
LOG_FORMAT_FILE = "[{time:YYYY-MM-DD HH:mm:ss}][{name}][{level}] {message}"


def setup_logging(log_level: str = "INFO") -> None:
    """Configure the console logger.

    Args:
        log_level: Log level (default: "INFO")
    """
    logger.remove()
    logger.add(sys.stdout, format=LOG_FORMAT, colorize=True, level=log_level.upper())


def setup_file_logging(cfg: Args, output_dir: str = "logs") -> Path:
    """Add a file sink and save config. Called by the @cli decorator.

    Args:
        cfg: Full Args config (saved to the run directory)
        output_dir: Base output directory (default: "logs")

    Returns:
        Path to the run directory
    """
    now = datetime.now()
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_dir = Path(output_dir, now.strftime("%Y-%m-%d"), f"{now.strftime('%H-%M-%S')}_{job_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)

    log_file = run_dir / "main.log"
    logger.add(log_file, format=LOG_FORMAT_FILE, level="DEBUG", colorize=False)

    logger.info(f"Logging to: {log_file.absolute()}")
    logger.info(f"Config saved to: {config_path.absolute()}")

    return run_dir
