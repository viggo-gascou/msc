"""CLI utilities for easy setup."""

from typing import Callable

import tyro
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from msc.config import Args
from msc.log_utils import setup_file_logging, setup_logging


def cli(
    config_class: type = Args,
) -> Callable[[Callable[[DictConfig], None]], Callable[[], None]]:
    """Decorator that parses CLI arguments and sets up logging.

    Args:
        config_class: Config dataclass to use (default: Config)

    Returns:
        Decorator function
    """

    def decorator(func: Callable[[DictConfig], None]) -> Callable[[], None]:
        def wrapper() -> None:
            # Parse CLI arguments
            cfg = tyro.cli(config_class)

            # Convert to OmegaConf
            cfg_omega = OmegaConf.structured(cfg)

            # Reconfigure console with the requested log level, then add file sink
            setup_logging(log_level=cfg_omega.log_level)
            run_dir = setup_file_logging(cfg_omega)

            # Log the run directory
            logger.info(f"Run directory: {run_dir.absolute()}")

            # Run the function
            return func(cfg_omega)

        return wrapper

    return decorator
