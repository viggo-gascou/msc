"""CLI utilities for easy setup."""

from typing import Callable

import tyro

from msc.config import Args
from msc.log_utils import setup_logging


def cli(
    config_class: type = Args,
) -> Callable[[Callable[[Args], None]], Callable[[], None]]:
    """Decorator that parses CLI arguments and sets up logging.

    Args:
        config_class: Config dataclass to use (default: Args)

    Returns:
        Decorator function
    """

    def decorator(func: Callable[[Args], None]) -> Callable[[], None]:
        def wrapper() -> None:
            cfg = tyro.cli(config_class)
            setup_logging(log_level=cfg.log_level)
            return func(cfg)

        return wrapper

    return decorator
