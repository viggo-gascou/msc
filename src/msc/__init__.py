"""a project."""

import sys

from loguru import logger

from .log_utils import setup_logging

# Default console logger
setup_logging()
