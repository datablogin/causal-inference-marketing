"""Logging setup compatible with analytics-backend-monorepo."""

import logging
import sys

from shared.config import CausalInferenceConfig, Environment


def setup_logging(config: CausalInferenceConfig | None = None) -> None:
    """Set up logging configuration."""
    if config is None:
        config = CausalInferenceConfig()

    # Configure log level based on environment
    if config.environment == Environment.PRODUCTION:
        log_level = logging.INFO
    elif config.environment == Environment.DEVELOPMENT:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Set up root logger
    logging.basicConfig(
        level=log_level, format=log_format, handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Configure specific loggers
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO
        if config.environment == Environment.DEVELOPMENT
        else logging.WARNING
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
