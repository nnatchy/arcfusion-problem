"""
Loguru logging configuration with colors and rotation.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(config):
    """
    Configure loguru logging with colors, rotation, and file output.

    Args:
        config: Config object with logging settings
    """
    # Remove default handler
    logger.remove()

    # Console handler with colors
    logger.add(
        sys.stdout,
        format=config.logging.format,
        level=config.logging.level,
        colorize=config.logging.colorize,
        backtrace=True,
        diagnose=True,  # Show variable values in exceptions
    )

    # Create logs directory if it doesn't exist
    log_dir = Path(config.logging.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # File handler with rotation
    logger.add(
        config.logging.log_file,
        format=config.logging.format,
        level=config.logging.level,
        rotation=config.logging.rotation,  # Rotate at specified size
        retention=config.logging.retention,  # Keep for specified time
        compression="zip",  # Compress old logs
        backtrace=True,
        diagnose=True,
    )

    logger.info("✅ Logging configured successfully")
    logger.debug(f"Log level: {config.logging.level}")
    logger.debug(f"Log file: {config.logging.log_file}")


# Export logger for easy import
__all__ = ["logger", "setup_logging"]
