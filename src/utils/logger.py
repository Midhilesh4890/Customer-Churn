import os
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.config import LOGS_DIR


def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Creates and returns a logger with the specified name and log level.

    Args:
        name (str): The name of the logger.
        log_level (int): The logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{name}_{timestamp}.log"
    log_path = Path(LOGS_DIR) / log_filename

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
