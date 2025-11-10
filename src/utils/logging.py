import logging
import os
from pathlib import Path
from typing import Optional


def create_logger(name: Optional[str] = None, level: str | int = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def configure_file_logging(log_dir: str | Path, filename: str = "run.log", level: str | int = "INFO") -> Path:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename

    logger = logging.getLogger()
    logger.setLevel(level if isinstance(level, int) else getattr(logging, level.upper(), logging.INFO))

    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == log_path.as_posix() for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    os.environ["LOG_PATH"] = str(log_path)
    return log_path
