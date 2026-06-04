import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler

from core.constants import LOG_DIR

_rich_console = Console()


class BotLogger:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        self._loggers = {}

    def get_logger(self, name: str, level: str = "INFO") -> logging.Logger:
        if name in self._loggers:
            return self._loggers[name]

        logger = logging.getLogger(f"forex_bot.{name}")
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.handlers.clear()

        ch = RichHandler(
            console=_rich_console,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            log_time_format="%Y-%m-%d %H:%M:%S",
        )
        logger.addHandler(ch)

        file_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        log_file = Path(LOG_DIR) / f"{name}.log"
        fh = RotatingFileHandler(
            str(log_file), maxBytes=10_485_760, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(file_fmt)
        logger.addHandler(fh)

        self._loggers[name] = logger
        return logger


bot_logger = BotLogger()


def get_logger(name: str) -> logging.Logger:
    return bot_logger.get_logger(name)


def get_console() -> Console:
    return _rich_console
