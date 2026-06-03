import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def setup_logger(name: str, log_dir: str = "./logs", level: str = "INFO") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"training_pipeline.{name}")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = RotatingFileHandler(
        str(Path(log_dir) / f"{name}.log"),
        maxBytes=10_485_760,
        backupCount=5,
        encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def safe_json_dump(obj: Any, path: str, indent: int = 2) -> None:
    class NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.bool_,)):
                return bool(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (pd.Timestamp,)):
                return str(o)
            return super().default(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, cls=NpEncoder, indent=indent)


def load_json(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def validate_dataframe(df: pd.DataFrame, required_cols: list) -> Optional[str]:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return f"Missing columns: {missing}"
    if df.empty:
        return "DataFrame is empty"
    return None
