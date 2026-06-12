"""Training aman untuk RAM 4GB — filter data ke 2022+ biar LSTM tidak OOM.

Usage:
    python train_safe.py
"""

import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from core.config import config
from core.constants import LOOKAHEAD_5, RESULTS_DIR
from data.data_loader import DataLoader
from features.feature_pipeline import FeaturePipeline
from ml.ensemble import VotingEnsemble
from ml.model_manager import ModelManager
from ml.trainer import ModelTrainer
from learning.walk_forward_validator import WalkForwardValidator
from utils.logger import get_logger

logger = get_logger("train_safe")

PRIMARY_TF = 5
PAIR = config.trading["pairs"][0] if config.trading["pairs"] else "EURUSD"
FILTER_YEAR = 2022


def load_filtered_m5(pair: str) -> pd.DataFrame:
    symbol = pair
    loader = DataLoader(symbol)
    aligned = loader.load_aligned()

    if aligned.empty:
        raise ValueError(f"No M5 data for {pair}")

    if "time" in aligned.columns:
        before = len(aligned)
        aligned["time"] = pd.to_datetime(aligned["time"])
        aligned = aligned[aligned["time"] >= pd.Timestamp(FILTER_YEAR, 1, 1)].copy()
        after = len(aligned)
        logger.info(f"Filtered data: {before} -> {after} rows ({FILTER_YEAR}+)")

    aligned["pair"] = pair
    aligned["timeframe"] = PRIMARY_TF
    logger.info(f"Loaded {len(aligned)} rows with {len(aligned.columns)} columns")
    return aligned


def main():
    start = time.monotonic()
    logger.info(f"=== SAFE TRAINING {PAIR} — data from {FILTER_YEAR}+ (RAM-safe) ===")

    # ── Load & filter data ──
    df = load_filtered_m5(PAIR)

    # ── Train ──
    trainer = ModelTrainer()
    model_manager = ModelManager()

    logger.info("Preparing features + training...")
    _bt = config.training["buy_threshold"]
    _st = config.training["sell_threshold"]
    X, y, features, df_clean = trainer.prepare_training_data(
        df, lookahead=LOOKAHEAD_5,
        buy_threshold=_bt, sell_threshold=_st,
        target_type="class",
        pair="EURUSD", timeframe=5,
    )

    recency = ModelTrainer.compute_recency_weights(df_clean["time"]) if "time" in df_clean.columns else None
    logger.info(f"Samples: {len(X)} | Features: {len(features)}")

    results = trainer.train_all_models(
        X, y,
        feature_cols=features,
        target_type="class",
        recency_weights=recency,
        tf_label="M5",
    )

    ensemble = trainer.get_ensemble()
    num_models = ensemble.get_num_models()
    if num_models == 0:
        logger.error("No models trained")
        return

    version = model_manager.save_ensemble(ensemble, timeframe=PRIMARY_TF)
    logger.info(f"Trained {num_models} models, saved as v{version}")

    # Save feature importance
    trainer.save_feature_importance(ensemble, features, PAIR, PRIMARY_TF, version)

    # Walk-forward validation
    logger.info("Running walk-forward validation...")
    validator = WalkForwardValidator()
    val_result = validator.validate(
        df, ensemble, trainer, timeframe_label="M5+CTX",
    )
    validator.save_results(val_result, PAIR, PRIMARY_TF, version)

    grade = val_result.get("grade", "N/A")
    passed = val_result.get("passed", False)
    logger.info(f"Validation: grade={grade}, passed={passed}")

    # Save report
    from pathlib import Path
    import json
    report_dir = Path(RESULTS_DIR) / PAIR / "training_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"train_safe_{timestamp}.json"
    report_data = {
        "pair": PAIR,
        "success": True,
        "version": version,
        "num_models": num_models,
        "samples": len(X),
        "features": len(features),
        "filter_year": FILTER_YEAR,
        "elapsed": round(time.monotonic() - start, 1),
    }
    if val_result:
        report_data["validation"] = {
            "grade": val_result.get("grade"),
            "passed": val_result.get("passed"),
            "windows": val_result.get("total_windows"),
        }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    elapsed = time.monotonic() - start
    logger.info(f"Done in {elapsed:.1f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
