"""Main training entry point — single model on M5 with M15/M30/H1/H4 context.

Prerequisite:
    python download_data.py --pair EURUSD    # download MT5 data since 2019

Usage:
    python train.py --pair EURUSD
    python train.py --pair EURUSD --validate --save-fi
    python train.py --all --validate
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import config
from core.constants import LOOKAHEAD_5, RESULTS_DIR
from data.data_loader import DataLoader, ALL_TFS, CONTEXT_TFS
from features.feature_pipeline import FeaturePipeline
from ml.ensemble import VotingEnsemble
from ml.model_manager import ModelManager
from ml.trainer import ModelTrainer
from learning.walk_forward_validator import WalkForwardValidator
from utils.logger import get_logger

logger = get_logger("train")

PRIMARY_TF = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Forex Trading Bot — Training Pipeline\n"
                    "M5 = entry TF, M15/M30/H1/H4 = context only"
    )
    parser.add_argument("--pair", default="EURUSD", help="Currency pair to train")
    parser.add_argument("--lookahead", type=int, default=LOOKAHEAD_5,
                        help="Lookahead candles for target (M5 bars)")
    parser.add_argument("--buy-threshold", type=float, default=0.001,
                        help="Return threshold for BUY signal")
    parser.add_argument("--sell-threshold", type=float, default=0.001,
                        help="Return threshold for SELL signal")
    parser.add_argument("--target-type", default="class", choices=["class", "regression"],
                        help="Target type: class (0/1/2) or regression")
    parser.add_argument("--validate", action="store_true",
                        help="Run walk-forward validation after training")
    parser.add_argument("--save-fi", action="store_true",
                        help="Save feature importance CSV + PNG")
    parser.add_argument("--promote", action="store_true",
                        help="Promote best candidate to production")
    parser.add_argument("--all", action="store_true",
                        help="Train all pairs from storage")
    return parser.parse_args()


def load_m5_with_context(pair: str) -> pd.DataFrame:
    """Load M5 data with M15/M30/H1/H4 context features aligned."""
    symbol = f"{pair}.fl"
    loader = DataLoader(symbol)
    aligned = loader.load_aligned()
    if aligned.empty:
        raise ValueError(f"No M5 data for {pair}")
    aligned["pair"] = pair
    aligned["timeframe"] = PRIMARY_TF
    logger.info(f"Loaded M5 ({len(aligned)} rows) with {len(aligned.columns)} columns")
    logger.info(f"Context TFs: {[c for c in aligned.columns if '_tf' in c][:10]}...")
    return aligned


def train_m5_model(
    df: pd.DataFrame,
    trainer: ModelTrainer,
    model_manager: ModelManager,
    args: argparse.Namespace,
) -> Optional[Dict]:
    """Train a single model on M5 with multi-TF context features."""
    start = time.monotonic()
    logger.info("Preparing M5 training data with M15/M30/H1/H4 context features...")

    try:
        X, y, features, df_clean = trainer.prepare_training_data(
            df,
            lookahead=args.lookahead,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            target_type=args.target_type,
        )
    except Exception as e:
        logger.error(f"Feature prep failed: {e}")
        return None

    from ml.trainer import ModelTrainer as _MT
    recency = _MT.compute_recency_weights(df_clean["time"]) if "time" in df_clean.columns else None
    logger.info(f"Training samples: {len(X)} | Features: {len(features)}")

    try:
        results = trainer.train_all_models(
            X, y,
            feature_cols=features,
            target_type=args.target_type,
            recency_weights=recency,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None

    ensemble = trainer.get_ensemble()
    num_models = ensemble.get_num_models()
    if num_models == 0:
        logger.warning("No models trained")
        return None

    version = model_manager.save_ensemble(ensemble, timeframe=PRIMARY_TF)
    logger.info(f"Trained {num_models} models on M5, saved as v{version}")

    if args.save_fi:
        trainer.save_feature_importance(
            ensemble, features, args.pair, PRIMARY_TF, version
        )

    elapsed = time.monotonic() - start
    return {
        "timeframe": PRIMARY_TF,
        "version": version,
        "num_models": num_models,
        "samples": len(X),
        "features": len(features),
        "ensemble": ensemble,
        "elapsed": round(elapsed, 1),
    }


def validate_m5_model(
    df: pd.DataFrame,
    ensemble: VotingEnsemble,
    trainer: ModelTrainer,
    args: argparse.Namespace,
    version: str,
) -> Dict:
    """Run walk-forward validation on M5 with multi-TF context."""
    logger.info("Running walk-forward validation on M5 (multi-TF context)...")
    validator = WalkForwardValidator()
    result = validator.validate(
        df, ensemble, trainer,
        timeframe_label="M5+CTX",
    )
    validator.save_results(result, args.pair, PRIMARY_TF, version)

    grade = result.get("grade", "N/A")
    passed = result.get("passed", False)
    logger.info(f"Validation: grade={grade}, passed={passed}")
    return result


def train_pair(args: argparse.Namespace) -> Dict:
    """Train a single M5 model with multi-TF context for one pair."""
    pair = args.pair
    logger.info(f"=== Training {pair}: M5 entry + M15/M30/H1/H4 context ===")

    df = load_m5_with_context(pair)

    trainer = ModelTrainer()
    model_manager = ModelManager()
    results = {"pair": pair, "success": False}

    res = train_m5_model(df, trainer, model_manager, args)
    if res is None:
        logger.error(f"Training failed for {pair}")
        return {"pair": pair, "success": False, "error": "training failed"}

    ensemble = res.pop("ensemble")
    results["training"] = res
    results["success"] = True

    if args.validate:
        val_result = validate_m5_model(df, ensemble, trainer, args, res["version"])
        results["validation"] = {
            "grade": val_result.get("grade"),
            "passed": val_result.get("passed"),
            "windows": val_result.get("total_windows"),
        }
        results["success"] = val_result.get("passed", False)

    # Save training report
    report_dir = Path(RESULTS_DIR) / pair / "training_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"train_m5_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Training report saved to {report_path}")

    return results


def find_pairs_in_storage() -> List[str]:
    """Discover available pairs from parquet storage."""
    storage_dir = Path(config.data.get("storage", "storage"))
    if not storage_dir.exists():
        logger.error(f"Storage dir not found: {storage_dir}")
        return []

    pair_files = set()
    for f in storage_dir.glob("*.parquet"):
        parts = f.stem.split("_")
        if len(parts) >= 1:
            pair = parts[0].upper()
            if len(pair) == 6:
                pair_files.add(pair)
    return sorted(pair_files)


def main() -> None:
    args = parse_args()

    if args.all:
        pairs = find_pairs_in_storage()
        if not pairs:
            logger.error("No pairs found in storage")
            sys.exit(1)

        logger.info(f"Found pairs: {', '.join(pairs)}")
        all_results = {}
        for pair in pairs:
            args.pair = pair
            result = train_pair(args)
            all_results[pair] = result.get("success", False)

        successes = sum(1 for v in all_results.values() if v)
        total = len(all_results)
        logger.info(f"All pairs done: {successes}/{total} succeeded")
    else:
        train_pair(args)


if __name__ == "__main__":
    main()
