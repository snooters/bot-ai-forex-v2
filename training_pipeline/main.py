#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import TrainingConfig
from .trainer import Trainer
from .evaluator import Evaluator
from .data_loader import DataLoader
from .features import FeatureEngineer
from .labeling import LabelEngine
from .model import XGBoostModel
from .utils import setup_logger


def cmd_train(args: argparse.Namespace):
    config = TrainingConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        log_level=args.log_level,
        rolling=args.rolling,
        multi_timeframe=args.multi_tf,
        window_days=args.window_days,
        step_days=args.step_days,
        prediction_horizon=args.horizon,
        threshold=args.threshold,
        num_rolls=args.num_rolls,
    )
    logger = setup_logger("main", config.log_dir, config.log_level)
    logger.info("=" * 60)
    logger.info("TRAINING COMMAND")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Rolling: {config.rolling}")
    logger.info(f"Multi-TF: {config.multi_timeframe}")
    logger.info(f"Window: {config.window_days}d, Step: {config.step_days}d")
    logger.info(f"Horizon: {config.prediction_horizon}, Threshold: {config.threshold}")
    logger.info("=" * 60)

    trainer = Trainer(config)
    report = trainer.train()

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    bt = report.get("backtest", {})
    print(f"  Test Accuracy:  {report.get('accuracy', 'N/A'):.4f}")
    print(f"  F1 (weighted):  {report.get('f1_weighted', 'N/A'):.4f}")
    print(f"  Total Trades:   {bt.get('total_trades', 0)}")
    print(f"  Win Rate:       {bt.get('win_rate', 0):.2%}")
    print(f"  Profit Factor:  {bt.get('profit_factor', 0):.2f}")
    print(f"  Sharpe Ratio:   {bt.get('sharpe_ratio', 0):.2f}")
    print(f"  Max DD:         {bt.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Total Return:   {bt.get('total_return_pct', 0):.2f}%")
    print(f"  Model saved to: {config.output_dir}")
    print("=" * 60)

    return 0


def cmd_evaluate(args: argparse.Namespace):
    config = TrainingConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        log_level=args.log_level,
        prediction_horizon=args.horizon,
        threshold=args.threshold,
    )

    logger = setup_logger("main", config.log_dir, config.log_level)
    logger.info("=" * 60)
    logger.info("EVALUATE COMMAND")
    logger.info("=" * 60)

    model = XGBoostModel(config)
    model.load(config.output_dir)
    logger.info(f"Loaded model with {len(model.feature_names)} features")

    df = DataLoader(config).load()
    df = FeatureEngineer(config).compute_all(df)
    df = LabelEngine(config).create_labels(df)

    features = [c for c in model.feature_names if c in df.columns]
    logger.info(f"Using {len(features)} features for evaluation")

    if not features:
        logger.error("No common features between model and data")
        return 1

    X = df[features].values.astype("float32")
    y = df["label_encoded"].values

    y_pred = model.predict(pd.DataFrame(X, columns=features))
    y_proba = model.predict_proba(pd.DataFrame(X, columns=features))

    evaluator = Evaluator(config)
    report = evaluator.full_report(pd.Series(y), y_pred, y_proba, df)

    eval_path = Path(config.output_dir) / "evaluation_report.json"
    from .utils import safe_json_dump
    safe_json_dump(report, str(eval_path))
    logger.info(f"Evaluation report saved to {eval_path}")

    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    bt = report.get("backtest", {})
    print(f"  Samples:        {report.get('num_samples', 0)}")
    print(f"  Accuracy:       {report.get('accuracy', 'N/A'):.4f}")
    print(f"  F1 (weighted):  {report.get('f1_weighted', 'N/A'):.4f}")
    print(f"  Precision:      {report.get('precision_weighted', 'N/A'):.4f}")
    print(f"  Recall:         {report.get('recall_weighted', 'N/A'):.4f}")
    print(f"  Total Trades:   {bt.get('total_trades', 0)}")
    print(f"  Win Rate:       {bt.get('win_rate', 0):.2%}")
    print(f"  Profit Factor:  {bt.get('profit_factor', 0):.2f}")
    print(f"  Sharpe Ratio:   {bt.get('sharpe_ratio', 0):.2f}")
    print(f"  Max DD:         {bt.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Total Return:   {bt.get('total_return_pct', 0):.2f}%")
    print("=" * 60)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="XGBoost Trading Model Training Pipeline"
    )
    parser.add_argument("command", nargs="?", default="train",
                        choices=["train", "evaluate"],
                        help="Command to run (default: train)")
    parser.add_argument("--data-path", type=str, default="./data",
                        help="Path to parquet data directory or file")
    parser.add_argument("--output-dir", type=str, default="./models/xgboost_model",
                        help="Output directory for model and reports")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Log directory")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log level")
    parser.add_argument("--rolling", action="store_true",
                        help="Enable rolling window training")
    parser.add_argument("--multi-tf", action="store_true",
                        help="Multi-timeframe training (load files, align, merge features)")
    parser.add_argument("--window-days", type=int, default=730,
                        help="Training window size in days")
    parser.add_argument("--step-days", type=int, default=30,
                        help="Step size in days for walk-forward")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Prediction horizon (candles ahead)")
    parser.add_argument("--threshold", type=float, default=0.001,
                        help="Label threshold for BUY/SELL")
    parser.add_argument("--num-rolls", type=int, default=4,
                        help="Number of rolling windows")

    args = parser.parse_args()

    if args.command == "train":
        return cmd_train(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
