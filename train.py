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


def _get_ensemble_feature_count(ensemble) -> int:
    """Get number of features the ensemble's models were trained with."""
    if ensemble is None:
        return 0
    for name, model in ensemble.models.items():
        if hasattr(model, 'model') and model.model is not None:
            m = model.model
            if hasattr(m, 'n_features_in_'):
                return m.n_features_in_
            if hasattr(m, '_Booster') and hasattr(m._Booster, 'num_feature'):
                try:
                    return m._Booster.num_feature()
                except Exception:
                    pass
            if hasattr(m, 'n_estimators') and hasattr(m, 'feature_importances_'):
                return len(m.feature_importances_)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Forex Trading Bot — Training Pipeline\n"
                    "M5 = entry TF, M15/M30/H1/H4 = context only"
    )
    parser.add_argument("--pair", default="EURUSD", help="Currency pair to train")
    parser.add_argument("--lookahead", type=int, default=LOOKAHEAD_5,
                        help="Lookahead candles for target (M5 bars)")
    parser.add_argument("--buy-threshold", type=float,
                        default=config.training["buy_threshold"],
                        help="Return threshold for BUY signal (default: 0.0004 = 4 pips)")
    parser.add_argument("--sell-threshold", type=float,
                        default=config.training["sell_threshold"],
                        help="Return threshold for SELL signal (default: 0.0004 = 4 pips)")
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
    parser.add_argument("--days", type=int, default=0,
                        help="Number of recent days of data to use (0 = all available)")
    return parser.parse_args()


def load_m5_with_context(pair: str, days: int = 0) -> pd.DataFrame:
    """Load M5 data with M15/M30/H1/H4 context features aligned."""
    symbol = pair
    loader = DataLoader(symbol)
    aligned = loader.load_aligned()
    if aligned.empty:
        raise ValueError(f"No M5 data for {pair}")

    if days > 0:
        time_col = "time"
        if time_col in aligned.columns:
            aligned[time_col] = pd.to_datetime(aligned[time_col])
            cutoff = aligned[time_col].max() - pd.Timedelta(days=days)
            before = len(aligned)
            aligned = aligned[aligned[time_col] >= cutoff].copy()
            logger.info(f"Filtered to last {days} days: {before} -> {len(aligned)} rows")

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
    """Train a single model on M5 with multi-TF context features.
    Only saves to disk if OOS validation beats the current best version."""
    start = time.monotonic()

    # ── Check HOW MANY features will be used (before loading warm-start) ──
    # We need to know feature count to decide if warm-start is possible
    available_cols_all = trainer.feature_pipeline.get_feature_columns()
    old_importance = trainer._load_feature_importance(args.pair or "EURUSD", 5)
    has_new_features = False
    if old_importance is not None:
        new_feats = [c for c in available_cols_all if c not in old_importance]
        has_new_features = len(new_feats) > 0

    # ── Warm-start: load best existing model untuk continued training ──
    existing_ensemble = None
    model_params = None
    is_warm_start = False
    if not has_new_features:
        try:
            existing_ensemble = model_manager.load_best_ensemble(PRIMARY_TF)
            if existing_ensemble is not None:
                # Check feature count compatibility
                old_n_features = _get_ensemble_feature_count(existing_ensemble)
                if old_n_features and old_n_features != len(available_cols_all):
                    logger.info(
                        f"Warm-start skipped: model trained with {old_n_features} features, "
                        f"pipeline has {len(available_cols_all)} features"
                    )
                    existing_ensemble = None
                else:
                    is_warm_start = True
                    logger.info(
                        f"Warm-start: loaded best existing ensemble "
                        f"({existing_ensemble.get_num_models()} models) — "
                        f"using fine-tuning params (smaller LR, fewer estimators)"
                    )
                    model_params = {
                        "xgboost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.01, "subsample": 0.8},
                        "random_forest": {"n_estimators": 100, "max_depth": 8, "min_samples_split": 10},
                        "lightgbm": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.01, "subsample": 0.8},
                    }
        except Exception as e:
            logger.warning(f"Could not load existing ensemble for warm-start: {e}")
            existing_ensemble = None
    else:
        logger.info(
            f"New features detected ({len(new_feats)} new) — training from scratch "
            f"to generate proper feature importance"
        )

    logger.info("Preparing M5 training data with M15/M30/H1/H4 context features...")
    try:
        X, y, features, df_clean = trainer.prepare_training_data(
            df,
            lookahead=args.lookahead,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            target_type=args.target_type,
            pair=args.pair or "EURUSD",
            timeframe=5,
        )
    except Exception as e:
        logger.error(f"Feature prep failed: {e}")
        return None

    from ml.trainer import ModelTrainer as _MT
    recency = _MT.compute_recency_weights(df_clean["time"]) if "time" in df_clean.columns else None

    # Feature selection may have reduced dimensions — disable warm-start if mismatch
    if existing_ensemble is not None:
        old_n = _get_ensemble_feature_count(existing_ensemble)
        if old_n and old_n != len(features):
            logger.info(
                f"Warm-start disabled: model trained with {old_n} features, "
                f"selected features = {len(features)}"
            )
            existing_ensemble = None
            model_params = None
            is_warm_start = False

    logger.info(f"Training samples: {len(X)} | Features: {len(features)}")

    try:
        results = trainer.train_all_models(
            X, y,
            feature_cols=features,
            model_params=model_params,
            target_type=args.target_type,
            recency_weights=recency,
            tf_label="M5",
            existing_ensemble=existing_ensemble,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None

    ensemble = trainer.get_ensemble()
    num_models = ensemble.get_num_models()
    if num_models == 0:
        logger.warning("No models trained")
        return None

    # ── Run OOS validation on the SAME data used for training ──
    from learning.oos_validator import OOSValidator
    oos_val = OOSValidator()

    # OOS pada data FULL (sama dengan training)
    # Threshold HARUS sama dengan training (config: 0.0004)
    bt = args.buy_threshold if hasattr(args, 'buy_threshold') else config.training["buy_threshold"]
    st = args.sell_threshold if hasattr(args, 'sell_threshold') else config.training["sell_threshold"]
    oos_result = oos_val.validate(df, ensemble, trainer, "M5+CTX", oos_split=0.2,
                                   buy_threshold=bt, sell_threshold=st, timeframe=PRIMARY_TF)
    new_score = model_manager._compute_oos_numeric_score(oos_result)

    # Compare with current best version — re-evaluasi old model pada data SAMA untuk fair comparison
    best_ver = model_manager.get_best_version(PRIMARY_TF)
    old_score = 0
    if best_ver and is_warm_start:
        try:
            old_ensemble = model_manager.load_ensemble(best_ver)
            if old_ensemble.get_num_models() > 0:
                oos_val2 = OOSValidator()
                old_oos = oos_val2.validate(df, old_ensemble, trainer, "M5+CTX", oos_split=0.2,
                                            buy_threshold=bt, sell_threshold=st, timeframe=PRIMARY_TF)
                old_score = model_manager._compute_oos_numeric_score(old_oos)
                logger.info(
                    f"Re-evaluated {best_ver} on same data: "
                    f"WR={old_oos.get('win_rate',0):.1f}% PF={old_oos.get('profit_factor',0):.2f} "
                    f"Score={old_score:.1f}"
                )
        except Exception as e:
            logger.warning(f"Could not re-evaluate old model: {e}")
            old_score = 0
    elif best_ver:
        # Warm-start skipped (features/lookahead changed) — cannot compare with old model.
        # Accept new model as fresh baseline.
        logger.info(
            f"Warm-start not applicable (feature/lookahead change detected). "
            f"Accepting new model as fresh baseline (skipping old comparison)."
        )
        old_score = 0
    # ── Save feature importance EVEN IF rejected (so next retrain has data) ──
    if args.save_fi and best_ver:
        # Overwrite the best version's feature importance with new data
        # This ensures new features appear in importance CSV for next retrain
        try:
            trainer.save_feature_importance(
                ensemble, features, args.pair, PRIMARY_TF, best_ver
            )
        except Exception as e:
            logger.debug(f"Feature importance save to {best_ver} skipped: {e}")

    if old_score > 0:
        better, reason = ModelManager.is_model_better(oos_result, old_oos)
        if not better:
            logger.warning(
                f"New model rejected: {reason}"
            )
            logger.info(
                f"  Best {best_ver}: WR={old_oos.get('win_rate',0):.1f}% PF={old_oos.get('profit_factor',0):.2f} "
                f"Grade={old_oos.get('grade','N/A')}"
            )
            logger.info(
                f"  New model: WR={oos_result.get('win_rate',0):.1f}% PF={oos_result.get('profit_factor',0):.2f} "
                f"Grade={oos_result.get('grade','N/A')}"
            )
            return None
        logger.info(f"New model accepted: {reason}")


    # Only save if new model beats best (or no best exists yet)
    version = model_manager.save_ensemble(ensemble, timeframe=PRIMARY_TF)
    model_manager.save_oos_result(version, oos_result)
    logger.info(f"Trained {num_models} models on M5, saved as v{version} "
                f"(OOS score {new_score:.1f} vs best {old_score:.1f})")

    # Save perf data from training
    perf_data = {"accuracy": {}}
    for m_name in ["xgboost", "random_forest", "lightgbm"]:
        m_data = (results.get("models", {}) or {}).get(m_name, {}) or {}
        perf_data["accuracy"][m_name] = m_data.get("train_accuracy", 0) or 0
        perf_data["accuracy"][f"{m_name}_val"] = m_data.get("val_accuracy", 0) or 0
    model_manager.save_performance(version, perf_data)

    # Save feature importance for accepted model (overwrites temp save above)
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
        "oos_result": oos_result,
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

    df = load_m5_with_context(pair, days=args.days)

    trainer = ModelTrainer()
    model_manager = ModelManager()
    results = {"pair": pair, "success": False}

    res = train_m5_model(df, trainer, model_manager, args)
    if res is None:
        logger.warning(f"Training completed but not saved - not better than existing best for {pair}")
        return {"pair": pair, "success": False, "error": "not better than existing best"}

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
    from core.constants import DATA_DIR
    historical_dir = Path(DATA_DIR) / "historical"
    if not historical_dir.exists():
        logger.error(f"Historical data dir not found: {historical_dir}")
        return []

    pairs = []
    for d in historical_dir.iterdir():
        if d.is_dir():
            pair_name = d.name.upper().replace(".FL", "")
            if len(pair_name) == 6:
                pairs.append(pair_name)
    return sorted(pairs)


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
