from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import TrainingConfig
from .data_loader import DataLoader, TF_LABELS, TF_SORT
from .features import FeatureEngineer
from .labeling import LabelEngine
from .model import XGBoostModel
from .evaluator import Evaluator
from .utils import setup_logger, safe_json_dump


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = setup_logger("trainer", config.log_dir, config.log_level)
        self.data_loader = DataLoader(config)
        self.feature_engineer = FeatureEngineer(config)
        self.label_engine = LabelEngine(config)
        self.model = XGBoostModel(config)
        self.evaluator = Evaluator(config)

    def train(self) -> Dict[str, Any]:
        self.logger.info("=" * 60)
        self.logger.info("STARTING TRAINING PIPELINE")
        self.logger.info("=" * 60)

        if self.config.multi_timeframe:
            return self._train_multi_timeframe()

        df = self.data_loader.load()
        self.logger.info(f"Loaded {len(df)} rows")

        df = self.feature_engineer.compute_all(df)
        if df.empty:
            raise ValueError("No data after feature engineering")

        df = self.label_engine.create_labels(df)
        if df.empty:
            raise ValueError("No data after labeling")

        features = self.feature_engineer.get_feature_columns(df)
        if not features:
            raise ValueError("No feature columns found")

        self.logger.info(f"Using {len(features)} features: {features}")
        self.model.build()
        self.model.feature_names = features

        X = df[features].values.astype(np.float32)
        y = df["label_encoded"].values

        if self.config.rolling:
            return self._train_rolling(df, features)
        return self._train_standard(df, features)

    def _select_base_timeframe(self, tf_data, tfs):
        window = timedelta(days=self.config.window_days)
        for tf in tfs:
            df = tf_data[tf]
            span = df["timestamp"].max() - df["timestamp"].min()
            self.logger.info(f"  {tf}: {len(df)} rows, span={span.days}d")
            if span >= window:
                self.logger.info(f"  -> Selected {tf} as base timeframe ({span.days}d >= {window.days}d)")
                return tf
        best = max(tfs, key=lambda k: tf_data[k]["timestamp"].max() - tf_data[k]["timestamp"].min())
        self.logger.warning(f"No timeframe covers {window.days}d window. Using {best} as base (longest span).")
        return best

    def _train_multi_timeframe(self) -> Dict[str, Any]:
        self.logger.info("Multi-timeframe training mode")
        tf_data = self.data_loader.load_multi_timeframe()
        tfs = sorted(tf_data.keys(), key=lambda k: TF_SORT.get(k, 999))

        fast_tf = self._select_base_timeframe(tf_data, tfs)
        self.logger.info(f"Base timeframe: {fast_tf}, Higher timeframes: {[t for t in tfs if t != fast_tf]}")

        featured = {}
        for tf, df in tf_data.items():
            self.logger.info(f"Computing features for {tf} ({len(df)} rows)...")
            fdf = self.feature_engineer.compute_all(df)
            if fdf.empty:
                raise ValueError(f"No data after feature engineering for {tf}")
            featured[tf] = fdf

        base_span = (featured[fast_tf]["timestamp"].max() - featured[fast_tf]["timestamp"].min()).days
        for tf in list(featured.keys()):
            if tf == fast_tf:
                continue
            span = (featured[tf]["timestamp"].max() - featured[tf]["timestamp"].min()).days
            min_span = max(30, int(base_span * 0.1))
            if span < min_span:
                self.logger.warning(f"Skipping {tf} for alignment: span={span}d < min={min_span}d (10% of base={base_span}d)")
                del featured[tf]

        aligned = self.data_loader.align_timeframes(featured, fast_tf)
        if aligned.empty:
            raise ValueError("No data after aligning timeframes")

        close_col = f"close_{fast_tf}"
        if "close" not in aligned.columns and close_col in aligned.columns:
            aligned["close"] = aligned[close_col]

        aligned = self.label_engine.create_labels(aligned)
        if aligned.empty:
            raise ValueError("No data after labeling")

        features = self.feature_engineer.get_feature_columns(aligned)
        if not features:
            raise ValueError("No feature columns found")

        self.logger.info(f"Using {len(features)} features from {len(tfs)} timeframes")
        self.model.build()
        self.model.feature_names = features

        if self.config.rolling:
            return self._train_rolling(aligned, features)
        return self._train_standard(aligned, features)

    def _train_standard(self, df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        train_df, val_df, test_df = self.data_loader.train_val_test_split(df)
        self.logger.info(
            f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

        X_train = train_df[features].values.astype(np.float32)
        y_train = train_df["label_encoded"].values
        X_val = val_df[features].values.astype(np.float32)
        y_val = val_df["label_encoded"].values
        X_test = test_df[features].values.astype(np.float32)
        y_test = test_df["label_encoded"].values

        train_result = self.model.train(
            pd.DataFrame(X_train, columns=features),
            pd.Series(y_train),
            pd.DataFrame(X_val, columns=features),
            pd.Series(y_val),
        )

        y_pred = self.model.predict(pd.DataFrame(X_test, columns=features))
        y_proba = self.model.predict_proba(pd.DataFrame(X_test, columns=features))

        report = self.evaluator.full_report(
            pd.Series(y_test), y_pred, y_proba, test_df
        )
        report["train_result"] = train_result
        report["num_features"] = len(features)
        report["config"] = {
            "window_days": self.config.window_days,
            "prediction_horizon": self.config.prediction_horizon,
            "threshold": self.config.threshold,
        }

        self._save_results(report)
        self.model.save(self.config.output_dir)

        return report

    def _train_rolling(self, df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        self.logger.info("Rolling window training enabled")
        df = df.sort_values("timestamp").reset_index(drop=True)
        timestamps = df["timestamp"]
        start = timestamps.min()
        end = timestamps.max()

        window = timedelta(days=self.config.window_days)
        step = timedelta(days=self.config.step_days)
        rolls = self.config.num_rolls

        all_reports = []
        current_start = start
        best_report = None
        best_score = -float("inf")

        for roll_idx in range(rolls):
            train_end = current_start + window
            if train_end > end:
                self.logger.info(f"Roll {roll_idx + 1}: not enough data, stopping")
                break

            test_end = train_end + step
            if test_end > end:
                test_end = end

            train_mask = (timestamps >= current_start) & (timestamps < train_end)
            test_mask = (timestamps >= train_end) & (timestamps < test_end)

            if train_mask.sum() < 100 or test_mask.sum() < 20:
                self.logger.warning(f"Roll {roll_idx + 1}: insufficient samples, skipping")
                current_start += step
                continue

            train_df = df[train_mask].reset_index(drop=True)
            test_df = df[test_mask].reset_index(drop=True)

            X_train = train_df[features].values.astype(np.float32)
            y_train = train_df["label_encoded"].values
            X_test = test_df[features].values.astype(np.float32)
            y_test = test_df["label_encoded"].values

            val_split = int(len(X_train) * 0.85)
            X_val, y_val = X_train[val_split:], y_train[val_split:]
            X_train_cv, y_train_cv = X_train[:val_split], y_train[:val_split]

            self.logger.info(
                f"Roll {roll_idx + 1}/{rolls}: "
                f"train={len(X_train_cv)}, val={len(X_val)}, test={len(X_test)}"
            )

            roll_model = XGBoostModel(self.config)
            roll_model.build()
            roll_model.feature_names = features

            roll_model.train(
                pd.DataFrame(X_train_cv, columns=features),
                pd.Series(y_train_cv),
                pd.DataFrame(X_val, columns=features),
                pd.Series(y_val),
            )

            y_pred = roll_model.predict(pd.DataFrame(X_test, columns=features))
            y_proba = roll_model.predict_proba(pd.DataFrame(X_test, columns=features))

            report = self.evaluator.full_report(
                pd.Series(y_test), y_pred, y_proba, test_df
            )
            report["roll_index"] = roll_idx + 1
            report["train_start"] = str(current_start)
            report["train_end"] = str(train_end)
            report["test_end"] = str(test_end)
            all_reports.append(report)

            score = report.get("accuracy", 0) + report.get("backtest", {}).get("win_rate", 0) * 0.5
            if score > best_score:
                best_score = score
                best_report = report
                self.model = roll_model
                self.logger.info(f"Roll {roll_idx + 1}: new best model (score={score:.4f})")

            current_start += step

        if not all_reports:
            data_span = (end - start).days
            raise ValueError(
                f"No valid rolling windows found. Total data: {len(df)} rows, "
                f"span={data_span}d from {start.date()} to {end.date()}. "
                f"window_days={self.config.window_days}d exceeds available data span. "
                f"Use --window-days (e.g. {max(30, data_span // 2)}) or point to a single "
                f"timeframe file with --data-path."
            )

        summary = {
            "num_rolls_completed": len(all_reports),
            "best_roll_index": best_report.get("roll_index", 0) if best_report else 0,
        }

        final_report = (best_report or all_reports[-1]).copy()
        final_report["rolling_summary"] = summary

        self._save_results(final_report)
        self.model.save(self.config.output_dir)

        return final_report

    def _save_results(self, report: Dict) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_json_dump(report, str(output_dir / "evaluation_report.json"))

        fi = self.model.get_feature_importance()
        if fi:
            safe_json_dump(fi, str(output_dir / "feature_importance.json"))

        self.logger.info(f"Results saved to {output_dir}")
