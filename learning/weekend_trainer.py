import json
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from pathlib import Path

import numpy as np
import pandas as pd

from core.constants import MODEL_DIR, DATA_DIR, Timeframe
from core.config import Config
from ml.model_manager import ModelManager
from ml.ensemble import VotingEnsemble
from ml.xgboost_model import XGBoostModel
from ml.random_forest_model import RandomForestModel
from ml.lightgbm_model import LightGBMModel
from learning.trade_memory import TradeMemory
from learning.oos_validator import OOSValidator
from learning.concept_drift import ConceptDriftDetector
from learning.skill_scorer import SkillScorer
from utils.logger import get_logger


class WeekendTrainer:
    def __init__(self):
        self.logger = get_logger("weekend_trainer")
        self.config = Config()
        self.model_manager = ModelManager()
        self.trade_memory = TradeMemory()
        self.oos_validator = OOSValidator()
        self.drift_detector = ConceptDriftDetector()
        self.skill_scorer = SkillScorer()
        self._training = False

    def should_train(self) -> bool:
        now = datetime.now()
        is_weekend = now.weekday() >= 5
        is_market_closed = not self._is_fx_market_open(now)
        if is_weekend or is_market_closed:
            last_train_file = Path(MODEL_DIR) / "last_weekend_train.txt"
            if last_train_file.exists():
                try:
                    last_train_str = last_train_file.read_text().strip()
                    last_train = datetime.fromisoformat(last_train_str)
                    if now - last_train < timedelta(hours=6):
                        return False
                except Exception:
                    pass
            return True
        return False

    def _is_fx_market_open(self, dt: datetime) -> bool:
        if dt.weekday() >= 5:
            return False
        return True

    def train_all_timeframes(self, force: bool = False) -> Dict[str, Dict]:
        if self._training:
            return {"status": "already_training"}
        self._training = True
        self.logger.info("Starting weekend training session")

        if not force and not self.should_train():
            self._training = False
            return {"status": "market_open_skip"}

        results = {}
        timeframes = self._get_timeframes_to_train()

        for tf in timeframes:
            try:
                result = self._train_timeframe(tf)
                results[self._tf_label(tf)] = result
                self.logger.info(f"Weekend training for tf{tf}: {result.get('status')}")
            except Exception as e:
                self.logger.error(f"Error training timeframe {tf}: {e}")
                results[self._tf_label(tf)] = {"status": "error", "error": str(e)}

        self._training = False
        last_train_file = Path(MODEL_DIR) / "last_weekend_train.txt"
        try:
            last_train_file.write_text(datetime.now().isoformat())
        except Exception:
            pass

        return results

    def _get_timeframes_to_train(self) -> List[int]:
        config_tfs = self.config.get("timeframes", [5, 15, 30])
        return config_tfs

    def _tf_label(self, timeframe: int) -> str:
        return Timeframe.LABELS.get(timeframe, f"tf{timeframe}")

    def _train_timeframe(self, timeframe: int) -> Dict:
        self.logger.info(f"Training timeframe {timeframe}")

        X_train, y_train, X_val, y_val, X_test = self._prepare_training_data(timeframe)
        if X_train is None or len(X_train) < 100:
            return {"status": "insufficient_data", "samples": 0}

        historical_data = self._load_historical_data(timeframe)
        trade_memory_data = self.trade_memory.get_trades_for_timeframe(timeframe)
        trade_features = self._extract_trade_features(trade_memory_data)

        if trade_features is not None:
            X_train = np.vstack([X_train, trade_features])
            y_train = np.hstack([y_train, np.ones(len(trade_features))])

        ensemble = VotingEnsemble()
        models_to_train = self._get_available_models()
        model_results = {}
        for name, model_class in models_to_train:
            try:
                model = model_class()
                result = model.train(X_train, y_train, X_val, y_val)
                model_results[name] = result
                ensemble.register_model(name, model)
                self.logger.info(f"Trained {name} for tf{timeframe}")
            except Exception as e:
                self.logger.warning(f"Failed to train {name}: {e}")

        if ensemble.get_num_models() == 0:
            return {"status": "no_models_trained"}

        candidate_version = self.model_manager.save_to_candidate(ensemble, timeframe)

        oos_result = self._validate_oos(ensemble, timeframe, X_val, y_val)
        self.model_manager.save_oos_result(candidate_version, oos_result)

        promoted, message = self.model_manager.promote_candidate(timeframe)

        if promoted:
            self.model_manager.increment_retrain_count(timeframe)
            # Save accuracy data for the promoted production version
            try:
                tf_label = self._tf_label(timeframe)
                current_file = Path(MODEL_DIR) / "production" / tf_label / "current.txt"
                if current_file.exists():
                    prod_version = current_file.read_text().strip()
                    perf_data = {"accuracy": {}}
                    for m_name in ["xgboost", "random_forest", "lightgbm"]:
                        m_data = model_results.get(m_name, {}) or {}
                        perf_data["accuracy"][m_name] = m_data.get("train_accuracy", 0) or 0
                        perf_data["accuracy"][f"{m_name}_val"] = m_data.get("val_accuracy", 0) or 0
                    prod_dir = Path(MODEL_DIR) / f"model_{prod_version}"
                    prod_dir.mkdir(parents=True, exist_ok=True)
                    with open(prod_dir / "performance.json", "w") as f:
                        json.dump(perf_data, f, indent=2)
                    # Also copy OOS data from candidate to production
                    cand_oos_path = Path(MODEL_DIR) / "candidate" / tf_label / "performance.json"
                    if cand_oos_path.exists():
                        try:
                            with open(cand_oos_path) as f:
                                cand_data = json.load(f)
                            prod_perf_path = prod_dir / "performance.json"
                            with open(prod_perf_path) as f:
                                prod_data = json.load(f)
                            prod_data["oos"] = cand_data.get("oos", {})
                            prod_data["oos_score"] = cand_data.get("oos_score", 0)
                            with open(prod_perf_path, "w") as f:
                                json.dump(prod_data, f, indent=2)
                        except Exception as e:
                            self.logger.warning(f"Failed to copy OOS to production: {e}")
            except Exception as e:
                self.logger.warning(f"Failed to save performance data: {e}")
            self.logger.info(f"Promoted candidate to production for tf{timeframe}")

        # Post-train verification: check buy/sell distribution on test set
        if X_test is not None and len(X_test[0]) > 0:
            try:
                X_test_data, y_test_data = X_test
                if len(X_test_data) > 0:
                    probs = ensemble.predict_proba(X_test_data)
                    if probs.ndim == 1:
                        preds = (probs > 0.5).astype(int)
                    else:
                        preds = np.argmax(probs, axis=1)
                    buy_pct = preds.mean() * 100
                    actual_buy_pct = y_test_data.mean() * 100
                    self.logger.info(
                        f"Post-train verification tf{timeframe}: "
                        f"predicted BUY={buy_pct:.0f}% SELL={100-buy_pct:.0f}% "
                        f"(actual: BUY={actual_buy_pct:.0f}% SELL={100-actual_buy_pct:.0f}%)"
                    )
                    if buy_pct < 15 or buy_pct > 85:
                        self.logger.warning(
                            f"Model bias detected for tf{timeframe}: "
                            f"predicted BUY only {buy_pct:.0f}% of test set — retrain recommended"
                        )
            except Exception as e:
                self.logger.warning(f"Post-train verification failed: {e}")

        drift_detected = self.drift_detector.should_retrain()

        return {
            "status": "trained",
            "candidate_version": candidate_version,
            "promoted": promoted,
            "promotion_message": message,
            "oos_grade": oos_result.get("grade", "N/A"),
            "oos_win_rate": oos_result.get("win_rate", 0),
            "oos_profit_factor": oos_result.get("profit_factor", 0),
            "drift_detected": drift_detected,
        }

    def _prepare_training_data(self, timeframe: int):
        try:
            data = self._load_historical_data(timeframe)
            if data is None or data.empty:
                self.logger.warning(f"No historical data for tf{timeframe}")
                return None, None, None, None, None
            features = self._compute_features(data)
            target = self._compute_target(data)

            if len(features) < 200:
                self.logger.warning(f"Too few samples for tf{timeframe}: {len(features)}")
                return None, None, None, None, None

            # Balanced sampling: equal buy/sell representation
            buy_idx = np.where(target == 1)[0]
            sell_idx = np.where(target == 0)[0]
            n_min = min(len(buy_idx), len(sell_idx))

            if len(buy_idx) != len(sell_idx):
                if len(buy_idx) > len(sell_idx):
                    step = len(buy_idx) / n_min
                    buy_sub = np.linspace(0, len(buy_idx) - 1, n_min, dtype=int)
                    selected = np.sort(np.concatenate([buy_idx[buy_sub], sell_idx]))
                else:
                    step = len(sell_idx) / n_min
                    sell_sub = np.linspace(0, len(sell_idx) - 1, n_min, dtype=int)
                    selected = np.sort(np.concatenate([buy_idx, sell_idx[sell_sub]]))
                X_bal = features[selected]
                y_bal = target[selected]
            else:
                X_bal = features
                y_bal = target

            split_idx = int(len(X_bal) * 0.7)
            val_split = int(len(X_bal) * 0.85)

            X_train = X_bal[:split_idx]
            y_train = y_bal[:split_idx]
            X_val = X_bal[split_idx:val_split]
            y_val = y_bal[split_idx:val_split]
            X_test = X_bal[val_split:]
            y_test = y_bal[val_split:]

            buy_pct = y_bal.mean() * 100
            self.logger.info(
                f"Balanced data for tf{timeframe}: {len(X_bal)} total, "
                f"{buy_pct:.0f}% BUY / {100-buy_pct:.0f}% SELL"
            )

            return X_train, y_train, X_val, y_val, (X_test, y_test)
        except Exception as e:
            self.logger.error(f"Error preparing training data for tf{timeframe}: {e}")
            return None, None, None, None, None

    def _load_historical_data(self, timeframe: int) -> Optional[pd.DataFrame]:
        tf_label = self._tf_label(timeframe)
        pair = self.config.get("symbol", "EURUSD.fl")
        data_path = (
            Path(DATA_DIR)
            / "historical"
            / pair
            / f"tf_{timeframe}.parquet"
        )
        if not data_path.exists():
            self.logger.warning(f"Historical data not found: {data_path}")
            return None
        try:
            return pd.read_parquet(data_path)
        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")
            return None

    def _compute_features(self, df: pd.DataFrame) -> np.ndarray:
        from features.feature_pipeline import FeaturePipeline
        pipeline = FeaturePipeline()
        try:
            processed = pipeline.compute_all(df)
            feature_cols = pipeline.get_feature_columns()
            available = [c for c in feature_cols if c in processed.columns]
            if not available:
                self.logger.warning("No feature columns available")
                return np.array([])
            result = processed[available].dropna().values
            return result
        except Exception as e:
            self.logger.error(f"Feature computation failed: {e}")
            return np.array([])

    def _compute_target(self, df: pd.DataFrame) -> np.ndarray:
        if "returns" in df.columns:
            return (df["returns"].shift(-1) > 0).astype(int).values[:-1]
        if "close" in df.columns:
            close = df["close"].values
            future_close = close[1:]
            current_close = close[:-1]
            target = (future_close > current_close).astype(int)
            return target
        return np.zeros(len(df))

    def _extract_trade_features(self, trades: List[Dict]) -> Optional[np.ndarray]:
        if not trades:
            return None
        features = []
        for trade in trades:
            if "indicators" in trade:
                indicators = trade["indicators"]
                feature_vector = []
                for key in ["rsi", "macd", "atr", "spread"]:
                    val = indicators.get(key, 0)
                    if val is None:
                        val = 0
                    if isinstance(val, (int, float)):
                        feature_vector.append(val)
                    else:
                        feature_vector.append(0)
                features.append(feature_vector)
            else:
                features.append([0, 0, 0, 0])
        return np.array(features) if features else None

    def _get_available_models(self) -> List:
        models = []
        try:
            models.append(("xgboost", XGBoostModel))
        except Exception:
            pass
        try:
            models.append(("random_forest", RandomForestModel))
        except Exception:
            pass
        try:
            models.append(("lightgbm", LightGBMModel))
        except Exception:
            pass
        return models

    def _validate_oos(self, ensemble: VotingEnsemble, timeframe: int, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        try:
            oos_results = self.oos_validator.validate(ensemble, self._load_historical_data(timeframe), timeframe)
            return oos_results
        except Exception as e:
            self.logger.warning(f"OOS validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "grade": "N/A",
            }

    def get_status(self) -> Dict:
        now = datetime.now()
        last_train_file = Path(MODEL_DIR) / "last_weekend_train.txt"
        last_train = None
        if last_train_file.exists():
            try:
                last_train = last_train_file.read_text().strip()
            except Exception:
                pass

        return {
            "is_training": self._training,
            "is_weekend": now.weekday() >= 5,
            "market_open": self._is_fx_market_open(now),
            "should_train": self.should_train(),
            "last_training": last_train,
            "timeframes": self._get_timeframes_to_train(),
        }
