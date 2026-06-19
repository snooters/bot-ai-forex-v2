import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from core.config import config
from core.constants import LOOKAHEAD_5, MODEL_DIR, RESULTS_DIR
from core.exceptions import ModelTrainingError
from features.feature_pipeline import FeaturePipeline
from ml.ensemble import VotingEnsemble
from learning.trade_memory import TradeMemory
from learning.trade_outcome_trainer import TradeOutcomeTrainer
from learning.mistake_weighting import MistakeWeighting
from utils.logger import get_logger
from utils.decorators import measure_time, safe_execute


# Max features: reduced from 135 to 50 based on v44_M5 feature importance.
# The bottom 85 features have importance < 79 and include many zero-importance
# columns (is_weekend, dist_to_support, etc.). Pruning reduces overfitting risk
# from the 135:5000 feature:sample ratio. Previous attempt at 40 was too aggressive
# (v33_M5 era), but with 581 sim trades + recency weighting, 50 is a reasonable target.
#
# IMPORTANT: _load_feature_importance() now SKIPS biased versions (<100 features)
# to break the chicken-and-egg cycle that previously dropped multi-TF features
# permanently. Only full-feature importance (e.g. v44_M5 with 135 features) is
# used for selection, ensuring multi-TF features get a fair ranking.
MAX_FEATURES_TARGET = 50


class ModelTrainer:
    def __init__(self, trade_memory: Optional[TradeMemory] = None):
        self.logger = get_logger("model_trainer")
        self.feature_pipeline = FeaturePipeline()
        self.ensemble = VotingEnsemble()
        self.trade_memory = trade_memory
        self.mistake_weighting = MistakeWeighting(trade_memory) if trade_memory else None
        self.trade_outcome_trainer = TradeOutcomeTrainer(trade_memory) if trade_memory else None

    def _create_model_instance(self, name: str):
        try:
            if name == "xgboost":
                from ml.xgboost_model import XGBoostModel
                return XGBoostModel()
            elif name == "random_forest":
                from ml.random_forest_model import RandomForestModel
                return RandomForestModel()
            elif name == "lightgbm":
                from ml.lightgbm_model import LightGBMModel
                return LightGBMModel()
            elif name == "lstm":
                from ml.lstm_model import LSTMModel
                return LSTMModel()
        except Exception as e:
            self.logger.warning(f"Cannot create {name} model: {e}")
            return None

    def _load_feature_importance(self, pair: str, timeframe: int) -> Optional[Dict[str, float]]:
        """Load saved feature importance from training run.

        Prefers the version with the MOST NON-ZERO features to break the
        iterative selection bias cycle. Biased versions (e.g. v60_M5 with
        only 49 non-zero features) are skipped in favour of full-feature
        versions (e.g. v44_M5 with 125 non-zero features), ensuring that
        multi-TF features get a fair ranking in the selection process.
        """
        try:
            results_dir = Path(RESULTS_DIR) / pair / str(timeframe)
            if not results_dir.exists():
                return None

            versions = sorted(results_dir.iterdir(), reverse=True)
            best_fi = None
            best_nz = 0  # track non-zero feature count
            best_total = 0

            for v_dir in versions:
                fi_file = v_dir / "feature_importance.csv"
                if fi_file.exists():
                    fi_df = pd.read_csv(fi_file)
                    if "feature" in fi_df.columns and "importance" in fi_df.columns:
                        fi_dict = dict(zip(fi_df["feature"], fi_df["importance"]))
                        nz = sum(1 for v in fi_dict.values() if v > 0)
                        total = len(fi_dict)

                        # Prefer version with most non-zero features
                        # (full-feature runs like v44_M5 have 125/135 non-zero;
                        #  biased runs have <50 non-zero)
                        if nz > best_nz:
                            best_fi = (fi_dict, fi_file, total, nz)
                            best_nz = nz
                            best_total = total

            if best_fi:
                fi_dict, fi_file, total, nz = best_fi
                level = "info" if nz >= 100 else "warning"
                getattr(self.logger, level)(
                    f"Loaded feature importance from {fi_file} "
                    f"({nz}/{total} non-zero features) — "
                    f"{'unbiased' if nz >= 100 else 'biased'} selection"
                )
                return fi_dict

            return None
        except Exception as e:
            self.logger.debug(f"Could not load feature importance: {e}")
            return None

    def _select_top_features(
        self,
        available_cols: List[str],
        importance: Optional[Dict[str, float]],
        max_features: int = MAX_FEATURES_TARGET,
    ) -> List[str]:
        """Select top N features by importance. Falls back to all if no importance data.

        If NEW features are detected (not in importance dict), ALL features are used
        so the new features get properly evaluated and ranked.
        """
        if importance is None:
            self.logger.info(f"No feature importance available — using all {len(available_cols)} features")
            return available_cols

        # Check for new features not in importance dict
        new_features = [c for c in available_cols if c not in importance]
        if new_features:
            self.logger.info(
                f"Found {len(new_features)} new features not in importance dict — "
                f"using ALL {len(available_cols)} features for this training run "
                f"(new: {new_features})"
            )
            return available_cols

        scored = [(col, importance.get(col, 0.0)) for col in available_cols]
        scored.sort(key=lambda x: x[1], reverse=True)

        if len(scored) <= max_features:
            self.logger.info(f"Only {len(scored)} features available — using all")
            return available_cols

        selected = [col for col, _ in scored[:max_features]]
        dropped = len(available_cols) - len(selected)
        dropped_names = [col for col, _ in scored[max_features:]]
        self.logger.info(
            f"Feature selection: {len(selected)}/{len(available_cols)} features kept "
            f"(dropped {dropped} low-importance features)"
        )
        if dropped_names:
            self.logger.info(
                f"Dropped features ({len(dropped_names)}): "
                f"{', '.join(dropped_names[:20])}"
                f"{'...' if len(dropped_names) > 20 else ''}"
            )
        return selected

    @measure_time
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        lookahead: int = LOOKAHEAD_5,
        buy_threshold: float = 0.0004,
        sell_threshold: float = 0.0004,
        target_type: str = "class",
        max_features: int = MAX_FEATURES_TARGET,
        pair: str = "EURUSD",
        timeframe: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
        if df.empty or len(df) < 250:
            raise ModelTrainingError(f"Insufficient data: {len(df)} rows")

        df = self.feature_pipeline.compute_all(df)
        feature_cols = self.feature_pipeline.get_feature_columns()
        available_cols = [c for c in feature_cols if c in df.columns]

        # Load importance from previous training and select top features
        importance = self._load_feature_importance(pair, timeframe or 5)
        selected_cols = self._select_top_features(available_cols, importance, max_features)
        self.ensemble.feature_cols = selected_cols

        # Drop NaN on the SELECTED columns (models will only use these)
        df = df.dropna(subset=selected_cols).copy()
        if df.empty or len(df) < 200:
            raise ModelTrainingError("Insufficient data after cleaning")

        future_close = df["close"].shift(-lookahead)
        current_close = df["close"]
        future_return = (future_close - current_close) / current_close

        if target_type == "regression":
            y = future_return.values
        else:
            y = np.zeros(len(df), dtype=int)
            y[future_return > buy_threshold] = 0
            y[future_return < -sell_threshold] = 1
            y[(future_return >= -sell_threshold) & (future_return <= buy_threshold)] = 2

        # Train on selected features only — consistent with feature_cols
        X = df[selected_cols].values
        mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        df_clean = df[mask].copy() if hasattr(df, 'iloc') else df

        if len(X) < 100:
            raise ModelTrainingError(f"Too few training samples: {len(X)}")

        if target_type != "regression":
            self.logger.info(f"Prepared {len(X)} training samples with {len(selected_cols)} features "
                             f"(selected from {len(available_cols)} avail). "
                             f"BUY: {(y==0).sum()}, SELL: {(y==1).sum()}, HOLD: {(y==2).sum()}")
        else:
            self.logger.info(f"Prepared {len(X)} training samples with {len(selected_cols)} features "
                             f"(selected from {len(available_cols)} avail, regression). "
                             f"return range=[{y.min():.4f}, {y.max():.4f}]")
        return X, y, selected_cols, df_clean

    def incorporate_trade_outcomes(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_cols: List[str],
        pair: str,
        timeframe: Optional[int] = None,
        min_trades: int = 10,
        upsample_wins: bool = True,
        win_weight: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merge trade-outcome training samples into OHLC-derived data.

        Closes the loop: real trade results become additional labeled samples.
        Returns (X, y, sample_weights) with trade data integrated.
        """
        if self.trade_outcome_trainer is None:
            self.logger.info("No trade_outcome_trainer available, skipping")
            return X, y, np.ones(len(y))

        trades = self.trade_outcome_trainer.get_recent_trades(
            pair=pair, timeframe=timeframe, min_trades=min_trades,
        )
        if not trades:
            self.logger.info("No recent closed trades for %s, skipping outcome incorporation", pair)
            return X, y, np.ones(len(y))

        stats = self.trade_outcome_trainer.get_trade_quality_stats(trades)
        self.logger.info("Trade outcome stats for %s: %s", pair, stats)

        X_trade, y_trade = self.trade_outcome_trainer.convert_to_samples(
            trades, feature_cols,
        )
        if len(X_trade) == 0:
            self.logger.info("No valid trade samples could be created")
            return X, y, np.ones(len(y))

        X_merged, y_merged, w_merged = self.trade_outcome_trainer.merge_with_ohlc(
            X, y, X_trade, y_trade,
            upsample_wins=upsample_wins, win_weight=win_weight,
        )
        return X_merged, y_merged, w_merged

    def _compute_sample_weights(self, y: np.ndarray, multiplier: float = 1.0) -> np.ndarray:
        classes, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(classes)
        weights = np.zeros(n_samples, dtype=float)
        for cls, count in zip(classes, counts):
            cls_mask = y == cls
            base_weight = n_samples / (n_classes * count)
            if cls in (0, 1):
                base_weight *= multiplier
            weights[cls_mask] = base_weight
        self.logger.info(f"Sample weights computed (multiplier={multiplier}): "
                         f"classes={dict(zip(classes, counts))}, "
                         f"weight_range=[{weights.min():.2f}, {weights.max():.2f}]")
        return weights



    @staticmethod
    def compute_recency_weights(time_col: pd.Series, now: Optional[datetime] = None) -> np.ndarray:
        """Weight by recency: newer data gets higher weight.
        0-30d=1.0, 31-90d=0.8, 91-180d=0.6, 181-365d=0.4, >365d=0.2.
        """
        if now is None:
            now = datetime.now()
        age_days = (now - time_col).dt.total_seconds() / 86400.0
        weights = np.ones(len(age_days), dtype=float)
        weights[age_days > 365] = 0.2
        for lo, hi, val in [(180, 365, 0.4), (90, 180, 0.6), (30, 90, 0.8)]:
            weights[(age_days > lo) & (age_days <= hi)] = val
        return weights

    @measure_time
    def train_all_models(self, X: np.ndarray, y: np.ndarray,
                         sample_weight_multiplier: float = 1.0,
                         model_params: Optional[Dict] = None,
                         feature_cols: Optional[List[str]] = None,
                         progress=None,
                         tf_label=None,
                         target_type: str = "class",
                         recency_weights: Optional[np.ndarray] = None,
                         trade_outcome_weights: Optional[np.ndarray] = None,
                         existing_ensemble: Optional[VotingEnsemble] = None,
                         is_warm_start: bool = False) -> Dict:
        """Train all enabled models with optional warm-start from existing ensemble.
        
        Args:
            existing_ensemble: If provided, models continue training from these existing models.
            is_warm_start: If True, uses reduced n_estimators/epochs for fine-tuning
                           (warm-start mode) rather than full training from scratch.
        """
        y_effective = y.copy()
        if self.mistake_weighting and feature_cols:
            y_effective = self.mistake_weighting.adjust_labels(X, y_effective, feature_cols)
            flips = int((y_effective != y).sum())
            if flips > 0:
                self.logger.info(f"Labels adjusted: {flips} samples flipped based on trade memory")

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_effective[:split_idx], y_effective[split_idx:]

        sample_weight = self._compute_sample_weights(y_train, multiplier=sample_weight_multiplier)

        # ── Defensive alignment: recency_weights & trade_outcome_weights ──
        # These can be shorter than X after merge_with_ohlc() appends sim trade
        # samples. Pad with neutral weight 1.0 to prevent shape mismatch.
        if recency_weights is not None and len(recency_weights) != len(X):
            self.logger.warning(
                f"Recency weights length {len(recency_weights)} != X length {len(X)} — "
                f"padding/truncating to match (after sim trade merge)"
            )
            if len(recency_weights) > len(X):
                recency_weights = recency_weights[:len(X)]
            else:
                recency_weights = np.pad(
                    recency_weights,
                    (0, len(X) - len(recency_weights)),
                    mode='constant',
                    constant_values=1.0,
                )
        if recency_weights is not None:
            rw_train = recency_weights[:split_idx]
            sample_weight = sample_weight * rw_train
            self.logger.info(f"Recency weights applied: "
                             f"range=[{rw_train.min():.2f}, {rw_train.max():.2f}]")

        if trade_outcome_weights is not None and len(trade_outcome_weights) != len(X):
            self.logger.warning(
                f"Trade outcome weights length {len(trade_outcome_weights)} != X length {len(X)} — "
                f"padding/truncating to match"
            )
            if len(trade_outcome_weights) > len(X):
                trade_outcome_weights = trade_outcome_weights[:len(X)]
            else:
                trade_outcome_weights = np.pad(
                    trade_outcome_weights,
                    (0, len(X) - len(trade_outcome_weights)),
                    mode='constant',
                    constant_values=1.0,
                )
        if trade_outcome_weights is not None:
            tow_train = trade_outcome_weights[:split_idx]
            sample_weight = sample_weight * tow_train
            self.logger.info(f"Trade outcome weights applied: "
                             f"range=[{tow_train.min():.2f}, {tow_train.max():.2f}]")

        if self.mistake_weighting and feature_cols:
            sample_weight = self.mistake_weighting.compute_weights(
                X_train, y_train, sample_weight, feature_cols
            )
            sample_weight = self.mistake_weighting.compute_no_trade_weights(
                X_train, sample_weight, feature_cols
            )

        model_params = model_params or {}

        results = {"models": {}, "ensemble": {}}
        ml_config = config.ml

        model_names = []
        if ml_config["enable_xgboost"]:
            model_names.append("xgboost")
        if ml_config["enable_random_forest"]:
            model_names.append("random_forest")
        if ml_config["enable_lightgbm"]:
            model_names.append("lightgbm")
        # LSTM hanya untuk timeframe H1 (60) ke atas — terlalu heavy untuk M5/M15
        if ml_config.get("enable_lstm", True):
            tf_label_lower = (tf_label or "").lower()
            # Cegah LSTM di timeframe rendah: cek dari parameter atau label
            is_low_tf = False
            if tf_label_lower:
                if any(x in tf_label_lower for x in ["m5", "m15", "m30"]):
                    is_low_tf = True
            if not is_low_tf:
                model_names.append("lstm")
            else:
                self.logger.info(f"Skipping LSTM for low timeframe {tf_label} (too heavy, prone to overfit)")

        for name in model_names:
            try:
                # ── Warm-start: use existing trained model if available ──
                init_model = None
                if existing_ensemble is not None and name in existing_ensemble.models:
                    existing_model = existing_ensemble.models[name]
                    if existing_model.is_trained:
                        init_model = existing_model.model
                        self.logger.info(
                            f"{name}: warm-start from existing model "
                            f"(trained={existing_model.is_trained})"
                        )

                self.logger.info(f"Training {name}...")
                model = self._create_model_instance(name)
                if model is None:
                    self.logger.warning(f"Skipping {name}: not available")
                    results["models"][name] = {"error": "not available"}
                    continue

                if init_model is not None:
                    # Warm-start: load existing model state into model instance
                    # The model instance already has load() — we borrow the loaded model
                    model.model = init_model
                    model._trained = True
                    self.logger.info(f"{name}: loaded existing model for continued training")
                else:
                    params = model_params.get(name, {}).copy()
                    if name == "lstm":
                        params["n_features"] = X.shape[1]
                    model.create_model(**params)

                if hasattr(model, "_available") and not model._available:
                    self.logger.warning(f"Skipping {name}: not available")
                    results["models"][name] = {"error": "not available"}
                    continue

                if progress and tf_label:
                    n_est = 200
                    if model.model and hasattr(model.model, "get_params"):
                        try:
                            n_est = model.model.get_params().get("n_estimators", 200)
                        except Exception:
                            pass
                    progress.begin_model(name, total=n_est)

                try:
                    result = model.train(X_train, y_train, X_val, y_val, sample_weight=sample_weight,
                                         progress_callback=progress.make_model_callback(name) if (progress and tf_label) else None,
                                         init_model=model.model if init_model is not None else None)
                except Exception as e:
                    if init_model is not None:
                        # Warm-start failed — retry from scratch
                        self.logger.warning(
                            f"{name}: warm-start failed ({e}), retrying from scratch..."
                        )
                        model = self._create_model_instance(name)
                        if model is not None:
                            params = model_params.get(name, {}).copy()
                            if name == "lstm":
                                params["n_features"] = X.shape[1]
                            model.create_model(**params)
                            result = model.train(
                                X_train, y_train, X_val, y_val,
                                sample_weight=sample_weight,
                                progress_callback=progress.make_model_callback(name) if (progress and tf_label) else None,
                                init_model=None,
                            )
                        else:
                            raise
                    else:
                        raise

                results["models"][name] = result
                self.ensemble.register_model(name, model)

                if progress and tf_label:
                    progress.end_model(name)

                if self.mistake_weighting and hasattr(model.model, "feature_importances_"):
                    try:
                        importances = model.model.feature_importances_
                        if feature_cols and len(importances) == len(feature_cols):
                            fi_dict = {feature_cols[i]: float(imp) for i, imp in enumerate(importances)}
                            self.mistake_weighting.set_feature_importance(fi_dict)
                    except Exception:
                        pass

                self.logger.info(f"{name} trained: {result}")
            except Exception as e:
                self.logger.error(f"Failed to train {name}: {e}")
                results["models"][name] = {"error": str(e)}

        if self.ensemble.get_num_models() > 0:
            results["ensemble"]["num_models"] = self.ensemble.get_num_models()
            results["ensemble"]["active_models"] = self.ensemble.get_active_models()
            self.logger.info(f"Ensemble ready with {self.ensemble.get_num_models()} models")

            # ── Set ensemble weights based on val_accuracy ──
            val_accs = {}
            for name, model_result in results.get("models", {}).items():
                if isinstance(model_result, dict):
                    va = model_result.get("val_accuracy", 0)
                    if va > 0:
                        val_accs[name] = va
            if val_accs:
                self.ensemble.set_weights_from_val_accuracy(val_accs)
                results["ensemble"]["weights"] = dict(self.ensemble.weights)
            else:
                self.logger.info("No val_accuracy from training — keeping uniform ensemble weights")

            try:
                val_probas = self.ensemble.predict_proba(X_val)
                from ml.probability_calibrator import ProbabilityCalibrator
                calibrator = ProbabilityCalibrator(method="platt")
                calibrator.train(y_val, val_probas)
                results["calibration"] = {
                    "brier_score": calibrator.brier_score,
                    "ece": calibrator.ece,
                }
                self.ensemble.calibrator = calibrator
                self.logger.info(f"Calibration: Brier={calibrator.brier_score:.4f} ECE={calibrator.ece:.4f}")
            except Exception as e:
                self.logger.warning(f"Calibration training failed: {e}")
                results["calibration"] = {"error": str(e)}

        # Safety net: pastikan feature_cols tersimpan di ensemble
        if feature_cols is not None:
            self.ensemble.feature_cols = feature_cols

        return results

    def save_feature_importance(self, ensemble: VotingEnsemble, feature_cols: List[str],
                                 pair: str, timeframe: int, version: str) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not available, skipping feature importance plot")
            return

        importance_dir = Path(RESULTS_DIR) / pair / str(timeframe) / version
        importance_dir.mkdir(parents=True, exist_ok=True)

        # Collect importances from trained models (only available for selected features)
        all_fi = {}
        for name, model in ensemble.models.items():
            if not model.is_trained:
                continue
            if hasattr(model.model, "feature_importances_"):
                fi = model.model.feature_importances_
                if len(fi) == len(feature_cols):
                    all_fi[name] = {feature_cols[i]: float(f) for i, f in enumerate(fi)}

        if not all_fi:
            self.logger.warning("No feature importances available from any model")
            return

        # Get ALL possible features from the pipeline for metadata
        all_possible = self.feature_pipeline.get_feature_columns()

        avg_fi = {}
        for col in feature_cols:
            # Only save features that were actually trained
            vals = [all_fi[m][col] for m in all_fi if col in all_fi[m]]
            if vals:
                avg_fi[col] = sum(vals) / len(vals)

        # Fill missing features with minimum importance so they're not dropped next time
        if avg_fi:
            min_imp = min(avg_fi.values()) * 0.5
        else:
            min_imp = 0.001

        for col in all_possible:
            if col not in avg_fi:
                avg_fi[col] = min_imp

        trained_count = len([c for c in feature_cols if c in avg_fi])
        seeded_count = len(all_possible) - trained_count
        self.logger.info(
            f"Saving feature importance: {len(avg_fi)} total "
            f"({trained_count} trained + {seeded_count} seeded at {min_imp:.6f})"
        )

        fi_df = pd.DataFrame(avg_fi.items(), columns=["feature", "importance"])
        fi_df = fi_df.sort_values("importance", ascending=False)
        fi_path = importance_dir / "feature_importance.csv"
        fi_df.to_csv(fi_path, index=False)
        self.logger.info(f"Feature importance saved to {fi_path}")

        try:
            top_n = min(30, len(fi_df))
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(range(top_n), fi_df.iloc[:top_n]["importance"][::-1])
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(fi_df.iloc[:top_n]["feature"][::-1])
            ax.set_xlabel("Importance")
            ax.set_title(f"Feature Importance — {pair} ({timeframe}m) v{version}")
            plt.tight_layout()
            png_path = importance_dir / "feature_importance.png"
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            self.logger.info(f"Feature importance plot saved to {png_path}")
        except Exception as e:
            self.logger.warning(f"Failed to create importance plot: {e}")

        json_path = importance_dir / "feature_importance.json"
        with open(json_path, "w") as f:
            json.dump(all_fi, f, indent=2)

    def get_ensemble(self) -> VotingEnsemble:
        return self.ensemble
