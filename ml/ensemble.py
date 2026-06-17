import numpy as np
from typing import Dict, List, Optional, Tuple

from core.config import config
from core.constants import ML_WEIGHT, INTELLIGENCE_WEIGHT
from core.exceptions import ModelError
from utils.logger import get_logger


class VotingEnsemble:
    def __init__(self):
        self.logger = get_logger("voting_ensemble")
        self.models: Dict[str, object] = {}
        self.weights: Dict[str, float] = {}
        self._trained = False
        self.feature_cols: Optional[List[str]] = None
        self.version: Optional[str] = None

    def register_model(self, name: str, model: object, weight: float = 1.0):
        self.models[name] = model
        self.weights[name] = weight
        self.logger.info(f"Registered model: {name} with weight {weight}")

    def set_model_weight(self, name: str, weight: float):
        if name in self.weights:
            self.weights[name] = weight
            self.logger.info(f"Updated weight for {name}: {weight}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise ModelError("No models registered in ensemble")

        weighted_probas = []
        total_weight = 0

        for name, model in self.models.items():
            if not model.is_trained:
                self.logger.warning(f"Model {name} not trained, skipping")
                continue

            try:
                proba = model.predict_proba(X)
                weight = self.weights.get(name, 1.0)
                weighted_probas.append((proba, weight, name))
                total_weight += weight
            except Exception as e:
                self.logger.warning(f"Model {name} prediction failed: {e}")
                continue

        if not weighted_probas:
            raise ModelError("No models could make predictions")

        n_samples = X.shape[0] if X.ndim > 1 else 1
        n_classes = 3

        aligned = []
        for proba, weight, name in weighted_probas:
            if proba.shape[0] == 1 and n_samples > 1:
                proba = np.repeat(proba, n_samples, axis=0)
            elif proba.shape[0] != n_samples:
                self.logger.warning(
                    f"Model {name} returned shape {proba.shape}, expected ({n_samples}, {n_classes}). Skipping."
                )
                total_weight -= weight
                continue
            aligned.append(proba * weight)

        if not aligned:
            raise ModelError("No models produced valid predictions")

        ensemble_proba = np.sum(aligned, axis=0) / total_weight
        return ensemble_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def get_buy_sell_hold(self, X: np.ndarray) -> Dict[str, float]:
        proba = self.predict_proba(X)[0]
        return {
            "BUY": float(proba[0]),
            "SELL": float(proba[1]),
            "HOLD": float(proba[2]),
        }

    def get_ml_signal(self, X: np.ndarray) -> Dict:
        proba = self.predict_proba(X)[0]
        prediction = int(np.argmax(proba))
        confidence = float(np.max(proba))

        label_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
        return {
            "signal": label_map.get(prediction, "HOLD"),
            "confidence": confidence,
            "buy_prob": float(proba[0]),
            "sell_prob": float(proba[1]),
            "hold_prob": float(proba[2]),
        }

    @property
    def is_trained(self) -> bool:
        if not self.models:
            return False
        return all(m.is_trained for m in self.models.values())

    def set_weights_from_val_accuracy(self, val_accuracies: Dict[str, float]):
        """Set model weights proportional to validation accuracy.

        Models with val_accuracy <= random baseline (0.33 for 3-class)
        get minimal weight so they don't pollute the ensemble.
        Linear normalization with a floor prevents any model from
        being completely silenced while still rewarding better models.

        Args:
            val_accuracies: Dict mapping model name -> validation accuracy (0-1).
                            Example: {'xgboost': 0.72, 'random_forest': 0.41, 'lightgbm': 0.56}
        """
        if not val_accuracies:
            self.logger.info("No val_accuracies provided — keeping default weights")
            return

        n_models = len(self.models)
        if n_models == 0:
            return

        # 1. Extract raw accuracies (ensure 0-1 range)
        raw_accs = {}
        for name in self.models:
            raw = val_accuracies.get(name, 0.0)
            if raw > 1.0:
                raw = raw / 100.0
            raw_accs[name] = raw

        # 2. Subtract random baseline (1/3 for 3-class), floor at 0
        adjusted = {}
        for name, raw in raw_accs.items():
            adj = max(0.0, raw - 0.33)
            adjusted[name] = adj

        values = np.array([adjusted[n] for n in self.models])
        if values.sum() == 0:
            # All models at or below random: fall back to uniform
            for name in self.models:
                self.weights[name] = 1.0 / n_models
            self.logger.info("All models at/below random baseline — using uniform weights")
            return

        # 3. If no model is convincingly above baseline, use uniform
        if values.max() < 0.10:
            for name in self.models:
                self.weights[name] = 1.0 / n_models
            self.logger.info("All models within 10% of random baseline — using uniform weights")
            return

        # 4. Linear normalization: proportional to adjusted accuracy above baseline
        # Use square root to compress the range slightly (prevent domination)
        sqrt_vals = np.sqrt(values)
        raw_weights = sqrt_vals / sqrt_vals.sum()

        # 5. Apply minimum weight floor — no model gets less than 10% weight
        min_weight = 0.10
        if n_models > 1 and raw_weights.min() < min_weight:
            clipped = np.maximum(raw_weights, min_weight)
            raw_weights = clipped / clipped.sum()

        for i, name in enumerate(self.models):
            self.weights[name] = float(raw_weights[i])
            self.logger.info(
                f"Ensemble weight for {name}: {self.weights[name]:.3f} "
                f"(val_acc={raw_accs[name]:.1%})"
            )

    def get_model_val_accuracies(self, performance_path: Optional[str] = None) -> Dict[str, float]:
        """Extract per-model val_accuracy from a performance.json file.
        
        The performance.json stores accuracy per model:
            accuracy.xgboost_val, accuracy.random_forest_val, accuracy.lightgbm_val
        
        Returns a dict like {'xgboost': 0.72, 'random_forest': 0.41, ...}
        """
        if performance_path:
            try:
                with open(performance_path) as f:
                    import json
                    perf = json.load(f)
                acc = perf.get("accuracy", {})
                result = {}
                for name in self.models:
                    val_key = f"{name}_val"
                    if val_key in acc:
                        result[name] = float(acc[val_key])
                    # Fallback: try direct key (e.g. lstm has no lstm_val)
                    elif name in acc:
                        result[name] = float(acc[name])
                if result:
                    return result
            except Exception as e:
                self.logger.debug(f"Cannot read performance data: {e}")
        
        # Fallback: try reading from standard model version directory
        return {}

    def get_num_models(self) -> int:
        return len(self.models)

    def get_active_models(self) -> List[str]:
        return [n for n, m in self.models.items() if m.is_trained]
