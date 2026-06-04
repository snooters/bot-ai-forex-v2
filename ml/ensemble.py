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
                weighted_probas.append(proba * weight)
                total_weight += weight
            except Exception as e:
                self.logger.warning(f"Model {name} prediction failed: {e}")
                continue

        if not weighted_probas:
            raise ModelError("No models could make predictions")

        ensemble_proba = np.sum(weighted_probas, axis=0) / total_weight
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

    def get_num_models(self) -> int:
        return len(self.models)

    def get_active_models(self) -> List[str]:
        return [n for n, m in self.models.items() if m.is_trained]
