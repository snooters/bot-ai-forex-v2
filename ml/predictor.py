import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Union

from core.config import config
from core.constants import LOOKAHEAD_5, LOOKAHEAD_10, LOOKAHEAD_20, Timeframe
from core.exceptions import ModelPredictionError
from features.feature_pipeline import FeaturePipeline
from ml.ensemble import VotingEnsemble
from utils.logger import get_logger
from utils.decorators import safe_execute


class MLPredictor:
    def __init__(self, ensembles: Union[VotingEnsemble, Dict[int, VotingEnsemble]]):
        self.logger = get_logger("ml_predictor")
        self.feature_pipeline = FeaturePipeline()
        self._feature_cols: Optional[List[str]] = None

        if isinstance(ensembles, VotingEnsemble):
            self._ensembles: Dict[int, VotingEnsemble] = {}
            if ensembles.is_trained:
                default_tf = Timeframe.M15
                self._ensembles[default_tf] = ensembles
                self.logger.info(f"MLPredictor initialized with single ensemble (assigned to {Timeframe.LABELS.get(default_tf, 'M15')})")
            else:
                self._ensembles = {}
        else:
            self._ensembles = ensembles
            tfs = [Timeframe.LABELS.get(tf, str(tf)) for tf in ensembles]
            self.logger.info(f"MLPredictor initialized with {len(ensembles)} ensembles: {tfs}")

    @property
    def available_timeframes(self) -> List[int]:
        return list(self._ensembles.keys())

    @property
    def is_trained(self) -> bool:
        return len(self._ensembles) > 0

    def _get_ensemble(self, timeframe: int) -> VotingEnsemble:
        if timeframe in self._ensembles:
            return self._ensembles[timeframe]
        if self._ensembles:
            fallback = list(self._ensembles.values())[0]
            self.logger.warning(f"No model for {Timeframe.LABELS.get(timeframe, timeframe)}, using fallback")
            return fallback
        raise ModelPredictionError("No trained ensembles available")

    def _align_features(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        if feature_cols is None:
            feature_cols = [c for c in self.feature_pipeline.get_feature_columns() if c in df.columns]
        self._feature_cols = feature_cols

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        for col in feature_cols:
            if col in df.columns and df[col].isna().any():
                if df[col].dtype in (np.float64, np.float32):
                    df[col] = df[col].ffill()
                if df[col].isna().any():
                    df[col] = df[col].bfill()
                if df[col].isna().any():
                    df[col] = df[col].fillna(0.0)

        latest = df.iloc[-1:]
        X = latest[feature_cols].values
        X = np.nan_to_num(X, nan=0)
        return X

    @safe_execute(default_return=None, raise_on_error=True)
    def predict(self, df: pd.DataFrame, timeframe: int = Timeframe.M15) -> Dict:
        ensemble = self._get_ensemble(timeframe)
        if not ensemble.is_trained:
            raise ModelPredictionError("Ensemble not trained yet")

        df = self.feature_pipeline.compute_all(df)
        X = self._align_features(df, ensemble.feature_cols)

        ml_signal = ensemble.get_ml_signal(X)
        predictions = {"5_candle": ml_signal}

        ml_signal_10 = self._predict_simple(df, timeframe)
        if ml_signal_10:
            predictions["10_candle"] = ml_signal_10

        ml_signal_20 = self._predict_simple(df, timeframe)
        if ml_signal_20:
            predictions["20_candle"] = ml_signal_20

        return predictions

    def _predict_simple(self, df: pd.DataFrame, timeframe: int = Timeframe.M15) -> Optional[Dict]:
        try:
            ensemble = self._get_ensemble(timeframe)
            X = self._align_features(df, ensemble.feature_cols)
            return ensemble.get_ml_signal(X)
        except Exception as e:
            self.logger.warning(f"Additional prediction failed: {e}")
            return None

    def get_buy_sell_hold(self, df: pd.DataFrame, timeframe: int = Timeframe.M15) -> Dict:
        predictions = self.predict(df, timeframe)
        if not predictions:
            return {"signal": "HOLD", "confidence": 0, "buy_prob": 33, "sell_prob": 33, "hold_prob": 34}

        return predictions.get("5_candle", {
            "signal": "HOLD", "confidence": 0,
            "buy_prob": 33, "sell_prob": 33, "hold_prob": 34
        })

    def get_prediction_confidence(self, df: pd.DataFrame, timeframe: int = Timeframe.M15) -> float:
        signal = self.get_buy_sell_hold(df, timeframe)
        return signal.get("confidence", 0)
