import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Union

from core.constants import Timeframe
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
        self._calibrators: Dict[int, object] = {}

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

        self._load_calibrators()

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
            feature_cols = self.feature_pipeline.get_feature_columns()
        self._feature_cols = feature_cols

        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            df = df.copy()
            for col in missing_cols:
                df[col] = 0.0
        else:
            df = df

        X = df[feature_cols].values
        X = np.nan_to_num(X, nan=0)

        latest = X[-1:, :]
        return latest

    def _resolve_feature_cols(self, ensemble: VotingEnsemble) -> Optional[List[str]]:
        """Resolve feature_cols yang benar untuk ensemble.
        Handle mismatch: jika feature_cols di metadata tidak sesuai
        dengan jumlah fitur yang diharapkan model, gunakan full set.
        """
        fcols = ensemble.feature_cols
        if fcols is None:
            return None

        # Cek model pertama untuk jumlah fitur yang diharapkan
        # Gunakan underlying model (model.model) bukan wrapper, karena wrapper
        # (XGBoostModel/RandomForestModel/LightGBMModel) tidak memiliki n_features_in_
        expected = None
        for name, model in ensemble.models.items():
            underlying = getattr(model, 'model', model)
            if hasattr(underlying, 'n_features_in_') and underlying.n_features_in_:
                expected = underlying.n_features_in_
                break

        if expected is not None and expected != len(fcols):
            self.logger.warning(
                f"Feature count mismatch: model expects {expected}, "
                f"feature_cols has {len(fcols)}. Using full feature set."
            )
            return None  # fallback ke full set

        return fcols

    @safe_execute(default_return=None, raise_on_error=True)
    def predict(self, df: pd.DataFrame, timeframe: int = Timeframe.M15) -> Dict:
        ensemble = self._get_ensemble(timeframe)
        if not ensemble.is_trained:
            raise ModelPredictionError("Ensemble not trained yet")

        fcols = self._resolve_feature_cols(ensemble)
        X = self._align_features(df, fcols)

        ml_signal = ensemble.get_ml_signal(X)

        calibrator = getattr(ensemble, "calibrator", None) or self._calibrators.get(timeframe)
        if calibrator is not None and calibrator.is_fitted:
            try:
                raw = np.array([[ml_signal["buy_prob"], ml_signal["sell_prob"], ml_signal["hold_prob"]]])
                cal = calibrator.calibrate(raw)[0]
                ml_signal["buy_prob"] = float(cal[0])
                ml_signal["sell_prob"] = float(cal[1])
                ml_signal["hold_prob"] = float(cal[2])
                ml_signal["confidence"] = float(np.max(cal))
            except Exception as e:
                self.logger.debug(f"Calibration failed: {e}")

        return {"5_candle": ml_signal}

    def get_buy_sell_hold(self, df: pd.DataFrame, timeframe: int = Timeframe.M15) -> Dict:
        predictions = self.predict(df, timeframe)
        if not predictions:
            return {"signal": "HOLD", "confidence": 0, "buy_prob": 33, "sell_prob": 33, "hold_prob": 34}

        return predictions.get("5_candle", {
            "signal": "HOLD", "confidence": 0,
            "buy_prob": 33, "sell_prob": 33, "hold_prob": 34
        })

    def _load_calibrators(self):
        from ml.probability_calibrator import ProbabilityCalibrator
        calib_base = Path("models/calibration")
        for tf in self._ensembles:
            calib_path = calib_base / str(tf)
            if calib_path.exists():
                cal = ProbabilityCalibrator()
                cal.load(str(calib_path))
                self._calibrators[tf] = cal

    def get_model_version(self, timeframe: int = Timeframe.M15) -> str:
        """Get model version string for a given timeframe."""
        try:
            ensemble = self._get_ensemble(timeframe)
            return ensemble.version or "unknown"
        except Exception:
            return "unknown"

    def get_prediction_confidence(self, df: pd.DataFrame, timeframe: int = Timeframe.M15) -> float:
        signal = self.get_buy_sell_hold(df, timeframe)
        return signal.get("confidence", 0)
