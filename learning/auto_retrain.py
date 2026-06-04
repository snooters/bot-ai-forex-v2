from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from core.config import config
from core.constants import RETRAIN_TRADE_COUNT
from learning.trade_logger import TradeLogger
from learning.trade_memory import TradeMemory
from learning.mistake_weighting import MistakeWeighting
from learning.concept_drift import ConceptDriftDetector
from learning.performance_analyzer import PerformanceAnalyzer
from ml.trainer import ModelTrainer
from ml.model_manager import ModelManager
from ml.ensemble import VotingEnsemble
from utils.logger import get_logger


RETRAIN_MIN_TRADES = 100
PERFORMANCE_DROP_THRESHOLD = 0.15
RETRAIN_COOLDOWN_HOURS = 4


class AutoRetrainEngine:
    def __init__(
        self,
        trade_logger: TradeLogger,
        model_trainer: ModelTrainer,
        model_manager: ModelManager,
        trade_memory: Optional[TradeMemory] = None,
    ):
        self.logger = get_logger("auto_retrain")
        self.trade_logger = trade_logger
        self.model_trainer = model_trainer
        self.model_manager = model_manager
        self.trade_memory = trade_memory
        self.drift_detector = ConceptDriftDetector()
        self.performance_analyzer = PerformanceAnalyzer()
        self._last_retrain_count = 0
        self._retrain_count = 0
        self._last_retrain_time = datetime.now()
        self._last_performance_baseline: Optional[Dict] = None

    def check_retrain_needed(self) -> Tuple[bool, str]:
        if not config.learning["auto_retrain"]:
            return False, "Auto retrain disabled"

        cooldown_hours = config.learning.get("retrain_cooldown_hours", RETRAIN_COOLDOWN_HOURS)
        hours_since = (datetime.now() - self._last_retrain_time).total_seconds() / 3600
        if hours_since < cooldown_hours:
            return False, f"Cooldown active ({hours_since:.0f}h < {cooldown_hours}h)"

        trades = self.trade_logger.get_closed_trades()
        current_count = len(trades)
        since_last = current_count - self._last_retrain_count

        interval_hours = config.learning.get("retrain_interval_hours", 24)
        hours_since_last = (datetime.now() - self._last_retrain_time).total_seconds() / 3600
        if hours_since_last >= interval_hours and since_last >= max(10, RETRAIN_MIN_TRADES // 2):
            return True, f"Time-based retrain ({hours_since_last:.0f}h since last)"

        retrain_threshold = config.learning.get("retrain_after_trades", RETRAIN_TRADE_COUNT)
        if since_last >= retrain_threshold:
            return True, f"Trade count threshold: {since_last} new trades"

        if since_last >= RETRAIN_MIN_TRADES:
            return True, f"Reached {RETRAIN_MIN_TRADES} new trades (comprehensive retrain)"

        if since_last >= 20:
            perf_drop = self._check_performance_drop(trades)
            if perf_drop["drop_detected"]:
                return True, perf_drop["reason"]

        drift_check, drift_reason = self.drift_detector.should_retrain(trades)
        if drift_check:
            return True, drift_reason

        if since_last >= 10:
            mistake_check, mistake_reason = self._check_mistake_report()
            if mistake_check:
                return True, mistake_reason

        return False, "No retrain needed"

    def retrain(self, training_data: pd.DataFrame, timeframe: Optional[int] = None,
                sample_weight_multiplier: float = 1.0,
                model_params: Optional[Dict] = None,
                progress=None,
                tf_label=None) -> Dict:
        self.logger.info(f"Starting auto retrain for timeframe={timeframe} "
                         f"(weight_mult={sample_weight_multiplier})...")

        self.model_trainer.ensemble = VotingEnsemble()

        try:
            X, y, features, df_clean = self.model_trainer.prepare_training_data(training_data)
        except Exception as e:
            self.logger.error(f"Failed to prepare training data: {e}")
            return {"success": False, "error": str(e)}

        from ml.trainer import ModelTrainer as _MT
        recency = _MT.compute_recency_weights(df_clean["time"]) if "time" in df_clean.columns else None

        self.logger.info(f"Training samples: {len(X)} | features: {len(features)} | labels: BUY={(y==0).sum()} SELL={(y==1).sum()} HOLD={(y==2).sum()}")

        try:
            results = self.model_trainer.train_all_models(
                X, y,
                sample_weight_multiplier=sample_weight_multiplier,
                model_params=model_params,
                progress=progress,
                tf_label=tf_label,
                recency_weights=recency,
            )
        except Exception as e:
            self.logger.error(f"Failed to train models: {e}")
            return {"success": False, "error": str(e)}

        if self.model_trainer.get_ensemble().get_num_models() > 0:
            model_version = self.model_manager.save_ensemble(
                self.model_trainer.get_ensemble(), timeframe=timeframe
            )
            self.model_manager.increment_retrain_count(timeframe)
            self._retrain_count += 1
            self._last_retrain_count = self.trade_logger.get_trade_count()
            self._last_retrain_time = datetime.now()

            self.drift_detector.reset_baseline()

            result = {
                "success": True,
                "version": model_version,
                "retrain_count": self._retrain_count,
                "models": results,
                "X": X,
                "y": y,
            }
            self.logger.info(f"Auto retrain complete. New model version: {model_version}")
            return result

        return {"success": False, "error": "No models trained"}

    def _check_performance_drop(self, trades: List) -> Dict:
        if not trades or len(trades) < 20:
            return {"drop_detected": False, "reason": "insufficient trades"}

        closed = [t for t in trades if t.get("profit") is not None]
        closed.sort(key=lambda t: t.get("exit_time", ""))

        recent = closed[-20:]
        older = closed[:-20]

        if not older:
            return {"drop_detected": False, "reason": "no baseline"}

        recent_win_rate = sum(1 for t in recent if t.get("profit", 0) > 0) / len(recent)
        older_win_rate = sum(1 for t in older if t.get("profit", 0) > 0) / len(older)

        recent_gp = sum(t.get("profit", 0) for t in recent if t.get("profit", 0) > 0)
        recent_gl = abs(sum(t.get("profit", 0) for t in recent if t.get("profit", 0) < 0))
        recent_pf = recent_gp / recent_gl if recent_gl > 0 else 0

        older_gp = sum(t.get("profit", 0) for t in older if t.get("profit", 0) > 0)
        older_gl = abs(sum(t.get("profit", 0) for t in older if t.get("profit", 0) < 0))
        older_pf = older_gp / older_gl if older_gl > 0 else 0

        wr_drop_pct = (older_win_rate - recent_win_rate) / older_win_rate if older_win_rate > 0 else 0
        pf_drop_pct = (older_pf - recent_pf) / older_pf if older_pf > 0 else 0

        if wr_drop_pct >= PERFORMANCE_DROP_THRESHOLD:
            return {
                "drop_detected": True,
                "reason": f"Win rate dropped {wr_drop_pct:.1%} (threshold: {PERFORMANCE_DROP_THRESHOLD:.0%})",
                "wr_drop": wr_drop_pct,
                "pf_drop": pf_drop_pct,
            }

        if pf_drop_pct >= PERFORMANCE_DROP_THRESHOLD:
            return {
                "drop_detected": True,
                "reason": f"Profit factor dropped {pf_drop_pct:.1%} (threshold: {PERFORMANCE_DROP_THRESHOLD:.0%})",
                "wr_drop": wr_drop_pct,
                "pf_drop": pf_drop_pct,
            }

        return {"drop_detected": False, "reason": "performance stable"}

    def _check_mistake_report(self) -> Tuple[bool, str]:
        if not self.trade_memory:
            return False, "no trade memory"

        patterns_to_check = ["BUY", "SELL"]
        for direction in patterns_to_check:
            result = self.trade_memory.find_by_pattern(
                direction=direction, min_trades=5
            )
            if result.get("closed", 0) >= 5:
                win_rate = result.get("win_rate", 0)
                losses = result.get("losses", 0)
                if win_rate < 0.30 and losses >= 4:
                    return True, (
                        f"Mistake pattern: {direction} has {win_rate:.0%} WR "
                        f"({losses} losses in {result['closed']} trades)"
                    )

        closed = self.trade_memory.get_all_trades()
        closed = [t for t in closed if t.get("result") in ("WIN", "LOSS")]
        recent = closed[-30:] if len(closed) >= 30 else closed
        if len(recent) >= 10:
            recent_wins = sum(1 for t in recent if t["result"] == "WIN")
            recent_wr = recent_wins / len(recent)
            if recent_wr < 0.30:
                return True, (
                    f"Recent performance crash: {recent_wr:.0%} WR "
                    f"({recent_wins}/{len(recent)} wins last {len(recent)} trades)"
                )

        hold_retrain, hold_reason = self._check_incorrect_holds()
        if hold_retrain:
            return True, hold_reason

        return False, "no mistake patterns detected"

    def _check_incorrect_holds(self) -> Tuple[bool, str]:
        mw = MistakeWeighting(self.trade_memory)
        incorrect_hold_rate = mw.get_incorrect_hold_rate()
        if incorrect_hold_rate > 0.4:
            return True, (
                f"High INCORRECT_HOLD rate: {incorrect_hold_rate:.0%} "
                f"— model may be missing opportunities"
            )
        return False, "incorrect hold rate acceptable"

    @property
    def retrain_count(self) -> int:
        return self._retrain_count
