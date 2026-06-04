import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.constants import DECISION_LOG_DIR, Timeframe
from data.data_storage import ParquetStorage
from data.market_data_engine import MarketDataEngine
from utils.logger import get_logger


class DecisionEvaluator:
    OUTCOME_CORRECT_TRADE = "CORRECT_TRADE"
    OUTCOME_INCORRECT_TRADE = "INCORRECT_TRADE"
    OUTCOME_CORRECT_HOLD = "CORRECT_HOLD"
    OUTCOME_INCORRECT_HOLD = "INCORRECT_HOLD"
    OUTCOME_PENDING = "PENDING"

    def __init__(self, data_engine: Optional[MarketDataEngine] = None):
        self.logger = get_logger("decision_evaluator")
        self._log_dir = Path(DECISION_LOG_DIR)
        self._eval_dir = self._log_dir / "evaluated"
        self._eval_dir.mkdir(parents=True, exist_ok=True)
        self._storage = ParquetStorage()
        self._data_engine = data_engine

    def evaluate_recent(self, symbol: str, lookback_hours: int = 24) -> List[Dict]:
        decisions = self._load_recent_decisions(symbol, lookback_hours)
        results = []
        for dec in decisions:
            outcome = self._evaluate_one(dec, symbol)
            if outcome:
                self._save_evaluation(outcome)
                results.append(outcome)
        return results

    def _load_recent_decisions(
        self, symbol: str, lookback_hours: int
    ) -> List[Dict]:
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        pattern = f"decision_{symbol}_"
        files = sorted(self._log_dir.glob(f"{pattern}*.json"), reverse=True)
        decisions = []
        for f in files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                ts = data.get("timestamp", "")
                if ts:
                    dt = datetime.fromisoformat(ts)
                    if dt < cutoff:
                        break
                    decisions.append(data)
            except Exception:
                continue
        decisions.reverse()
        return decisions

    def _evaluate_one(self, decision_data: Dict, symbol: str) -> Optional[Dict]:
        ts = decision_data.get("timestamp", "")
        if not ts:
            return None
        decision_dt = datetime.fromisoformat(ts)

        dec = decision_data.get("decision", {})
        action = dec.get("action", "HOLD")
        is_trade = action in ("BUY", "SELL", "WEAK_BUY", "WEAK_SELL")
        no_trade = dec.get("no_trade", True)

        context = decision_data.get("context", {})
        tf = context.get("timeframe", Timeframe.M15)
        if isinstance(tf, str):
            tf = Timeframe.M15

        future_df = self._get_future_prices(symbol, tf, decision_dt, lookforward=12)
        if future_df is None or len(future_df) < 3:
            return None

        entry_price = future_df["close"].iloc[0]
        future_high = future_df["high"].max()
        future_low = future_df["low"].min()
        future_close = future_df["close"].iloc[-1]

        price_move_pct = (future_close - entry_price) / entry_price * 100
        range_pct = (future_high - future_low) / entry_price * 100

        direction = None
        if is_trade and not no_trade:
            direction = "BUY" if "BUY" in action else "SELL"

        if direction == "BUY":
            trade_correct = future_close > entry_price
            outcome = (
                self.OUTCOME_CORRECT_TRADE if trade_correct
                else self.OUTCOME_INCORRECT_TRADE
            )
        elif direction == "SELL":
            trade_correct = future_close < entry_price
            outcome = (
                self.OUTCOME_CORRECT_TRADE if trade_correct
                else self.OUTCOME_INCORRECT_TRADE
            )
        else:
            high_move = abs(future_high - entry_price) / entry_price * 100
            low_move = abs(future_low - entry_price) / entry_price * 100
            significant_move = max(high_move, low_move) > 0.15
            if significant_move:
                outcome = self.OUTCOME_INCORRECT_HOLD
            else:
                outcome = self.OUTCOME_CORRECT_HOLD

        timestamp_key = datetime.fromisoformat(ts).strftime("%Y%m%d_%H%M%S")
        evaluation = {
            "timestamp": ts,
            "symbol": symbol,
            "decision_action": action,
            "no_trade": no_trade,
            "confidence": dec.get("confidence", 0),
            "market_score": dec.get("market_score", 0),
            "outcome": outcome,
            "price_move_pct": round(price_move_pct, 4),
            "range_pct": round(range_pct, 4),
            "entry_price": entry_price,
            "future_high": future_high,
            "future_low": future_low,
            "future_close": future_close,
            "timeframe": context.get("timeframe"),
            "trend": context.get("trend"),
            "regime": context.get("regime"),
            "volatility": context.get("volatility"),
        }
        return evaluation

    def _get_future_prices(
        self, symbol: str, timeframe: int, from_dt: datetime, lookforward: int = 12
    ) -> Optional[pd.DataFrame]:
        try:
            if self._data_engine:
                df = self._data_engine.get_rates(
                    symbol, timeframe, count=lookforward + 1, use_cache=True
                )
            else:
                df = self._storage.load_data(symbol, timeframe)
            if df is None or df.empty:
                return None
            if "time" not in df.columns:
                return None
            df = df[df["time"] >= from_dt]
            if len(df) < lookforward:
                df_all = self._storage.load_data(symbol, timeframe)
                if df_all is not None and not df_all.empty:
                    df_all = df_all[df_all["time"] >= from_dt]
                    if len(df_all) >= lookforward:
                        df = df_all
            return df.iloc[:lookforward + 1] if len(df) > lookforward else df
        except Exception:
            return None

    def _save_evaluation(self, evaluation: Dict):
        ts = evaluation.get("timestamp", "")
        symbol = evaluation.get("symbol", "UNKNOWN")
        dt_key = datetime.fromisoformat(ts).strftime("%Y%m%d_%H%M%S") if ts else "unknown"
        filename = f"eval_{symbol}_{dt_key}.json"
        filepath = self._eval_dir / filename
        try:
            with open(filepath, "w") as f:
                json.dump(evaluation, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save evaluation: {e}")

    def get_evaluation_stats(self, symbol: Optional[str] = None) -> Dict:
        pattern = f"eval_{symbol}_" if symbol else "eval_"
        files = list(self._eval_dir.glob(f"{pattern}*.json"))
        outcomes = {
            self.OUTCOME_CORRECT_TRADE: 0,
            self.OUTCOME_INCORRECT_TRADE: 0,
            self.OUTCOME_CORRECT_HOLD: 0,
            self.OUTCOME_INCORRECT_HOLD: 0,
        }
        total = 0
        for f in files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                outcome = data.get("outcome")
                if outcome in outcomes:
                    outcomes[outcome] += 1
                    total += 1
            except Exception:
                continue
        correct = outcomes[self.OUTCOME_CORRECT_TRADE] + outcomes[self.OUTCOME_CORRECT_HOLD]
        incorrect = outcomes[self.OUTCOME_INCORRECT_TRADE] + outcomes[self.OUTCOME_INCORRECT_HOLD]
        return {
            "total_evaluated": total,
            "outcomes": outcomes,
            "accuracy": round(correct / total, 4) if total > 0 else 0,
            "incorrect_hold_rate": round(
                outcomes[self.OUTCOME_INCORRECT_HOLD] / total, 4
            ) if total > 0 else 0,
            "incorrect_trade_rate": round(
                outcomes[self.OUTCOME_INCORRECT_TRADE] / total, 4
            ) if total > 0 else 0,
        }

    def get_incorrect_holds(self, symbol: Optional[str] = None, n: int = 50) -> List[Dict]:
        pattern = f"eval_{symbol}_" if symbol else "eval_"
        files = sorted(self._eval_dir.glob(f"{pattern}*.json"), reverse=True)
        results = []
        for f in files[:n * 2]:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if data.get("outcome") == self.OUTCOME_INCORRECT_HOLD:
                    results.append(data)
                    if len(results) >= n:
                        break
            except Exception:
                continue
        return results

    def get_evaluated_decisions(self, lookback_hours: int = 48) -> List[Dict]:
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        files = list(self._eval_dir.glob("eval_*.json"))
        results = []
        for f in files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                ts = data.get("timestamp", "")
                if ts and datetime.fromisoformat(ts) >= cutoff:
                    results.append(data)
            except Exception:
                continue
        return results
