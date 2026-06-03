from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

import numpy as np

from core.constants import TRADE_HISTORY_DIR
from utils.logger import get_logger


MEMORY_DECAY = {
    7: 1.0,
    30: 0.8,
    90: 0.6,
    180: 0.3,
}

DEFAULT_WEIGHT = 0.2


class AdaptiveMemory:
    def __init__(self):
        self.logger = get_logger("adaptive_memory")
        self._memory_dir = Path(TRADE_HISTORY_DIR)
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._pattern_file = self._memory_dir / "market_patterns.json"
        self._patterns: Dict[str, Dict] = {}

    def compute_weight(self, trade_date: datetime) -> float:
        if not trade_date:
            return DEFAULT_WEIGHT
        days_ago = (datetime.now() - trade_date).days

        sorted_days = sorted(MEMORY_DECAY.keys())
        for day_threshold in sorted_days:
            if days_ago <= day_threshold:
                return MEMORY_DECAY[day_threshold]

        return DEFAULT_WEIGHT

    def compute_weights(self, trade_dates: List[datetime]) -> np.ndarray:
        if not trade_dates:
            return np.array([])
        return np.array([self.compute_weight(d) for d in trade_dates])

    def get_weighted_trades(self, trades: List[Dict]) -> List[Dict]:
        if not trades:
            return trades
        weighted = []
        for trade in trades:
            entry_time_str = trade.get("entry_time")
            try:
                entry_time = datetime.fromisoformat(entry_time_str) if entry_time_str else None
            except (ValueError, TypeError):
                entry_time = None
            weight = self.compute_weight(entry_time)
            t = trade.copy()
            t["memory_weight"] = weight
            weighted.append(t)
        return weighted

    def filter_recent_trades(self, trades: List[Dict], max_days: int = 90) -> List[Dict]:
        cutoff = datetime.now() - timedelta(days=max_days)
        result = []
        for trade in trades:
            entry_time_str = trade.get("entry_time")
            if entry_time_str:
                try:
                    entry_time = datetime.fromisoformat(entry_time_str)
                    if entry_time >= cutoff:
                        result.append(trade)
                except (ValueError, TypeError):
                    result.append(trade)
            else:
                result.append(trade)
        return result

    def learn_from_trade(self, trade: Dict):
        pattern_key = self._build_pattern_key(trade)
        if not pattern_key:
            return

        if pattern_key not in self._patterns:
            self._patterns[pattern_key] = {
                "count": 0,
                "wins": 0,
                "losses": 0,
                "total_profit": 0.0,
                "last_seen": None,
                "weighted_score": 0.0,
            }

        p = self._patterns[pattern_key]
        p["count"] += 1
        p["total_profit"] += trade.get("profit", 0)
        p["last_seen"] = trade.get("exit_time", datetime.now().isoformat())

        if trade.get("result") == "WIN":
            p["wins"] += 1
        elif trade.get("result") == "LOSS":
            p["losses"] += 1

        p["weighted_score"] = self._compute_pattern_score(p)
        self._save_patterns()

    def get_best_patterns(self, top_n: int = 5) -> List[Dict]:
        scored = []
        for key, data in self._patterns.items():
            if data["count"] >= 3:
                scored.append((data["weighted_score"], key, data))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"pattern": k, "score": s, "data": d}
            for s, k, d in scored[:top_n]
        ]

    def get_worst_patterns(self, top_n: int = 5) -> List[Dict]:
        scored = []
        for key, data in self._patterns.items():
            if data["count"] >= 3:
                scored.append((data["weighted_score"], key, data))
        scored.sort(key=lambda x: x[0])
        return [
            {"pattern": k, "score": s, "data": d}
            for s, k, d in scored[:top_n]
        ]

    def should_use_pattern(self, trade: Dict) -> Optional[Dict]:
        pattern_key = self._build_pattern_key(trade)
        if pattern_key in self._patterns:
            p = self._patterns[pattern_key]
            if p["count"] >= 3 and p["weighted_score"] > 0.6:
                return {"pattern": pattern_key, "score": p["weighted_score"], "wins": p["wins"], "losses": p["losses"]}
        return None

    def _build_pattern_key(self, trade: Dict) -> str:
        pair = trade.get("pair", "")
        timeframe = trade.get("timeframe", "")
        direction = trade.get("direction", "")
        regime = trade.get("market_conditions", {}).get("regime", "UNKNOWN")
        if not pair or not direction:
            return ""
        return f"{pair}_{timeframe}_{direction}_{regime}"

    def _compute_pattern_score(self, pattern: Dict) -> float:
        if pattern["count"] == 0:
            return 0.0
        win_rate = pattern["wins"] / pattern["count"]
        avg_profit = pattern["total_profit"] / pattern["count"]
        profit_factor = 0.0
        if pattern["losses"] > 0:
            gross_wins = pattern.get("total_profit", 0) if pattern["total_profit"] > 0 else 0
            gross_losses = abs(pattern.get("total_profit", 0)) if pattern["total_profit"] < 0 else 0
            profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0

        recency = 1.0
        if pattern.get("last_seen"):
            try:
                last = datetime.fromisoformat(pattern["last_seen"])
                days_since = (datetime.now() - last).days
                recency = max(0.1, 1.0 - days_since / 180)
            except (ValueError, TypeError):
                pass

        return win_rate * 0.5 + min(profit_factor, 3.0) / 3 * 0.3 + recency * 0.2

    def _save_patterns(self):
        try:
            with open(self._pattern_file, "w") as f:
                json.dump(self._patterns, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save patterns: {e}")
