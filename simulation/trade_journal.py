from __future__ import annotations

import csv
from datetime import datetime
from typing import Dict, List, Optional


class TradeJournal:
    def __init__(self):
        self.trades: List[Dict] = []
        self._current_pnl: float = 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.get("pnl", 0) for t in self.trades)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    def record_trade(self, trade: Dict):
        entry = {
            "pair": trade.get("pair", ""),
            "side": trade.get("side", ""),
            "volume": trade.get("volume", 0.0),
            "signal": trade.get("signal", ""),
            "confidence": trade.get("confidence", 0.0),
            "market_score": trade.get("market_score", 0),
            "regime": trade.get("regime", "UNKNOWN"),
            "trend": trade.get("trend", ""),
            "volatility": trade.get("volatility", ""),
            "rsi": trade.get("rsi", 50.0),
            "macd": trade.get("macd", ""),
            "adx": trade.get("adx", 25.0),
            "entry_price": trade.get("entry_price", 0.0),
            "exit_price": trade.get("exit_price", 0.0),
            "pnl": trade.get("pnl", 0.0),
            "commission": trade.get("commission", 0.0),
            "swap": trade.get("swap", 0.0),
            "net_pnl": trade.get("pnl", 0.0) - trade.get("commission", 0.0) - trade.get("swap", 0.0),
            "exit_reason": trade.get("exit_reason", ""),
            "entry_time": trade.get("entry_time", None),
            "close_time": trade.get("close_time", None),
            "holding_time_minutes": trade.get("holding_time_minutes", 0),
            "atr_entry": trade.get("atr_entry", 0.0),
            "spread": trade.get("spread", 0.0),
            "rr_ratio": trade.get("rr_ratio", 0.0),
            "tp1": trade.get("tp1", 0.0),
            "tp2": trade.get("tp2", 0.0),
            "sl_price": trade.get("sl_price", 0.0),
            "magic": trade.get("magic", 0),
            "multi_tf_agreement": trade.get("multi_tf_agreement", 0),
            "skill_score": trade.get("skill_score", 0),
            "trade_quality_score": trade.get("trade_quality_score", 0.0),
        }
        self.trades.append(entry)
        self._current_pnl += entry["net_pnl"]

    def get_trades_by_regime(self) -> Dict[str, List[Dict]]:
        result: Dict[str, List[Dict]] = {}
        for t in self.trades:
            regime = t.get("regime", "UNKNOWN")
            if regime not in result:
                result[regime] = []
            result[regime].append(t)
        return result

    def get_trades_by_confidence_range(self, bins: int = 5) -> Dict[str, List[Dict]]:
        step = 1.0 / bins
        result: Dict[str, List[Dict]] = {}
        for i in range(bins):
            lo = round(i * step, 2)
            hi = round((i + 1) * step, 2)
            label = f"{lo}-{hi}"
            result[label] = [
                t for t in self.trades
                if lo <= t.get("confidence", 0) < hi
            ]
        return result

    def get_trades_by_hour(self) -> Dict[int, List[Dict]]:
        result: Dict[int, List[Dict]] = {}
        for t in self.trades:
            et = t.get("entry_time")
            if et is None:
                continue
            hour = et.hour if hasattr(et, "hour") else 0
            if hour not in result:
                result[hour] = []
            result[hour].append(t)
        return result

    def get_trades_by_pair(self) -> Dict[str, List[Dict]]:
        result: Dict[str, List[Dict]] = {}
        for t in self.trades:
            pair = t.get("pair", "")
            if pair not in result:
                result[pair] = []
            result[pair].append(t)
        return result

    def get_trades_by_side(self) -> Dict[str, List[Dict]]:
        result: Dict[str, List[Dict]] = {}
        for t in self.trades:
            side = t.get("side", "")
            if side not in result:
                result[side] = []
            result[side].append(t)
        return result

    def get_winning_trades(self) -> List[Dict]:
        return [t for t in self.trades if t.get("net_pnl", 0) > 0]

    def get_losing_trades(self) -> List[Dict]:
        return [t for t in self.trades if t.get("net_pnl", 0) <= 0]

    def export_csv(self, path: str):
        if not self.trades:
            return
        fields = list(self.trades[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.trades)

    def reset(self):
        self.trades.clear()
        self._current_pnl = 0.0
