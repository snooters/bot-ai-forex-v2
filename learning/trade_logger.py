import json
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

import pandas as pd

from core.constants import TRADE_HISTORY_DIR
from utils.logger import get_logger


class TradeLogger:
    def __init__(self):
        self.logger = get_logger("trade_logger")
        self._trade_dir = Path(TRADE_HISTORY_DIR)
        self._trade_dir.mkdir(parents=True, exist_ok=True)
        self._trades: List[Dict] = []
        self._load_existing()

    def _load_existing(self):
        trade_file = self._trade_dir / "trade_history.json"
        if trade_file.exists():
            try:
                with open(trade_file) as f:
                    self._trades = json.load(f)
                self.logger.info(f"Loaded {len(self._trades)} existing trades")
            except Exception as e:
                self.logger.warning(f"Failed to load trade history: {e}")
                self._trades = []

    def log_trade_open(self, trade: Dict):
        trade_record = {
            "ticket": trade.get("ticket", 0),
            "symbol": trade.get("symbol", ""),
            "direction": trade.get("direction", trade.get("type", "")),
            "volume": trade.get("volume", trade.get("lot_size", 0)),
            "entry_price": trade.get("entry_price", trade.get("price", 0)),
            "stop_loss": trade.get("sl", 0),
            "take_profit": trade.get("tp", 0),
            "entry_time": datetime.now().isoformat(),
            "exit_time": None,
            "exit_price": None,
            "profit": None,
            "profit_pips": None,
            "exit_reason": None,
            "confidence": trade.get("decision", {}).get("confidence", 0),
            "market_score": trade.get("decision", {}).get("market_score", 0),
            "model_version": trade.get("model_version", "unknown"),
            "timeframe": trade.get("timeframe", "M15"),
            "market_conditions": {
                "trend": trade.get("decision", {}).get("trend", ""),
                "regime": trade.get("decision", {}).get("regime", ""),
                "volatility": trade.get("decision", {}).get("volatility", ""),
            },
        }
        self._trades.append(trade_record)
        self._save()
        self.logger.info(
            f"Trade opened: {trade_record['direction']} {trade_record['symbol']} "
            f"at {trade_record['entry_price']}"
        )

    def log_trade_close(self, ticket: int, exit_price: float, exit_reason: str = ""):
        for trade in self._trades:
            if trade["ticket"] == ticket and trade["exit_time"] is None:
                trade["exit_time"] = datetime.now().isoformat()
                trade["exit_price"] = exit_price
                trade["exit_reason"] = exit_reason

                entry_price = trade["entry_price"] or 0
                direction = trade["direction"]

                if direction == "BUY":
                    trade["profit"] = (exit_price - entry_price) * trade["volume"] * 100000
                    pips = (exit_price - entry_price) / 0.0001
                else:
                    trade["profit"] = (entry_price - exit_price) * trade["volume"] * 100000
                    pips = (entry_price - exit_price) / 0.0001

                if "JPY" in trade.get("symbol", "").upper():
                    pips = pips / 100
                trade["profit_pips"] = pips

                self._save()
                self.logger.info(
                    f"Trade closed: {trade['direction']} {trade['symbol']} "
                    f"profit=${trade['profit']:.2f} pips={pips:.1f} reason={exit_reason}"
                )
                return trade

        self.logger.warning(f"Trade {ticket} not found or already closed")
        return None

    def get_all_trades(self) -> List[Dict]:
        return self._trades

    def get_open_trades(self) -> List[Dict]:
        return [t for t in self._trades if t["exit_time"] is None]

    def get_closed_trades(self) -> List[Dict]:
        return [t for t in self._trades if t["exit_time"] is not None]

    def get_recent_trades(self, n: int = 50) -> List[Dict]:
        closed = self.get_closed_trades()
        return closed[-n:]

    def get_trades_by_date(self, target_date: date) -> List[Dict]:
        result = []
        for t in self._trades:
            entry_str = t.get("entry_time")
            if entry_str:
                try:
                    t_date = datetime.fromisoformat(entry_str).date()
                    if t_date == target_date:
                        result.append(t)
                except (ValueError, TypeError):
                    pass
        return result

    def get_trades_since(self, since_date: date) -> List[Dict]:
        result = []
        for t in self._trades:
            entry_str = t.get("entry_time")
            if entry_str:
                try:
                    t_date = datetime.fromisoformat(entry_str).date()
                    if t_date >= since_date:
                        result.append(t)
                except (ValueError, TypeError):
                    pass
        return result

    def get_trade_count(self) -> int:
        return len(self.get_closed_trades())

    def _save(self):
        trade_file = self._trade_dir / "trade_history.json"
        try:
            with open(trade_file, "w") as f:
                json.dump(self._trades, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save trade history: {e}")

    def export_to_csv(self, path: Optional[str] = None):
        if path is None:
            path = self._trade_dir / "trade_history.csv"
        try:
            df = pd.DataFrame(self._trades)
            df.to_csv(path, index=False)
            self.logger.info(f"Trade history exported to {path}")
        except Exception as e:
            self.logger.error(f"Failed to export trades: {e}")
