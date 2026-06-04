from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
from pathlib import Path
import json
import sqlite3
import uuid
import threading

from core.constants import TRADE_HISTORY_DIR, Timeframe
from utils.logger import get_logger


INDICATOR_FIELDS = [
    "rsi", "macd", "macd_signal", "macd_histogram",
    "atr", "adx", "plus_di", "minus_di",
    "bb_upper", "bb_lower", "bb_mid", "bb_width", "bb_pct",
    "stoch_k", "stoch_d",
    "ema_20", "ema_50", "ema_200",
    "momentum_10", "momentum_20",
    "volatility", "williams_r", "cci",
    "spread_pips", "obv",
    "nearest_support", "nearest_resistance",
    "dist_to_support", "dist_to_resistance",
    "volume_ratio",
]


class TradeMemory:
    def __init__(self):
        self.logger = get_logger("trade_memory")
        self._memory_dir = Path(TRADE_HISTORY_DIR)
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._memory_dir / "trade_memory.db"
        self._json_path = self._memory_dir / "trade_memory.json"
        self._lock = threading.Lock()
        self._trades: List[Dict] = []
        self._init_db()
        self._load()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    pair TEXT,
                    timeframe TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    volume REAL,
                    profit REAL,
                    profit_pips REAL,
                    result TEXT,
                    exit_reason TEXT,
                    entry_time TEXT,
                    exit_time TEXT,
                    model_version TEXT,
                    confidence REAL,
                    trade_duration_minutes REAL,
                    max_dd_during_trade REAL,
                    spread REAL,
                    commission REAL,
                    swap REAL,
                    session TEXT,
                    indicators TEXT,
                    market_conditions TEXT,
                    created_at TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pair ON trades(pair)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_result ON trades(result)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entry_time ON trades(entry_time)")
            conn.commit()
            conn.close()

    def record_trade(
        self,
        pair: str,
        timeframe: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        volume: float,
        profit: float,
        profit_pips: float,
        result: str,
        exit_reason: str,
        entry_time: str,
        exit_time: str,
        indicators: Optional[Dict] = None,
        market_conditions: Optional[Dict] = None,
        model_version: str = "unknown",
        confidence: float = 0.0,
        trade_duration_minutes: Optional[float] = None,
        max_dd_during_trade: Optional[float] = None,
        spread: float = 0.0,
        commission: float = 0.0,
        swap: float = 0.0,
        session: str = "",
    ):
        if indicators is None:
            indicators = {}
        if market_conditions is None:
            market_conditions = {}

        trade_id = str(uuid.uuid4())[:8]
        dur = trade_duration_minutes or self._calc_duration(entry_time, exit_time)

        record = {
            "trade_id": trade_id,
            "pair": pair,
            "timeframe": timeframe,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "volume": volume,
            "profit": round(profit, 2),
            "profit_pips": round(profit_pips, 1),
            "result": result,
            "exit_reason": exit_reason,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "model_version": model_version,
            "confidence": round(confidence, 2),
            "trade_duration_minutes": dur,
            "max_dd_during_trade": round(max_dd_during_trade, 4) if max_dd_during_trade else 0,
            "spread": round(spread, 1),
            "commission": round(commission, 2),
            "swap": round(swap, 2),
            "session": session,
            "indicators": {k: round(v, 6) if isinstance(v, float) else v
                          for k, v in indicators.items() if k in INDICATOR_FIELDS},
            "market_conditions": market_conditions,
            "created_at": datetime.now().isoformat(),
        }

        self._trades.append(record)
        self._insert_sqlite(record)
        return record

    def record_from_trade_log(self, trade: Dict, indicators: Optional[Dict] = None):
        entry_time_str = trade.get("entry_time", "")
        exit_time_str = trade.get("exit_time", "")
        entry_price = trade.get("entry_price", 0) or 0
        exit_price = trade.get("exit_price", 0) or 0
        profit = trade.get("profit", 0) or 0
        profit_pips = trade.get("profit_pips", 0) or 0
        direction = trade.get("direction", "HOLD")

        result = "WIN" if profit > 0 else "LOSS" if profit < 0 else "BREAK"

        return self.record_trade(
            pair=trade.get("symbol", "UNKNOWN"),
            timeframe=trade.get("timeframe", "M15"),
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            volume=trade.get("volume", 0),
            profit=profit,
            profit_pips=profit_pips,
            result=result,
            exit_reason=trade.get("exit_reason", ""),
            entry_time=entry_time_str,
            exit_time=exit_time_str,
            indicators=indicators,
            market_conditions=trade.get("market_conditions", {}),
            model_version=trade.get("model_version", "unknown"),
            confidence=trade.get("confidence", 0),
            spread=trade.get("spread", trade.get("spread_at_entry", 0)),
            commission=trade.get("commission", 0),
            swap=trade.get("swap", 0),
            session=trade.get("session", ""),
        )

    def _insert_sqlite(self, record: Dict):
        try:
            with self._lock:
                conn = sqlite3.connect(str(self._db_path))
                conn.execute("""
                    INSERT OR REPLACE INTO trades
                    (trade_id, pair, timeframe, direction, entry_price, exit_price,
                     volume, profit, profit_pips, result, exit_reason,
                     entry_time, exit_time, model_version, confidence,
                     trade_duration_minutes, max_dd_during_trade,
                     spread, commission, swap, session,
                     indicators, market_conditions, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    record["trade_id"], record["pair"], record["timeframe"],
                    record["direction"], record["entry_price"], record["exit_price"],
                    record["volume"], record["profit"], record["profit_pips"],
                    record["result"], record["exit_reason"],
                    record["entry_time"], record["exit_time"],
                    record["model_version"], record["confidence"],
                    record["trade_duration_minutes"], record["max_dd_during_trade"],
                    record["spread"], record["commission"], record["swap"],
                    record["session"],
                    json.dumps(record.get("indicators", {})),
                    json.dumps(record.get("market_conditions", {})),
                    record["created_at"],
                ))
                conn.commit()
                conn.close()
        except Exception as e:
            self.logger.error(f"SQLite insert failed: {e}")

    def get_all_trades(self) -> List[Dict]:
        return list(self._trades)

    def get_wins(self) -> List[Dict]:
        return [t for t in self._trades if t["result"] == "WIN"]

    def get_losses(self) -> List[Dict]:
        return [t for t in self._trades if t["result"] == "LOSS"]

    def get_by_pair(self, pair: str) -> List[Dict]:
        return [t for t in self._trades if t["pair"] == pair]

    def get_by_timeframe(self, tf: str) -> List[Dict]:
        return [t for t in self._trades if t["timeframe"] == tf]

    def get_trades_for_timeframe(self, timeframe: int) -> List[Dict]:
        tf_label = Timeframe.LABELS.get(timeframe, f"M{timeframe}")
        return [t for t in self._trades if t.get("timeframe") == tf_label]

    def get_by_date_range(self, start_date: date, end_date: date) -> List[Dict]:
        result = []
        for t in self._trades:
            try:
                t_date = datetime.fromisoformat(t["entry_time"]).date()
                if start_date <= t_date <= end_date:
                    result.append(t)
            except (ValueError, TypeError):
                pass
        return result

    def get_recent(self, n: int = 100) -> List[Dict]:
        return self._trades[-n:]

    def get_win_rate(self, trades: Optional[List[Dict]] = None) -> float:
        source = trades if trades is not None else self._trades
        if not source:
            return 0.0
        wins = sum(1 for t in source if t["result"] == "WIN")
        return wins / len(source) * 100

    def get_average_profit(self, trades: Optional[List[Dict]] = None) -> float:
        source = trades if trades is not None else self._trades
        if not source:
            return 0.0
        return sum(t.get("profit", 0) for t in source) / len(source)

    def count(self, trades: Optional[List[Dict]] = None) -> int:
        return len(trades) if trades is not None else len(self._trades)

    def find_by_pattern(
        self,
        direction: Optional[str] = None,
        regime: Optional[str] = None,
        timeframe: Optional[str] = None,
        min_trades: int = 3,
    ) -> Dict:
        matching = []
        for t in self._trades:
            if direction and t.get("direction") != direction:
                continue
            if timeframe and t.get("timeframe") != timeframe:
                continue
            if regime:
                mc = t.get("market_conditions", {})
                t_regime = mc.get("regime", "") if mc else ""
                if regime not in t_regime:
                    continue
            matching.append(t)

        closed = [t for t in matching if t.get("result") in ("WIN", "LOSS")]
        result = {
            "total": len(matching),
            "closed": len(closed),
        }
        if len(closed) >= min_trades:
            wins = sum(1 for t in closed if t["result"] == "WIN")
            losses = len(closed) - wins
            gross_profit = sum(t.get("profit", 0) for t in closed if t.get("profit", 0) > 0)
            gross_loss = abs(sum(t.get("profit", 0) for t in closed if t.get("profit", 0) < 0))
            result.update({
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / len(closed), 4),
                "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else float("inf"),
                "avg_profit": round(sum(t.get("profit", 0) for t in closed) / len(closed), 4),
            })
        return result

    def find_similar(
        self,
        indicators: Dict,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict]:
        if not indicators or not self._trades:
            return []

        def _cosine_sim(a: Dict, b: Dict) -> float:
            common = set(a.keys()) & set(b.keys())
            if len(common) < 3:
                return 0.0
            dot = sum(a[k] * b[k] for k in common)
            na = sum(a[k] ** 2 for k in common) ** 0.5
            nb = sum(b[k] ** 2 for k in common) ** 0.5
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        scored = []
        for t in self._trades[-500:]:
            t_inds = t.get("indicators", {})
            if not t_inds:
                continue
            sim = _cosine_sim(indicators, t_inds)
            if sim < min_score:
                continue
            scored.append((sim, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k]]

    def export_json(self) -> str:
        return json.dumps(self._trades, indent=2, default=str)

    def _calc_duration(self, entry_time: str, exit_time: str) -> float:
        try:
            entry = datetime.fromisoformat(entry_time)
            exit_dt = datetime.fromisoformat(exit_time) if exit_time else datetime.now()
            return (exit_dt - entry).total_seconds() / 60
        except (ValueError, TypeError):
            return 0.0

    def _load(self):
        try:
            with self._lock:
                conn = sqlite3.connect(str(self._db_path))
                rows = conn.execute("SELECT * FROM trades ORDER BY created_at").fetchall()
                col_names = [d[1] for d in conn.execute("PRAGMA table_info(trades)").fetchall()]
                conn.close()
            self._trades = []
            for row in rows:
                rec = dict(zip(col_names, row))
                rec["indicators"] = json.loads(rec.get("indicators") or "{}")
                rec["market_conditions"] = json.loads(rec.get("market_conditions") or "{}")
                self._trades.append(rec)
            if not self._trades and self._json_path.exists():
                self.logger.info("SQLite empty but JSON exists — migrating")
                self._load_json_fallback()
            else:
                self.logger.info(f"Loaded {len(self._trades)} trades from SQLite")
        except Exception as e:
            self.logger.warning(f"SQLite load failed ({e}), trying JSON fallback")
            self._load_json_fallback()

    def _load_json_fallback(self):
        if self._json_path.exists():
            try:
                with open(self._json_path) as f:
                    data = json.load(f)
                self._trades = []
                for rec in data:
                    rec["trade_id"] = rec.get("trade_id", str(uuid.uuid4())[:8])
                    rec["spread"] = rec.get("spread", rec.get("spread_at_entry", 0))
                    rec["commission"] = rec.get("commission", 0)
                    rec["swap"] = rec.get("swap", 0)
                    rec["session"] = rec.get("session", "")
                    rec["created_at"] = rec.get("created_at", rec.get("timestamp", datetime.now().isoformat()))
                    self._trades.append(rec)
                    self._insert_sqlite(rec)
                self.logger.info(f"Migrated {len(self._trades)} trades from JSON to SQLite")
            except Exception as e2:
                self.logger.warning(f"JSON fallback also failed: {e2}")
                self._trades = []

    def get_summary_stats(self) -> Dict:
        if not self._trades:
            return {}
        wins = self.get_wins()
        losses = self.get_losses()
        total = len(self._trades)
        win_rate = len(wins) / total * 100 if total > 0 else 0
        gross_profit = sum(t["profit"] for t in wins)
        gross_loss = abs(sum(t["profit"] for t in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        profits = [t["profit"] for t in self._trades]
        avg_profit = sum(profits) / len(profits) if profits else 0
        return {
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(pf, 2),
            "avg_profit": round(avg_profit, 2),
            "net_profit": round(sum(profits), 2),
        }
