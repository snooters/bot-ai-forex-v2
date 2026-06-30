from __future__ import annotations

import json
import logging
import time as time_module
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import config
from core.constants import Timeframe
from data.data_storage import ParquetStorage
from decision.decision_engine import DecisionEngine
from features.feature_pipeline import FeaturePipeline

from intelligence.market_regime import MarketRegimeDetector
from intelligence.momentum_analysis import MomentumAnalyzer
from intelligence.trend_analysis import TrendAnalyzer
from intelligence.volatility_analysis import VolatilityAnalyzer
from learning.skill_scorer import SkillScorer

from simulation.trade_journal import TradeJournal
from simulation.virtual_account import VirtualAccount

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None


class ReplayEngine:
    """Iterate M5 data bar-by-bar, building context TF from available data only."""

    def __init__(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        warmup_bars: int = 500,
    ):
        self.symbol = symbol
        self.from_date = from_date
        self.to_date = to_date
        self.warmup_bars = warmup_bars
        self.storage = ParquetStorage()
        self.loaded: bool = False
        self._m5_data: Optional[pd.DataFrame] = None
        self._feat_data: Optional[pd.DataFrame] = None
        self._total_bars: int = 0
        self._current_idx: int = 0

    def load_and_precompute(self) -> None:
        raw = self.storage.load_data(self.symbol, Timeframe.M5, self.from_date, self.to_date)
        if raw is None or raw.empty:
            raise ValueError(f"No M5 data for {self.symbol} in date range")

        raw = raw.sort_values("time").reset_index(drop=True)
        if len(raw) < self.warmup_bars + 100:
            raise ValueError(
                f"Not enough M5 data for {self.symbol}: {len(raw)} bars "
                f"(need at least {self.warmup_bars + 100})"
            )

        self._m5_data = raw

        pipeline = FeaturePipeline()
        self._feat_data = pipeline.compute_all(raw.copy())
        self._total_bars = len(self._feat_data)
        self.loaded = True
        self._current_idx = 0

        logger.info(
            "ReplayEngine loaded %d bars for %s (warmup=%d)",
            self._total_bars, self.symbol, self.warmup_bars,
        )

    def __len__(self) -> int:
        if not self.loaded:
            return 0
        return max(0, self._total_bars - self.warmup_bars)

    def __iter__(self):
        if not self.loaded:
            self.load_and_precompute()
        self._current_idx = self.warmup_bars
        return self

    MAX_WINDOW = 200

    def __next__(self) -> Tuple[int, pd.DataFrame]:
        if self._current_idx >= self._total_bars:
            raise StopIteration

        idx = self._current_idx
        start = max(0, idx + 1 - self.MAX_WINDOW)
        window = self._feat_data.iloc[start: idx + 1]
        self._current_idx += 1
        return idx, window

    @property
    def current_time(self) -> Optional[datetime]:
        if self._feat_data is None or self._current_idx <= 0:
            return None
        return self._feat_data.iloc[self._current_idx - 1].get("time")


class Simulator:
    def __init__(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        initial_balance: float = 10000.0,
        volume_fixed: Optional[float] = None,
        warmup_bars: int = 500,
        max_positions: int = 2,
        atr_multiplier_tp1: float = 3.0,
        atr_multiplier_tp2: float = 5.0,
        atr_multiplier_trailing: float = 1.5,
        max_hold_hours: float = 12.0,
        min_rr: float = 2.0,
        pair_suffix: str = "",
        decision_engine: Optional[DecisionEngine] = None,
    ):
        self.symbol = symbol
        self.from_date = from_date
        self.to_date = to_date
        self.initial_balance = initial_balance
        self.volume_fixed = volume_fixed
        self.warmup_bars = warmup_bars
        self.max_positions = max_positions
        self.atr_multiplier_tp1 = atr_multiplier_tp1
        self.atr_multiplier_tp2 = atr_multiplier_tp2
        self.atr_multiplier_trailing = atr_multiplier_trailing
        self.max_hold_hours = max_hold_hours
        self.min_rr = min_rr
        self.pair_suffix = pair_suffix

        self.replay = ReplayEngine(symbol, from_date, to_date, warmup_bars)
        self.account = VirtualAccount(initial_balance, leverage=100)
        self.journal = TradeJournal()
        self.feature_pipeline = FeaturePipeline()
        self.decision_engine = decision_engine
        self.trend_analyzer = TrendAnalyzer()
        self.vol_analyzer = VolatilityAnalyzer()
        self.momentum_analyzer = MomentumAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        self.skill_scorer = SkillScorer()
        self._position_id_counter: int = 0
        self._total_candles_processed: int = 0
        self._signals_generated: int = 0
        self._trades_opened: int = 0
        self._trades_closed: int = 0

    async def run(self, show_progress: bool = True) -> Dict[str, Any]:
        logger.info("Starting simulation for %s", self.symbol)
        self.replay.load_and_precompute()

        total_bars = len(self.replay)
        pair_symbol = self.symbol + self.pair_suffix if self.pair_suffix else self.symbol

        # Progress bar setup
        use_progress = show_progress and HAS_TQDM
        pbar = None
        if use_progress:
            pbar = tqdm(
                total=total_bars,
                desc=f"Simulating {self.symbol}",
                unit="bar",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] trades={postfix}",
            )
        for idx, window in self.replay:
            self._total_candles_processed += 1
            current_row = window.iloc[-1]
            current_time = current_row.get("time")
            if isinstance(current_time, str):
                current_time = pd.to_datetime(current_time)

            high = float(current_row.get("high", 0))
            low = float(current_row.get("low", 0))
            close = float(current_row.get("close", 0))
            spread = float(current_row.get("spread", 0))

            atr_val = float(current_row.get("atr", 0))

            if self.account.positions:
                self.account.check_sl_tp_all(high, low, current_time)
                self.account.apply_trailing_all(close, atr_val, self.atr_multiplier_trailing)
                self.account.check_time_exit_all(self.max_hold_hours, current_time)

            open_positions = [
                p for p in self.account.positions if p.close_price is None
            ]
            if open_positions:
                if pbar is not None:
                    pbar.set_postfix_str(f"{self._trades_opened}open", refresh=False)
                    pbar.update(1)
                continue

            if self.decision_engine is None:
                if pbar is not None:
                    pbar.update(1)
                continue

            trend_result = self.trend_analyzer.analyze_trend(window)
            vol_result = self.vol_analyzer.analyze_volatility(window)
            momentum_result = self.momentum_analyzer.analyze_momentum(window)
            regime_result = self.regime_detector.detect_regime(
                trend_result, vol_result, momentum_result, window
            )

            sr_info = self.feature_pipeline.support_resistance.detect_levels(window)
            feature_summary = self.feature_pipeline.compute_features_summary(window, sr_info=sr_info)
            multi_tf_trends = self._extract_multi_tf_trends(current_row)

            pair_skill_score = None
            try:
                skills = self.skill_scorer.get_pair_skills()
                if pair_symbol in skills:
                    pair_skill_score = skills[pair_symbol]
            except Exception:
                pass

            news = {"sentiment": "neutral", "impact": "low", "headlines": []}
            llm_analysis = None

            decision = self.decision_engine.make_decision(
                symbol=pair_symbol,
                df_entry={Timeframe.M5: window},
                trend_result=trend_result,
                vol_result=vol_result,
                momentum_result=momentum_result,
                regime_result=regime_result,
                sr_info=sr_info,
                feature_summary=feature_summary,
                spread=spread,
                timeframe=Timeframe.M5,
                multi_tf_trends=multi_tf_trends,
                pair_skill_score=pair_skill_score,
                news_analysis=news,
                llm_analysis=llm_analysis,
            )

            action = decision.get("action", "HOLD")
            no_trade = decision.get("no_trade", True)
            confidence = decision.get("confidence", 0.0)
            market_score = decision.get("market_score", 0)

            if action in ("BUY", "SELL") and not no_trade and confidence > 0:
                self._signals_generated += 1
                pos = self._execute_trade(
                    side=action,
                    price=close,
                    atr=atr_val,
                    decision=decision,
                    current_time=current_time,
                    spread=spread,
                    trend_result=trend_result,
                    regime_result=regime_result,
                    feature_summary=feature_summary,
                    pair_skill_score=pair_skill_score,
                    feature_row=current_row,
                )
                if pos:
                    self._trades_opened += 1

            if pbar is not None:
                pbar.set_postfix_str(f"{self._trades_opened}open", refresh=False)
                pbar.update(1)

        # Close progress bar
        if pbar is not None:
            pbar.close()

        closed_from_mgmt = self.account.close_all_positions(
            self._get_last_price(), "SIMULATION_END", self.replay.current_time or datetime.now()
        )
        self._trades_closed = len(self.account.closed_positions)

        result = self._build_result()
        logger.info(
            "Simulation complete for %s: %d trades, %.2f%% return",
            self.symbol,
            result["total_trades"],
            result.get("return_pct", 0),
        )
        return result

    def _execute_trade(
        self,
        side: str,
        price: float,
        atr: float,
        decision: Dict,
        current_time: datetime,
        spread: float,
        trend_result: Dict,
        regime_result: Dict,
        feature_summary: Dict,
        pair_skill_score: Optional[float] = None,
        feature_row: Optional[pd.Series] = None,
    ):
        volume = self.volume_fixed or 0.01

        atr_pts = atr
        # Match live engine: SL = ATR × 2.0 (min 15 pips), TP1 = ATR × 3.0, TP2 = ATR × 5.0
        pip_size = 0.0001
        sl_pts = max(atr_pts * 2.0, 15 * pip_size)
        tp1_pts = atr_pts * self.atr_multiplier_tp1
        tp2_pts = atr_pts * self.atr_multiplier_tp2
        if side == "BUY":
            sl_price = price - sl_pts
            tp1_price = price + tp1_pts
            tp2_price = price + tp2_pts
        else:
            sl_price = price + sl_pts
            tp1_price = price - tp1_pts
            tp2_price = price - tp2_pts

        tp1_dist = abs(tp1_price - price)
        sl_dist = abs(price - sl_price)
        rr1 = tp1_dist / sl_dist if sl_dist > 0 else 0

        entry_vol = volume / 2

        pos = self.account.open_position(
            pair=self.symbol,
            side=side,
            price=price,
            volume=entry_vol,
            current_time=current_time,
            sl=sl_price,
            tp=tp1_price,
            atr_entry=atr,
            comment=f"sim_{side}_{self._total_candles_processed}",
            trailing_activated=True,
        )

        if pos:
            feat_vec = None
            if feature_row is not None:
                try:
                    fv = {}
                    for k, v in feature_row.items():
                        if k in ("time", "symbol") or isinstance(v, (pd.Timestamp, datetime)):
                            continue
                        try:
                            fv[k] = float(v)
                        except (ValueError, TypeError):
                            continue
                    feat_vec = json.dumps(fv) if fv else None
                except Exception:
                    pass

            entry_record = {
                "pair": self.symbol + self.pair_suffix,
                "side": side,
                "volume": entry_vol,
                "signal": decision.get("ml_signal", ""),
                "confidence": decision.get("confidence", 0),
                "market_score": decision.get("market_score", 0),
                "model_version": decision.get("model_version", "unknown"),
                "regime": str(regime_result.get("regime", "UNKNOWN")),
                "trend": str(trend_result.get("trend", "")),
                "volatility": str(trend_result.get("volatility", "")),
                "rsi": feature_summary.get("indicators", {}).get("rsi", 50),
                "macd": str(feature_summary.get("indicators", {}).get("macd", "")),
                "adx": feature_summary.get("indicators", {}).get("adx", 25),
                "entry_price": price,
                "exit_price": 0,
                "pnl": 0.0,
                "commission": pos.commission,
                "swap": 0.0,
                "exit_reason": "",
                "entry_time": current_time,
                "close_time": None,
                "holding_time_minutes": 0,
                "atr_entry": atr,
                "spread": spread,
                "rr_ratio": rr1,
                "tp1": tp1_price,
                "tp2": tp2_price,
                "sl_price": sl_price,
                "magic": pos.magic,
                "multi_tf_agreement": 0,
                "skill_score": pair_skill_score or 0,
                "trade_quality_score": 0.0,
                "feature_vector": feat_vec,
            }
            pos.comment_data = entry_record

        return pos

    def _extract_multi_tf_trends(self, row: pd.Series) -> Dict[str, int]:
        trends = {}
        for col in ["trend240", "trend60", "trend30", "trend15"]:
            if col in row.index:
                val = row.get(col)
                try:
                    trends[col] = int(val) if pd.notna(val) else 0
                except (ValueError, TypeError):
                    trends[col] = 0
        return trends

    def _get_last_price(self) -> float:
        if self.replay._feat_data is not None and len(self.replay._feat_data) > 0:
            return float(self.replay._feat_data.iloc[-1].get("close", 0))
        return 0.0

    def _build_result(self) -> Dict[str, Any]:
        from simulation.learning_engine import LearningEngine
        from simulation.performance_tracker import PerformanceTracker

        trades = self.journal.trades

        all_trade_records = []
        for pos in self.account.closed_positions:
            cd = getattr(pos, "comment_data", None) or {}
            holding = 0
            if pos.open_time and pos.close_time:
                holding = (pos.close_time - pos.open_time).total_seconds() / 60.0
            record = {
                **cd,
                "exit_price": pos.close_price or 0,
                "pnl": pos.pnl,
                "commission": pos.commission,
                "swap": pos.swap,
                "net_pnl": pos.pnl - pos.commission - pos.swap,
                "exit_reason": pos.exit_reason or "",
                "close_time": pos.close_time,
                "holding_time_minutes": holding,
            }
            all_trade_records.append(record)

        for t in trades:
            t.setdefault("net_pnl", t.get("pnl", 0) - t.get("commission", 0) - t.get("swap", 0))

        all_trades = all_trade_records + trades

        tracker = PerformanceTracker()
        stats = tracker.compute(all_trades, self.initial_balance)

        engine = LearningEngine()
        learning = engine.analyze(all_trades, self.initial_balance)

        return {
            "symbol": self.symbol,
            "from_date": self.from_date,
            "to_date": self.to_date,
            "initial_balance": self.initial_balance,
            "final_balance": round(self.account.balance, 2),
            "total_candles_processed": self._total_candles_processed,
            "signals_generated": self._signals_generated,
            "trades_opened": self._trades_opened,
            "trades_closed": self._trades_closed,
            "total_trades": len(all_trades),
            "return_pct": stats.get("return_pct", 0),
            "stats": stats,
            "learning": learning,
            "trades": all_trades,
        }

    def reset(self):
        self.account.reset()
        self.journal.reset()
        self._total_candles_processed = 0
        self._signals_generated = 0
        self._trades_opened = 0
        self._trades_closed = 0
