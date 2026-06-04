from typing import Dict, Optional, Tuple

from core.config import config
from core.constants import ML_WEIGHT, INTELLIGENCE_WEIGHT, TrendDirection, TradeDirection, Timeframe
from core.exceptions import DecisionError
from decision.confidence_calculator import ConfidenceCalculator
from decision.no_trade_engine import NoTradeEngine
from decision.trend_reversal_detector import TrendReversalDetector
from intelligence.market_scorer import MarketScorer
from learning.trade_memory import TradeMemory
from ml.predictor import MLPredictor
from utils.logger import get_logger


class DecisionEngine:
    def __init__(
        self,
        ml_predictor: MLPredictor,
        market_scorer: MarketScorer,
        trade_memory: Optional[TradeMemory] = None,
    ):
        self.logger = get_logger("decision_engine")
        self.ml_predictor = ml_predictor
        self.market_scorer = market_scorer
        self.trade_memory = trade_memory
        self.confidence_calculator = ConfidenceCalculator()
        self.no_trade_engine = NoTradeEngine()
        self._last_decision: Optional[Dict] = None

    def make_decision(
        self,
        symbol: str,
        df_entry: Dict,
        trend_result: Dict,
        vol_result: Dict,
        momentum_result: Dict,
        regime_result: Dict,
        sr_info: Dict,
        feature_summary: Dict,
        account_info: Optional[Dict] = None,
        positions: Optional[list] = None,
        news_analysis: Optional[Dict] = None,
        llm_analysis: Optional[Dict] = None,
        spread: float = 0.0,
        timeframe: Optional[int] = None,
        consensus: Optional[Dict] = None,
        reversal_info: Optional[Dict] = None,
        multi_tf_trends: Optional[Dict] = None,
    ) -> Dict:
        entry_tfs = list(df_entry.keys()) if isinstance(df_entry, dict) else []
        entry_tf = timeframe or (entry_tfs[0] if entry_tfs else Timeframe.M15)
        df = df_entry[entry_tf] if isinstance(df_entry, dict) else df_entry

        decision = {
            "symbol": symbol,
            "action": TradeDirection.HOLD.value,
            "confidence": 0.0,
            "market_score": 0,
            "reasons": [],
            "no_trade": True,
            "no_trade_reasons": [],
            "ml_signal": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "position_size": 0,
            "timeframe": entry_tf,
            "reversal_force_close": False,
        }

        # ── Check trend reversal first ──
        if reversal_info and reversal_info.get("severity", 0) >= TrendReversalDetector.CRITICAL:
            decision["action"] = TradeDirection.HOLD.value
            decision["no_trade"] = True
            decision["no_trade_reasons"].append(f"Trend reversal: {reversal_info.get('reason', '')}")
            decision["reasons"].append(f"Blocked by reversal: {reversal_info.get('to_trend', '')}")
            decision["reversal_force_close"] = True
            self._last_decision = decision
            return decision

        try:
            # ── Use multi-TF consensus ML signal if available ──
            if consensus:
                ml_signal = {
                    "signal": consensus.get("signal", "HOLD"),
                    "confidence": consensus.get("confidence", 0),
                    "buy_prob": consensus.get("buy_prob", 0),
                    "sell_prob": consensus.get("sell_prob", 0),
                    "hold_prob": consensus.get("hold_prob", 0),
                }
            else:
                ml_signal = self.ml_predictor.get_buy_sell_hold(df, timeframe=entry_tf)
            decision["ml_signal"] = ml_signal

            market_score = self.market_scorer.compute_market_score(
                trend_result, vol_result, momentum_result,
                regime_result, sr_info, feature_summary
            )
            decision["market_score"] = market_score

            confidence = self.confidence_calculator.calculate_confidence(
                ml_signal=ml_signal,
                market_score=market_score,
                trend_result=trend_result,
                regime_result=regime_result,
                sr_info=sr_info,
                news_analysis=news_analysis,
                llm_analysis=llm_analysis,
            )
            decision["confidence"] = confidence

            balance = (account_info or {}).get("balance", 0)

            no_trade_severity = self.no_trade_engine.should_no_trade(
                confidence=confidence,
                market_score=market_score,
                spread=spread,
                news_analysis=news_analysis,
                regime_result=regime_result,
                existing_positions=positions,
                balance=balance,
                trend_result=trend_result,
            )

            trend_dir = trend_result.get("direction", "SIDEWAYS")
            ml_signal_dir = ml_signal.get("signal", "HOLD")
            buy_prob = ml_signal.get("buy_prob", 0)
            sell_prob = ml_signal.get("sell_prob", 0)
            hold_prob = ml_signal.get("hold_prob", 0)

            if no_trade_severity >= NoTradeEngine.CRITICAL:
                decision["no_trade"] = True
                decision["no_trade_reasons"] = self.no_trade_engine.reasons
                decision["action"] = TradeDirection.HOLD.value
                critical_reason = "; ".join(self.no_trade_engine.reasons)
                decision["reasons"].append(f"Blocked: {critical_reason}")
                self._last_decision = decision
                return decision

            memory_check = self._check_trade_memory(
                direction=ml_signal_dir,
                regime=regime_result.get("regime", ""),
                timeframe=Timeframe.LABELS.get(entry_tf, "M15"),
            )
            if memory_check.get("block"):
                decision["no_trade"] = True
                decision["no_trade_reasons"] = [memory_check["reason"]]
                decision["action"] = TradeDirection.HOLD.value
                decision["reasons"].append(f"Memory block: {memory_check['reason']}")
                self._last_decision = decision
                return decision

            reduce_size = memory_check.get("reduce_size", False)

            dynamic_min = config.get_dynamic_min_confidence(balance)
            min_conf = max(config.ai_filter["min_confidence"], dynamic_min)

            if confidence >= min_conf and confidence < 0.70:
                trade_type = "WEAK_SIGNAL"
            elif confidence >= 0.70:
                trade_type = "STRONG_SIGNAL"
            else:
                trade_type = "NO_SIGNAL"

            if trade_type == "NO_SIGNAL":
                direction_bias = self._get_direction_bias(
                    ml_signal_dir, buy_prob, sell_prob, trend_dir,
                    momentum_result=momentum_result,
                )
                if direction_bias:
                    decision["action"] = f"WEAK_{direction_bias}"
                    decision["no_trade"] = False
                    decision["reasons"].append(
                        f"Weak {direction_bias} bias (confidence={confidence:.1%}, score={market_score})"
                    )
                else:
                    decision["action"] = TradeDirection.HOLD.value
                    decision["no_trade"] = True
                    decision["no_trade_reasons"].append(
                        f"Below threshold: conf={confidence:.1%} < {min_conf:.0%})"
                    )
            else:
                intelligence_weight = INTELLIGENCE_WEIGHT * market_score / 100.0
                combined_buy = (
                    ml_signal.get("buy_prob", 0) * ML_WEIGHT +
                    intelligence_weight * 100 * INTELLIGENCE_WEIGHT
                )
                combined_sell = (
                    ml_signal.get("sell_prob", 0) * ML_WEIGHT +
                    intelligence_weight * 100 * INTELLIGENCE_WEIGHT
                )

                if combined_buy > combined_sell and combined_buy >= min_conf:
                    if self._validate_entry(direction="BUY", confidence=confidence,
                                             trend_result=trend_result, sr_info=sr_info, df=df):
                        decision["action"] = TradeDirection.BUY.value
                        decision["no_trade"] = False
                        label = "STRONG" if confidence >= 0.70 else "WEAK"
                        decision["reasons"].append(f"{label} BUY (conf={confidence:.0%}, score={market_score})")
                    else:
                        direction_bias = self._get_direction_bias(
                            ml_signal_dir, buy_prob, sell_prob, trend_dir,
                            momentum_result=momentum_result,
                        )
                        decision["action"] = f"WEAK_{direction_bias}" if direction_bias else TradeDirection.HOLD.value
                        decision["no_trade"] = False if direction_bias else True
                        if direction_bias:
                            decision["reasons"].append(f"BUY validation failed, weak {direction_bias} bias")

                elif combined_sell > combined_buy and combined_sell >= min_conf:
                    if self._validate_entry(direction="SELL", confidence=confidence,
                                             trend_result=trend_result, sr_info=sr_info, df=df):
                        decision["action"] = TradeDirection.SELL.value
                        decision["no_trade"] = False
                        label = "STRONG" if confidence >= 0.70 else "WEAK"
                        decision["reasons"].append(f"{label} SELL (conf={confidence:.0%}, score={market_score})")
                    else:
                        direction_bias = self._get_direction_bias(
                            ml_signal_dir, buy_prob, sell_prob, trend_dir,
                            momentum_result=momentum_result,
                        )
                        decision["action"] = f"WEAK_{direction_bias}" if direction_bias else TradeDirection.HOLD.value
                        decision["no_trade"] = False if direction_bias else True
                        if direction_bias:
                            decision["reasons"].append(f"SELL validation failed, weak {direction_bias} bias")
                else:
                    decision["action"] = TradeDirection.HOLD.value
                    decision["no_trade"] = True
                    decision["no_trade_reasons"].append("Combined signal below min confidence")

            # ── Trend-based direction override ──
            if not decision["no_trade"]:
                trend_dir = trend_result.get("direction", "")
                current_action = decision.get("action", TradeDirection.HOLD.value)
                if trend_dir == TrendDirection.STRONG_BULLISH.value:
                    if current_action in (TradeDirection.HOLD.value, TradeDirection.SELL.value):
                        decision["action"] = TradeDirection.BUY.value
                        decision["reasons"].append(f"Trend override: STRONG_BULLISH → BUY")
                elif trend_dir == TrendDirection.STRONG_BEARISH.value:
                    if current_action in (TradeDirection.HOLD.value, TradeDirection.BUY.value):
                        decision["action"] = TradeDirection.SELL.value
                        decision["reasons"].append(f"Trend override: STRONG_BEARISH → SELL")

            # ── RSI + MACD safety override ──
            if not decision["no_trade"]:
                ind = feature_summary.get("indicators", {})
                rsi_val = ind.get("rsi", 50)
                macd_val = ind.get("macd", 0)
                macd_sig = ind.get("macd_signal", 0)
                current_action = decision.get("action", TradeDirection.HOLD.value)
                if rsi_val < 30 and macd_val < macd_sig:
                    if current_action in (TradeDirection.BUY.value, "WEAK_BUY"):
                        decision["action"] = TradeDirection.HOLD.value
                        decision["no_trade"] = True
                        decision["no_trade_reasons"].append(
                            f"Safety: RSI={rsi_val:.1f} < 30 + MACD bearish, no BUY"
                        )
                        decision["reasons"].append("RSI+MACD safety override")
                elif rsi_val > 70 and macd_val > macd_sig:
                    if current_action in (TradeDirection.SELL.value, "WEAK_SELL"):
                        decision["action"] = TradeDirection.HOLD.value
                        decision["no_trade"] = True
                        decision["no_trade_reasons"].append(
                            f"Safety: RSI={rsi_val:.1f} > 70 + MACD bullish, no SELL"
                        )
                        decision["reasons"].append("RSI+MACD safety override")

            # ── Multi-TF trend alignment filter ──
            if multi_tf_trends and not decision["no_trade"]:
                h4 = multi_tf_trends.get("trend240", 0)
                h1 = multi_tf_trends.get("trend60", 0)
                m30 = multi_tf_trends.get("trend30", 0)
                current_action = decision.get("action", TradeDirection.HOLD.value)

                if current_action in (TradeDirection.BUY.value, "WEAK_BUY"):
                    if h4 < 0 or h1 < 0 or m30 < 0:
                        decision["action"] = TradeDirection.HOLD.value
                        decision["no_trade"] = True
                        violating = [k for k, v in [("H4",h4),("H1",h1),("M30",m30)] if v < 0]
                        decision["no_trade_reasons"].append(
                            f"MTF filter: BUY blocked by {'/'.join(violating)} trend"
                        )
                        decision["reasons"].append(
                            f"MTF filter: BUY→HOLD (H4={h4:+d} H1={h1:+d} M30={m30:+d})"
                        )

                elif current_action in (TradeDirection.SELL.value, "WEAK_SELL"):
                    if h4 > 0 or h1 > 0 or m30 > 0:
                        decision["action"] = TradeDirection.HOLD.value
                        decision["no_trade"] = True
                        violating = [k for k, v in [("H4",h4),("H1",h1),("M30",m30)] if v > 0]
                        decision["no_trade_reasons"].append(
                            f"MTF filter: SELL blocked by {'/'.join(violating)} trend"
                        )
                        decision["reasons"].append(
                            f"MTF filter: SELL→HOLD (H4={h4:+d} H1={h1:+d} M30={m30:+d})"
                        )

            # ── Trend alignment enforcement from multi-TF consensus ──
            if consensus and not decision["no_trade"]:
                if not consensus.get("trend_aligned", True):
                    decision["action"] = TradeDirection.HOLD.value
                    decision["no_trade"] = True
                    decision["no_trade_reasons"].append(
                        f"Against dominant trend: {consensus.get('dominant_trend', 'unknown')}"
                    )
                    decision["reasons"].append(
                        f"Consensus {consensus.get('signal','HOLD')} vs trend {consensus.get('dominant_trend','SIDEWAYS')}"
                    )

            # ── Reversal WARNING: reduce confidence ──
            if reversal_info and reversal_info.get("severity") == TrendReversalDetector.WARNING and not decision["no_trade"]:
                decision["confidence"] *= 0.7
                decision["reasons"].append(f"Trend warning: {reversal_info.get('reason', '')}")

            if no_trade_severity == NoTradeEngine.WARNING and not decision["no_trade"]:
                decision["reasons"].append(f"Warnings: {'; '.join(self.no_trade_engine.reasons)}")
                decision["no_trade_reasons"] = self.no_trade_engine.reasons

        except Exception as e:
            self.logger.error(f"Decision error: {e}")
            decision["action"] = TradeDirection.HOLD.value
            decision["no_trade"] = True
            decision["reasons"].append(f"Error: {e}")

        self._last_decision = decision
        return decision

    def _check_trade_memory(
        self,
        direction: str,
        regime: str,
        timeframe: str,
    ) -> Dict:
        if not self.trade_memory or direction in ("HOLD", ""):
            return {"block": False, "reduce_size": False}

        result = self.trade_memory.find_by_pattern(
            direction=direction,
            regime=regime,
            timeframe=timeframe,
            min_trades=3,
        )

        if result.get("closed", 0) < 3:
            return {"block": False, "reduce_size": False}

        win_rate = result.get("win_rate", 0)
        losses = result.get("losses", 0)

        if win_rate < 0.25 and losses >= 3:
            return {
                "block": True,
                "reduce_size": False,
                "reason": (
                    f"Pattern {direction}/{regime} win rate {win_rate:.0%} "
                    f"({result['closed']} trades, {losses} losses)"
                ),
            }

        if win_rate < 0.40 and losses >= 3:
            return {
                "block": False,
                "reduce_size": True,
                "reason": f"Pattern {direction}/{regime} win rate {win_rate:.0%} — reducing size",
            }

        if result.get("profit_factor", 0) < 0.5 and result.get("closed", 0) >= 5:
            return {
                "block": False,
                "reduce_size": True,
                "reason": f"PF={result['profit_factor']:.2f} — reducing size",
            }

        return {"block": False, "reduce_size": False}

    def _get_direction_bias(
        self,
        ml_signal_dir: str,
        buy_prob: float,
        sell_prob: float,
        trend_dir: str,
        momentum_result: Optional[Dict] = None,
    ) -> Optional[str]:
        if ml_signal_dir in ("BUY", "SELL"):
            return ml_signal_dir
        margin = abs(buy_prob - sell_prob)
        if margin < 0.05:
            # If sell_prob >= 20% and bearish momentum, allow SELL
            if sell_prob >= 0.20 and momentum_result:
                mom_dir = momentum_result.get("direction", 0)
                if mom_dir < 0:
                    return "SELL"
            if buy_prob >= 0.20 and momentum_result:
                mom_dir = momentum_result.get("direction", 0)
                if mom_dir > 0:
                    return "BUY"
            return None
        dominant = "BUY" if buy_prob > sell_prob else "SELL"
        if "BULLISH" in trend_dir and dominant == "BUY":
            return dominant
        if "BEARISH" in trend_dir and dominant == "SELL":
            return dominant
        if margin > 0.10:
            return dominant
        return None

    def _validate_entry(
        self,
        direction: str,
        confidence: float,
        trend_result: Dict,
        sr_info: Dict,
        df,
    ) -> bool:
        min_conf = config.ai_filter["min_confidence"]
        if confidence < min_conf * 0.8:
            return False

        if "atr" in df.columns and not df["atr"].empty:
            atr = df["atr"].iloc[-1]
            if atr > 0:
                atr_pct = atr / df["close"].iloc[-1]
                if atr_pct > 0.02:
                    return False

        return True

    @property
    def last_decision(self) -> Optional[Dict]:
        return self._last_decision
