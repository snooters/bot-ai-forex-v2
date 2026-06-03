from typing import Dict, Optional, Tuple

from core.config import config
from core.constants import ML_WEIGHT, INTELLIGENCE_WEIGHT, TradeDirection, Timeframe
from core.exceptions import DecisionError
from decision.confidence_calculator import ConfidenceCalculator
from decision.no_trade_engine import NoTradeEngine
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
        }

        try:
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

            no_trade_severity = self.no_trade_engine.should_no_trade(
                confidence=confidence,
                market_score=market_score,
                spread=spread,
                news_analysis=news_analysis,
                regime_result=regime_result,
                existing_positions=positions,
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
                decision["reasons"].append(f"Blocked by critical filter")
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

            min_conf = config.ai_filter["min_confidence"] * 100

            if confidence >= min_conf and confidence < 70:
                trade_type = "WEAK_SIGNAL"
            elif confidence >= 70:
                trade_type = "STRONG_SIGNAL"
            else:
                trade_type = "NO_SIGNAL"

            if trade_type == "NO_SIGNAL":
                direction_bias = self._get_direction_bias(
                    ml_signal_dir, buy_prob, sell_prob, trend_dir
                )
                if direction_bias:
                    decision["action"] = f"WEAK_{direction_bias}"
                    decision["no_trade"] = False
                    decision["reasons"].append(
                        f"Weak {direction_bias} bias (confidence={confidence:.1f}%, score={market_score})"
                    )
                else:
                    decision["action"] = TradeDirection.HOLD.value
                    decision["no_trade"] = True
                    decision["no_trade_reasons"].append(
                        f"Below threshold: conf={confidence:.1f}% < {min_conf:.0f}%"
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
                        label = "STRONG" if confidence >= 70 else "WEAK"
                        decision["reasons"].append(f"{label} BUY (conf={confidence:.1f}%, score={market_score})")
                    else:
                        direction_bias = self._get_direction_bias(
                            ml_signal_dir, buy_prob, sell_prob, trend_dir
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
                        label = "STRONG" if confidence >= 70 else "WEAK"
                        decision["reasons"].append(f"{label} SELL (conf={confidence:.1f}%, score={market_score})")
                    else:
                        direction_bias = self._get_direction_bias(
                            ml_signal_dir, buy_prob, sell_prob, trend_dir
                        )
                        decision["action"] = f"WEAK_{direction_bias}" if direction_bias else TradeDirection.HOLD.value
                        decision["no_trade"] = False if direction_bias else True
                        if direction_bias:
                            decision["reasons"].append(f"SELL validation failed, weak {direction_bias} bias")
                else:
                    decision["action"] = TradeDirection.HOLD.value
                    decision["no_trade"] = True
                    decision["no_trade_reasons"].append("Combined signal below min confidence")

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
    ) -> Optional[str]:
        if ml_signal_dir in ("BUY", "SELL"):
            return ml_signal_dir
        margin = abs(buy_prob - sell_prob)
        if margin < 5:
            return None
        dominant = "BUY" if buy_prob > sell_prob else "SELL"
        if "BULLISH" in trend_dir and dominant == "BUY":
            return dominant
        if "BEARISH" in trend_dir and dominant == "SELL":
            return dominant
        if margin > 10:
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
        min_conf = config.ai_filter["min_confidence"] * 100
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
