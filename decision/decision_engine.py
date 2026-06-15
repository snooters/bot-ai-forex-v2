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
        self._regime_classifier = None

    def _init_regime_classifier(self):
        if self._regime_classifier is not None:
            return
        try:
            from analysis.regime_classifier import RegimeClassifier
            self._regime_classifier = RegimeClassifier()
        except Exception as e:
            self.logger.debug(f"RegimeClassifier not available: {e}")

    def _classify_regime(self, df) -> Optional[Dict]:
        self._init_regime_classifier()
        if self._regime_classifier is None:
            return None
        try:
            return self._regime_classifier.classify(df)
        except Exception as e:
            self.logger.warning(f"Regime classification failed: {e}")
            return None

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
        pair_skill_score: Optional[float] = None,
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

            new_regime = self._classify_regime(df)
            if new_regime is not None and new_regime.get("confidence", 0) > 0.5:
                # Only override caller's regime if classifier has meaningful confidence
                # (>0.5 avoids replacing a pre-classified regime with fallback RANGING)
                regime_result = new_regime
                decision["regime_classifier"] = new_regime.get("regime")

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
                pair_skill_score=pair_skill_score,
            )
            decision["confidence"] = confidence

            balance = (account_info or {}).get("balance", 0)
            trend_dir = trend_result.get("direction", "SIDEWAYS")
            ml_signal_dir = ml_signal.get("signal", "HOLD")

            no_trade_severity = self.no_trade_engine.should_no_trade(
                confidence=confidence,
                market_score=market_score,
                spread=spread,
                news_analysis=news_analysis,
                regime_result=regime_result,
                existing_positions=positions,
                balance=balance,
                trend_result=trend_result,
                signal=ml_signal_dir,
            )
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
            min_conf = max(0.50, dynamic_min)

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
                intel_base = market_score / 100.0
                buy_boost = intel_base
                sell_boost = 1.0 - intel_base
                combined_buy = (
                    ml_signal.get("buy_prob", 0) * ML_WEIGHT +
                    buy_boost * INTELLIGENCE_WEIGHT
                )
                combined_sell = (
                    ml_signal.get("sell_prob", 0) * ML_WEIGHT +
                    sell_boost * INTELLIGENCE_WEIGHT
                )

                counter_trend_min_conf = config.ai_filter.get("counter_trade_min_confidence", 0.60)

                if combined_buy > combined_sell:
                    # ── Counter-trade filter: BUY when M5 trend is bearish (non-STRONG) ──
                    # STRONG_BEARISH skipped here — handled by trend override later
                    if trend_dir in ("BEARISH", "WEAK_BEARISH") and confidence < counter_trend_min_conf:
                        decision["action"] = TradeDirection.HOLD.value
                        decision["no_trade"] = True
                        decision["no_trade_reasons"].append(
                            f"Counter-trade BUY blocked: trend={trend_dir}, conf={confidence:.0%} < {counter_trend_min_conf:.0%}"
                        )
                        decision["reasons"].append(f"Counter-trade BUY needs ≥{counter_trend_min_conf:.0%} conf (got {confidence:.0%})")
                    elif self._validate_entry(direction="BUY", confidence=confidence,
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

                elif combined_sell > combined_buy:
                    # ── Counter-trade filter: SELL when M5 trend is bullish (non-STRONG) ──
                    # STRONG_BULLISH skipped here — handled by trend override later
                    if trend_dir in ("BULLISH", "WEAK_BULLISH") and confidence < counter_trend_min_conf:
                        decision["action"] = TradeDirection.HOLD.value
                        decision["no_trade"] = True
                        decision["no_trade_reasons"].append(
                            f"Counter-trade SELL blocked: trend={trend_dir}, conf={confidence:.0%} < {counter_trend_min_conf:.0%}"
                        )
                        decision["reasons"].append(f"Counter-trade SELL needs ≥{counter_trend_min_conf:.0%} conf (got {confidence:.0%})")
                    elif self._validate_entry(direction="SELL", confidence=confidence,
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
                    decision["no_trade_reasons"].append("Combined signal equal - HOLD")

            # ── Trend-based direction override (ML-gated) ──
            if not decision["no_trade"]:
                trend_dir = trend_result.get("direction", "")
                current_action = decision.get("action", TradeDirection.HOLD.value)
                ml_buy_prob = ml_signal.get("buy_prob", 0)
                ml_sell_prob = ml_signal.get("sell_prob", 0)

                if trend_dir == TrendDirection.STRONG_BULLISH.value:
                    if current_action in (TradeDirection.HOLD.value, TradeDirection.SELL.value):
                        # Don't override if ML confidently disagrees
                        if ml_sell_prob < 0.50:
                            decision["action"] = TradeDirection.BUY.value
                            decision["reasons"].append("Trend override: STRONG_BULLISH -> BUY")
                        else:
                            decision["reasons"].append(
                                f"Trend override suppressed: ML sell_prob={ml_sell_prob:.2f}"
                            )
                elif trend_dir == TrendDirection.STRONG_BEARISH.value:
                    if current_action in (TradeDirection.HOLD.value, TradeDirection.BUY.value):
                        # Don't override if ML confidently disagrees
                        if ml_buy_prob < 0.50:
                            decision["action"] = TradeDirection.SELL.value
                            decision["reasons"].append("Trend override: STRONG_BEARISH -> SELL")
                        else:
                            decision["reasons"].append(
                                f"Trend override suppressed: ML buy_prob={ml_buy_prob:.2f}"
                            )

            # ── RSI + MACD safety override ──
            if not decision["no_trade"]:
                ind = feature_summary.get("indicators", {})
                rsi_val = ind.get("rsi", 50)
                macd_val = ind.get("macd", 0)
                macd_sig = ind.get("macd_signal", 0)
                current_action = decision.get("action", TradeDirection.HOLD.value)

                # Overbought: RSI > 70 → no BUY
                if rsi_val > 70 and current_action in (TradeDirection.BUY.value, "WEAK_BUY"):
                    decision["action"] = TradeDirection.HOLD.value
                    decision["no_trade"] = True
                    decision["no_trade_reasons"].append(
                        f"Safety: RSI={rsi_val:.1f} > 70 overbought, no BUY"
                    )
                    decision["reasons"].append("RSI overbought safety override")
                # Oversold: RSI < 30 → no SELL
                elif rsi_val < 30 and current_action in (TradeDirection.SELL.value, "WEAK_SELL"):
                    decision["action"] = TradeDirection.HOLD.value
                    decision["no_trade"] = True
                    decision["no_trade_reasons"].append(
                        f"Safety: RSI={rsi_val:.1f} < 30 oversold, no SELL"
                    )
                    decision["reasons"].append("RSI oversold safety override")
                # MACD-confirmed extreme: keep existing combined checks
                elif rsi_val < 30 and macd_val < macd_sig:
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
            # H4, H1, M30, M15 = context TFs
            # M5  = Entry (primary)
            # Require >= 3 of 4 context TFs to agree with signal direction
            if multi_tf_trends and not decision["no_trade"]:
                h4 = multi_tf_trends.get("trend240", 0)
                h1 = multi_tf_trends.get("trend60", 0)
                m30 = multi_tf_trends.get("trend30", 0)
                m15 = multi_tf_trends.get("trend15", 0)
                current_action = decision.get("action", TradeDirection.HOLD.value)

                tf_values = [("H4", h4), ("H1", h1), ("M30", m30), ("M15", m15)]

                if current_action in (TradeDirection.BUY.value, "WEAK_BUY"):
                    agree = sum(1 for _, v in tf_values if v > 0)
                    disagree = sum(1 for _, v in tf_values if v < 0)
                    if agree < 3 and disagree >= 2:
                        decision["action"] = TradeDirection.HOLD.value
                        decision["no_trade"] = True
                        detail = " ".join(f"{k}={v:+d}" for k, v in tf_values)
                        decision["no_trade_reasons"].append(
                            f"MTF filter: BUY blocked (agree={agree}/4, disagree={disagree})"
                        )
                        decision["reasons"].append(f"MTF filter: BUY->HOLD ({detail})")
                    elif agree < 3:
                        decision["reasons"].append(
                            f"MTF note: only {agree}/4 TFs bullish"
                        )

                elif current_action in (TradeDirection.SELL.value, "WEAK_SELL"):
                    agree = sum(1 for _, v in tf_values if v < 0)
                    disagree = sum(1 for _, v in tf_values if v > 0)
                    if agree < 3 and disagree >= 2:
                        decision["action"] = TradeDirection.HOLD.value
                        decision["no_trade"] = True
                        detail = " ".join(f"{k}={v:+d}" for k, v in tf_values)
                        decision["no_trade_reasons"].append(
                            f"MTF filter: SELL blocked (agree={agree}/4, disagree={disagree})"
                        )
                        decision["reasons"].append(f"MTF filter: SELL->HOLD ({detail})")
                    elif agree < 3:
                        decision["reasons"].append(
                            f"MTF note: only {agree}/4 TFs bearish"
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
        if "atr" in df.columns and not df["atr"].empty:
            atr = df["atr"].iloc[-1]
            if atr > 0:
                atr_pct = atr / df["close"].iloc[-1]
                if atr_pct > config.risk.get("max_atr_pct", 0.02):
                    return False

        # ── Candle confirmation: prevent fake breakouts ──
        # Require the last 2 COMPLETED candles to show momentum in signal direction.
        # NOTE: df.iloc[-1] is the CURRENT (possibly incomplete) candle in LIVE mode
        # because MT5 copy_rates_from_pos(start_pos=0) includes the forming candle.
        # We use df.iloc[-3] (signal) and df.iloc[-2] (confirm) which are always completed.
        if len(df) >= 4:
            signal_candle = df.iloc[-3]
            confirm_candle = df.iloc[-2]
            sig_close = float(signal_candle.get("close", 0))
            con_close = float(confirm_candle.get("close", 0))
            con_open = float(confirm_candle.get("open", 0))

            if direction == "BUY":
                # Confirm candle must be bullish AND close higher than signal close
                if con_close <= con_open:
                    return False  # Bearish candle — does not confirm
                if con_close <= sig_close:
                    return False  # No upward momentum
            elif direction == "SELL":
                # Confirm candle must be bearish AND close lower than signal close
                if con_close >= con_open:
                    return False  # Bullish candle — does not confirm
                if con_close >= sig_close:
                    return False  # No downward momentum

        return True

    @property
    def last_decision(self) -> Optional[Dict]:
        return self._last_decision
