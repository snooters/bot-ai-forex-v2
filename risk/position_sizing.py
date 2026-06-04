import math
from typing import Dict, Optional, Tuple

from core.config import config
from utils.logger import get_logger


class PositionSizer:
    BALANCE_TIERS = [
        (0, 200, 0.0025, 0.01),
        (200, 500, 0.0035, 0.02),
        (500, 2000, 0.0050, 0.05),
        (2000, 5000, 0.0050, 0.20),
        (5000, float("inf"), 0.0050, 0.50),
    ]

    def __init__(self):
        self.logger = get_logger("position_sizer")

    def get_balance_tier_risk_pct(self, balance: float) -> float:
        for lo, hi, risk, _ in self.BALANCE_TIERS:
            if lo <= balance < hi:
                return risk
        return 0.005

    def get_balance_tier_max_lot(self, balance: float) -> float:
        for lo, hi, _, max_lot in self.BALANCE_TIERS:
            if lo <= balance < hi:
                return max_lot
        return 0.50

    def calculate_lot_size(
        self,
        balance: float,
        risk_pct: float,
        stop_loss_pips: float,
        pip_val: float = 0.0001,
        leverage: int = 100,
        volatility_multiplier: float = 1.0,
        aggressiveness_mult: float = 1.0,
    ) -> float:
        if stop_loss_pips <= 0 or pip_val <= 0:
            return 0.01

        tier_risk_pct = self.get_balance_tier_risk_pct(balance)
        effective_risk = min(risk_pct, tier_risk_pct)
        risk_amount = balance * effective_risk * volatility_multiplier * aggressiveness_mult
        max_risk = balance * config.risk["max_risk_pct"] * 2
        risk_amount = min(risk_amount, max_risk)

        contract_size = 100000
        lot_size = risk_amount / (stop_loss_pips * pip_val * (contract_size / 100000))

        max_lot = self.get_balance_tier_max_lot(balance)
        lot_size = min(lot_size, max_lot)

        lot_size = max(lot_size, 0.01)
        lot_size = round(lot_size, 2)

        self.logger.info(
            f"Position sizing: balance=${balance:.2f}, risk={effective_risk*100:.2f}%, "
            f"SL={stop_loss_pips:.0f}pips, mult={volatility_multiplier:.2f}x, "
            f"aggr={aggressiveness_mult:.2f}x, lot={lot_size} (max_lot={max_lot})"
        )
        return lot_size

    def calculate_atr_based_sl(
        self,
        atr: float,
        direction: str,
        entry_price: float,
        atr_multiplier: float = 1.5,
        pip_size: float = 0.0001,
    ) -> Tuple[float, float]:
        sl_distance = atr * atr_multiplier
        sl_pips = sl_distance / pip_size if pip_size > 0 else 0

        if direction == "BUY":
            stop_loss = entry_price - sl_distance
        else:
            stop_loss = entry_price + sl_distance

        return stop_loss, sl_pips

    def calculate_atr_based_tp(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str,
        rr_ratio: float = 2.0,
    ) -> float:
        sl_distance = abs(entry_price - stop_loss)
        tp_distance = sl_distance * rr_ratio

        if direction == "BUY":
            take_profit = entry_price + tp_distance
        else:
            take_profit = entry_price - tp_distance

        return take_profit

    def calculate_pip_distance(
        self,
        entry: float,
        stop_loss: float,
        pip_size: float = 0.0001,
    ) -> float:
        return abs(entry - stop_loss) / pip_size if pip_size > 0 else 0

    def adjust_for_volatility(
        self,
        lot_size: float,
        atr: float,
        avg_atr: float,
    ) -> float:
        if avg_atr <= 0:
            return lot_size
        vol_ratio = atr / avg_atr
        if vol_ratio > 1.2:
            lot_size *= 1.2 / vol_ratio
        elif vol_ratio < 0.8:
            lot_size *= min(1.3, 0.8 / vol_ratio)
        return round(max(lot_size, 0.01), 2)

    def calculate_position_value(self, lot_size: float, entry_price: float, contract_size: int = 100000) -> float:
        return lot_size * contract_size * entry_price

    def margin_required(self, lot_size: float, entry_price: float, leverage: int, contract_size: int = 100000) -> float:
        return (lot_size * contract_size * entry_price) / leverage if leverage > 0 else float("inf")
