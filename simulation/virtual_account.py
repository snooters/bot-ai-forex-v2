from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from simulation.virtual_position import VirtualPosition


class VirtualAccount:
    def __init__(self, initial_balance: float = 10000.0, leverage: int = 100):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.positions: List[VirtualPosition] = []
        self._next_magic: int = 1001
        self.closed_positions: List[VirtualPosition] = []

    @property
    def equity(self) -> float:
        return self.balance + self.floating_pnl

    @property
    def floating_pnl(self) -> float:
        return sum(
            p.calculate_pnl(p.open_price)
            for p in self.positions
            if p.close_price is None
        )

    @property
    def used_margin(self) -> float:
        total = 0.0
        for p in self.positions:
            if p.close_price is not None:
                continue
            margin = (p.volume * 100000 * p.open_price) / self.leverage
            total += margin
        return total

    @property
    def free_margin(self) -> float:
        return self.equity - self.used_margin

    @property
    def margin_level(self) -> float:
        if self.used_margin == 0:
            return 0.0
        return (self.equity / self.used_margin) * 100

    @property
    def total_positions(self) -> int:
        return len([p for p in self.positions if p.close_price is None])

    def can_open_position(self, volume: float, price: float) -> Tuple[bool, str]:
        required_margin = (volume * 100000 * price) / self.leverage
        if required_margin > self.free_margin:
            return False, "Insufficient margin"
        if self.total_positions >= 50:
            return False, "Max positions reached"
        return True, "OK"

    def open_position(
        self,
        pair: str,
        side: str,
        price: float,
        volume: float,
        current_time: datetime,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        atr_entry: Optional[float] = None,
        comment: str = "",
        trailing_activated: bool = False,
    ) -> Optional[VirtualPosition]:
        ok, msg = self.can_open_position(volume, price)
        if not ok:
            return None

        pos = VirtualPosition(
            pair=pair,
            side=side,
            open_time=current_time,
            open_price=price,
            volume=volume,
            sl=sl,
            tp=tp,
            atr_entry=atr_entry,
            comment=comment,
            trailing_activated=trailing_activated,
            magic=self._next_magic,
        )
        self._next_magic += 1
        self.positions.append(pos)
        commission = self._calc_commission(volume, price)
        self.balance -= commission
        pos.commission = commission
        return pos

    def close_position(
        self,
        position_id: str,
        close_price: float,
        exit_reason: str,
        current_time: datetime,
    ) -> Optional[Dict]:
        for p in self.positions:
            if p.position_id == position_id and p.close_price is None:
                pnl = p.calculate_pnl(close_price)
                p.close(close_price, exit_reason, current_time)
                p.pnl = pnl
                swap = self._calc_swap(p, current_time)
                p.swap = swap
                self.balance += pnl - swap
                self.closed_positions.append(p)
                return {
                    "position_id": p.position_id,
                    "pair": p.pair,
                    "side": p.side,
                    "pnl": pnl,
                    "commission": p.commission,
                    "swap": swap,
                    "exit_reason": exit_reason,
                    "open_time": p.open_time,
                    "close_time": current_time,
                    "open_price": p.open_price,
                    "close_price": close_price,
                    "volume": p.volume,
                }
        return None

    def close_all_positions(self, price: float, reason: str, current_time: datetime) -> List[Dict]:
        results = []
        for p in list(self.positions):
            if p.close_price is None:
                r = self.close_position(p.position_id, price, reason, current_time)
                if r:
                    results.append(r)
        return results

    def update_floating(self, current_prices: Dict[str, float]) -> None:
        pass

    def check_sl_tp_all(self, high: float, low: float, current_time: datetime) -> List[Dict]:
        closed = []
        for p in list(self.positions):
            if p.close_price is not None:
                continue
            if p.check_tp_hit(high, low):
                r = self.close_position(p.position_id, p.tp, "TP_HIT", current_time)
                if r:
                    closed.append(r)
            elif p.check_sl_hit(high, low):
                r = self.close_position(p.position_id, p.sl, "SL_HIT", current_time)
                if r:
                    closed.append(r)
            elif p.check_trailing_hit(high, low):
                r = self.close_position(p.position_id, p.trailing_stop_price, "TRAILING_HIT", current_time)
                if r:
                    closed.append(r)
        return closed

    def apply_trailing_all(self, current_price: float, atr: float, atr_multiplier: float = 1.5) -> None:
        for p in self.positions:
            if p.close_price is None:
                p.update_trailing(current_price, atr, atr_multiplier)

    def check_time_exit_all(self, max_hold_hours: float, current_time: datetime) -> List[Dict]:
        closed = []
        for p in list(self.positions):
            if p.close_price is None and p.check_time_exit(max_hold_hours, current_time):
                r = self.close_position(p.position_id, p.open_price, "TIME_EXIT", current_time)
                if r:
                    closed.append(r)
        return closed

    def _calc_commission(self, volume: float, price: float) -> float:
        return volume * 100000 * price * 0.00006

    def _calc_swap(self, position: VirtualPosition, current_time: datetime) -> float:
        if position.open_time is None:
            return 0.0
        days_held = (current_time - position.open_time).days
        overnight_swaps = max(0, days_held)
        return overnight_swaps * position.volume * 0.5

    def get_position_by_id(self, position_id: str) -> Optional[VirtualPosition]:
        for p in self.positions:
            if p.position_id == position_id:
                return p
        return None

    def reset(self):
        self.balance = self.initial_balance
        self.positions.clear()
        self.closed_positions.clear()
        self._next_magic = 1001
