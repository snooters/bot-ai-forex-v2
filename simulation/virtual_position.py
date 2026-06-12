from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class VirtualPosition:
    pair: str
    side: str
    open_time: datetime
    open_price: float
    volume: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    position_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    comment: str = ""
    trailing_activated: bool = False
    trailing_stop_price: Optional[float] = None
    atr_entry: Optional[float] = None
    close_time: Optional[datetime] = None
    close_price: Optional[float] = None
    pnl: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    exit_reason: Optional[str] = None
    magic: int = 0

    def calculate_pnl(self, current_price: float) -> float:
        if self.side == "BUY":
            return (current_price - self.open_price) * self.volume * 100000
        else:
            return (self.open_price - current_price) * self.volume * 100000

    def check_sl_hit(self, high: float, low: float) -> bool:
        if self.sl is None:
            return False
        if self.side == "BUY":
            return low <= self.sl
        else:
            return high >= self.sl

    def check_tp_hit(self, high: float, low: float) -> bool:
        if self.tp is None:
            return False
        if self.side == "BUY":
            return high >= self.tp
        else:
            return low <= self.tp

    def update_trailing(self, current_price: float, atr: float, atr_multiplier: float = 1.5) -> None:
        if not self.trailing_activated:
            return
        if self.side == "BUY":
            new_stop = current_price - atr * atr_multiplier
            if self.trailing_stop_price is None or new_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_stop
                self.sl = new_stop
        else:
            new_stop = current_price + atr * atr_multiplier
            if self.trailing_stop_price is None or new_stop < self.trailing_stop_price:
                self.trailing_stop_price = new_stop
                self.sl = new_stop

    def check_trailing_hit(self, high: float, low: float) -> bool:
        if self.trailing_stop_price is None:
            return False
        if self.side == "BUY":
            return low <= self.trailing_stop_price
        else:
            return high >= self.trailing_stop_price

    def check_time_exit(self, max_hold_hours: float, current_time: datetime) -> bool:
        if self.open_time is None:
            return False
        elapsed = (current_time - self.open_time).total_seconds() / 3600
        return elapsed >= max_hold_hours

    def close(self, close_price: float, exit_reason: str, current_time: Optional[datetime] = None):
        self.close_price = close_price
        self.exit_reason = exit_reason
        self.close_time = current_time or datetime.now()
        self.pnl = self.calculate_pnl(close_price)
