from datetime import datetime, date, timedelta
from typing import Dict, Optional, List

from core.config import config
from core.constants import EmergencyLevel
from core.exceptions import EmergencyStop
from utils.logger import get_logger


class AccountMonitor:
    def __init__(self):
        self.logger = get_logger("account_monitor")
        self._initial_balance: float = 0
        self._daily_initial_balance: float = 0
        self._weekly_initial_balance: float = 0
        self._monthly_initial_balance: float = 0
        self._current_date: Optional[date] = None
        self._peak_balance: float = 0
        self._peak_equity: float = 0
        self._daily_trades: int = 0
        self._weekly_trades: int = 0
        self._monthly_trades: int = 0
        self._emergency_level: EmergencyLevel = EmergencyLevel.NORMAL
        self._trading_paused = False

    def initialize(self, balance: float):
        self._initial_balance = balance
        self._daily_initial_balance = balance
        self._weekly_initial_balance = balance
        self._monthly_initial_balance = balance
        self._peak_balance = balance
        self._peak_equity = balance
        self._current_date = date.today()

    def update(self, balance: float, equity: float):
        today = date.today()

        if self._current_date != today:
            self._daily_initial_balance = balance
            self._daily_trades = 0
            self._current_date = today

            if today.weekday() == 0:
                self._weekly_initial_balance = balance
                self._weekly_trades = 0

            if today.day == 1:
                self._monthly_initial_balance = balance
                self._monthly_trades = 0

        if balance > self._peak_balance:
            self._peak_balance = balance
        if equity > self._peak_equity:
            self._peak_equity = equity

        self._check_emergency_level(balance, equity)

    def _check_emergency_level(self, balance: float, equity: float):
        peak = max(self._peak_balance, self._peak_equity)
        if peak <= 0:
            return

        total_dd = (peak - equity) / peak

        if total_dd >= 0.05:
            self._emergency_level = EmergencyLevel.CRITICAL
            self._trading_paused = True
        elif total_dd >= 0.04:
            self._emergency_level = EmergencyLevel.DANGER
        elif total_dd >= 0.03:
            self._emergency_level = EmergencyLevel.CAUTION
        else:
            self._emergency_level = EmergencyLevel.NORMAL
            self._trading_paused = False

    def get_account_status(self, account_info: Dict) -> Dict:
        balance = account_info.get("balance", 0)
        equity = account_info.get("equity", 0)
        margin = account_info.get("margin", 0)
        margin_free = account_info.get("margin_free", 0)
        margin_level = account_info.get("margin_level", 0)
        profit = account_info.get("profit", 0)

        if self._initial_balance == 0:
            self._initial_balance = balance
            self._peak_balance = balance
            self._peak_equity = equity

        current_dd = self._compute_drawdown(balance, equity)
        daily_dd = self._compute_drawdown(self._daily_initial_balance, equity)
        weekly_dd = self._compute_drawdown(self._weekly_initial_balance, equity)
        monthly_dd = self._compute_drawdown(self._monthly_initial_balance, equity)
        total_dd = self._compute_drawdown(self._initial_balance, equity)
        peak_dd = self._compute_drawdown(self._peak_balance, equity)

        return {
            "balance": balance,
            "equity": equity,
            "margin": margin,
            "free_margin": margin_free,
            "margin_level": margin_level,
            "floating_profit": profit,
            "floating_loss": min(profit, 0),
            "current_drawdown": current_dd,
            "daily_drawdown": daily_dd,
            "weekly_drawdown": weekly_dd,
            "monthly_drawdown": monthly_dd,
            "total_drawdown": total_dd,
            "peak_drawdown": peak_dd,
            "peak_balance": self._peak_balance,
            "daily_trades": self._daily_trades,
            "weekly_trades": self._weekly_trades,
            "monthly_trades": self._monthly_trades,
            "emergency_level": self._emergency_level.value,
            "trading_paused": self._trading_paused,
        }

    def _compute_drawdown(self, reference: float, current: float) -> float:
        if reference <= 0:
            return 0.0
        return (reference - current) / reference * 100

    def check_limits(self, status: Dict) -> List[str]:
        violations = []

        daily_dd = status.get("daily_drawdown", 0)
        max_daily = config.risk["max_daily_loss_pct"] * 100
        if daily_dd >= max_daily:
            violations.append(f"Daily drawdown limit: {daily_dd:.1f}% >= {max_daily:.1f}%")

        total_dd = status.get("total_drawdown", 0)
        hard_stop = 5.0
        if total_dd >= hard_stop:
            violations.append(f"Hard stop loss: {total_dd:.1f}% >= {hard_stop:.1f}%")

        margin_level = status.get("margin_level", 0)
        if margin_level > 0 and margin_level < 100:
            violations.append(f"Low margin level: {margin_level:.1f}%")

        return violations

    def is_daily_limit_exceeded(self, status: Dict) -> bool:
        return status.get("daily_drawdown", 0) >= config.risk["max_daily_loss_pct"] * 100

    def is_hard_stop_reached(self, status: Dict) -> bool:
        return status.get("total_drawdown", 0) >= 5.0

    def record_trade(self):
        self._daily_trades += 1
        self._weekly_trades += 1
        self._monthly_trades += 1

    def get_emergency_level(self) -> EmergencyLevel:
        return self._emergency_level

    def get_aggressiveness_multiplier(self) -> float:
        multipliers = {
            EmergencyLevel.NORMAL: 1.0,
            EmergencyLevel.CAUTION: 0.5,
            EmergencyLevel.DANGER: 0.25,
            EmergencyLevel.CRITICAL: 0.0,
        }
        return multipliers.get(self._emergency_level, 1.0)

    def is_trading_paused(self) -> bool:
        return self._trading_paused

    def reset_daily(self):
        self._daily_trades = 0
        self._daily_initial_balance = self._peak_balance

    def get_max_position_size(self, balance: float) -> float:
        mult = self.get_aggressiveness_multiplier()
        max_risk = balance * config.risk["max_risk_pct"] * 2 * mult
        return max_risk
