from typing import Dict, Optional, Tuple, List

from core.config import config
from core.constants import EmergencyLevel
from core.exceptions import RiskError, RiskLimitExceeded, EmergencyStop
from risk.account_monitor import AccountMonitor
from risk.position_sizing import PositionSizer
from utils.logger import get_logger


class RiskManager:
    def __init__(self):
        self.logger = get_logger("risk_manager")
        self.account_monitor = AccountMonitor()
        self.position_sizer = PositionSizer()
        self._emergency_mode = False
        self._emergency_reasons: List[str] = []

    def initialize(self, balance: float):
        self.account_monitor.initialize(balance)
        self.logger.info(f"Risk Manager initialized with balance ${balance:.2f}")

    def evaluate_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        atr: float,
        account_info: Dict,
        existing_positions: List,
    ) -> Dict:
        result = {
            "allowed": False,
            "lot_size": 0.0,
            "risk_amount": 0.0,
            "risk_pct": 0.0,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "sl_pips": 0,
            "tp_pips": 0,
            "rr_ratio": 0.0,
            "reasons": [],
            "emergency_level": EmergencyLevel.NORMAL.value,
        }

        account_status = self.account_monitor.get_account_status(account_info)
        result["emergency_level"] = account_status.get("emergency_level", EmergencyLevel.NORMAL.value)

        if self.account_monitor.is_trading_paused():
            result["reasons"].append("Emergency: trading paused - critical drawdown")
            self.logger.warning("EMERGENCY: Trading paused by account monitor")
            return result

        violations = self.account_monitor.check_limits(account_status)
        if violations:
            result["reasons"].extend(violations)
            self.logger.warning(f"Risk check failed: {'; '.join(violations)}")
            return result

        balance = account_status["balance"]
        sl_pips = self.position_sizer.calculate_pip_distance(entry_price, stop_loss)
        tp_pips = self.position_sizer.calculate_pip_distance(entry_price, take_profit)

        if sl_pips <= 0:
            result["reasons"].append("Invalid stop loss distance")
            return result

        rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 0
        min_rr = config.risk["rr_ratio"]
        if rr_ratio < min_rr:
            result["reasons"].append(f"Risk/reward {rr_ratio:.1f} below minimum {min_rr:.1f}")
            return result

        pip_val = 0.01 if "JPY" in symbol.upper() else 0.0001

        aggr_mult = self.account_monitor.get_aggressiveness_multiplier()
        vol_mult = 1.0

        risk_pct = config.risk["max_risk_pct"] * aggr_mult

        lot_size = self.position_sizer.calculate_lot_size(
            balance=balance,
            risk_pct=risk_pct,
            stop_loss_pips=sl_pips,
            pip_val=pip_val,
            leverage=account_info.get("leverage", 100),
            volatility_multiplier=vol_mult,
            aggressiveness_mult=aggr_mult,
        )

        risk_amount = balance * risk_pct
        risk_pct_actual = min(risk_pct * vol_mult, config.risk["max_risk_pct"] * 2 * aggr_mult)

        self.account_monitor.record_trade()

        result["allowed"] = True
        result["lot_size"] = lot_size
        result["risk_amount"] = risk_amount
        result["risk_pct"] = risk_pct_actual * 100
        result["sl_pips"] = sl_pips
        result["tp_pips"] = tp_pips
        result["rr_ratio"] = rr_ratio
        result["aggressiveness_mult"] = aggr_mult
        result["account_status"] = account_status

        self.logger.info(
            f"Trade validated: {direction} {lot_size} {symbol}, "
            f"SL={sl_pips:.0f}pips, RR={rr_ratio:.1f}, "
            f"risk={risk_pct_actual*100:.2f}%, "
            f"aggr={aggr_mult:.2f}x"
        )
        return result

    def update_account(self, account_info: Dict):
        self.account_monitor.update(
            account_info.get("balance", 0),
            account_info.get("equity", 0),
        )

    def is_trading_allowed(self, account_info: Dict) -> Tuple[bool, List[str]]:
        status = self.account_monitor.get_account_status(account_info)
        violations = self.account_monitor.check_limits(status)

        if self.account_monitor.is_trading_paused():
            violations.append("Trading paused by emergency mode")

        return len(violations) == 0, violations

    def get_emergency_status(self) -> Dict:
        return {
            "level": self.account_monitor.get_emergency_level().value,
            "trading_paused": self.account_monitor.is_trading_paused(),
            "aggressiveness_mult": self.account_monitor.get_aggressiveness_multiplier(),
        }

    def check_emergency(self, account_info: Dict) -> Optional[Dict]:
        status = self.account_monitor.get_account_status(account_info)
        level = self.account_monitor.get_emergency_level()

        if level == EmergencyLevel.NORMAL:
            return None

        alert = {
            "level": level.value,
            "drawdown": status.get("current_drawdown", 0),
            "daily_dd": status.get("daily_drawdown", 0),
            "balance": status.get("balance", 0),
            "equity": status.get("equity", 0),
            "action": "Monitoring",
        }

        if level == EmergencyLevel.CAUTION:
            alert["action"] = "Reducing position size by 50%"
        elif level == EmergencyLevel.DANGER:
            alert["action"] = "Reducing position size by 75%"
        elif level == EmergencyLevel.CRITICAL:
            alert["action"] = "STOP ALL TRADING - Close positions recommended"

        return alert
