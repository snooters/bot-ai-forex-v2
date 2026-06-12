from __future__ import annotations

import math
from typing import Dict, List, Optional


class PerformanceTracker:
    def compute(self, trades: List[Dict], initial_balance: float) -> Dict:
        total = len(trades)
        if total == 0:
            return self._empty_result(initial_balance)

        winning = [t for t in trades if t.get("net_pnl", 0) > 0]
        losing = [t for t in trades if t.get("net_pnl", 0) <= 0]

        win_count = len(winning)
        loss_count = len(losing)
        win_rate = win_count / total if total > 0 else 0.0
        loss_rate = loss_count / total if total > 0 else 0.0

        gross_profit = sum(t["net_pnl"] for t in winning) if winning else 0.0
        gross_loss = abs(sum(t["net_pnl"] for t in losing)) if losing else 0.0
        net_profit = gross_profit - gross_loss

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

        avg_win = gross_profit / win_count if win_count > 0 else 0.0
        avg_loss = gross_loss / loss_count if loss_count > 0 else 0.0

        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        pnls = [t.get("net_pnl", 0) for t in trades]
        sharpe = self._compute_sharpe(pnls)
        sortino = self._compute_sortino(pnls)

        max_dd, max_dd_pct = self._compute_max_drawdown(pnls, initial_balance)

        avg_rr = self._compute_avg_rr(trades)

        holding_times = [t.get("holding_time_minutes", 0) for t in trades if t.get("holding_time_minutes", 0) > 0]
        avg_holding = sum(holding_times) / len(holding_times) if holding_times else 0.0

        consec_wins, consec_losses = self._compute_consecutive(trades)

        best_trade = max(trades, key=lambda t: t.get("net_pnl", 0))
        worst_trade = min(trades, key=lambda t: t.get("net_pnl", 0))

        by_regime = self._group_stats(trades, "regime")
        by_confidence = self._group_stats_confidence(trades)
        by_hour = self._group_stats(trades, "entry_hour")
        by_side = self._group_stats(trades, "side")
        by_exit_reason = self._group_stats(trades, "exit_reason")
        by_pair = self._group_stats(trades, "pair")

        return {
            "total_trades": total,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": round(win_rate, 4),
            "loss_rate": round(loss_rate, 4),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_profit": round(net_profit, 2),
            "profit_factor": round(profit_factor, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy": round(expectancy, 2),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_dd, 2),
            "max_drawdown_pct": round(max_dd_pct, 4),
            "avg_rr": round(avg_rr, 4),
            "avg_holding_minutes": round(avg_holding, 2),
            "consecutive_wins": consec_wins,
            "consecutive_losses": consec_losses,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "by_regime": by_regime,
            "by_confidence": by_confidence,
            "by_hour": by_hour,
            "by_side": by_side,
            "by_exit_reason": by_exit_reason,
            "by_pair": by_pair,
            "return_pct": round(net_profit / initial_balance * 100, 4) if initial_balance > 0 else 0.0,
        }

    def _empty_result(self, initial_balance: float) -> Dict:
        return {
            "total_trades": 0, "win_count": 0, "loss_count": 0,
            "win_rate": 0.0, "loss_rate": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0, "net_profit": 0.0,
            "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "expectancy": 0.0, "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
            "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
            "avg_rr": 0.0, "avg_holding_minutes": 0.0,
            "consecutive_wins": 0, "consecutive_losses": 0,
            "best_trade": {}, "worst_trade": {},
            "by_regime": {}, "by_confidence": {}, "by_hour": {}, "by_side": {},
            "by_exit_reason": {}, "by_pair": {}, "return_pct": 0.0,
        }

    def _compute_sharpe(self, pnls: List[float]) -> float:
        if len(pnls) < 2:
            return 0.0
        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        if variance <= 0:
            return 0.0
        std = math.sqrt(variance)
        if std == 0:
            return 0.0
        return (mean_pnl / std) * math.sqrt(288)

    def _compute_sortino(self, pnls: List[float]) -> float:
        if len(pnls) < 2:
            return 0.0
        mean_pnl = sum(pnls) / len(pnls)
        downside = [p for p in pnls if p < 0]
        if not downside:
            return 0.0
        downside_var = sum((p - mean_pnl) ** 2 for p in downside) / len(pnls)
        if downside_var <= 0:
            return 0.0
        downside_std = math.sqrt(downside_var)
        if downside_std == 0:
            return 0.0
        return (mean_pnl / downside_std) * math.sqrt(288)

    def _compute_max_drawdown(self, pnls: List[float], initial_balance: float) -> tuple:
        balance = initial_balance
        peak = initial_balance
        max_dd = 0.0
        max_dd_pct = 0.0
        for p in pnls:
            balance += p
            if balance > peak:
                peak = balance
            dd = peak - balance
            dd_pct = dd / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        return max_dd, max_dd_pct

    def _compute_avg_rr(self, trades: List[Dict]) -> float:
        rrs = [t.get("rr_ratio", 0) for t in trades if t.get("rr_ratio", 0) > 0]
        return sum(rrs) / len(rrs) if rrs else 0.0

    def _compute_consecutive(self, trades: List[Dict]) -> tuple:
        max_wins = cur_wins = 0
        max_losses = cur_losses = 0
        for t in trades:
            if t.get("net_pnl", 0) > 0:
                cur_wins += 1
                cur_losses = 0
                if cur_wins > max_wins:
                    max_wins = cur_wins
            else:
                cur_losses += 1
                cur_wins = 0
                if cur_losses > max_losses:
                    max_losses = cur_losses
        return max_wins, max_losses

    def _group_stats(self, trades: List[Dict], key: str) -> Dict:
        groups: Dict[str, List[Dict]] = {}
        for t in trades:
            k = str(t.get(key, "UNKNOWN"))
            if k not in groups:
                groups[k] = []
            groups[k].append(t)
        result = {}
        for k, g in groups.items():
            total = len(g)
            wins = sum(1 for t in g if t.get("net_pnl", 0) > 0)
            gross_p = sum(t["net_pnl"] for t in g if t.get("net_pnl", 0) > 0)
            gross_l = abs(sum(t["net_pnl"] for t in g if t.get("net_pnl", 0) <= 0))
            result[k] = {
                "trades": total,
                "wins": wins,
                "win_rate": round(wins / total, 4) if total > 0 else 0,
                "profit_factor": round(gross_p / gross_l, 4) if gross_l > 0 else (gross_p if gross_p > 0 else 0),
                "net_pnl": round(gross_p - gross_l, 2),
            }
        return result

    def _group_stats_confidence(self, trades: List[Dict]) -> Dict:
        buckets: Dict[str, List[Dict]] = {}
        for t in trades:
            c = t.get("confidence", 0)
            label = f"{int(c * 100 // 20 * 20)}-{int(c * 100 // 20 * 20 + 20)}"
            if label not in buckets:
                buckets[label] = []
            buckets[label].append(t)
        result = {}
        for k, g in buckets.items():
            total = len(g)
            wins = sum(1 for t in g if t.get("net_pnl", 0) > 0)
            gross_p = sum(t["net_pnl"] for t in g if t.get("net_pnl", 0) > 0)
            gross_l = abs(sum(t["net_pnl"] for t in g if t.get("net_pnl", 0) <= 0))
            result[k] = {
                "trades": total,
                "wins": wins,
                "win_rate": round(wins / total, 4) if total > 0 else 0,
                "profit_factor": round(gross_p / gross_l, 4) if gross_l > 0 else (gross_p if gross_p > 0 else 0),
                "net_pnl": round(gross_p - gross_l, 2),
            }
        return result
