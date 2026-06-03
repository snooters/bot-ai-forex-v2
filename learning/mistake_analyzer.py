from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from pathlib import Path
import json
from collections import Counter, defaultdict

import numpy as np

from core.constants import TRADE_HISTORY_DIR
from utils.logger import get_logger


class MistakeAnalyzer:
    def __init__(self):
        self.logger = get_logger("mistake_analyzer")
        self._report_dir = Path(TRADE_HISTORY_DIR)
        self._report_dir.mkdir(parents=True, exist_ok=True)
        self._report_file = self._report_dir / "mistake_report.json"

    def analyze_losses(self, trades: List[Dict]) -> Dict:
        losses = [t for t in trades if t.get("result") == "LOSS"]
        all_closed = [t for t in trades if t.get("result") in ("WIN", "LOSS")]

        if not losses:
            report = self._empty_report()
            self._save_report(report)
            return report

        report = {
            "total_losses": len(losses),
            "total_trades": len(all_closed),
            "loss_rate": round(len(losses) / len(all_closed) * 100, 2) if all_closed else 0,
            "total_loss_amount": round(abs(sum(t.get("profit", 0) for t in losses)), 2),
            "avg_loss": round(abs(np.mean([t.get("profit", 0) for t in losses])), 2) if losses else 0,
            "worst_loss": round(abs(min(t.get("profit", 0) for t in losses)), 2) if losses else 0,
            "generated_at": datetime.now().isoformat(),
        }

        report["by_pair"] = self._analyze_by_field(losses, all_closed, "pair")
        report["by_timeframe"] = self._analyze_by_field(losses, all_closed, "timeframe")
        report["by_direction"] = self._analyze_by_field(losses, all_closed, "direction")
        report["by_exit_reason"] = self._analyze_exit_reasons(losses, all_closed)
        report["by_hour"] = self._analyze_by_hour(losses, all_closed)
        report["by_session"] = self._analyze_by_session(losses, all_closed)
        report["by_regime"] = self._analyze_market_condition(losses, all_closed, "regime")
        report["by_volatility"] = self._analyze_market_condition(losses, all_closed, "volatility")
        report["by_trend"] = self._analyze_market_condition(losses, all_closed, "trend")
        report["by_model_version"] = self._analyze_by_field(losses, all_closed, "model_version")
        report["by_confidence_bucket"] = self._analyze_by_confidence(losses, all_closed)

        report["indicator_failures"] = self._analyze_indicator_failures(losses)
        report["worst_combinations"] = self._analyze_combinations(losses)

        concurrency = self._analyze_loss_streaks(all_closed)
        report.update(concurrency)

        report["summary"] = self._generate_summary(report)

        self._save_report(report)
        self.logger.info(
            f"Mistake analysis: {report['total_losses']} losses, "
            f"avg=${report['avg_loss']}, "
            f"worst pairs: {', '.join(report['by_pair']['worst'][:3])}"
        )
        return report

    def _analyze_by_field(self, losses: List[Dict], all_trades: List[Dict], field: str) -> Dict:
        loss_counts = Counter()
        total_counts = Counter()
        loss_amounts = defaultdict(float)

        for t in losses:
            val = str(t.get(field, "unknown"))
            loss_counts[val] += 1
            loss_amounts[val] += abs(t.get("profit", 0))

        for t in all_trades:
            val = str(t.get(field, "unknown"))
            total_counts[val] += 1

        results = {}
        for val in sorted(set(list(loss_counts.keys()) + list(total_counts.keys()))):
            lc = loss_counts.get(val, 0)
            tc = total_counts.get(val, 0)
            results[val] = {
                "losses": lc,
                "total": tc,
                "loss_rate": round(lc / tc * 100, 2) if tc > 0 else 0,
                "total_loss_amount": round(loss_amounts.get(val, 0), 2),
            }

        sorted_items = sorted(results.items(), key=lambda x: x[1]["loss_rate"], reverse=True)
        return {
            "details": results,
            "worst": [k for k, v in sorted_items[:5] if v["total"] >= 3],
            "best": [k for k, v in sorted_items[-5:] if v["total"] >= 3],
        }

    def _analyze_exit_reasons(self, losses: List[Dict], all_trades: List[Dict]) -> Dict:
        return self._analyze_by_field(losses, all_trades, "exit_reason")

    def _analyze_by_hour(self, losses: List[Dict], all_trades: List[Dict]) -> Dict:
        loss_hours = Counter()
        total_hours = Counter()

        for t in losses:
            entry_str = t.get("entry_time")
            if entry_str:
                try:
                    hour = datetime.fromisoformat(entry_str).hour
                    loss_hours[hour] += 1
                except (ValueError, TypeError):
                    pass

        for t in all_trades:
            entry_str = t.get("entry_time")
            if entry_str:
                try:
                    hour = datetime.fromisoformat(entry_str).hour
                    total_hours[hour] += 1
                except (ValueError, TypeError):
                    pass

        results = {}
        for h in range(24):
            lc = loss_hours.get(h, 0)
            tc = total_hours.get(h, 0)
            results[f"{h:02d}:00"] = {
                "losses": lc,
                "total": tc,
                "loss_rate": round(lc / tc * 100, 2) if tc > 0 else 0,
            }

        sorted_items = sorted(results.items(), key=lambda x: x[1]["loss_rate"], reverse=True)
        return {
            "details": results,
            "worst_hours": [k for k, v in sorted_items[:3] if v["total"] >= 2],
            "best_hours": [k for k, v in sorted_items[-3:] if v["total"] >= 2],
        }

    def _analyze_by_session(self, losses: List[Dict], all_trades: List[Dict]) -> Dict:
        def get_session(hour: int) -> str:
            if 0 <= hour < 8:
                return "Asia"
            elif 8 <= hour < 16:
                return "London"
            else:
                return "New_York"

        loss_sessions = Counter()
        total_sessions = Counter()

        for t in losses:
            entry_str = t.get("entry_time")
            if entry_str:
                try:
                    hour = datetime.fromisoformat(entry_str).hour
                    loss_sessions[get_session(hour)] += 1
                except (ValueError, TypeError):
                    pass

        for t in all_trades:
            entry_str = t.get("entry_time")
            if entry_str:
                try:
                    hour = datetime.fromisoformat(entry_str).hour
                    total_sessions[get_session(hour)] += 1
                except (ValueError, TypeError):
                    pass

        results = {}
        for session in ["Asia", "London", "New_York"]:
            lc = loss_sessions.get(session, 0)
            tc = total_sessions.get(session, 0)
            results[session] = {
                "losses": lc,
                "total": tc,
                "loss_rate": round(lc / tc * 100, 2) if tc > 0 else 0,
            }
        return results

    def _analyze_market_condition(self, losses: List[Dict], all_trades: List[Dict], condition: str) -> Dict:
        loss_conds = Counter()
        total_conds = Counter()

        for t in losses:
            val = str(t.get("market_conditions", {}).get(condition, "unknown"))
            loss_conds[val] += 1

        for t in all_trades:
            val = str(t.get("market_conditions", {}).get(condition, "unknown"))
            total_conds[val] += 1

        results = {}
        for val in sorted(set(list(loss_conds.keys()) + list(total_conds.keys()))):
            lc = loss_conds.get(val, 0)
            tc = total_conds.get(val, 0)
            results[val] = {
                "losses": lc,
                "total": tc,
                "loss_rate": round(lc / tc * 100, 2) if tc > 0 else 0,
            }

        sorted_items = sorted(results.items(), key=lambda x: x[1]["loss_rate"], reverse=True)
        return {
            "details": results,
            "worst": [k for k, v in sorted_items[:3] if v["total"] >= 3],
        }

    def _analyze_by_confidence(self, losses: List[Dict], all_trades: List[Dict]) -> Dict:
        def bucket(conf: float) -> str:
            if conf >= 90:
                return "90-100"
            elif conf >= 80:
                return "80-90"
            elif conf >= 70:
                return "70-80"
            elif conf >= 60:
                return "60-70"
            elif conf >= 50:
                return "50-60"
            else:
                return "<50"

        loss_buckets = Counter()
        total_buckets = Counter()

        for t in losses:
            loss_buckets[bucket(t.get("confidence", 0))] += 1
        for t in all_trades:
            total_buckets[bucket(t.get("confidence", 0))] += 1

        results = {}
        for b in ["<50", "50-60", "60-70", "70-80", "80-90", "90-100"]:
            lc = loss_buckets.get(b, 0)
            tc = total_buckets.get(b, 0)
            results[b] = {
                "losses": lc,
                "total": tc,
                "loss_rate": round(lc / tc * 100, 2) if tc > 0 else 0,
            }
        return results

    def _analyze_indicator_failures(self, losses: List[Dict]) -> Dict:
        indicator_ranges = {
            "rsi": {"oversold": (0, 30), "overbought": (70, 100)},
            "adx": {"weak": (0, 25), "strong": (25, 100)},
            "bb_pct": {"low": (0, 0.2), "high": (0.8, 1.0)},
            "volatility": {"low": (0, 0.001), "high": (0.005, float("inf"))},
        }

        failures = {}
        for losses_list in [losses]:
            for field, ranges in indicator_ranges.items():
                field_failures = {label: {"count": 0, "total_loss": 0.0} for label in ranges}
                for t in losses_list:
                    ind = t.get("indicators", {})
                    val = ind.get(field)
                    if val is not None:
                        for label, (lo, hi) in ranges.items():
                            if lo <= val <= hi:
                                field_failures[label]["count"] += 1
                                field_failures[label]["total_loss"] += abs(t.get("profit", 0))
                failures[field] = field_failures

        worst_indicators = []
        for field, ranges in failures.items():
            for label, data in ranges.items():
                if data["count"] >= 2:
                    worst_indicators.append({
                        "indicator": f"{field}_{label}",
                        "loss_count": data["count"],
                        "total_loss": round(data["total_loss"], 2),
                    })

        worst_indicators.sort(key=lambda x: x["loss_count"], reverse=True)
        return {
            "details": failures,
            "worst_indicators": worst_indicators[:5],
        }

    def _analyze_combinations(self, losses: List[Dict]) -> List[Dict]:
        combos = Counter()
        combo_amounts = defaultdict(float)

        for t in losses:
            pair = t.get("pair", "?")
            tf = t.get("timeframe", "?")
            direction = t.get("direction", "?")
            regime = t.get("market_conditions", {}).get("regime", "?")
            key = f"{pair}_{tf}_{direction}_{regime}"
            combos[key] += 1
            combo_amounts[key] += abs(t.get("profit", 0))

        sorted_combos = sorted(combos.items(), key=lambda x: x[1], reverse=True)
        return [
            {"combination": k, "losses": v, "total_loss": round(combo_amounts[k], 2)}
            for k, v in sorted_combos[:10] if v >= 2
        ]

    def _analyze_loss_streaks(self, all_trades: List[Dict]) -> Dict:
        sorted_trades = sorted(all_trades, key=lambda t: t.get("exit_time", ""))
        current_streak = 0
        max_streak = 0
        streak_start = None
        streak_trades = []
        worst_streak = {"count": 0, "total_loss": 0, "trades": []}

        for t in sorted_trades:
            if t.get("result") == "LOSS":
                if current_streak == 0:
                    streak_start = t.get("exit_time", "")
                    streak_trades = [t]
                else:
                    streak_trades.append(t)
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
                    streak_loss = sum(abs(tt.get("profit", 0)) for tt in streak_trades)
                    worst_streak = {
                        "count": current_streak,
                        "total_loss": round(streak_loss, 2),
                        "start_time": streak_start,
                        "end_time": t.get("exit_time", ""),
                    }
            else:
                current_streak = 0
                streak_trades = []

        return {
            "max_consecutive_losses": max_streak,
            "worst_loss_streak": worst_streak,
        }

    def _generate_summary(self, report: Dict) -> str:
        parts = []

        by_pair = report.get("by_pair", {})
        worst_pairs = by_pair.get("worst", [])
        if worst_pairs:
            parts.append(f"Worst pairs: {', '.join(worst_pairs)}")

        by_session = report.get("by_session", {})
        bad_sessions = [s for s, d in by_session.items() if d.get("loss_rate", 0) > 60 and d.get("total", 0) >= 3]
        if bad_sessions:
            parts.append(f"Bad sessions: {', '.join(bad_sessions)}")

        by_hour = report.get("by_hour", {})
        worst_hours = by_hour.get("worst_hours", [])
        if worst_hours:
            parts.append(f"Bad hours: {', '.join(worst_hours)}")

        worst_streak = report.get("worst_loss_streak", {})
        if worst_streak.get("count", 0) >= 3:
            parts.append(f"Worst streak: {worst_streak['count']} losses (${worst_streak['total_loss']})")

        worst_inds = report.get("indicator_failures", {}).get("worst_indicators", [])
        if worst_inds:
            parts.append(f"Indicator failures: {', '.join(i['indicator'] for i in worst_inds[:3])}")

        return "; ".join(parts) if parts else "No significant patterns detected"

    def _empty_report(self) -> Dict:
        return {
            "total_losses": 0,
            "total_trades": 0,
            "loss_rate": 0,
            "total_loss_amount": 0,
            "avg_loss": 0,
            "worst_loss": 0,
            "summary": "No losses to analyze",
            "generated_at": datetime.now().isoformat(),
        }

    def _save_report(self, report: Dict):
        try:
            with open(self._report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save mistake report: {e}")
