from __future__ import annotations

from typing import Any, Dict, Optional


def render_simulation_panel(sim_result: Optional[Dict[str, Any]]) -> str:
    if sim_result is None:
        return _empty_panel()

    lr = sim_result.get("learning", {})
    stats = sim_result.get("stats", {})
    skill = lr.get("skill", {})
    sep = "=" * 44
    lines = []
    lines.append(sep)
    lines.append(" SELF LEARNING ".center(42, "="))
    lines.append(sep)
    lines.append(f" Simulation Trades:    {stats.get('total_trades', 0):>6}")
    lines.append(f" Win Rate:            {stats.get('win_rate', 0)*100:>6.1f}%")
    lines.append(f" Profit Factor:       {stats.get('profit_factor', 0):>6.2f}")
    lines.append(f" Net Profit:          ${stats.get('net_profit', 0):>8.2f}")
    lines.append(f" Return:              {stats.get('return_pct', 0):>6.2f}%")
    lines.append(f" Max DD:              {stats.get('max_drawdown_pct', 0)*100:>6.1f}%")
    lines.append(f" Sharpe:              {stats.get('sharpe_ratio', 0):>6.2f}")
    lines.append(f" Expectancy:          ${stats.get('expectancy', 0):>8.2f}")

    skill_score = skill.get("score", 0)
    skill_level = skill.get("level", "N/A")
    lines.append(f" Skill:               {skill_level:>12} ({skill_score:>2}/100)")

    readiness = lr.get("real_account_readiness", "N/A")
    lines.append(f" Real Account:        {readiness:>12}")

    by_regime = lr.get("regime_analysis", {})
    if by_regime:
        best_regime = max(by_regime.items(), key=lambda x: x[1].get("win_rate", 0))
        worst_regime = min(by_regime.items(), key=lambda x: x[1].get("win_rate", 0))
        lines.append(f" Best Regime:         {best_regime[0]:>12} ({best_regime[1].get('win_rate', 0)*100:.0f}%)")
        lines.append(f" Worst Regime:        {worst_regime[0]:>12} ({worst_regime[1].get('win_rate', 0)*100:.0f}%)")

    pair_skills = skill.get("pair_skills", {})
    if pair_skills:
        best_pair = max(pair_skills.items(), key=lambda x: x[1])
        worst_pair = min(pair_skills.items(), key=lambda x: x[1])
        lines.append(f" Best Pair:           {best_pair[0]:>12} ({best_pair[1]:>2}/100)")
        lines.append(f" Worst Pair:          {worst_pair[0]:>12} ({worst_pair[1]:>2}/100)")

    rec_conf = lr.get("recommended_confidence", 0)
    rec_ms = lr.get("recommended_market_score", 0)
    lines.append(f" Rec. Confidence:     {rec_conf:>7.2f}")
    lines.append(f" Rec. Market Score:   {rec_ms:>6}")

    blocked = lr.get("blocked_regimes", [])
    if blocked:
        lines.append(f" Blocked Regimes:     {', '.join(blocked):>12}")

    lines.append("=" * 44)
    return "\n".join(lines)


def render_monte_carlo_panel(mc_result: Optional[Dict[str, Any]]) -> str:
    if mc_result is None:
        return ""

    lines = []
    lines.append(" MONTE CARLO ".center(42, "="))
    lines.append(f" Iterations:          {mc_result.get('iterations', 0):>8}")
    lines.append(f" Profit Probability:  {mc_result.get('profit_probability', 0)*100:>6.1f}%")
    lines.append(f" DD > 20% Prob:       {mc_result.get('dd_gt_20pct_probability', 0)*100:>6.1f}%")
    lines.append(f" Avg Profit:          ${mc_result.get('avg_profit', 0):>8.2f}")
    lines.append(f" Median Profit:       ${mc_result.get('median_profit', 0):>8.2f}")
    lines.append(f" Avg DD:              {mc_result.get('avg_drawdown_pct', 0)*100:>6.1f}%")
    lines.append(f" 95% CI:              ${mc_result.get('percentile_5', 0):>8.2f} to ${mc_result.get('percentile_95', 0):>8.2f}")
    lines.append("=" * 44)
    return "\n".join(lines)


def render_walk_forward_panel(wf_result: Optional[Any]) -> str:
    if wf_result is None:
        return ""

    lines = []
    lines.append(" WALK FORWARD ".center(42, "="))
    lines.append(f" Symbol:              {wf_result.symbol:>12}")
    lines.append(f" Grade:               {wf_result.grade:>12}")
    lines.append(f" Windows:             {len(wf_result.windows):>8}")
    lines.append(f" Avg Train WR:        {wf_result.avg_train_wr*100:>6.1f}%")
    lines.append(f" Avg Val WR:          {wf_result.avg_val_wr*100:>6.1f}%")
    lines.append(f" OOS Decay (WR):      {wf_result.oos_decay_wr*100:>6.1f}%")
    lines.append(f" Total Train Trades:  {wf_result.total_trades_train:>8}")
    lines.append(f" Total Val Trades:    {wf_result.total_trades_val:>8}")
    lines.append("=" * 44)

    for w in wf_result.windows[:5]:
        lines.append(f" {w['label']}: Train WR {w['train_win_rate']*100:.0f}% "
                     f"Val WR {w['val_win_rate']*100:.0f}% "
                     f"({w['val_trades']} trades)")

    if len(wf_result.windows) > 5:
        lines.append(f" ... and {len(wf_result.windows) - 5} more windows")
    lines.append("=" * 44)
    return "\n".join(lines)


def _empty_panel() -> str:
    lines = []
    lines.append("=" * 44)
    lines.append(" SELF LEARNING ".center(42, "="))
    lines.append("=" * 44)
    lines.append(" No simulation data available.".center(42))
    lines.append(" Run simulation first.".center(42))
    lines.append("=" * 44)
    return "\n".join(lines)
