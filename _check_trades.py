from learning.trade_logger import TradeLogger
tl = TradeLogger()
trades = tl.get_closed_trades()
closed = [t for t in trades if t.get("profit") is not None]
recent = sorted(closed, key=lambda x: x.get("exit_time", "") or "", reverse=True)[:5]
for t in recent:
    et = (t.get("exit_time") or "")[:19]
    print(f'{et} | Profit: {t.get("profit",0):+.2f} | {t.get("symbol","?")} | {t.get("direction","?")}')
