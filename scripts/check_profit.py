"""Check today's profit from MT5"""
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.WARNING)
from datetime import datetime, date
from data.mt5_connector import MT5Connector

# Use MT5 directly for quick check
import MetaTrader5 as mt5
if mt5.initialize():
    account_info = mt5.account_info()
    if account_info:
        print(f"Balance: ${account_info.balance:.2f}")
        print(f"Equity: ${account_info.equity:.2f}")
        print(f"Floating P/L: ${account_info.profit:.2f}")
    
    # Get today's deal history
    today = date.today()
    from_dt = datetime(today.year, today.month, today.day, 0, 0, 0)
    to_dt = datetime.now()
    deals = mt5.history_deals_get(from_dt, to_dt)
    if deals:
        total_profit = sum(d.profit for d in deals if d.profit != 0)
        win_trades = sum(1 for d in deals if d.profit > 0)
        loss_trades = sum(1 for d in deals if d.profit < 0)
        commissions = sum(d.commission for d in deals if d.commission != 0)
        swaps = sum(d.swap for d in deals if d.swap != 0)
        print(f"\nToday's deals: {len(deals)}")
        print(f"  Wins: {win_trades}, Losses: {loss_trades}")
        print(f"  Commissions: ${commissions:.2f}")
        print(f"  Swaps: ${swaps:.2f}")
        print(f"  Total P/L (closed): ${total_profit:.2f}")
        print(f"  Total P/L (incl floating): ${total_profit + account_info.profit:.2f}")
        print(f"\nLast 10 deals:")
        for d in deals[-10:]:
            print(f"  {d.time} | type={d.type} | volume={d.volume} | price={d.price} | profit={d.profit:.2f} | commission={d.commission:.2f}")
    else:
        print("No closed deals today")
    
    # Check open positions
    positions = mt5.positions_get()
    if positions:
        print(f"\nOpen positions: {len(positions)}")
        for p in positions:
            print(f"  Ticket #{p.ticket} | {'BUY' if p.type==0 else 'SELL'} | {p.volume} lots | Open={p.price_open} | SL={p.sl} TP={p.tp} | Profit={p.profit:.2f}")
    else:
        print("No open positions")
    
    mt5.shutdown()
else:
    print("Failed to connect to MT5")
