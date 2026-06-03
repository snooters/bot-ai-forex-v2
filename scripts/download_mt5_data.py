import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import MetaTrader5 as mt5

SYMBOL = "EURUSD.fl"
TIMEFRAMES = {
    1: "M1",
    5: "M5",
    15: "M15",
    30: "M30",
    60: "H1",
    240: "H4",
}
OUTPUT_DIR = Path("data/historical/EURUSD.fl")
START_DATE = datetime(2019, 1, 1)

MAX_CANDLES_PER_CALL = 100000

def download_timeframe(tf: int, label: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Downloading {SYMBOL} {label} (tf={tf})")
    print(f"{'='*60}")

    if tf == 1:
        chunks = _chunk_months(START_DATE, datetime.now(), 1)
    elif tf == 5:
        chunks = _chunk_months(START_DATE, datetime.now(), 3)
    elif tf == 15:
        chunks = _chunk_months(START_DATE, datetime.now(), 6)
    else:
        chunks = _chunk_years(START_DATE, datetime.now())

    all_dfs = []
    for i, (start, end) in enumerate(chunks):
        rates = mt5.copy_rates_range(SYMBOL, tf, start, end)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            print(f"  Chunk {i+1}/{len(chunks)}: {start.date()} -> {end.date()} -> no data ({err})")
            continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        all_dfs.append(df)
        print(f"  Chunk {i+1}/{len(chunks)}: {start.date()} -> {end.date()} -> {len(df)} candles")

    if not all_dfs:
        print(f"  No data downloaded for {label}")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.sort_values("time", inplace=True)
    combined.drop_duplicates(subset=["time"], keep="last", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    out = [
        "time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"
    ]
    keep = [c for c in out if c in combined.columns]
    combined = combined[keep]

    span_days = (combined["time"].max() - combined["time"].min()).days
    print(f"\n  {label}: {len(combined)} rows, "
          f"{combined['time'].min().date()} -> {combined['time'].max().date()}, "
          f"span={span_days}d")
    return combined


def _chunk_months(start: datetime, end: datetime, step_months: int):
    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = cursor + timedelta(days=step_months * 30)
        if chunk_end > end:
            chunk_end = end + timedelta(seconds=1)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def _chunk_years(start: datetime, end: datetime):
    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = cursor + timedelta(days=365)
        if chunk_end > end:
            chunk_end = end + timedelta(seconds=1)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def main():
    print("=" * 60)
    print("MT5 DATA DOWNLOADER")
    print(f"Symbol: {SYMBOL}")
    print(f"From: {START_DATE.date()}")
    print(f"To: {datetime.now().date()}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        sys.exit(1)

    print(f"MT5 terminal: {mt5.terminal_info().name}")
    acct = mt5.account_info()
    print(f"Account: {acct.login} ({acct.server}), Balance: {acct.balance:.2f}")

    mt5.symbol_select(SYMBOL, True)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for tf, label in sorted(TIMEFRAMES.items()):
        df = download_timeframe(tf, label)
        if df.empty:
            print(f"  WARNING: No data for {label}, skipping")
            continue

        filepath = OUTPUT_DIR / f"tf_{tf}.parquet"
        old_rows = 0
        if filepath.exists():
            try:
                old = pd.read_parquet(filepath)
                old_rows = len(old)
            except Exception:
                pass

        df.to_parquet(str(filepath), index=False)
        new_rows = len(df)
        diff = new_rows - old_rows
        sign = "+" if diff >= 0 else ""
        print(f"  Saved: {filepath.name} ({new_rows} rows, {sign}{diff} from previous)")

    mt5.shutdown()
    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*60}")

    print(f"\nFiles in {OUTPUT_DIR}:")
    for f in sorted(OUTPUT_DIR.glob("*.parquet")):
        sz = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {sz:.1f} MB")


if __name__ == "__main__":
    main()
