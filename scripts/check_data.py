import pandas as pd
from pathlib import Path

d = Path("data/historical/EURUSD.fl")
print(f"Data directory: {d.resolve()}")
print(f"Files found: {len(list(d.glob('*.parquet')))}")
print()

for f in sorted(d.glob("*.parquet")):
    df = pd.read_parquet(f)
    ts_col = [c for c in ["time", "timestamp"] if c in df.columns][0]
    ts = pd.to_datetime(df[ts_col])
    print(f"  {f.name}: {len(df):>7} rows, {ts.min().date()} -> {ts.max().date()}, span={(ts.max()-ts.min()).days}d")
