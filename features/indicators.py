import numpy as np
import pandas as pd
from typing import Optional, Tuple

from core.constants import (
    EME_FAST, EME_MEDIUM, EME_SLOW,
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ADX_PERIOD, ATR_PERIOD, BB_PERIOD, BB_STD,
    STOCH_K, STOCH_D
)
from utils.logger import get_logger


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    series: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = ADX_PERIOD
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr.replace(0, np.nan)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(window=period).mean()

    return adx, plus_di, minus_di


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def compute_bollinger_bands(
    series: pd.Series, period: int = BB_PERIOD, std: float = BB_STD
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = compute_sma(series, period)
    std_dev = series.rolling(window=period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    return upper, sma, lower


def compute_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = STOCH_K, d_period: int = STOCH_D
) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan))
    d = k.rolling(window=d_period).mean()
    return k, d


def compute_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    typical_price = (high + low + close) / 3
    cum_pv = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def compute_momentum(series: pd.Series, period: int = 10) -> pd.Series:
    return series / series.shift(period) - 1


def compute_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    log_returns = np.log(series / series.shift(1))
    return log_returns.rolling(window=period).std()


def compute_mass_index(high: pd.Series, low: pd.Series, period: int = 25) -> pd.Series:
    hl_range = high - low
    ema1 = compute_ema(hl_range, 9)
    ema2 = compute_ema(ema1, 9)
    ratio = ema1 / ema2.replace(0, np.nan)
    mass_index = ratio.rolling(window=period).sum()
    return mass_index


def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = compute_sma(tp, period)
    mad = (tp - sma).abs().rolling(window=period).mean()
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    return cci


def compute_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()
    wr = -100 * (highest - close) / (highest - lowest).replace(0, np.nan)
    return wr


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (volume * direction).cumsum()


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df.get("volume", df.get("tick_volume", pd.Series(index=df.index, dtype=float)))

    df["ema_20"] = compute_ema(close, EME_FAST)
    df["ema_50"] = compute_ema(close, EME_MEDIUM)
    df["ema_200"] = compute_ema(close, EME_SLOW)

    df["rsi"] = compute_rsi(close, RSI_PERIOD)

    macd_line, signal_line, histogram = compute_macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = histogram

    adx, plus_di, minus_di = compute_adx(high, low, close)
    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    df["atr"] = compute_atr(high, low, close)

    df["vwap"] = compute_vwap(high, low, close, volume)

    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close)
    df["bb_upper"] = bb_upper
    df["bb_mid"] = bb_mid
    df["bb_lower"] = bb_lower
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    df["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    stoch_k, stoch_d = compute_stochastic(high, low, close)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d

    df["momentum_10"] = compute_momentum(close, 10)
    df["momentum_20"] = compute_momentum(close, 20)
    df["volatility"] = compute_volatility(close, 20)
    df["obv"] = compute_obv(close, volume)
    df["cci"] = compute_cci(high, low, close)
    df["williams_r"] = compute_williams_r(high, low, close)
    df["mass_index"] = compute_mass_index(high, low)

    hl_range = high - low
    df["hl_range"] = hl_range
    df["body"] = (close - open_).abs() if "open" in df.columns else pd.Series(index=df.index)
    df["upper_wick"] = high - pd.concat([close, open_], axis=1).max(axis=1) if "open" in df.columns else 0
    df["lower_wick"] = pd.concat([close, open_], axis=1).min(axis=1) - low if "open" in df.columns else 0

    df["spread_pips"] = df["spread"] * 1e-5 if "spread" in df.columns else 0
    df["volume_ratio"] = volume / volume.rolling(20).mean().replace(0, np.nan)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


class IndicatorEngine:
    def __init__(self):
        self.logger = get_logger("indicator_engine")

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 30:
            self.logger.warning(f"Insufficient data for indicators: {len(df)} rows")
            return df
        df = compute_all_indicators(df)
        return df
