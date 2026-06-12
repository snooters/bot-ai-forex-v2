from datetime import datetime
from typing import Dict, List
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pytest


# ── Force test environment ──
os.environ["ENV"] = "test"


@pytest.fixture
def sample_ohlc() -> pd.DataFrame:
    n = 500
    base = 1.0500
    noise = np.random.default_rng(42).normal(0, 0.0005, n).cumsum()
    close = base + noise
    high = close + np.abs(np.random.default_rng(42).normal(0, 0.0003, n))
    low = close - np.abs(np.random.default_rng(42).normal(0, 0.0003, n))
    times = pd.date_range("2025-01-01", periods=n, freq="5min")

    df = pd.DataFrame({
        "time": times,
        "open": close - np.random.default_rng(42).normal(0, 0.0001, n),
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": np.random.default_rng(42).poisson(100, n),
        "spread": np.full(n, 10),
    })
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["ema_200"] = df["close"].ewm(span=200).mean()
    df["atr"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()
    df["atr"] = df["atr"].rolling(14).mean().fillna(0.001)
    df["rsi"] = 50.0
    df["macd"] = df["ema_20"] - df["ema_50"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["adx"] = 25.0
    return df


@pytest.fixture
def sample_ml_signal() -> Dict:
    return {
        "signal": "BUY",
        "confidence": 0.72,
        "buy_prob": 0.65,
        "sell_prob": 0.15,
        "hold_prob": 0.20,
    }


@pytest.fixture
def sample_trend_bullish() -> Dict:
    return {
        "direction": "BULLISH",
        "strength": 0.6,
        "score": 0.55,
    }


@pytest.fixture
def sample_trend_bearish() -> Dict:
    return {
        "direction": "BEARISH",
        "strength": 0.5,
        "score": -0.45,
    }


@pytest.fixture
def sample_regime_sideways() -> Dict:
    return {
        "regime": "SIDEWAYS",
        "confidence": 0.7,
        "volatility_score": 25,
    }


@pytest.fixture
def sample_regime_trending_bullish() -> Dict:
    return {
        "regime": "STRONG_TRENDING_BULLISH",
        "confidence": 0.85,
        "volatility_score": 50,
    }


@pytest.fixture
def sample_regime_news_driven() -> Dict:
    return {
        "regime": "NEWS_DRIVEN",
        "confidence": 0.8,
        "volatility_score": 90,
    }


@pytest.fixture
def sample_trade_buy_win() -> Dict:
    return {
        "ticket": 1001,
        "symbol": "EURUSD",
        "direction": "BUY",
        "volume": 0.05,
        "entry_price": 1.0500,
        "exit_price": 1.0600,
        "stop_loss": 1.0450,
        "take_profit": 1.0650,
        "profit": 50.0,
        "profit_pips": 100.0,
        "exit_reason": "TP_HIT",
        "confidence": 0.80,
        "market_score": 75,
        "market_conditions": {
            "trend": "BULLISH",
            "regime": "STRONG_TRENDING_BULLISH",
            "volatility": "medium",
        },
    }


@pytest.fixture
def sample_trade_buy_loss() -> Dict:
    return {
        "ticket": 1002,
        "symbol": "EURUSD",
        "direction": "BUY",
        "volume": 0.05,
        "entry_price": 1.0500,
        "exit_price": 1.0450,
        "stop_loss": 1.0450,
        "take_profit": 1.0650,
        "profit": -25.0,
        "profit_pips": -50.0,
        "exit_reason": "SL_HIT",
        "confidence": 0.55,
        "market_score": 35,
        "market_conditions": {
            "trend": "SIDEWAYS",
            "regime": "SIDEWAYS",
            "volatility": "low",
        },
    }
