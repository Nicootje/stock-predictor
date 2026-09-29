import pandas as pd
import numpy as np
from numbers import Integral


def _validate_period(period):
    if isinstance(period, bool) or not isinstance(period, Integral) or period < 1:
        raise ValueError("De periode moet een positief geheel getal zijn.")
    return int(period)


def _single_ticker(df, ticker=None):
    """Maak yfinance-kolommen vlak, zonder meerdere instrumenten te vermengen."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    levels = [i for i in range(df.columns.nlevels)
              if 'Close' in df.columns.get_level_values(i)]
    if df.columns.nlevels != 2 or len(levels) != 1:
        raise ValueError("Onbekende indeling van koerskolommen.")
    ticker_level = 1 - levels[0]
    names = df.columns.get_level_values(ticker_level).unique()
    matches = [name for name in names if ticker is not None and str(name).upper() == ticker.upper()]
    if ticker is None and len(names) == 1:
        matches = list(names)
    if len(matches) != 1:
        raise ValueError("Selecteer één ticker uit de koersdata.")
    return df.xs(matches[0], axis=1, level=ticker_level).copy()


def _wilder_mean(series, period):
    """Eerste gemiddelde over n waarden; daarna Wilder's recursie. Gaten resetten."""
    period = _validate_period(period)
    result = np.full(len(series), np.nan)
    seed = []
    average = np.nan
    for i, value in enumerate(series.to_numpy(dtype=float)):
        if not np.isfinite(value):
            seed, average = [], np.nan
        elif np.isnan(average):
            seed.append(value)
            if len(seed) == period:
                average = float(np.mean(seed))
                result[i] = average
        else:
            average = (average * (period - 1) + value) / period
            result[i] = average
    return pd.Series(result, index=series.index)

def calc_sma_ema(df, periods):
    df = _single_ticker(df)
    for p in periods:
        p = _validate_period(p)
        df[f'SMA{p}'] = df['Close'].rolling(window=p).mean()
        df[f'EMA{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
    return df

def calc_rsi(df, period=14):
    df = _single_ticker(df)
    period = _validate_period(period)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _wilder_mean(gain, period)
    avg_loss = _wilder_mean(loss, period)
    total = avg_gain + avg_loss
    # Alleen winst: 100; alleen verlies: 0; volledig vlak: neutraal 50.
    df['RSI'] = (100 * avg_gain / total.replace(0, np.nan)).mask(total.eq(0), 50.0)
    return df

def calc_bollinger_bands(df, period=20, std_dev=2):
    df = _single_ticker(df)
    period = _validate_period(period)
    if not np.isfinite(std_dev) or std_dev <= 0:
        raise ValueError("std_dev moet positief en eindig zijn.")
    df['BB_Middle'] = df['Close'].rolling(window=period).mean()
    df['BB_Std'] = df['Close'].rolling(window=period).std(ddof=0)
    df['BB_Upper'] = df['BB_Middle'] + std_dev * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - std_dev * df['BB_Std']
    return df

def calc_macd(df, fast=12, slow=26, signal=9):
    df = _single_ticker(df)
    fast, slow, signal = [_validate_period(p) for p in (fast, slow, signal)]
    if fast >= slow:
        raise ValueError("MACD fast moet kleiner zijn dan slow.")
    df['EMA_Fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    return df

def calc_stochastic(df, period=14, smooth_k=3, smooth_d=3):
    df = _single_ticker(df)
    period, smooth_k, smooth_d = [_validate_period(p) for p in (period, smooth_k, smooth_d)]
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    raw_k = 100 * (df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan)
    df['%K'] = raw_k.rolling(window=smooth_k).mean()
    df['%D'] = df['%K'].rolling(window=smooth_d).mean()  # Smooth %K to get %D
    return df
