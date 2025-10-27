import pandas as pd
import numpy as np

def calc_sma_ema(df, periods):
    for p in periods:
        df[f'SMA{p}'] = df['Close'].rolling(window=p).mean()
        df[f'EMA{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
    return df

def calc_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calc_bollinger_bands(df, period=20, std_dev=2):
    df['BB_Middle'] = df['Close'].rolling(window=period).mean()
    df['BB_Std'] = df['Close'].rolling(window=period).std()
    df['BB_Upper'] = df['BB_Middle'] + std_dev * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - std_dev * df['BB_Std']
    return df

def calc_macd(df, fast=12, slow=26, signal=9):
    df['EMA_Fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    return df

def calc_stochastic(df, period=14, smooth_k=3, smooth_d=3):
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    df['%K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['%D'] = df['%K'].rolling(window=smooth_d).mean()  # Smooth %K to get %D
    return df