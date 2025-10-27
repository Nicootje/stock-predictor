import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from src.calc_indicators import calc_sma_ema, calc_rsi, calc_bollinger_bands, calc_macd

# Volume Moving Average
def calculate_volume_ma(df, period=20):
    df['Vol_MA'] = df['Volume'].rolling(window=period).mean()
    return df

# Main function: Assess rebound vs trendswitch, with volume integrated
def assess_trend(df, start_date, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, bb_period=20, bb_std=2, vol_period=20, periods=[20, 50, 200], plot=True):
    """
    Assesses if the trend is a rebound or trendswitch based on TA indicators, including volume.
    
    Args:
    - df: pandas DataFrame from yfinance with DatetimeIndex and 'Open', 'High', 'Low', 'Close', 'Volume'
    - start_date: str (e.g., '2025-01-01') to filter data from
    - rsi_period, macd_fast, macd_slow, macd_signal, bb_period, bb_std, vol_period: Indicator parameters
    - periods: list of periods for SMA/EMA (default [20, 50, 200])
    - plot: bool, if True, generates a plot with volume subplot
    
    Returns:
    - assessment: str ('Rebound (Temporary Bounce)', 'Trendswitch (Bullish Reversal)', or 'Uncertain or Continuing Trend')
    - updated_df: DataFrame with added indicator columns
    """
    # Ensure DatetimeIndex and sort
    df = df.sort_index()
    
    # Flatten MultiIndex columns if present (as per your working solution)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    # Filter from start_date using boolean indexing
    start_date = pd.to_datetime(start_date)
    df = df[df.index >= start_date]
    
    # Check if filtering resulted in empty DataFrame
    if df.empty:
        return "No data available after filtering from start_date.", df
    
    # Check for sufficient data
    min_periods = max(rsi_period, macd_slow, bb_period, vol_period, max(periods))
    if len(df) < min_periods:
        return "Not enough data for analysis (less than required periods).", df
    
    # Ensure required columns are present
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise KeyError(f"DataFrame missing required columns. Expected {required_cols}, got {df.columns.tolist()}")
    
    # Calculate indicators
    df = calc_sma_ema(df, periods)
    df = calc_rsi(df, rsi_period)
    df = calc_macd(df, macd_fast, macd_slow, macd_signal)
    df = calc_bollinger_bands(df, bb_period, bb_std)
    # Adjust Bollinger Bands columns to match expected names
    df['BB_Upper'] = df['BB_Middle'] + bb_std * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - bb_std * df['BB_Std']
    df = df.drop(columns=['BB_Std'])  # Remove intermediate Std column
    
    df = calculate_volume_ma(df, vol_period)
    
    # Drop NaNs only for indicator columns to preserve data
    indicator_cols = ['RSI', 'MACD', 'Signal', 'Histogram', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'Vol_MA']
    df = df.dropna(subset=indicator_cols)
    
    # Check if DataFrame is empty after processing
    if df.empty:
        return "No valid data after indicator calculations.", df
    
    # Recent values (last row) for assessment
    recent = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else recent
    
    # Volume analysis
    vol_increase = recent['Volume'] > recent['Vol_MA'] * 1.5
    vol_trend_up = recent['Volume'] > prev['Volume']
    
    # Criteria for Trendswitch
    is_breakout = recent['Close'] > recent['BB_Upper']
    macd_bullish = (recent['MACD'] > recent['Signal']) and (recent['Histogram'] > 0) and (recent['Histogram'] > prev['Histogram'])
    rsi_bullish = recent['RSI'] > 50 and (recent['RSI'] > prev['RSI'])
    price_up = recent['Close'] > prev['Close']
    if is_breakout and macd_bullish and rsi_bullish and (vol_increase or vol_trend_up):
        assessment = "Trendswitch (Bullish Reversal)"
    
    # Criteria for Rebound
    elif (recent['Close'] > recent['BB_Lower']) and (recent['MACD'] > recent['Signal']) and (40 < recent['RSI'] < 60) and not vol_increase:
        assessment = "Rebound (Temporary Bounce)"
    
    else:
        assessment = "Uncertain or Continuing Trend"
    
    # Generate plot if requested
    if plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # Price with Bollinger Bands and SMA/EMA
        ax1.plot(df.index, df['Close'], label='Close')
        ax1.plot(df.index, df['BB_Upper'], label='BB Upper', linestyle='--')
        ax1.plot(df.index, df['BB_Middle'], label='BB Middle', linestyle='--')
        ax1.plot(df.index, df['BB_Lower'], label='BB Lower', linestyle='--')
        for p in periods:
            ax1.plot(df.index, df[f'SMA{p}'], label=f'SMA{p}', linestyle='-.')
            ax1.plot(df.index, df[f'EMA{p}'], label=f'EMA{p}', linestyle=':')
        ax1.set_title('Price with Bollinger Bands, SMA, and EMA')
        ax1.legend()
        
        # RSI
        ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='green', linestyle='--')
        ax2.axhline(50, color='gray', linestyle='--')
        ax2.set_title('RSI')
        ax2.legend()
        
        # MACD
        ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
        ax3.plot(df.index, df['Signal'], label='Signal', color='orange')
        ax3.bar(df.index, df['Histogram'], label='Histogram', color='gray')
        ax3.set_title('MACD')
        ax3.legend()
        
        # Volume with MA
        ax4.bar(df.index, df['Volume'], label='Volume', color='green', alpha=0.5)
        ax4.plot(df.index, df['Vol_MA'], label=f'Volume MA ({vol_period})', color='red')
        ax4.set_title('Volume with Moving Average')
        ax4.legend()
        
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.show()
    
    return assessment