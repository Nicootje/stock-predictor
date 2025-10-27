import matplotlib.pyplot as plt
import pandas as pd
from src.calc_indicators import calc_sma_ema, calc_rsi, calc_bollinger_bands, calc_macd

def plot_sma_ema_with_rsi(df, ticker, sma_ema_periods=[20, 50, 200], rsi_period=14, start_plot_date=None):
    plot_df = df.copy()

    # Calculate SMA/EMA if not present
    for p in sma_ema_periods:
        if f'SMA{p}' not in plot_df.columns or f'EMA{p}' not in plot_df.columns:
            plot_df = calc_sma_ema(plot_df, sma_ema_periods)

    # Calculate RSI if not present
    if 'RSI' not in plot_df.columns:
        plot_df = calc_rsi(plot_df, rsi_period)

    # Filter by start_plot_date
    if start_plot_date:
        plot_df = plot_df[plot_df.index >= start_plot_date]

    # Create figure with 2 subplots, remove vertical spacing
    fig, (ax_price, ax_rsi) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0}  # hspace=0 removes the gap
    )

    # --- Price subplot ---
    ax_price.plot(plot_df.index, plot_df['Close'], label='Close', color='blue')
    colors = {20: "green", 50: "red", 200: "black"}
    for p in sma_ema_periods:
        ax_price.plot(plot_df.index, plot_df[f'SMA{p}'], label=f'SMA{p}', color=colors.get(p, 'gray'))
        ax_price.plot(plot_df.index, plot_df[f'EMA{p}'], label=f'EMA{p}', color=colors.get(p, 'gray'), linestyle='--')
    ax_price.set_title(f"{ticker.upper()} — SMA/EMA with RSI")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle='--', alpha=0.3)
    ax_price.legend()

    # Optional: Draw thin line connecting top and bottom subplot (just under price)
    ax_price.axhline(y=ax_price.get_ylim()[0], color='black', linewidth=0.5, alpha=0.5)

    # --- RSI subplot ---
    ax_rsi.plot(plot_df.index, plot_df['RSI'], color='purple', label=f'RSI ({rsi_period})')
    ax_rsi.axhline(70, color='red', linestyle='--')
    ax_rsi.axhline(30, color='green', linestyle='--')
    ax_rsi.fill_between(plot_df.index, 70, 30, color='gray', alpha=0.1)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.grid(True, linestyle='--', alpha=0.3)
    ax_rsi.legend()

    plt.tight_layout()
    plt.show()
    

def plot_bollinger_bands(df, ticker, period=20, std_dev=2, start_plot_date=None):
    plot_df = df.copy()
    if 'BB_Middle' not in plot_df.columns:
        plot_df = calc_bollinger_bands(plot_df, period, std_dev)
    if start_plot_date:
        plot_df = plot_df[plot_df.index >= start_plot_date]

    plt.figure(figsize=(14, 4))
    plt.plot(plot_df.index, plot_df['Close'], label='Close', color='blue')
    plt.plot(plot_df.index, plot_df['BB_Upper'], label='BB Upper', color='green', linestyle='--')
    plt.plot(plot_df.index, plot_df['BB_Middle'], label='BB Middle', color='purple')
    plt.plot(plot_df.index, plot_df['BB_Lower'], label='BB Lower', color='red', linestyle='--')
    plt.fill_between(plot_df.index, plot_df['BB_Upper'], plot_df['BB_Lower'], color='gray', alpha=0.1)
    plt.title(f"{ticker.upper()} — Bollinger Bands")
    plt.ylabel("Price")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.show()


def plot_rsi(df, ticker, period=14, start_plot_date=None):
    plot_df = df.copy()
    if 'RSI' not in plot_df.columns:
        plot_df = calc_rsi(plot_df, period)
    if start_plot_date:
        plot_df = plot_df[plot_df.index >= start_plot_date]

    plt.figure(figsize=(14, 3))
    plt.plot(plot_df.index, plot_df['RSI'], color='purple', label='RSI')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.fill_between(plot_df.index, 70, 30, color='gray', alpha=0.1)
    plt.title(f"{ticker.upper()} — RSI ({period})")
    plt.ylabel("RSI")
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.show()


def plot_macd(df, ticker, fast=12, slow=26, signal=9, start_plot_date=None):
    plot_df = df.copy()
    if not {'MACD', 'Signal', 'Histogram'}.issubset(plot_df.columns):
        plot_df = calc_macd(plot_df, fast, slow, signal)
    if start_plot_date:
        plot_df = plot_df[plot_df.index >= start_plot_date]

    plt.figure(figsize=(14, 4))
    plt.plot(plot_df.index, plot_df['MACD'], color='blue', label='MACD')
    plt.plot(plot_df.index, plot_df['Signal'], color='orange', label='Signal')
    plt.bar(plot_df.index, plot_df['Histogram'], color='gray', alpha=0.5, label='Histogram')
    plt.title(f"{ticker.upper()} — MACD ({fast},{slow},{signal})")
    plt.ylabel("MACD")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.show()

def plot_full_chart(df, ticker, 
                    sma_ema_periods=[20, 50, 200], 
                    start_plot_date= None,
                    sma=True, ema=True,
                    bb_period=20, bb_std=2,
                    rsi_period=14,
                    macd_fast=12, macd_slow=26, macd_signal=9,
                    ):
    plot_df = df.copy()

    # --- Calculate indicators if not present ---
    # SMA/EMA
    for p in sma_ema_periods:
        if sma and f'SMA{p}' not in plot_df.columns:
            plot_df = calc_sma_ema(plot_df, sma_ema_periods)
        if ema and f'EMA{p}' not in plot_df.columns:
            plot_df = calc_sma_ema(plot_df, sma_ema_periods)
    
    # Bollinger Bands
    if 'BB_Middle' not in plot_df.columns:
        plot_df = calc_bollinger_bands(plot_df, bb_period, bb_std)

    # RSI
    if 'RSI' not in plot_df.columns:
        plot_df = calc_rsi(plot_df, rsi_period)

    # MACD
    if not {'MACD', 'Signal', 'Histogram'}.issubset(plot_df.columns):
        plot_df = calc_macd(plot_df, macd_fast, macd_slow, macd_signal)

    # Filter by start_plot_date
    if start_plot_date:
        plot_df = plot_df[plot_df.index >= start_plot_date]

    # --- Create figure with 3 subplots ---
    fig, (ax_price, ax_rsi, ax_macd) = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={'height_ratios': [3, 1, 1], 'hspace': 0.03}
    )

    # --- Price + SMA/EMA + Bollinger ---
    ax_price.plot(plot_df.index, plot_df['Close'], label='Close', color='blue')
    colors = {20: "green", 50: "red", 200: "black"}
    for p in sma_ema_periods:
        if sma:
            ax_price.plot(plot_df.index, plot_df[f'SMA{p}'], label=f'SMA{p}', color=colors.get(p, 'gray'))
        if ema:
            ax_price.plot(plot_df.index, plot_df[f'EMA{p}'], label=f'EMA{p}', color=colors.get(p, 'gray'), linestyle='--')
    
    # Bollinger Bands
    ax_price.plot(plot_df.index, plot_df['BB_Upper'], label='BB Upper', color='green', linestyle='--')
    ax_price.plot(plot_df.index, plot_df['BB_Middle'], label='BB Middle', color='purple')
    ax_price.plot(plot_df.index, plot_df['BB_Lower'], label='BB Lower', color='red', linestyle='--')
    ax_price.fill_between(plot_df.index, plot_df['BB_Upper'], plot_df['BB_Lower'], color='gray', alpha=0.1)
    
    ax_price.set_title(f"{ticker.upper()} — Price with SMA/EMA + Bollinger Bands")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle='--', alpha=0.3)
    ax_price.legend()

    # --- RSI subplot ---
    ax_rsi.plot(plot_df.index, plot_df['RSI'], color='purple', label=f'RSI ({rsi_period})')
    ax_rsi.axhline(70, color='red', linestyle='--')
    ax_rsi.axhline(30, color='green', linestyle='--')
    ax_rsi.fill_between(plot_df.index, 70, 30, color='gray', alpha=0.1)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.grid(True, linestyle='--', alpha=0.3)
    ax_rsi.legend()

    # --- MACD subplot ---
    ax_macd.plot(plot_df.index, plot_df['MACD'], color='blue', label='MACD')
    ax_macd.plot(plot_df.index, plot_df['Signal'], color='orange', label='Signal')
    ax_macd.bar(plot_df.index, plot_df['Histogram'], color='gray', alpha=0.5, label='Histogram')
    ax_macd.set_ylabel("MACD")
    ax_macd.grid(True, linestyle='--', alpha=0.3)
    ax_macd.legend()

    plt.tight_layout()
    plt.show()

    # --- Print info about which indicators are plotted ---
    indicators = []
    if sma: indicators.append(f"SMA: {sma_ema_periods}")
    if ema: indicators.append(f"EMA: {sma_ema_periods}")
    indicators.append(f"Bollinger Bands: period={bb_period}, std={bb_std}")
    indicators.append(f"RSI: period={rsi_period}")
    indicators.append(f"MACD: fast={macd_fast}, slow={macd_slow}, signal={macd_signal}")
    
    print(f"Plotted {ticker.upper()} with indicators: {', '.join(indicators)}")

def plot_volume(df, ticker, period=20, start_plot_date=None):
    plot_df = df.copy()
    
    # Flatten MultiIndex columns
    if isinstance(plot_df.columns, pd.MultiIndex):
        plot_df.columns = [col[0] for col in plot_df.columns]

    # Volume MA
    if 'Vol_MA' not in plot_df.columns:
        plot_df['Vol_MA'] = plot_df['Volume'].rolling(window=period).mean()

    # Filter by start_plot_date
    if start_plot_date:
        start_plot_date = pd.to_datetime(start_plot_date)
        plot_df = plot_df[plot_df.index >= start_plot_date]

    if plot_df.empty:
        print(f"No data to plot for {ticker} after {start_plot_date}")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.bar(plot_df.index, plot_df['Volume'], label='Volume', color='green', alpha=0.5)
    ax.plot(plot_df.index, plot_df['Vol_MA'], label=f'Volume MA ({period})', color='red', linewidth=2)
    ax.set_title(f"{ticker.upper()} — Volume with Moving Average")
    ax.set_ylabel("Volume")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_stochastic(df, ticker, period=14, smooth_k=3, smooth_d=3, start_plot_date=None):
    from src.calc_indicators import calc_stochastic  # Importeer calc_stochastic
    
    plot_df = df.copy()
    
    # Flatten MultiIndex columns
    if isinstance(plot_df.columns, pd.MultiIndex):
        plot_df.columns = [col[0] for col in plot_df.columns]

    # Calculate Stochastic if not present
    if not {'%K', '%D'}.issubset(plot_df.columns):
        plot_df = calc_stochastic(plot_df, period, smooth_k, smooth_d)

    # Filter by start_plot_date
    if start_plot_date:
        start_plot_date = pd.to_datetime(start_plot_date)
        plot_df = plot_df[plot_df.index >= start_plot_date]

    if plot_df.empty:
        print(f"No data to plot for {ticker} after {start_plot_date}")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(plot_df.index, plot_df['%K'], label=f'%K ({period})', color='blue')
    ax.plot(plot_df.index, plot_df['%D'], label=f'%D ({smooth_d})', color='orange')
    ax.axhline(80, color='red', linestyle='--')
    ax.axhline(20, color='green', linestyle='--')
    ax.fill_between(plot_df.index, 20, 80, color='gray', alpha=0.1)
    ax.set_title(f"{ticker.upper()} — Stochastic Oscillator")
    ax.set_ylabel("Stochastic")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_fibonacci_levels(df, ticker, low_price=None, high_price=None, start_plot_date=None, auto_detect=False, lookback_period=60):
    plot_df = df.copy()
    
    # Flatten MultiIndex columns
    if isinstance(plot_df.columns, pd.MultiIndex):
        plot_df.columns = [col[0] for col in plot_df.columns]

    # Filter by start_plot_date
    if start_plot_date:
        start_plot_date = pd.to_datetime(start_plot_date)
        plot_df = plot_df[plot_df.index >= start_plot_date]

    if plot_df.empty:
        print(f"No data to plot for {ticker} after {start_plot_date}")
        return

    # Automatische detectie van low en high als niet handmatig gespecificeerd
    if auto_detect and (low_price is None or high_price is None):
        lookback_df = plot_df.tail(lookback_period)  # Laatste 60 dagen
        low_price = lookback_df['Close'].min()
        high_price = lookback_df['Close'].max()

    # Gebruik handmatige waarden als opgegeven
    if low_price is None or high_price is None:
        print("Please specify low_price and high_price or set auto_detect=True with a lookback_period.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(plot_df.index, plot_df['Close'], label='Close', color='blue')

    # Calculate Fibonacci levels
    price_range = high_price - low_price
    levels = [0, 23.6, 38.2, 50, 61.8, 100]
    for level in levels:
        price = high_price - (price_range * level / 100)
        ax.axhline(y=price, color='gray', linestyle='--', alpha=0.5)
        ax.text(plot_df.index[0], price, f'{level}%', va='center', ha='right', alpha=0.7)

    ax.set_title(f"{ticker.upper()} — Fibonacci Levels (Low: {low_price}, High: {high_price})")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
