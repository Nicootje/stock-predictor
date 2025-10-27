from src.calc_indicators import calc_sma_ema, calc_rsi, calc_bollinger_bands, calc_macd

def summary_technical_indicators(ticker,
                                 df,
                                 rsi_period,
                                 periods,
                                 start_plot_date,
                                 bb_period=20, bb_std_dev=2,
                                 macd_fast=12, macd_slow=26, macd_signal=9):
    """
    Analyseert RSI, Bollinger Bands, MACD, SMA & EMA.
    Geeft een korte samenvatting en retourneert enkel de summary string.
    """

    # ---------- 1. Data ----------
    close_df = df[['Close']].copy()

    # ---------- 2. Indicatoren berekenen ----------
    close_df = calc_sma_ema(close_df, periods)
    close_df = calc_rsi(close_df, rsi_period)
    close_df = calc_bollinger_bands(close_df, period=bb_period, std_dev=bb_std_dev)
    close_df = calc_macd(close_df, fast=macd_fast, slow=macd_slow, signal=macd_signal)

    # ---------- 3. Filter voor de plot ----------
    plot_df = close_df[close_df.index >= start_plot_date].copy()

    # ---------- 4. Laatste waarden ----------
    last = close_df.iloc[-1]
    prev = close_df.iloc[-2] if len(close_df) > 1 else last

    rsi_val = last['RSI'].item()
    price = last['Close'].item()
    bb_upper = last['BB_Upper'].item()
    bb_middle = last['BB_Middle'].item()
    bb_lower = last['BB_Lower'].item()
    macd = last['MACD'].item()
    signal = last['Signal'].item()
    histogram = last['Histogram'].item()

    # ---------- 5. Analyse ----------
    # RSI
    if rsi_val > 70:
        rsi_status = "Overbought (mogelijk bearish, verkoopsignaal)"
    elif rsi_val < 30:
        rsi_status = "Oversold (mogelijk bullish, koopsignaal)"
    else:
        rsi_status = "Neutraal"

    # Bollinger Bands
    if price >= bb_upper:
        bb_status = "Boven bovenste band (mogelijk overbought, bearish)"
    elif price <= bb_lower:
        bb_status = "Onder onderste band (mogelijk oversold, bullish)"
    elif abs(price - bb_upper) < abs(price - bb_middle):
        bb_status = "Dicht bij bovenste band (bullish trend)"
    elif abs(price - bb_lower) < abs(price - bb_middle):
        bb_status = "Dicht bij onderste band (bearish trend)"
    else:
        bb_status = "Rond middenband (neutraal, consolidatie)"

    # MACD
    if macd > signal and histogram > 0:
        macd_status = "Bullish (MACD boven signaallijn, koopsignaal)"
    elif macd < signal and histogram < 0:
        macd_status = "Bearish (MACD onder signaallijn, verkoopsignaal)"
    else:
        macd_status = "Neutraal (geen duidelijke crossover)"

    # SMA / EMA + Golden/Death Cross
    above_all_sma = all(price > last[f'SMA{p}'].item() for p in periods)
    below_all_sma = all(price < last[f'SMA{p}'].item() for p in periods)
    above_all_ema = all(price > last[f'EMA{p}'].item() for p in periods)
    below_all_ema = all(price < last[f'EMA{p}'].item() for p in periods)

    cross = ""
    if prev['SMA50'].item() <= prev['SMA200'].item() and last['SMA50'].item() > last['SMA200'].item():
        cross = "Golden Cross (sterk bullish, koopsignaal)"
    elif prev['SMA50'].item() >= prev['SMA200'].item() and last['SMA50'].item() < last['SMA200'].item():
        cross = "Death Cross (sterk bearish, verkoopsignaal)"

    if above_all_sma and above_all_ema:
        ma_status = "Sterk bullish (koers boven alle SMA/EMA)"
    elif below_all_sma and below_all_ema:
        ma_status = "Sterk bearish (koers onder alle SMA/EMA)"
    elif above_all_sma or above_all_ema:
        ma_status = "Bullish (koers boven meeste SMA/EMA)"
    elif below_all_sma or below_all_ema:
        ma_status = "Bearish (koers onder meeste SMA/EMA)"
    else:
        ma_status = "Neutraal (koers tussen SMA/EMA)"

    if cross:
        ma_status += f", {cross}"

    # ---------- 6. Samenvatting ----------
    summary = f"""Analyse voor **{ticker.upper()}** (laatste datum: {close_df.index[-1].strftime('%Y-%m-%d')}):
- RSI ({rsi_period}): {rsi_val:.2f} → **{rsi_status}**
- Bollinger Bands: **{bb_status}**
- MACD ({macd_fast},{macd_slow},{macd_signal}): **{macd_status}**
- SMA/EMA ({', '.join(map(str,periods))}): **{ma_status}**
"""
    print(summary)
    return summary
