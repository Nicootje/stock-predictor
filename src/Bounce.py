import pandas as pd
import yfinance as yf
from typing import Dict, Any
from src.calc_indicators import calc_sma_ema, calc_rsi, calc_bollinger_bands, calc_macd
from tqdm import tqdm

def _calc_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df['Vol_MA'] = df['Volume'].rolling(period).mean()
    return df

def _detect_candles(df: pd.DataFrame) -> list:
    signals = []
    for i in range(-2, 0):
        idx = len(df) + i - 1
        if idx < 1: continue
        p, c = df.iloc[idx-1], df.iloc[idx]
        if (p['Close'] < p['Open'] and c['Close'] > c['Open'] and
            c['Open'] < p['Close'] and c['Close'] > p['Open']):
            signals.append(("Bullish Engulfing", idx)); continue
        o, h, l, c_val = c[['Open','High','Low','Close']]
        body = abs(c_val - o); total = h - l
        if total > 0 and body <= total * 0.3:
            lower = min(o, c_val) - l
            upper = h - max(o, c_val)
            if lower >= body * 2 and upper <= body * 0.5:
                signals.append(("Hammer", idx))
    return signals

def _detect_bearish_candles(df: pd.DataFrame) -> list:
    signals = []
    for i in range(-2, 0):
        idx = len(df) + i - 1
        if idx < 1: continue
        p, c = df.iloc[idx-1], df.iloc[idx]
        if (p['Close'] > p['Open'] and c['Close'] < c['Open'] and
            c['Open'] > p['Close'] and c['Close'] < p['Open']):
            signals.append(("Bearish Engulfing", idx)); continue
        o, h, l, c_val = c[['Open','High','Low','Close']]
        body = abs(c_val - o); total = h - l
        if total > 0 and body <= total * 0.3:
            upper = h - max(o, c_val)
            lower = min(o, c_val) - l
            if upper >= body * 2 and lower <= body * 0.5:
                signals.append(("Shooting Star", idx))
    return signals

def _detect_divergence(df: pd.DataFrame, lookback: int = 30) -> str:
    recent = df.iloc[-lookback:]
    lows = recent['Low'].idxmin()
    prev_lows = recent['Low'][:recent.index.get_loc(lows)]
    if len(prev_lows) < 5: return ""
    prev_low_idx = prev_lows.idxmin()
    curr_low_idx = lows
    if recent.loc[curr_low_idx, 'Low'] >= recent.loc[prev_low_idx, 'Low']: return ""
    rsi_curr = recent.loc[curr_low_idx, 'RSI']; rsi_prev = recent.loc[prev_low_idx, 'RSI']
    if rsi_curr > rsi_prev: return f"RSI Bull Div (+{rsi_curr - rsi_prev:.1f})"
    macd_curr = recent.loc[curr_low_idx, 'Histogram']; macd_prev = recent.loc[prev_low_idx, 'Histogram']
    if macd_curr > macd_prev: return f"MACD Bull Div (+{macd_curr - macd_prev:.3f})"
    return ""

def _detect_bearish_divergence(df: pd.DataFrame, lookback: int = 30) -> str:
    recent = df.iloc[-lookback:]
    highs = recent['High'].idxmax()
    prev_highs = recent['High'][:recent.index.get_loc(highs)]
    if len(prev_highs) < 5: return ""
    prev_high_idx = prev_highs.idxmax()
    curr_high_idx = highs
    if recent.loc[curr_high_idx, 'High'] <= recent.loc[prev_high_idx, 'High']: return ""
    rsi_curr = recent.loc[curr_high_idx, 'RSI']; rsi_prev = recent.loc[prev_high_idx, 'RSI']
    if rsi_curr < rsi_prev: return f"RSI Bear Div (-{rsi_prev - rsi_curr:.1f})"
    macd_curr = recent.loc[curr_high_idx, 'Histogram']; macd_prev = recent.loc[prev_high_idx, 'Histogram']
    if macd_curr < macd_prev: return f"MACD Bear Div (-{macd_prev - macd_curr:.3f})"
    return ""

def _detect_volume_spike(df: pd.DataFrame, threshold: float = 3.0) -> str:
    r = df.iloc[-1]
    if 'Vol_MA' not in df.columns: return ""
    ratio = r['Volume'] / r['Vol_MA']
    if ratio > threshold and r['Close'] < r['SMA20']: return f"Vol Spike x{ratio:.1f}"
    return ""

def _detect_double_bottom(df: pd.DataFrame, window: int = 50, tolerance: float = 0.02) -> str:
    recent = df.iloc[-window:]
    lows = recent['Low']; low_idx = lows.idxmin(); price_low = lows.loc[low_idx]
    prev_lows = lows[:recent.index.get_loc(low_idx)]
    similar_lows = prev_lows[abs(prev_lows - price_low) / price_low < tolerance]
    if len(similar_lows) == 0: return ""
    prev_low_idx = similar_lows.idxmin()
    if low_idx <= prev_low_idx: return ""
    between = recent.loc[prev_low_idx:low_idx]
    if between['High'].max() < price_low * 1.05: return ""
    neckline = between['High'].max()
    if recent.iloc[-1]['Close'] > neckline: return f"DB → BO @ {neckline:.2f}"
    return f"DB (neck {neckline:.2f})"

def _detect_double_top(df: pd.DataFrame, window: int = 50, tolerance: float = 0.02) -> str:
    recent = df.iloc[-window:]
    highs = recent['High']; high_idx = highs.idxmax(); price_high = highs.loc[high_idx]
    prev_highs = highs[:recent.index.get_loc(high_idx)]
    similar_highs = prev_highs[abs(prev_highs - price_high) / price_high < tolerance]
    if len(similar_highs) == 0: return ""
    prev_high_idx = similar_highs.idxmax()
    if high_idx <= prev_high_idx: return ""
    between = recent.loc[prev_high_idx:high_idx]
    if between['Low'].min() > price_high * 0.95: return ""
    neckline = between['Low'].min()
    if recent.iloc[-1]['Close'] < neckline: return f"DT → BD @ {neckline:.2f}"
    return f"DT (neck {neckline:.2f})"

def _detect_head_and_shoulders(df: pd.DataFrame, window: int = 60, tolerance: float = 0.03) -> str:
    """
    Detecteert Head and Shoulders patroon.
    Retourneert: 'H&S → BD @ X.XX' of 'H&S (neck X.XX)'
    """
    recent = df.iloc[-window:]
    highs = recent['High']
    lows = recent['Low']

    # Zoek 3 hoogste pieken
    peak_indices = highs.nlargest(3).index
    if len(peak_indices) < 3:
        return ""
    
    # Sorteer op tijd (oud → nieuw)
    peak_indices = sorted(peak_indices)
    ls_idx, head_idx, rs_idx = peak_indices[-3], peak_indices[-2], peak_indices[-1]
    
    ls_price = highs.loc[ls_idx]
    head_price = highs.loc[head_idx]
    rs_price = highs.loc[rs_idx]

    # Hoofd moet duidelijk hoger zijn
    if not (head_price > ls_price * (1 + tolerance) and head_price > rs_price * (1 + tolerance)):
        return ""
    # Schouders mogen niet te ver uit elkaar liggen
    if abs(ls_price - rs_price) / head_price > tolerance * 2:
        return ""

    # Neckline = gemiddelde van de twee valleien
    left_valley = lows[ls_idx:head_idx].min()
    right_valley = lows[head_idx:rs_idx].min()
    neckline = (left_valley + right_valley) / 2

    current_close = recent.iloc[-1]['Close']
    if current_close < neckline * 0.98:  # 2% onder neckline = breakdown
        return f"H&S → BD @ {neckline:.2f}"
    else:
        return f"H&S (neck {neckline:.2f})"

def assess_trend(df: pd.DataFrame, start_date: str, periods=[20,50,200]) -> tuple:
    df = df.sort_index()
    if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
    df = df[df.index >= pd.to_datetime(start_date)]
    if len(df) < 200: return "Onvoldoende data", df
    df = calc_sma_ema(df, periods); df = calc_rsi(df); df = calc_macd(df); df = calc_bollinger_bands(df)
    df = _calc_volume_ma(df); df = df.dropna(subset=['RSI','MACD','BB_Lower','Vol_MA'])
    if df.empty: return "Geen data", df
    r, p = df.iloc[-1], df.iloc[-2]
    vol_up = r['Volume'] > r['Vol_MA'] * 1.5
    if (r['Close'] > r['BB_Upper'] and r['MACD'] > r['Signal'] and r['Histogram'] > p['Histogram'] and
        r['RSI'] > 50 and r['RSI'] > p['RSI'] and vol_up):
        return "Trendswitch", df
    elif (r['Close'] > r['BB_Lower'] and r['MACD'] > r['Signal'] and 40 < r['RSI'] < 60 and not vol_up):
        return "Rebound", df
    return "Uncertain", df

def trend_bounce_score(df: pd.DataFrame, start_date: str = '2025-01-01') -> Dict[str, Any]:
    trend, df_ind = assess_trend(df, start_date)
    candles = _detect_candles(df_ind)
    bear_candles = _detect_bearish_candles(df_ind)
    r = df_ind.iloc[-1]
    p = df_ind.iloc[-2] if len(df_ind) > 1 else r

    # === MOMENTUM: RSI + MACD + VOLUME (altijd pijl + %) ===
    momentum = []
    if r['RSI'] > 50:
        momentum.append(f"RSI ↑ {r['RSI']:.0f}")
    elif r['RSI'] < 50:
        momentum.append(f"RSI ↓ {r['RSI']:.0f}")
    else:
        momentum.append(f"RSI → {r['RSI']:.0f}")

    macd_val = r['MACD']
    signal_val = r['Signal']
    if macd_val > signal_val:
        momentum.append(f"MACD ↑ {macd_val:.3f}")
    elif macd_val < signal_val:
        momentum.append(f"MACD ↓ {macd_val:.3f}")
    else:
        momentum.append(f"MACD → {macd_val:.3f}")

    vol_ratio = r['Volume'] / r['Vol_MA'] if r['Vol_MA'] > 0 else 1.0
    if vol_ratio > 1.1:
        momentum.append(f"Vol ↑ {vol_ratio:.0%}")
    elif vol_ratio < 0.9:
        momentum.append(f"Vol ↓ {vol_ratio:.0%}")
    else:
        momentum.append(f"Vol → {vol_ratio:.0%}")

    momentum_str = " | ".join(momentum)

    # === LONG ===
    long_score = 0
    long_details = []
    if trend == "Trendswitch":
        long_score += 4
        long_details.append("Trendswitch")
    elif trend == "Rebound":
        long_score += 2
        long_details.append("Rebound")

    if (r['Close'] > r['BB_Lower'] and r['RSI'] > 30 and
        r['Volume'] > r['Vol_MA'] * 1.1 and candles):
        long_score += 5
        long_details.append(f"{candles[0][0]} + RSI {r['RSI']:.1f}")

    div = _detect_divergence(df_ind)
    if div:
        long_score += 3
        long_details.append(div)

    vol_spike = _detect_volume_spike(df_ind)
    if vol_spike:
        long_score += 2
        long_details.append(vol_spike)

    db = _detect_double_bottom(df_ind)
    if db:
        long_score += 4
        long_details.append(db)

    if r['Volume'] > r['Vol_MA'] * 1.5:
        long_score += 1
        long_details.append(f"Vol +{r['Volume']/r['Vol_MA']:.0%}")

    long_score = min(long_score, 10)

    # === SHORT ===
    short_score = 0
    short_details = []
    vol_down = r['Volume'] > r['Vol_MA'] * 1.5
    if (r['Close'] < r['BB_Lower'] and r['MACD'] < r['Signal'] and
        r['Histogram'] < p['Histogram'] and r['RSI'] < 50 and
        r['RSI'] < p['RSI'] and vol_down):
        short_score += 4
        short_details.append("Trendbreak")
    elif (r['Close'] < r['BB_Upper'] and r['MACD'] < r['Signal'] and
          40 < r['RSI'] < 60 and not vol_down):
        short_score += 2
        short_details.append("Oversold")

    if (r['Close'] < r['BB_Upper'] and r['RSI'] < 70 and
        r['Volume'] > r['Vol_MA'] * 1.1 and bear_candles):
        short_score += 5
        short_details.append(f"{bear_candles[0][0]} + RSI {r['RSI']:.1f}")

    bear_div = _detect_bearish_divergence(df_ind)
    if bear_div:
        short_score += 3
        short_details.append(bear_div)

    if r['Volume'] > r['Vol_MA'] * 3.0 and r['Close'] > r['SMA20']:
        short_score += 2
        short_details.append(f"Vol Spike x{r['Volume']/r['Vol_MA']:.1f}")

    dt = _detect_double_top(df_ind)
    if dt:
        short_score += 4
        short_details.append(dt)

    # === H&S TOEGEVOEGD ===
    hs = _detect_head_and_shoulders(df_ind)
    if hs:
        if "→ BD" in hs:
            short_score += 5
        else:
            short_score += 3
        short_details.append(hs)

    if r['Volume'] > r['Vol_MA'] * 1.5:
        short_score += 1
        short_details.append(f"Vol +{r['Volume']/r['Vol_MA']:.0%}")

    short_score = min(short_score, 10)

    # === PULLBACK ===
    pullback = False
    if long_score >= 9:
        in_lower_band = r['BB_Lower'] < r['Close'] < r['BB_Middle']
        rsi_ok = 45 <= r['RSI'] <= 60
        vol_calm = r['Volume'] < r['Vol_MA'] * 1.3
        above_sma20 = r['Close'] > r['SMA20']
        if in_lower_band and rsi_ok and vol_calm and above_sma20:
            pullback = True
            long_details.append(f"Pullback @ {r['Close']:.2f}")

    # === ACTIE ===
    if pullback:
        actie = "KOOP DIP"
    elif 6 <= long_score <= 8:
        actie = "KOOP VROEG"
    elif long_score >= 9:
        actie = "TE LAAT"
    elif short_score >= 7:
        actie = "STERKE SHORT"
    elif short_score >= 4:
        actie = "SHORT"
    elif long_score >= 4 or short_score >= 4:
        actie = "WACHT"
    else:
        actie = "NEUTRAAL"

    return {
        "long_score": long_score,
        "long_details": " | ".join(long_details) if long_details else "Geen long",
        "short_score": short_score,
        "short_details": " | ".join(short_details) if short_details else "Geen short",
        "actie": actie,
        "momentum": momentum_str
    }

def scan_portfolio(tickers: list, start_date: str = '2025-01-01', min_data_days: int = 50) -> pd.DataFrame:
    results = []
    print(f"Scanning {len(tickers)} aandelen...")
    for t in tqdm(tickers):
        try:
            df = yf.download(t, start="2023-01-01", progress=False, auto_adjust=True)
            if df.empty or len(df) < min_data_days: continue
            res = trend_bounce_score(df, start_date)
            results.append({"ticker": t.upper(), **res})
        except Exception as e:
            print(f"Fout bij {t}: {e}")
            continue

    if not results:
        print("Geen resultaten.")
        return pd.DataFrame()

    df_out = pd.DataFrame(results).sort_values("long_score", ascending=False).reset_index(drop=True)
    pd.set_option('display.max_colwidth', 80); pd.set_option('display.width', 600)

    print("\n" + "="*160)
    print("VOLLEDIGE SCANNER + DETAILS + MOMENTUM")
    print("="*160)
    display(df_out[["ticker", "long_score", "short_score", "actie", "momentum"]])

    # === KOOPKANSEN ===
    koop = df_out[df_out["actie"].isin(["KOOP DIP", "KOOP VROEG"])]
    if not koop.empty:
        print("\n" + "KOOPKANSEN".center(80, " "))
        print("="*80)
        display(koop[["ticker", "actie", "long_details", "momentum"]])

    # === VERKOOPKANSEN ===
    verkoop = df_out[df_out["actie"].isin(["STERKE SHORT", "SHORT"])]
    if not verkoop.empty:
        print("\n" + "VERKOOPKANSEN".center(80, " "))
        print("="*80)
        display(verkoop[["ticker", "actie", "short_details", "momentum"]])

    if koop.empty and verkoop.empty:
        print("\nGeen directe kansen.")

    return df_out