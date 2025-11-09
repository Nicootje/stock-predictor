import pandas as pd
import yfinance as yf
import numpy as np
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from src.calc_indicators import calc_sma_ema, calc_rsi, calc_bollinger_bands, calc_macd

# ============================================================
#  HELPER FUNCTIES - OPTIMIZED + ENHANCED
# ============================================================

def _safe_last_value(df: pd.DataFrame, col: str) -> float:
    """Geeft laatste waarde van een kolom (scalar, zonder FutureWarning)."""
    if col not in df.columns:
        for c in df.columns:
            if isinstance(c, tuple) and c[0] == col:
                return df[c].iloc[-1]
        return np.nan
    return df[col].iloc[-1]

def _calc_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Bereken ADX (Average Directional Index) voor trendsterkte."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    
    df['ADX'] = adx
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    return df

def _calc_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Volume moving average + ratio."""
    df['Vol_MA'] = df['Volume'].rolling(period).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Vol_MA']
    return df

def _get_market_regime(df: pd.DataFrame) -> str:
    """Bepaal marktregime op basis van ADX en prijsactie."""
    r = df.iloc[-1]
    
    if pd.isna(r.get('ADX', np.nan)):
        return "UNCERTAIN"
    
    adx = r['ADX']
    plus_di = r.get('Plus_DI', 0)
    minus_di = r.get('Minus_DI', 0)
    
    if adx > 25:
        if plus_di > minus_di:
            return "STRONG_UPTREND"
        else:
            return "STRONG_DOWNTREND"
    elif adx < 20:
        return "RANGING"
    else:
        return "WEAK_TREND"

# ============================================================
#  CANDLE PATROON DETECTIE - CONTEXTUAL (v2) + ENHANCED
# ============================================================

def _detect_candles(df: pd.DataFrame) -> List[Tuple[str, int]]:
    """Detecteert bullish candles met context."""
    signals = []
    if len(df) < 3:
        return signals
        
    for i in range(-3, 0):
        idx = len(df) + i
        if idx < 2:
            continue
        p2, p1, c = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
        
        # Bullish Engulfing - alleen in downtrend
        if (p1['Close'] < p1['Open'] and c['Close'] > c['Open'] and
            c['Open'] < p1['Close'] and c['Close'] > p1['Open'] and
            p1['Close'] < p1['SMA20']):
            signals.append(("Bullish Engulfing", idx))
            continue
            
        # Hammer - near support
        body = abs(c['Close'] - c['Open'])
        total_range = c['High'] - c['Low']
        if total_range > 0 and body <= total_range * 0.3:
            lower_shadow = min(c['Open'], c['Close']) - c['Low']
            upper_shadow = c['High'] - max(c['Open'], c['Close'])
            if lower_shadow >= body * 2 and upper_shadow <= body:
                if c['Low'] <= c['BB_Lower'] * 1.02:
                    signals.append(("Hammer", idx))
    
    return signals

def _detect_bearish_candles(df: pd.DataFrame) -> List[Tuple[str, int]]:
    """Detecteert bearish candles met context."""
    signals = []
    if len(df) < 3:
        return signals
        
    for i in range(-3, 0):
        idx = len(df) + i
        if idx < 2:
            continue
        p2, p1, c = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
        
        # Bearish Engulfing - alleen in uptrend
        if (p1['Close'] > p1['Open'] and c['Close'] < c['Open'] and
            c['Open'] > p1['Close'] and c['Close'] < p1['Open'] and
            p1['Close'] > p1['SMA20']):
            signals.append(("Bearish Engulfing", idx))
            continue
            
        # Shooting Star - near resistance
        body = abs(c['Close'] - c['Open'])
        total_range = c['High'] - c['Low']
        if total_range > 0 and body <= total_range * 0.3:
            upper_shadow = c['High'] - max(c['Open'], c['Close'])
            lower_shadow = min(c['Open'], c['Close']) - c['Low']
            if upper_shadow >= body * 2 and lower_shadow <= body:
                if c['High'] >= c['BB_Upper'] * 0.98:
                    signals.append(("Shooting Star", idx))
    
    return signals

# ============================================================
#  DIVERGENCE DETECTIE - ROBUST (v2)
# ============================================================

def _detect_divergences(df: pd.DataFrame, lookback: int = 30) -> Tuple[str, str]:
    """Detecteert bullish/bearish divergences (RSI or MACD)."""
    recent = df.iloc[-lookback:]
    bullish_div = bearish_div = ""
    
    price_lows = recent['Low'].nsmallest(3)
    price_highs = recent['High'].nlargest(3)
    
    # Bullish divergence
    if len(price_lows) >= 2:
        low1_idx, low2_idx = price_lows.index[-2], price_lows.index[-1]
        if low1_idx < low2_idx:
            price_lower = recent.loc[low2_idx, 'Low'] < recent.loc[low1_idx, 'Low']
            rsi_higher = recent.loc[low2_idx, 'RSI'] > recent.loc[low1_idx, 'RSI']
            macd_higher = recent.loc[low2_idx, 'Histogram'] > recent.loc[low1_idx, 'Histogram']
            if price_lower and (rsi_higher or macd_higher):
                if rsi_higher:
                    bullish_div = f"RSI Bull Div (+{recent.loc[low2_idx, 'RSI'] - recent.loc[low1_idx, 'RSI']:.1f})"
                elif macd_higher:
                    bullish_div = f"MACD Bull Div (+{recent.loc[low2_idx, 'Histogram'] - recent.loc[low1_idx, 'Histogram']:.3f})"
    
    # Bearish divergence
    if len(price_highs) >= 2:
        high1_idx, high2_idx = price_highs.index[-2], price_highs.index[-1]
        if high1_idx < high2_idx:
            price_higher = recent.loc[high2_idx, 'High'] > recent.loc[high1_idx, 'High']
            rsi_lower = recent.loc[high2_idx, 'RSI'] < recent.loc[high1_idx, 'RSI']
            macd_lower = recent.loc[high2_idx, 'Histogram'] < recent.loc[high1_idx, 'Histogram']
            if price_higher and (rsi_lower or macd_lower):
                if rsi_lower:
                    bearish_div = f"RSI Bear Div (-{recent.loc[high1_idx, 'RSI'] - recent.loc[high2_idx, 'RSI']:.1f})"
                elif macd_lower:
                    bearish_div = f"MACD Bear Div (-{recent.loc[high1_idx, 'Histogram'] - recent.loc[high2_idx, 'Histogram']:.3f})"
    
    return bullish_div, bearish_div

# ============================================================
#  PATROON DETECTIE - ENHANCED (v1 style necklines)
# ============================================================

def _detect_double_bottom(df: pd.DataFrame, window: int = 50, tolerance: float = 0.02) -> str:
    recent = df.iloc[-window:]
    lows = recent['Low']
    low_idx = lows.idxmin()
    price_low = lows.loc[low_idx]
    prev_lows = lows[:recent.index.get_loc(low_idx)]
    similar_lows = prev_lows[abs(prev_lows - price_low) / price_low < tolerance]
    if len(similar_lows) == 0:
        return ""
    prev_low_idx = similar_lows.idxmin()
    if low_idx <= prev_low_idx:
        return ""
    between = recent.loc[prev_low_idx:low_idx]
    if between['High'].max() < price_low * 1.05:
        return ""
    neckline = between['High'].max()
    current_close = recent.iloc[-1]['Close']
    if current_close > neckline:
        return f"DB → BO @ {neckline:.2f}"
    return f"DB (neck {neckline:.2f})"

def _detect_double_top(df: pd.DataFrame, window: int = 50, tolerance: float = 0.02) -> str:
    recent = df.iloc[-window:]
    highs = recent['High']
    high_idx = highs.idxmax()
    price_high = highs.loc[high_idx]
    prev_highs = highs[:recent.index.get_loc(high_idx)]
    similar_highs = prev_highs[abs(prev_highs - price_high) / price_high < tolerance]
    if len(similar_highs) == 0:
        return ""
    prev_high_idx = similar_highs.idxmax()
    if high_idx <= prev_high_idx:
        return ""
    between = recent.loc[prev_high_idx:high_idx]
    if between['Low'].min() > price_high * 0.95:
        return ""
    neckline = between['Low'].min()
    current_close = recent.iloc[-1]['Close']
    if current_close < neckline:
        return f"DT → BD @ {neckline:.2f}"
    return f"DT (neck {neckline:.2f})"

def _detect_head_and_shoulders(df: pd.DataFrame, window: int = 60, tolerance: float = 0.03) -> str:
    recent = df.iloc[-window:]
    highs = recent['High']
    lows = recent['Low']
    peak_indices = highs.nlargest(3).index
    if len(peak_indices) < 3:
        return ""
    peak_indices = sorted(peak_indices)
    ls_idx, head_idx, rs_idx = peak_indices[-3], peak_indices[-2], peak_indices[-1]
    ls_price, head_price, rs_price = highs.loc[ls_idx], highs.loc[head_idx], highs.loc[rs_idx]
    if not (head_price > ls_price * (1 + tolerance) and head_price > rs_price * (1 + tolerance)):
        return ""
    if abs(ls_price - rs_price) / head_price > tolerance * 2:
        return ""
    left_valley = lows[ls_idx:head_idx].min()
    right_valley = lows[head_idx:rs_idx].min()
    neckline = (left_valley + right_valley) / 2
    current_close = recent.iloc[-1]['Close']
    if current_close < neckline * 0.98:
        return f"H&S → BD @ {neckline:.2f}"
    else:
        return f"H&S (neck {neckline:.2f})"

# ============================================================
#  TREND ASSESSMENT
# ============================================================

def assess_trend_with_regime(df: pd.DataFrame, start_date: str) -> Tuple[str, pd.DataFrame]:
    df = df.sort_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df[df.index >= pd.to_datetime(start_date)]
    if len(df) < 200:
        return "INSUFFICIENT_DATA", df
    
    df = calc_sma_ema(df, [20, 50, 200])
    df = calc_rsi(df)
    df = calc_macd(df)
    df = calc_bollinger_bands(df)
    df = _calc_volume_ma(df)
    df = _calc_adx(df)
    df = df.dropna(subset=['RSI', 'MACD', 'BB_Lower', 'Vol_MA', 'ADX'])
    
    if df.empty:
        return "NO_DATA", df
    
    return _get_market_regime(df), df

# ============================================================
#  SCORING ENGINE - BEST OF BOTH
# ============================================================

def calculate_scores(df: pd.DataFrame, regime: str) -> Dict[str, Any]:
    r = df.iloc[-1]
    p = df.iloc[-2] if len(df) > 1 else r
    
    rsi_dir = 1 if r['RSI'] > p['RSI'] else -1 if r['RSI'] < p['RSI'] else 0
    macd_dir = 1 if r['MACD'] > r['Signal'] else -1
    volume_ratio = r['Volume_Ratio']
    
    long_score = short_score = 0
    long_signals = []
    short_signals = []
    
    # MARKT REGIME
    if regime == "STRONG_UPTREND":
        long_score += 2
    elif regime == "STRONG_DOWNTREND":
        short_score += 2
    
    # MOMENTUM
    if rsi_dir == 1 and macd_dir == 1:
        long_signals.append("Momentum ↑")
    if rsi_dir == -1 and macd_dir == -1:
        short_signals.append("Momentum ↓")
    
    # OVERSOLD / OVERBOUGHT
    if r['RSI'] < 35 and rsi_dir == 1:
        long_signals.append("Oversold Bounce")
    if r['RSI'] > 65 and rsi_dir == -1:
        short_signals.append("Overbought Rejection")
    
    # SUPPORT / RESISTANCE
    if r['Close'] <= r['BB_Lower'] * 1.02:
        long_signals.append("Near Support")
    if r['Close'] >= r['BB_Upper'] * 0.98:
        short_signals.append("Near Resistance")
    
    # VOLUME
    if volume_ratio > 1.5 and rsi_dir == 1:
        long_signals.append("Volume Support")
    if volume_ratio > 1.5 and rsi_dir == -1:
        short_signals.append("Volume Pressure")
    
    # VOLUME SPIKE (v1 style)
    if volume_ratio > 2.0:
        spike = f"Vol Spike x{volume_ratio:.1f}"
        if long_score > short_score:
            long_signals.append(spike)
        elif short_score > long_score:
            short_signals.append(spike)
    
    # PATTERNS
    candles = _detect_candles(df)
    bear_candles = _detect_bearish_candles(df)
    bullish_div, bearish_div = _detect_divergences(df)
    db = _detect_double_bottom(df)
    dt = _detect_double_top(df)
    hs = _detect_head_and_shoulders(df)
    
    if candles:
        long_signals.append(candles[0][0])
    if bear_candles:
        short_signals.append(bear_candles[0][0])
    if bullish_div:
        long_signals.append(bullish_div)
    if bearish_div:
        short_signals.append(bearish_div)
    if db:
        long_signals.append(db)
    if dt:
        short_signals.append(dt)
    if hs:
        short_signals.append(hs)
    
    # FINAL SCORE
    long_score += min(len(long_signals) * 2, 6)
    short_score += min(len(short_signals) * 2, 6)
    
    # Volume bonus
    if volume_ratio > 2.0:
        if long_score > short_score:
            long_score += 1
        elif short_score > long_score:
            short_score += 1
    
    long_score = min(long_score, 10)
    short_score = min(short_score, 10)
    
    return {
        'long_score': long_score,
        'short_score': short_score,
        'long_details': " | ".join(long_signals) if long_signals else "Geen signals",
        'short_details': " | ".join(short_signals) if short_signals else "Geen signals",
        'regime': regime,
        'volume_ratio': volume_ratio
    }

# ============================================================
#  HOOFDFUNCTIE
# ============================================================

def trend_bounce_score_v3(df: pd.DataFrame, start_date: str = '2025-01-01') -> Dict[str, Any]:
    regime, df_ind = assess_trend_with_regime(df, start_date)
    
    if regime in ["INSUFFICIENT_DATA", "NO_DATA"]:
        return {
            "long_score": 0, "long_details": regime,
            "short_score": 0, "short_details": regime,
            "actie": "GEEN DATA", "momentum": "Onvoldoende data", "regime": regime
        }
    
    r = df_ind.iloc[-1]
    p = df_ind.iloc[-2] if len(df_ind) > 1 else r
    scores = calculate_scores(df_ind, regime)
    
    # Momentum string
    rsi_dir = "↑" if r['RSI'] > p['RSI'] else "↓" if r['RSI'] < p['RSI'] else "→"
    macd_dir = "↑" if r['MACD'] > r['Signal'] else "↓" if r['MACD'] < r['Signal'] else "→"
    vol_dir = "↑" if r['Volume_Ratio'] > 1.2 else "↓" if r['Volume_Ratio'] < 0.8 else "→"
    momentum_str = f"RSI {rsi_dir} {r['RSI']:.0f} | MACD {macd_dir} | Vol {vol_dir} | ADX {r['ADX']:.0f}"
    
    # Pullback
    pullback = False
    if regime == "STRONG_UPTREND" and scores['long_score'] >= 6:
        in_pullback = (r['Close'] < r['SMA20'] and r['Close'] > r['BB_Lower'] and 40 <= r['RSI'] <= 60)
        if in_pullback:
            pullback = True
            scores['long_details'] += " | Pullback"
    
    # ACTIE
    long_score = scores['long_score']
    short_score = scores['short_score']
    
    if regime == "RANGING":
        if long_score >= 7 and long_score > short_score + 2:
            actie = "RANGE LONG"
        elif short_score >= 7 and short_score > long_score + 2:
            actie = "RANGE SHORT"
        else:
            actie = "RANGE NEUTRAL"
    elif pullback and long_score >= 6:
        actie = "TREND PULLBACK"
    elif long_score >= 8:
        actie = "STRONG LONG"
    elif short_score >= 8:
        actie = "STRONG SHORT"
    elif long_score >= 6:
        actie = "MILD LONG"
    elif short_score >= 6:
        actie = "MILD SHORT"
    else:
        actie = "NEUTRAL"
    
    return {
        "long_score": long_score,
        "long_details": scores['long_details'],
        "short_score": short_score,
        "short_details": scores['short_details'],
        "actie": actie,
        "momentum": momentum_str,
        "regime": regime
    }

# ============================================================
#  PORTFOLIO SCANNER
# ============================================================

def scan_portfolio(tickers: list, start_date: str = '2025-01-01', min_data_days: int = 50) -> pd.DataFrame:
    results = []
    print(f"Scanning {len(tickers)} aandelen...")
    
    for t in tqdm(tickers):
        try:
            df = yf.download(t, start="2023-01-01", progress=False, auto_adjust=True)
            if df.empty or len(df) < min_data_days:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
                
            res = trend_bounce_score_v3(df, start_date)
            price = _safe_last_value(df, 'Close')
            price = round(float(price), 2) if not pd.isna(price) else None
            
            results.append({"ticker": t.upper(), "price": price, **res})
        except Exception as e:
            continue

    if not results:
        print("Geen resultaten.")
        return pd.DataFrame()

    df_out = pd.DataFrame(results)
    df_out['net_score'] = df_out['long_score'] - df_out['short_score']
    df_out = df_out.sort_values(["actie", "net_score"], ascending=[True, False])
    
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.width', 800)
    
    print("\n" + "="*120)
    print("TREND BOUNCE SCANNER V3 - BEST OF BOTH WORLDS")
    print("="*120)
    
    display_columns = ["ticker", "price", "regime", "long_score", "short_score", "actie", "momentum"]
    display(df_out[display_columns])
    
    long_ops = df_out[df_out["actie"].str.contains("LONG")]
    pullback_ops = df_out[df_out["actie"] == "TREND PULLBACK"]
    short_ops = df_out[df_out["actie"].str.contains("SHORT")]
    
    if not long_ops.empty:
        print("\n" + "LONG KANSEN".center(80, " "))
        print("="*80)
        display(long_ops[["ticker", "price", "actie", "long_details", "regime"]])
    
    if not pullback_ops.empty:
        print("\n" + "PULLBACK KANSEN".center(80, " "))
        print("="*80)
        display(pullback_ops[["ticker", "price", "actie", "long_details", "regime"]])
    
    if not short_ops.empty:
        print("\n" + "SHORT KANSEN".center(80, " "))
        print("="*80)
        display(short_ops[["ticker", "price", "actie", "short_details", "regime"]])
    
    if long_ops.empty and short_ops.empty and pullback_ops.empty:
        print("\nGeen duidelijke kansen - markt in consolidatie.")
    
    return df_out

# ============================================================
#  USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    test_tickers = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD", "META", "GOOGL"]
    results_df = scan_portfolio(test_tickers)
    print(f"\nTotaal signals: {len(results_df[results_df['actie'] != 'NEUTRAL'])}")