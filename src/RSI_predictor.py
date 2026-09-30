import numpy as np
import pandas as pd
from src.calc_indicators import _single_ticker, _wilder_mean, _validate_period


def price_for_target_rsi(df, target_rsi=70, rsi_period=14, ticker=None):
    """Volgende slotkoers voor de gewenste RSI, vanuit de volledige historie.

    Gebruikt dezelfde Wilder-toestand als calc_rsi, zonder invoer te wijzigen.
    De laatste aangeleverde slotkoers is het uitgangspunt. Dit is een scenario
    voor één volgende candle, geen voorspelling van prijs of aankomsttijd.
    Retourneert een float, of None bij een onbereikbaar doel/niet-positieve
    koers. Vanuit vlakke historie hebben RSI 0/100 geen unieke doelprijs;
    ook dan volgt None. Bij dezelfde RSI blijft de koers ongewijzigd.
    """
    rsi_period = _validate_period(rsi_period)
    if rsi_period < 2:
        raise ValueError("rsi_period moet minstens 2 zijn voor dit scenario.")
    if not np.isfinite(target_rsi) or not 0 <= target_rsi <= 100:
        raise ValueError("target_rsi moet tussen 0 en 100 liggen.")

    # ---- Lees de slotkoersen zonder de aangeleverde data te wijzigen ----
    data = _single_ticker(df, ticker)
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.hasnans or data.index.has_duplicates:
        raise ValueError("Koersdata vereist unieke, geldige datums.")
    close = pd.to_numeric(data.sort_index()['Close'], errors='raise').astype(float)
    if len(close) < rsi_period + 1:
        raise ValueError(f"Minstens {rsi_period + 1} slotkoersen nodig.")
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Close moet positieve, eindige koersen bevatten zonder ontbrekende waarden.")

    # ---- Dezelfde gemiddelde winst en verlies als in calc_rsi ----
    delta = close.diff()
    avg_gain = float(_wilder_mean(delta.clip(lower=0), rsi_period).iloc[-1])
    avg_loss = float(_wilder_mean(-delta.clip(upper=0), rsi_period).iloc[-1])
    price = float(close.iloc[-1])
    total = avg_gain + avg_loss
    current_rsi = 50.0 if total == 0 else 100 * avg_gain / total
    if abs(target_rsi - current_rsi) < 1e-12:
        return price
    if target_rsi in (0, 100) or total == 0:
        return None

    # ---- Los Wilder's volgende stap op naar de prijsverandering ----
    target_ratio = target_rsi / (100 - target_rsi)
    if target_rsi > current_rsi:
        change = (rsi_period - 1) * (target_ratio * avg_loss - avg_gain)
    else:
        change = (rsi_period - 1) * (avg_loss - avg_gain / target_ratio)
    target_price = price + change
    return float(target_price) if np.isfinite(target_price) and target_price > 0 else None
