"""Standalone price/RSI divergence chart using the project's existing RSI formula.

No downloads, changes to the input frame, or changes to existing notebook code.
All periods refer to observations (daily candles in Beurs.ipynb).
"""
from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calc_indicators import calc_rsi


_PRESETS = {
    "short": dict(window=126, order=2, min_distance=3, max_distance=30),
    "medium": dict(window=252, order=5, min_distance=10, max_distance=90),
    "long": dict(window=756, order=10, min_distance=20, max_distance=180),
}
_COLUMNS = ["direction", "first_pivot", "second_pivot", "confirmed_on",
            "first_price", "second_price", "first_rsi", "second_rsi",
            "price_change_pct", "rsi_change", "age_bars"]


def _close_data(df, ticker):
    if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df moet een DataFrame met een DatetimeIndex zijn.")
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        price_levels = [i for i in range(data.columns.nlevels)
                        if "Close" in data.columns.get_level_values(i)]
        if data.columns.nlevels != 2 or len(price_levels) != 1:
            raise ValueError("Onbekende indeling van koerskolommen.")
        level = 1 - price_levels[0]
        matches = [name for name in data.columns.get_level_values(level).unique()
                   if str(name).upper() == ticker.upper()]
        if len(matches) != 1:
            raise ValueError(f"Ticker {ticker} niet eenduidig gevonden in df.")
        data = data.xs(matches[0], axis=1, level=level)
    if data.columns.has_duplicates or "Close" not in data:
        raise ValueError("df moet precies één Close-kolom voor de ticker bevatten.")
    data = data[["Close"]].copy()
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    if data.index.has_duplicates or data.index.hasnans:
        raise ValueError("Koersdatums mogen niet dubbel of ontbrekend zijn.")
    data = data.sort_index()
    data["Close"] = pd.to_numeric(data.Close, errors="raise").astype(float)
    if not np.isfinite(data.Close).all() or (data.Close <= 0).any():
        raise ValueError("Close bevat ontbrekende, niet-eindige of niet-positieve koersen.")
    return data


def _pivots(values, order, low):
    if len(values) < 2 * order + 1:
        return np.array([], dtype=int)
    windows = np.lib.stride_tricks.sliding_window_view(values, 2 * order + 1)
    center = windows[:, order]
    reducer = np.min if low else np.max
    left = reducer(windows[:, :order], axis=1)
    right = reducer(windows[:, order + 1:], axis=1)
    extreme = ((center < left) & (center < right) if low else
               (center > left) & (center > right))
    return np.flatnonzero(extreme) + order


def plot_rsi_divergence(df, ticker, trend="medium", rsi_period=14,
                        start_plot_date=None, *, last_bar_complete=False, show=True):
    """Plot regular bullish/bearish divergence between consecutive Close pivots.

    short: 126 visible bars, 2 confirmation bars, pivots 3–30 bars apart.
    medium: 252 visible bars, 5 confirmation bars, pivots 10–90 bars apart.
    long: 756 visible bars, 10 confirmation bars, pivots 20–180 bars apart.

    Trend selects swing scale, not long/short trade direction. These settings
    are exploratory, not calibrated predictors. RSI uses calc_rsi unchanged;
    its first rsi_period observations are excluded from detection as warm-up.
    A bullish event has a lower Close low and higher RSI at that SAME pivot;
    a bearish event has a higher Close high and lower RSI at that SAME pivot.
    Flat/tied extrema are not pivots. An event exists only order bars later.

    The last supplied bar is excluded unless last_bar_complete=True. Compute
    RSI and events on full history BEFORE limiting the chart to the preset's
    window and optional start_plot_date. Only fully visible pairs are drawn.
    Returns figure, divergences (chart IDs), all_divergences and settings.
    """
    if trend not in _PRESETS:
        raise ValueError("trend moet 'short', 'medium' of 'long' zijn.")
    if isinstance(rsi_period, bool) or not isinstance(rsi_period, Integral) or rsi_period < 2:
        raise ValueError("rsi_period moet een geheel getal van minstens 2 zijn.")
    if not isinstance(last_bar_complete, bool):
        raise ValueError("last_bar_complete moet True of False zijn.")
    settings = _PRESETS[trend].copy()
    data = _close_data(df, ticker)
    if not last_bar_complete:
        data = data.iloc[:-1].copy()
    if data.empty:
        raise ValueError("Geen candles beschikbaar voor divergentieanalyse.")
    data = calc_rsi(data, rsi_period)
    data.loc[data.index[:rsi_period], "RSI"] = np.nan
    close, rsi = data.Close.to_numpy(), data.RSI.to_numpy()
    events = []
    order = settings["order"]
    for low, direction in ((True, "bullish"), (False, "bearish")):
        pivots = _pivots(close, order, low)
        for a, b in zip(pivots, pivots[1:]):
            if not settings["min_distance"] <= b - a <= settings["max_distance"]:
                continue
            if not np.isfinite([rsi[a], rsi[b]]).all():
                continue
            match = (close[b] < close[a] and rsi[b] > rsi[a]) if low else (
                close[b] > close[a] and rsi[b] < rsi[a])
            if match:
                events.append(dict(direction=direction, first_pivot=data.index[a],
                    second_pivot=data.index[b], confirmed_on=data.index[b + order],
                    first_price=close[a], second_price=close[b], first_rsi=rsi[a], second_rsi=rsi[b],
                    price_change_pct=(close[b] / close[a] - 1) * 100,
                    rsi_change=rsi[b] - rsi[a], age_bars=len(data) - 1 - (b + order)))
    all_events = pd.DataFrame(events, columns=_COLUMNS).sort_values(
        ["confirmed_on", "direction"], kind="stable").reset_index(drop=True)
    visible = data.tail(settings["window"])
    if start_plot_date is not None:
        start = pd.Timestamp(start_plot_date)
        if pd.isna(start):
            raise ValueError("Ongeldige start_plot_date.")
        if start.tzinfo is not None:
            start = start.tz_localize(None)
        visible = visible.loc[visible.index >= start]
    if visible.empty:
        raise ValueError("Geen candles binnen de gekozen plotperiode.")
    plotted = all_events.loc[all_events.first_pivot >= visible.index[0]].copy().reset_index(drop=True)
    plotted.insert(0, "id", np.arange(1, len(plotted) + 1))

    fig, (price_ax, rsi_ax) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
        layout="constrained", gridspec_kw={"height_ratios": [3, 1.5]})
    price_ax.plot(visible.index, visible.Close, color="blue", label="Slotkoers", linewidth=1.2)
    rsi_ax.plot(visible.index, visible.RSI, color="purple", label=f"RSI ({rsi_period})")
    rsi_ax.axhline(30, color="gray", linestyle="--", linewidth=.8)
    rsi_ax.axhline(70, color="gray", linestyle="--", linewidth=.8)
    seen = set()
    for event in plotted.itertuples():
        bull = event.direction == "bullish"
        color, marker, style = ("#0072B2", "^", "-") if bull else ("#D55E00", "v", "--")
        label = ("Bullish (B)" if bull else "Bearish (S)") if event.direction not in seen else None
        seen.add(event.direction)
        dates = [event.first_pivot, event.second_pivot]
        price_ax.plot(dates, [event.first_price, event.second_price], color=color, linestyle=style, marker="o", linewidth=2)
        rsi_ax.plot(dates, [event.first_rsi, event.second_rsi], color=color, linestyle=style, marker="o", linewidth=2)
        price = data.loc[event.confirmed_on, "Close"]
        price_ax.scatter(event.confirmed_on, price, color=color, marker=marker, s=65, label=label, zorder=5)
        price_ax.annotate(f"{'B' if bull else 'S'}{event.id}", (event.confirmed_on, price),
                          xytext=(4, 8), textcoords="offset points", fontsize=9)
        for ax in (price_ax, rsi_ax):
            ax.axvline(event.confirmed_on, color="gray", linestyle=":", alpha=.35)
    price_ax.set(title=f"{ticker.upper()} — RSI-divergentie | {trend} | {data.index[-1]:%Y-%m-%d}",
                 ylabel="Koers (noteringsvaluta)")
    info = f"Bevestiging na {order} candles. B / omhoog: bullish; S / omlaag: bearish."
    if plotted.empty:
        info += "\nGeen bevestigde divergenties in dit venster."
    if not last_bar_complete:
        info += "\nLaatste aangeleverde candle uitgesloten."
    price_ax.text(.01, .01, info, transform=price_ax.transAxes, fontsize=8,
                  bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    rsi_ax.set(ylabel="RSI", ylim=(0, 100))
    for ax in (price_ax, rsi_ax):
        ax.grid(linestyle="--", alpha=.25)
        ax.legend(loc="upper left")
    if show:
        plt.show()
    return dict(figure=fig, divergences=plotted, all_divergences=all_events,
                settings=dict(trend=trend, rsi_period=rsi_period, **settings,
                              last_bar_complete=last_bar_complete, as_of=data.index[-1]))
