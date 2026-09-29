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
    "short": dict(order=2, min_distance=5, max_distance=21),
    "medium": dict(order=5, min_distance=21, max_distance=63),
    "long": dict(order=10, min_distance=63, max_distance=126),
}
_COLUMNS = ["direction", "first_pivot", "second_pivot", "confirmed_on",
            "first_price", "second_price", "first_rsi", "second_rsi",
            "price_change_pct", "rsi_change", "span_bars", "age_bars", "status", "detected_on"]


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


def _current_divergences(latest, data, settings):
    """Gedeelde voorlopige detectie voor de grafiek en de watchlistscanner."""
    close, rsi = data.Close.to_numpy(), data.RSI.to_numpy()
    order = settings["order"]
    current_events = []
    b = len(latest) - 1
    for low, direction in ((True, "bullish"), (False, "bearish")):
        if b < order or not np.isfinite(latest.RSI.iloc[b]):
            continue
        price, current_rsi = latest.Close.iloc[b], latest.RSI.iloc[b]
        previous = latest.Close.iloc[b-order:b]
        extreme = price < previous.min() if low else price > previous.max()
        if not extreme:
            continue
        pivots = _pivots(close, order, low)
        eligible = pivots[(b - pivots >= settings["min_distance"]) &
                          (b - pivots <= settings["max_distance"])]
        if not len(eligible):
            continue
        a = eligible[-1]
        if not np.isfinite(rsi[a]):
            continue
        match = (price < close[a] and current_rsi > rsi[a]) if low else (
            price > close[a] and current_rsi < rsi[a])
        if match:
            current_events.append(dict(direction=direction, first_pivot=data.index[a],
                second_pivot=latest.index[b], confirmed_on=pd.NaT,
                first_price=close[a], second_price=price, first_rsi=rsi[a], second_rsi=current_rsi,
                price_change_pct=(price / close[a] - 1) * 100, rsi_change=current_rsi-rsi[a],
                span_bars=int(b-a), age_bars=0, status="PRELIMINARY", detected_on=latest.index[b]))
    return pd.DataFrame(current_events, columns=_COLUMNS)


def plot_rsi_divergence(df, ticker, trend="medium", rsi_period=14,
                        start_plot_date=None, *, last_bar_complete=False, show=True,
                        include_current=True):
    """Plot regular bullish/bearish divergence at the selected swing scale.

    short: pivots 5–21 bars apart (about 1–4 weeks), 2 confirmation bars.
    medium: pivots 21–63 bars apart (about 1–3 months), 5 confirmation bars.
    long: pivots 63–126 bars apart (about 3–6 months), 10 confirmation bars.
    These calendar approximations assume daily exchange sessions.
    Each pivot is compared with the most recent earlier pivot of the same
    type INSIDE the distance range. Closer pivots are skipped; we do not pick
    an older pair just because it yields a divergence. span_bars reports the
    actual distance. The same rule is evaluated throughout the plot period.

    Trend selects swing scale, not long/short trade direction. These settings
    are exploratory, not calibrated predictors. RSI uses calc_rsi unchanged;
    its first rsi_period observations are excluded from detection as warm-up.
    A bullish event has a lower Close low and higher RSI at that SAME pivot;
    a bearish event has a higher Close high and lower RSI at that SAME pivot.
    Flat/tied extrema are not pivots. An event exists only order bars later.

    include_current=True also compares the latest supplied price/RSI with an
    earlier confirmed pivot, without waiting for future bars. The latest price
    must be a strict extremum relative to the preceding order bars. This is a
    PRELIMINARY snapshot, not a confirmed turning point; it may disappear on
    the next update. No quote is downloaded: freshness depends on supplied df.
    last_bar_complete=False excludes the latest bar from CONFIRMED detection,
    but still includes it in the preliminary snapshot and chart. Set
    include_current=False for the previous confirmed-only behaviour. Compute
    RSI and events on full history BEFORE applying start_plot_date, which is
    the sole display cutoff in EVERY mode. None shows all available history.
    Only fully visible pairs are drawn; a wider plot may be needed for long
    divergences. Changing the plot start never changes computed indicators.
    Returns figure, divergences (chart IDs), all_divergences (confirmed history),
    current_divergences (latest snapshot, even outside the display) and settings.
    """
    if trend not in _PRESETS:
        raise ValueError("trend moet 'short', 'medium' of 'long' zijn.")
    if isinstance(rsi_period, bool) or not isinstance(rsi_period, Integral) or rsi_period < 2:
        raise ValueError("rsi_period moet een geheel getal van minstens 2 zijn.")
    if not isinstance(last_bar_complete, bool):
        raise ValueError("last_bar_complete moet True of False zijn.")
    if not isinstance(include_current, bool):
        raise ValueError("include_current moet True of False zijn.")
    settings = _PRESETS[trend].copy()
    latest = calc_rsi(_close_data(df, ticker), rsi_period)
    data = latest
    if not last_bar_complete:
        data = data.iloc[:-1].copy()
    if data.empty:
        raise ValueError("Geen candles beschikbaar voor divergentieanalyse.")
    close, rsi = data.Close.to_numpy(), data.RSI.to_numpy()
    events = []
    order = settings["order"]
    for low, direction in ((True, "bullish"), (False, "bearish")):
        pivots = _pivots(close, order, low)
        for i, b in enumerate(pivots):
            earlier = pivots[:i]
            eligible = earlier[(b - earlier >= settings["min_distance"]) &
                               (b - earlier <= settings["max_distance"])]
            if not len(eligible):
                continue
            a = eligible[-1]
            if not np.isfinite([rsi[a], rsi[b]]).all():
                continue
            match = (close[b] < close[a] and rsi[b] > rsi[a]) if low else (
                close[b] > close[a] and rsi[b] < rsi[a])
            if match:
                events.append(dict(direction=direction, first_pivot=data.index[a],
                    second_pivot=data.index[b], confirmed_on=data.index[b + order],
                    first_price=close[a], second_price=close[b], first_rsi=rsi[a], second_rsi=rsi[b],
                    price_change_pct=(close[b] / close[a] - 1) * 100,
                    rsi_change=rsi[b] - rsi[a], span_bars=int(b - a),
                    age_bars=len(data) - 1 - (b + order), status="CONFIRMED",
                    detected_on=data.index[b + order]))
    all_events = pd.DataFrame(events, columns=_COLUMNS).sort_values(
        ["confirmed_on", "direction"], kind="stable").reset_index(drop=True)
    current = (_current_divergences(latest, data, settings) if include_current
               else pd.DataFrame(columns=_COLUMNS))
    current_events = current.to_dict('records')
    visible = latest if include_current else data
    if start_plot_date is not None:
        start = pd.Timestamp(start_plot_date)
        if pd.isna(start):
            raise ValueError("Ongeldige start_plot_date.")
        if start.tzinfo is not None:
            start = start.tz_localize(None)
        visible = visible.loc[visible.index >= start]
    if visible.empty:
        raise ValueError("Geen candles binnen de gekozen plotperiode.")
    combined = pd.DataFrame(events + current_events, columns=_COLUMNS).sort_values(
        ["detected_on", "direction"], kind="stable")
    plotted = combined.loc[combined.first_pivot >= visible.index[0]].copy().reset_index(drop=True)
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
        preliminary = event.status == "PRELIMINARY"
        color, marker, style = ("#0072B2", "^", "-") if bull else ("#D55E00", "v", "--")
        if preliminary:
            marker, style = "D", ":"
        key = (event.direction, event.status)
        label = (("Voorlopig " if preliminary else "Bevestigd ") + event.direction) if key not in seen else None
        seen.add(key)
        dates = [event.first_pivot, event.second_pivot]
        price_ax.plot(dates, [event.first_price, event.second_price], color=color, linestyle=style, marker="o", linewidth=2)
        rsi_ax.plot(dates, [event.first_rsi, event.second_rsi], color=color, linestyle=style, marker="o", linewidth=2)
        date = event.detected_on
        price = latest.loc[date, "Close"]
        price_ax.scatter(date, price, edgecolors=color, facecolors="none" if preliminary else color,
                         marker=marker, s=85 if preliminary else 65, label=label, zorder=5)
        price_ax.annotate(f"{'P' if preliminary else ''}{'B' if bull else 'S'}{event.id}", (date, price),
                          xytext=(4, 8), textcoords="offset points", fontsize=9)
        for ax in (price_ax, rsi_ax):
            ax.axvline(date, color="gray", linestyle=":", alpha=.35)
    price_ax.set(title=f"{ticker.upper()} — RSI-divergentie | {trend} | {visible.index[-1]:%Y-%m-%d}",
                 ylabel="Koers (noteringsvaluta)")
    info = (f"Draaipunten {settings['min_distance']}–{settings['max_distance']} candles uit elkaar; "
            f"bevestiging na {order} candles.\nB / omhoog: bullish; S / omlaag: bearish.")
    if include_current:
        info += ("\nLaatste koers: voorlopige divergentie (open ruit, P); kan verdwijnen." if len(current)
                 else "\nLaatste koers: geen voorlopige divergentie volgens deze instellingen.")
        if len(current) and not plotted.status.eq("PRELIMINARY").any():
            info += " Anker vóór plotstart; zie current_divergences."
    elif plotted.empty:
        info += "\nGeen bevestigde divergenties in dit venster."
    if not last_bar_complete:
        info += "\nLaatste aangeleverde candle uitgesloten van bevestigde detectie."
    price_ax.text(.01, .01, info, transform=price_ax.transAxes, fontsize=8,
                  bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    rsi_ax.set(ylabel="RSI", ylim=(0, 100))
    for ax in (price_ax, rsi_ax):
        ax.grid(linestyle="--", alpha=.25)
        ax.legend(loc="upper left")
    if show:
        plt.show()
    return dict(figure=fig, divergences=plotted, all_divergences=all_events,
                current_divergences=current,
                settings=dict(trend=trend, rsi_period=rsi_period, **settings,
                              start_plot_date=start_plot_date,
                              last_bar_complete=last_bar_complete, include_current=include_current,
                              as_of=visible.index[-1]))
