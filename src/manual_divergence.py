"""Handmatig geselecteerde RSI-extremen met koers op exact dezelfde datums."""
from numbers import Integral, Real

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.calc_indicators import calc_rsi
from src.divergence import _close_data


def plot_manual_rsi_divergence(df, ticker, rsi_period=14, start_plot_date=None,
                              *, mode='top', selected_peak_indices=None,
                              prominence=6, distance=10, show=True, interactive=True):
    """Number RSI extrema from zero and connect selected IDs in both panels.

    First call with [] to inspect the numbers, then supply e.g. [0, 2, 3, 5].
    Selection is connected chronologically. All visible extrema get matching
    dots and IDs on the price chart, using Close on the exact RSI pivot date.
    Price points need not themselves be price extrema. No automatic divergence
    filter is applied and a drawn connection is not a confirmed trading signal.

    RSI and peaks are computed before the display cutoff. IDs enumerate visible
    peaks and can change with data, mode, cutoff, prominence or distance.
    prominence is measured in RSI points; distance in candles. Smaller values
    expose more extrema. Endpoint candles cannot be peaks (no right neighbour).
    Returns figure, extrema (id/date/price/rsi), selected_points and settings.
    With an interactive Matplotlib backend (e.g. %matplotlib widget), click
    numbered RSI dots to toggle any number of points. Both lines and the
    returned selected_points update immediately. Inline plots support IDs only.
    """
    if mode not in ('top', 'bottom'):
        raise ValueError("mode moet 'top' of 'bottom' zijn.")
    for name, value, minimum in [('rsi_period', rsi_period, 2), ('distance', distance, 1)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f'{name} moet een geheel getal van minstens {minimum} zijn.')
    if (isinstance(prominence, bool) or not isinstance(prominence, Real)
            or not np.isfinite(prominence) or prominence < 0):
        raise ValueError('prominence moet niet-negatief en eindig zijn.')
    data = calc_rsi(_close_data(df, ticker), rsi_period)
    valid = data.dropna(subset=['RSI'])
    values = valid.RSI.to_numpy()
    positions, _ = find_peaks(values if mode == 'top' else -values,
                              prominence=prominence, distance=distance)
    extrema = valid.iloc[positions]
    visible = data
    if start_plot_date is not None:
        start = pd.Timestamp(start_plot_date)
        if pd.isna(start):
            raise ValueError('Ongeldige start_plot_date.')
        if start.tzinfo is not None:
            start = start.tz_localize(None)
        visible = data.loc[data.index >= start]
        extrema = extrema.loc[extrema.index >= start]
    if visible.empty or visible.RSI.notna().sum() == 0:
        raise ValueError('Geen berekenbare RSI in de gekozen periode.')
    points = pd.DataFrame({'id': np.arange(len(extrema)), 'date': extrema.index,
                           'price': extrema.Close.to_numpy(), 'rsi': extrema.RSI.to_numpy()})
    ids = [] if selected_peak_indices is None else list(selected_peak_indices)
    if any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 or i >= len(points) for i in ids):
        available = f'0 t/m {len(points)-1}' if len(points) else 'geen'
        raise ValueError(f'Ongeldig extremumnummer; beschikbare nummers: {available}.')
    selected = points.iloc[sorted(set(ids))].copy()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             layout='constrained', gridspec_kw={'height_ratios': [3, 1.5]})
    label = 'RSI-toppen' if mode == 'top' else 'RSI-bodems'
    selection_lines = []
    rsi_dots = None
    for ax, series, column, color, name in [
            (axes[0], visible.Close, 'price', 'blue', 'Slotkoers'),
            (axes[1], visible.RSI, 'rsi', 'purple', f'RSI ({rsi_period})')]:
        ax.plot(visible.index, series, color=color, linewidth=1.2, label=name)
        dots = ax.scatter(points.date, points[column], color='black', s=24, zorder=4,
                   label=label if column == 'rsi' else 'Koers op RSI-extremen')
        if column == 'rsi':
            rsi_dots = dots
            if interactive:
                dots.set_picker(8)
        for point in points.itertuples(index=False):
            ax.annotate(str(point.id), (point.date, getattr(point, column)),
                        xytext=(0, 8 if mode == 'top' else -14),
                        textcoords='offset points', ha='center', fontsize=8)
        line, = ax.plot(selected.date, selected[column], color='#D55E00', marker='o',
                       linewidth=2, markersize=6, label='Handmatige selectie', zorder=5)
        selection_lines.append(line)
        ax.grid(linestyle='--', alpha=.25)
        ax.legend(loc='upper left')
    axes[0].set(title=f'{ticker.upper()} — Handmatige RSI-divergentie | {mode}',
                ylabel='Koers (noteringsvaluta)')
    axes[1].set(ylabel='RSI', ylim=(0, 100), xlabel='Datum')
    for level in (30, 70):
        axes[1].axhline(level, color='gray', linestyle='--', linewidth=.8)
    if points.empty:
        message = 'Geen extremen gevonden; verlaag eventueel prominence of distance.'
    elif selected.empty:
        message = 'Kies de genummerde punten met selected_peak_indices=[...]. Nummering begint bij 0.'
    else:
        message = 'Geselecteerde nummers: ' + ', '.join(map(str, selected.id))
    message_text = axes[0].text(.01, .02, message, transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    result = dict(figure=fig, extrema=points, selected_points=selected,
                  settings=dict(mode=mode, rsi_period=rsi_period, prominence=prominence,
                                distance=distance, start_plot_date=start_plot_date,
                                interactive=interactive))
    chosen = set(ids)

    def on_pick(event):
        if event.artist is not rsi_dots or not len(event.ind):
            return
        if event.mouseevent.button != 1 or getattr(fig.canvas.toolbar, 'mode', ''):
            return
        # If dots overlap, select the one nearest the mouse in screen pixels.
        candidates = np.asarray(event.ind, dtype=int)
        coordinates = rsi_dots.get_offset_transform().transform(
            rsi_dots.get_offsets()[candidates])
        mouse = np.array([event.mouseevent.x, event.mouseevent.y])
        index = int(candidates[np.argmin(np.sum((coordinates - mouse) ** 2, axis=1))])
        if index in chosen:
            chosen.remove(index)
        else:
            chosen.add(index)
        selection = points.iloc[sorted(chosen)].copy()
        result['selected_points'] = selection
        for line, column in zip(selection_lines, ('price', 'rsi')):
            line.set_data(selection.date, selection[column])
        message_text.set_text('Geselecteerde nummers: ' + ', '.join(map(str, selection.id))
                              if chosen else 'Klik op RSI-stippen om punten te selecteren.')
        fig.canvas.draw_idle()

    if interactive:
        result['selection_callback_id'] = fig.canvas.mpl_connect('pick_event', on_pick)
        if selected.empty and not points.empty:
            message_text.set_text('Klik op RSI-stippen, of gebruik selected_peak_indices=[...].')
    if show:
        plt.show()
    return result
