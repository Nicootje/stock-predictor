"""Handmatig geselecteerde RSI-extremen met koers op exact dezelfde datums."""
from numbers import Integral, Real

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.calc_indicators import calc_rsi, _single_ticker


def _close_data(df, ticker):
    if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df moet een DataFrame met een DatetimeIndex zijn.")
    data = _single_ticker(df, ticker)
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


def plot_manual_rsi_divergence(df, ticker, rsi_period=14, start_plot_date=None,
                              *, prominence=6, distance=10, dof=1, future_days=30,
                              plot_level=False, show=True, interactive=True):
    """Select RSI tops and bottoms by clicking, without numbered point inputs.

    Filtered RSI tops and bottoms are shown, with matching Close values in the
    price panel. prominence is the minimum prominence in RSI points; distance
    is the minimum number of candles between extrema of the same type. Higher
    values filter more small or nearby swings. Use 0 and 1 for all local extrema. Endpoints are not extrema because they have no neighbour on both sides.
    Click again to deselect. Selection is sorted chronologically.
    dof is the requested polynomial degree. With at least two points, the
    effective degree is min(dof, number of selected points - 1). Both price
    and RSI use that degree; the status reports any temporary reduction.
    The RSI fit is solid between selected points; both projections are dotted for future_days
    calendar days after the last selected point. Values outside 0..100 are hidden.
    plot_level shows the mean selected RSI. Returns figure, extrema,
    selected_points, fit, projection, rsi_level and settings; these update on clicks.
    update_settings(dof=..., future_days=...) refits the current selection in place.
    """
    for name, value, minimum in [('rsi_period', rsi_period, 2),
                                  ('dof', dof, 0), ('distance', distance, 1)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f'{name} moet een geheel getal van minstens {minimum} zijn.')
    if (isinstance(prominence, bool) or not isinstance(prominence, Real)
            or not np.isfinite(prominence) or prominence < 0):
        raise ValueError('prominence moet niet-negatief en eindig zijn.')
    if (isinstance(future_days, bool) or not isinstance(future_days, Real)
            or not np.isfinite(future_days) or future_days < 0):
        raise ValueError('future_days moet niet-negatief en eindig zijn.')
    if not isinstance(plot_level, bool):
        raise ValueError('plot_level moet True of False zijn.')
    data = calc_rsi(_close_data(df, ticker), rsi_period)
    valid = data.dropna(subset=['RSI'])
    values = valid.RSI.to_numpy()
    tops, _ = find_peaks(values, prominence=prominence, distance=distance)
    bottoms, _ = find_peaks(-values, prominence=prominence, distance=distance)
    positions = np.sort(np.concatenate((tops, bottoms)))
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
    selected = points.iloc[:0].copy()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             layout='constrained', gridspec_kw={'height_ratios': [3, 1.5]})
    label = 'RSI-toppen en -bodems'
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
        line, = ax.plot(selected.date, selected[column], color='#D55E00', marker='o',
                       linewidth=2, markersize=6, linestyle='-',
                       label='Handmatige selectie', zorder=5)
        selection_lines.append(line)
        ax.grid(linestyle='--', alpha=.25)
        ax.legend(loc='upper left')
    axes[0].set(title=f'{ticker.upper()} — Handmatige RSI-divergentie',
                ylabel='Koers (noteringsvaluta)')
    axes[1].set(ylabel='RSI', ylim=(0, 100), xlabel='Datum')
    for level in (30, 70):
        axes[1].axhline(level, color='gray', linestyle='--', linewidth=.8)
    message = ('Geen lokale RSI-toppen of -bodems gevonden.' if points.empty
               else 'Klik op RSI-stippen om punten te selecteren of te verwijderen.')
    message_text = axes[0].text(.01, .02, message, transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    result = dict(figure=fig, extrema=points, selected_points=selected,
                  settings=dict(rsi_period=rsi_period, start_plot_date=start_plot_date,
                                prominence=prominence, distance=distance,
                                interactive=interactive, dof=dof,
                                future_days=future_days, plot_level=plot_level))
    chosen = set()
    fit_line, = axes[1].plot([], [], color='#009E73', linestyle='-', linewidth=2.5, zorder=6,
                             label=f'RSI-regressie (dof={dof})')
    future_line, = axes[1].plot([], [], color='#CC79A7', linestyle=':', linewidth=2.5, zorder=6,
                                label='Doortrekking')
    price_future, = axes[0].plot([], [], color='#CC79A7', linestyle=':', linewidth=2.5,
                               zorder=6, label='Doortrekking')
    level_line = axes[1].axhline(0, color='blue', linestyle='--', linewidth=1.5,
                                visible=False, label='RSI-niveau')
    fit_message = axes[1].text(.01, .02, '', transform=axes[1].transAxes, fontsize=8,
                              bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))

    fit_cache = {}
    legend_signatures = {}

    def update_projection(selection):
        effective_dof = min(dof, len(selection)-1) if len(selection) >= 2 else None
        result['effective_dof'] = effective_dof
        fit_line.set_label(f'RSI-regressie (dof={effective_dof})')
        result['fit'] = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'),
                                      'rsi': pd.Series(dtype=float), 'price': pd.Series(dtype=float)})
        result['projection'] = result['fit'].copy()
        result['rsi_level'] = float(selection.rsi.mean()) if plot_level and len(selection) else None
        level = result['rsi_level']
        level_line.set_visible(level is not None)
        if level is not None:
            level_line.set_ydata([level, level])
            level_line.set_label(f'RSI-niveau: {level:.2f}')
        fit_line.set_data([], [])
        future_line.set_data([], [])
        future_line.set_visible(False)
        price_future.set_data([], [])
        price_future.set_visible(False)
        required = 2
        fit_line.set_visible(len(selection) >= required)
        if len(selection) >= required:
            origin = selection.date.iloc[0]
            days = (selection.date - origin).dt.total_seconds().to_numpy() / 86400
            key = (tuple(selection.id), effective_dof)
            if fit_cache.get('key') != key:
                fit_cache.update(
                    key=key,
                    rsi=np.polynomial.Polynomial.fit(days, selection.rsi.to_numpy(), deg=effective_dof),
                    price=np.polynomial.Polynomial.fit(days, selection.price.to_numpy(), deg=effective_dof),
                )
            polynomial, price_polynomial = fit_cache['rsi'], fit_cache['price']
            def evaluate(x):
                y = polynomial(x)
                y = np.where(np.isfinite(y) & (y >= 0) & (y <= 100), y, np.nan)
                price = price_polynomial(x)
                price = np.where(np.isfinite(price) & (price > 0), price, np.nan)
                return pd.DataFrame({'date': origin + pd.to_timedelta(x, unit='D'), 'rsi': y, 'price': price})

            result['fit'] = evaluate(np.linspace(0, days[-1], 200))
            fit_line.set_data(result['fit'].date, result['fit'].rsi)
            if future_days > 0:
                result['projection'] = evaluate(np.linspace(days[-1], days[-1] + future_days, 100))
                future_line.set_data(result['projection'].date, result['projection'].rsi)
                future_line.set_visible(True)
                price_future.set_data(result['projection'].date, result['projection'].price)
                price_future.set_visible(True)
            status = f'{len(selection)} punten geselecteerd; koers- en RSI-regressie met DOF {effective_dof}.'
            if effective_dof != dof:
                status += f' Gekozen DOF {dof} vereist {dof+1} punten; tijdelijk DOF {effective_dof}.'
            if future_days == 0:
                status += ' Future days is 0: geen stippellijn.'
            elif result['projection'].rsi.notna().sum() < 2:
                status += ' Doortrekking valt buiten het zichtbare RSI-bereik 0-100.'
            else:
                status += f' Stippellijn: {future_days:g} dagen vanaf de laatste geselecteerde stip.'
                if result['projection'].rsi.isna().any():
                    status += ' Het deel buiten RSI 0-100 wordt verborgen.'
            result['status'] = status
            fit_message.set_text('')
        else:
            result['status'] = (f'{len(selection)} punten geselecteerd. Minimaal {required} punten nodig: '
                                f'klik nog {required-len(selection)} RSI-stip(pen) aan voor de lijn en doortrekking.')
            fit_message.set_text(f'Geen fit: minstens {required} geselecteerde punten nodig.')
        # Recompute the shared date range so removing points also removes old projection space.
        end = visible.index[-1]
        if not result['projection'].empty:
            end = max(end, result['projection'].date.iloc[-1])
        padding = max((end - visible.index[0]) * .03, pd.Timedelta(days=1))
        axes[1].set_xlim(visible.index[0] - padding, end + padding)
        axes[0].relim()
        axes[0].autoscale_view(scalex=False, scaley=True)
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            shown = [(h, label) for h, label in zip(handles, labels) if h.get_visible()]
            signature = tuple((id(handle), label) for handle, label in shown)
            if legend_signatures.get(ax) != signature:
                ax.legend(*zip(*shown), loc='upper left', fontsize=8)
                legend_signatures[ax] = signature

    def update_settings(*, dof=None, future_days=None):
        new_dof = result['settings']['dof'] if dof is None else dof
        new_days = result['settings']['future_days'] if future_days is None else future_days
        if isinstance(new_dof, bool) or not isinstance(new_dof, Integral) or new_dof < 0:
            raise ValueError('dof moet een niet-negatief geheel getal zijn.')
        if (isinstance(new_days, bool) or not isinstance(new_days, Real)
                or not np.isfinite(new_days) or new_days < 0):
            raise ValueError('future_days moet niet-negatief en eindig zijn.')
        if (new_dof, new_days) != (result['settings']['dof'], result['settings']['future_days']):
            apply_settings(new_dof, new_days)

    def apply_settings(new_dof, new_days):
        nonlocal dof, future_days
        dof, future_days = new_dof, new_days
        result['settings'].update(dof=dof, future_days=future_days)
        update_projection(result['selected_points'])
        fig.canvas.draw()

    result['update_settings'] = update_settings
    update_projection(selected)

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
        update_projection(selection)
        for line, column in zip(selection_lines, ('price', 'rsi')):
            line.set_data(selection.date, selection[column])
        message_text.set_text(f'{len(chosen)} punten geselecteerd; klik nogmaals om te verwijderen.'
                              if chosen else 'Klik op RSI-stippen om punten te selecteren.')
        fig.canvas.draw()

    if interactive:
        result['selection_callback_id'] = fig.canvas.mpl_connect('pick_event', on_pick)
    if show:
        plt.show()
    return result
