"""Actuele voorlopige RSI-divergentie voor een lijst tickers, zonder grafieken."""
import numpy as np
import pandas as pd
import yfinance as yf
from IPython.display import display, Markdown

from src.calc_indicators import calc_rsi, _single_ticker, _validate_period
from src.divergence import _close_data, _current_divergences, _PRESETS


def scan_rsi_divergence(tickers, start_date="2019-01-01", rsi_period=14, *,
                        data_by_ticker=None, last_bar_complete=False, show=True):
    """Scan de nieuwste aangeleverde dagkoers op short/medium/long divergentie.

    Zelfde regels als plot_rsi_divergence(include_current=True). De nieuwste
    candle telt ALTIJD mee als voorlopig eindpunt, ook als last_bar_complete
    False is. Die optie bepaalt alleen of hij eerdere draaipunten mag bevestigen.
    Signalen kunnen veranderen/verdwijnen. Geen controle op actuele beursstatus;
    as_of en data_age_days tonen hoe oud de nieuwste waarneming is.

    Eén batchdownload plus maximaal één herpoging voor een mislukte ticker.
    data_by_ticker gebruikt bestaande frames, zonder downloads. RSI wordt één
    keer per ticker berekend, niet opnieuw per termijn. Geen plots aangemaakt.
    Toont matches en afzonderlijke dataproblemen; retourneert ALLE resultaatrijen.
    '-' betekent geen signaal, 'ONVOLDOENDE DATA' is geen neutraal resultaat.
    """
    rsi_period = _validate_period(rsi_period)
    if rsi_period < 2:
        raise ValueError("rsi_period moet minstens 2 zijn.")
    if not isinstance(last_bar_complete, bool):
        raise ValueError("last_bar_complete moet True of False zijn.")
    if isinstance(tickers, str):
        tickers = [tickers]
    names = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    terms = list(_PRESETS)
    columns = ['ticker', 'as_of', 'price', 'RSI', *terms, 'has_divergence',
               'status', 'data_age_days', 'history_bars', 'details', 'message']
    supplied = None if data_by_ticker is None else {t.upper(): f for t, f in data_by_ticker.items()}
    raw, batch_error = None, ''
    if names and supplied is None:
        try:
            raw = yf.download(names, start=start_date, interval='1d', auto_adjust=True,
                              progress=False, threads=False, timeout=10)
        except Exception as exc:
            batch_error = str(exc)
    rows = []
    for ticker in names:
        row = dict(ticker=ticker, as_of=pd.NaT, price=np.nan, RSI=np.nan,
                   **{term: 'NIET BEREKEND' for term in terms}, has_divergence=False,
                   status='ERROR', data_age_days=np.nan, history_bars=0, details={}, message='')
        try:
            if supplied is not None:
                if ticker not in supplied:
                    raise ValueError('Geen data aangeleverd voor deze ticker.')
                frame = supplied[ticker]
            else:
                try:
                    if raw is None:
                        raise ValueError(batch_error or 'Geen batchdata ontvangen.')
                    if len(names) > 1 and not isinstance(raw.columns, pd.MultiIndex):
                        raise ValueError('Tickers niet eenduidig in batchdata.')
                    frame = _single_ticker(raw, ticker).dropna(how='all')
                    if frame.empty:
                        raise ValueError('Ticker ontbreekt in batchdownload.')
                except (ValueError, KeyError, AttributeError):
                    frame = yf.download(ticker, start=start_date, interval='1d', auto_adjust=True,
                                        progress=False, threads=False, timeout=10)
            # Verwijder uitsluitend volledig lege uitgelijnde downloadrijen.
            selected = _single_ticker(frame, ticker)
            if supplied is None:
                selected = selected.dropna(how='all')
            latest = _close_data(selected, ticker)
            if latest.empty:
                row.update(status='NO_DATA', message='Geen koersdata ontvangen.')
                rows.append(row)
                continue
            latest = calc_rsi(latest, rsi_period)
            data = latest if last_bar_complete else latest.iloc[:-1]
            row.update(as_of=latest.index[-1], price=float(latest.Close.iloc[-1]),
                       RSI=float(latest.RSI.iloc[-1]), history_bars=len(latest),
                       data_age_days=(pd.Timestamp.now().normalize()-latest.index[-1].normalize()).days)
            missing = []
            for term, settings in _PRESETS.items():
                needed = max(rsi_period, settings['order']) + settings['min_distance'] + 1
                if len(latest) < needed or not np.isfinite(row['RSI']):
                    row[term] = 'ONVOLDOENDE DATA'
                    missing.append(f'{term}: minstens {needed} candles nodig')
                    continue
                signals = _current_divergences(latest, data, settings)
                row[term] = ' / '.join(s.direction.capitalize() for s in signals.itertuples()) or '-'
                if not signals.empty:
                    row['has_divergence'] = True
                    row['details'][term] = signals.to_dict('records')
            row['status'] = 'INSUFFICIENT_DATA' if len(missing) == 3 else 'PARTIAL_HISTORY' if missing else 'OK'
            row['message'] = '; '.join(missing)
        except Exception as exc:
            row.update(status='ERROR', message=str(exc))
        rows.append(row)
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values('has_divergence', ascending=False, kind='stable').reset_index(drop=True)
    if show:
        matches = result.loc[result.has_divergence.eq(True)]
        display(matches[['ticker', 'as_of', 'price', 'RSI', *terms, 'data_age_days']].round(3))
        problems = result.loc[result.status.ne('OK'), ['ticker', 'status', 'message']]
        if not problems.empty:
            display(problems)
    return result
