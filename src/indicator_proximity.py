"""Zoek compacte groepen koers-, SMA-, EMA- en Bollingerwaarden."""
import numpy as np
import pandas as pd
import yfinance as yf

from src.calc_indicators import calc_sma_ema, calc_bollinger_bands, _single_ticker, _validate_period


def scan_indicator_proximity(tickers, max_distance_pct=2.0, min_values=4,
                             start_date="2019-01-01", periods=(20, 50, 200),
                             bb_period=20, bb_std=2, *, data_by_ticker=None,
                             last_bar_complete=False):
    """Eén resultaatrij per ticker; geen koop-/verkoopadvies of voorspelscore.

    Een groep past als (hoogste - laagste waarde) / slotkoers * 100 <=
    max_distance_pct. Kies de grootste groep, bij gelijk aantal de smalste.
    Dit voorkomt ketens van nabije paren die samen een brede groep vormen.
    BB_Middle telt niet dubbel als SMA met dezelfde periode aanwezig is.
    near_price bevat afzonderlijk alle waarden binnen de tolerantie van Close.

    Download één batch, met één losse herpoging per mislukte ticker. Met
    data_by_ticker worden geen downloads gedaan. Standaard wordt de laatste
    aangeleverde candle uitgesloten; True gebruikt die ook als hij nog loopt.
    Minstens max(periods, bb_period) geldige candles vereist. Geen forward-fill.
    Percentages gebruiken ongeronde waarden; rond alleen de weergave af.
    """
    if not np.isfinite(max_distance_pct) or max_distance_pct <= 0:
        raise ValueError("max_distance_pct moet positief en eindig zijn.")
    min_values = _validate_period(min_values)
    periods = tuple(dict.fromkeys(_validate_period(p) for p in periods))
    bb_period = _validate_period(bb_period)
    if not np.isfinite(bb_std) or bb_std <= 0:
        raise ValueError("bb_std moet positief en eindig zijn.")
    if not isinstance(last_bar_complete, bool):
        raise ValueError("last_bar_complete moet True of False zijn.")
    if isinstance(tickers, str):
        tickers = [tickers]
    names = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    averages = [f'{kind}{p}' for kind in ('SMA', 'EMA') for p in periods]
    value_columns = ['Close', *averages, 'BB_Lower', 'BB_Middle', 'BB_Upper']
    columns = ['ticker', 'status', 'as_of', 'match', 'n_values', 'cluster',
               'cluster_values', 'cluster_width_pct', 'near_price', 'ma_width_pct',
               'bb_width_pct', *value_columns, 'message']
    if not names:
        return pd.DataFrame(columns=columns)
    supplied = None if data_by_ticker is None else {t.upper(): f for t, f in data_by_ticker.items()}
    raw, batch_error = None, ''
    if supplied is None:
        try:
            raw = yf.download(names, start=start_date, interval='1d', auto_adjust=True,
                              progress=False, threads=False, timeout=10)
        except Exception as exc:
            batch_error = str(exc)

    results = []
    for ticker in names:
        row = dict.fromkeys(columns, np.nan)
        row.update(ticker=ticker, status='NO_DATA', as_of=pd.NaT, match=False,
                   cluster='', cluster_values='', near_price='', message='')
        try:
            if supplied is not None:
                if ticker not in supplied:
                    raise ValueError('Geen data aangeleverd voor deze ticker.')
                frame = _single_ticker(supplied[ticker], ticker).copy()
            else:
                try:
                    if raw is None:
                        raise ValueError(batch_error or 'Geen batchdata ontvangen.')
                    if not isinstance(raw.columns, pd.MultiIndex) and len(names) > 1:
                        raise ValueError('Batchdata bevat geen eenduidige tickers.')
                    frame = _single_ticker(raw, ticker).dropna(how='all').copy()
                    if frame.empty:
                        raise ValueError('Geen koersdata in de batch.')
                except (ValueError, KeyError, AttributeError):
                    retry = yf.download(ticker, start=start_date, interval='1d', auto_adjust=True,
                                        progress=False, threads=False, timeout=10)
                    frame = _single_ticker(retry, ticker).copy()
            frame = frame.dropna(how='all').sort_index()
            if frame.empty:
                row['message'] = 'Geen koersdata ontvangen.'
                results.append(row)
                continue
            if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates or frame.index.hasnans:
                raise ValueError('Koersdata vereist unieke, geldige datums.')
            if not last_bar_complete:
                frame = frame.iloc[:-1].copy()
            frame['Close'] = pd.to_numeric(frame['Close'], errors='raise').astype(float)
            if not np.isfinite(frame.Close).all() or (frame.Close <= 0).any():
                raise ValueError('Close bevat ontbrekende of ongeldige koersen.')
            if len(frame):
                row['as_of'] = frame.index[-1]
                row['Close'] = float(frame.Close.iloc[-1])
            needed = max((*periods, bb_period))
            if len(frame) < needed:
                row.update(status='INSUFFICIENT_DATA', message=f'{len(frame)} candles; minstens {needed} nodig.')
                results.append(row)
                continue
            data = calc_bollinger_bands(calc_sma_ema(frame, periods), bb_period, bb_std)
            last = data.iloc[-1]
            price = float(last.Close)
            row.update({column: float(last[column]) for column in value_columns})
            # BB-midden is wiskundig dezelfde waarde als SMA(bb_period).
            unique_columns = [c for c in value_columns if c != 'BB_Middle' or bb_period not in periods]
            labels = {c: f'{c}/BB_Middle' if c == f'SMA{bb_period}' else c for c in unique_columns}
            ordered = sorted(unique_columns, key=lambda c: row[c])
            best, best_width = [], np.inf
            for left in range(len(ordered)):
                group = [c for c in ordered[left:]
                         if (row[c] - row[ordered[left]]) / price * 100 <= max_distance_pct]
                width = (row[group[-1]] - row[group[0]]) / price * 100
                if len(group) > len(best) or (len(group) == len(best) and width < best_width):
                    best, best_width = group, width
            near = [c for c in unique_columns if c != 'Close' and abs(row[c]/price-1)*100 <= max_distance_pct]
            row.update(status='OK', match=len(best) >= min_values, n_values=len(best),
                cluster=', '.join(labels[c] for c in best),
                cluster_values=' | '.join(f'{labels[c]}={row[c]:.4f}' for c in best),
                cluster_width_pct=best_width, near_price=', '.join(labels[c] for c in near),
                ma_width_pct=(max(row[c] for c in averages)-min(row[c] for c in averages))/price*100
                             if averages else np.nan,
                bb_width_pct=(row['BB_Upper']-row['BB_Lower'])/row['BB_Middle']*100,
                message='Laatste aangeleverde candle meegenomen; mogelijk nog onvoltooid.' if last_bar_complete
                        else 'Laatste aangeleverde candle uitgesloten.')
        except Exception as exc:
            row.update(status='ERROR', message=str(exc))
        results.append(row)
    return pd.DataFrame(results, columns=columns).sort_values(
        ['match', 'n_values', 'cluster_width_pct'], ascending=[False, False, True],
        na_position='last', kind='stable').reset_index(drop=True)
