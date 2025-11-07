import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
from src.calc_indicators import calc_rsi

def plot_rsi_peaks(df, rsi_period, start_plot_date, ticker, 
                   mode='top', dof=1, future_days=30, 
                   selected_peak_indices=None, plot_level=False):
    """
    Plot RSI met toppen of bodems, geselecteerde punten, lijn en regressie.

    Parameters:
    - df: DataFrame met 'Close' kolom (RSI wordt berekend als ontbreekt)
    - rsi_period: Aantal periodes voor RSI
    - start_plot_date: Startdatum voor plot
    - ticker: Symbool voor titel
    - mode: 'top' of 'bottom' om te plotten
    - dof: Degree of polynomial voor regressie
    - future_days: Aantal dagen in de toekomst voor voorspelling
    - selected_peak_indices: Lijst met indices van handmatig geselecteerde punten
    - plot_level: bool, True om horizontale RSI-lijn te tekenen
    """

    # ---- Bereken RSI indien ontbreekt ----
    if 'RSI' not in df.columns:
        df['RSI'] = calc_rsi(df, rsi_period)['RSI']

    # ---- Filter data ----
    plot_df = df[df.index >= start_plot_date].copy()
    rsi_values = plot_df['RSI'].values
    dates = plot_df.index

    # ---- Detecteer alle toppen of bodems ----
    if mode == 'top':
        peaks_idx, _ = find_peaks(rsi_values, prominence=6, distance=10)
    elif mode == 'bottom':
        peaks_idx, _ = find_peaks(-rsi_values, prominence=6, distance=10)
    else:
        raise ValueError("Mode moet 'top' of 'bottom' zijn.")

    peak_dates = dates[peaks_idx]
    peak_values = rsi_values[peaks_idx]

    # ---- Filter peaks voor vloeiende lijn ----
    if len(peak_values) > 0:
        filtered_idx = [0]
        for i in range(1, len(peak_values)-1):
            prev_i, curr_i, next_i = i-1, i, i+1
            interp = peak_values[prev_i] + (peak_values[next_i] - peak_values[prev_i]) * \
                     (peak_dates[i] - peak_dates[prev_i]).days / (peak_dates[next_i] - peak_dates[prev_i]).days
            if (mode=='top' and peak_values[curr_i] >= interp) or (mode=='bottom' and peak_values[curr_i] <= interp):
                filtered_idx.append(i)
        filtered_idx.append(len(peak_values)-1)
        filtered_idx = sorted(set(filtered_idx))
        filtered_dates = peak_dates[filtered_idx]
        filtered_values = peak_values[filtered_idx]
    else:
        filtered_dates = np.array([])
        filtered_values = np.array([])

    # ---- Handmatig geselecteerde toppen/bodems ----
    if selected_peak_indices is not None and len(filtered_values) > 0:
        selected_peak_indices = [i for i in selected_peak_indices if i < len(filtered_dates)]
        selected_dates = filtered_dates[selected_peak_indices]
        selected_values = filtered_values[selected_peak_indices]

        # ---- Bereken horizontale lijnniveau (optioneel) ----
        rsi_level = np.mean(selected_values) if plot_level else None
    else:
        selected_dates = np.array([])
        selected_values = np.array([])
        rsi_level = None

    # ---- Regressielijn ----
    if len(selected_values) >= 2:
        dates_num = np.array([(d - selected_dates[0]).days for d in selected_dates])
        coefs = np.polyfit(dates_num, selected_values, deg=dof)
        poly = np.poly1d(coefs)

        x_fit = np.linspace(dates_num[0], dates_num[-1] + future_days, 100)
        y_fit = poly(x_fit)
        fit_dates = [selected_dates[0] + pd.Timedelta(days=int(x)) for x in x_fit]
    else:
        fit_dates = np.array([])
        y_fit = np.array([])

    # ---- Plot ----
    plt.figure(figsize=(14, 5))
    plt.plot(dates, rsi_values, color='purple', linewidth=1.6, label=f'RSI ({rsi_period})')
    plt.axhline(70, color='red', linestyle='--', linewidth=1)
    plt.axhline(30, color='green', linestyle='--', linewidth=1)
    plt.fill_between(dates, 70, 30, color='gray', alpha=0.1)

    if len(filtered_values) > 0:
        plt.scatter(filtered_dates, filtered_values, color='black', s=50, label=f'Alle {mode}pen')

    if len(selected_values) > 0:
        plt.scatter(selected_dates, selected_values, color='green', s=70, label=f'Geselecteerde {mode}pen')

    if len(selected_values) >= 2:
        plt.plot(selected_dates, selected_values, color='darkgreen', linewidth=2.5, label=f'{mode.capitalize()}lijn')

    if rsi_level is not None:
        plt.axhline(rsi_level, color='blue', linestyle='--', linewidth=1.5,
                    label=f'RSI-niveau ≈ {rsi_level:.2f}')

    if len(fit_dates) > 0:
        plt.plot(fit_dates, y_fit, color='orange', linewidth=2, linestyle='--', 
                 label=f'Regressielijn (DOF={dof})')

    plt.title(f"{ticker} — RSI ({rsi_period}) met {mode}pen en regressielijn", fontsize=14, fontweight='bold')
    plt.ylabel("RSI")
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
