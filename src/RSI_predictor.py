import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
from src.calc_indicators import calc_rsi

def plot_rsi_peaks(df, rsi_period, start_plot_date, ticker, 
                   dof=1, n_toppen=3, future_days=30, selected_peak_indices=None):
    """
    Plot RSI with peaks, filtered peak line, and regression prediction.
    All detected peaks are black; manually selected peaks are green.
    
    Parameters:
    - df: DataFrame with 'Close' column (RSI will be calculated if missing)
    - rsi_period: Number of periods for RSI
    - start_plot_date: Start date for plotting
    - ticker: Stock symbol for title
    - dof: Degree of polynomial for regression
    - n_toppen: Number of peaks used for regression
    - future_days: Number of days into the future for prediction
    - selected_peak_indices: List of indices for manually selected peaks (optional)
    """

    # ---- Calculate RSI if missing ----
    if 'RSI' not in df.columns:
        df['RSI'] = calc_rsi(df, rsi_period)['RSI']

    # ---- Filter data for plotting ----
    plot_df = df[df.index >= start_plot_date].copy()
    rsi_values = plot_df['RSI'].values
    dates = plot_df.index

    # ---- Detect all peaks ----
    peaks_idx, _ = find_peaks(rsi_values, prominence=6, distance=10)
    peak_dates = dates[peaks_idx]
    peak_values = rsi_values[peaks_idx]

    # ---- Filter peaks for smooth line ----
    if len(peak_values) > 0:
        filtered_idx = [0]
        for i in range(1, len(peak_values)-1):
            prev_i, curr_i, next_i = i-1, i, i+1
            interp = peak_values[prev_i] + (peak_values[next_i] - peak_values[prev_i]) * \
                     (peak_dates[i] - peak_dates[prev_i]).days / (peak_dates[next_i] - peak_dates[prev_i]).days
            if peak_values[curr_i] >= interp:
                filtered_idx.append(i)
        filtered_idx.append(len(peak_values)-1)
        filtered_idx = sorted(set(filtered_idx))
        filtered_dates = peak_dates[filtered_idx]
        filtered_values = peak_values[filtered_idx]
    else:
        filtered_dates = np.array([])
        filtered_values = np.array([])

    # ---- Manual selection of peaks (optional) ----
    if selected_peak_indices is not None and len(filtered_values) > 0:
        selected_peak_indices = [i for i in selected_peak_indices if i < len(filtered_dates)]
        selected_dates = filtered_dates[selected_peak_indices]
        selected_values = filtered_values[selected_peak_indices]
    else:
        selected_dates = np.array([])
        selected_values = np.array([])

    # ---- Regression line using selected peaks ----
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

    # ---- Plot RSI + peaks + regression ----
    plt.figure(figsize=(14, 5))
    plt.plot(dates, rsi_values, color='purple', linewidth=1.6, label=f'RSI ({rsi_period})')
    plt.axhline(70, color='red', linestyle='--', linewidth=1)
    plt.axhline(30, color='green', linestyle='--', linewidth=1)
    plt.fill_between(dates, 70, 30, color='gray', alpha=0.1)

    # ---- All detected peaks in black ----
    if len(filtered_values) > 0:
        plt.scatter(filtered_dates, filtered_values, color='black', s=50, label='Alle toppen')

    # ---- Selected peaks in green ----
    if len(selected_values) > 0:
        plt.scatter(selected_dates, selected_values, color='green', s=70, label='Geselecteerde toppen')

    # ---- Peak line connecting selected peaks ----
    if len(selected_values) >= 2:
        plt.plot(selected_dates, selected_values, color='darkgreen', linewidth=2.5, label='Toppenlijn')

    # ---- Regression line ----
    if len(fit_dates) > 0:
        plt.plot(fit_dates, y_fit, color='orange', linewidth=2, linestyle='--', 
                 label=f'Regressielijn (DOF={dof})')

    plt.title(f"{ticker} — RSI ({rsi_period}) met toppen en regressielijn", fontsize=14, fontweight='bold')
    plt.ylabel("RSI")
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
