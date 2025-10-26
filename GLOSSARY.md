# Glossary — Technical indicators and terms used in this project

This glossary lists the technical indicators and common terms used in `notebooks/Beurs.ipynb` and the code under `src/`.

Each entry contains a short definition, the key parameters used in the notebook, and a short usage note.

---

## Ticker
The symbol used to identify a traded instrument (e.g., `^GSPC`, `AAPL`, `HIMS`). The notebook uses `yfinance` to download historical data for a `ticker`.

## Close (Close price)
The end-of-period price for the asset (daily close when `interval='1d'`). Most indicators here are computed from the `Close` price.

## RSI — Relative Strength Index
- Purpose: Momentum oscillator that measures the speed and change of price movements.
- Common parameter: `period` (default in notebook: 14).
- Calculation (Wilder smoothing):
  - delta = close.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
  - avg_gain = EWMA(gain, alpha=1/period)
  - avg_loss = EWMA(loss, alpha=1/period)
  - RS = avg_gain / avg_loss
  - RSI = 100 - (100 / (1 + RS))
- Interpretation: RSI > 70 commonly considered overbought; RSI < 30 considered oversold. The notebook plots lines at 70 and 30.

## SMA — Simple Moving Average
- Purpose: Average of the last N closing prices. Smooths price and shows trend.
- Parameter: window size `N` (e.g., 20, 50, 200 used in notebook).
- Calculation: SMA_N[t] = mean(close[t-N+1 : t]).

## EMA — Exponential Moving Average
- Purpose: Similar to SMA but gives more weight to recent prices.
- Parameter: `span` (e.g., 20, 50, 200 in the notebook). Implementation uses pandas `.ewm(span=..., adjust=False).mean()`.

## Bollinger Bands
- Purpose: Volatility bands around an N-period moving average.
- Parameters: period `N` (middle band is SMA_N) and `std_dev` (factor, commonly 2).
- Calculation:
  - Middle = SMA_N
  - Std = rolling_std(close, N)
  - Upper = Middle + Std * std_dev
  - Lower = Middle - Std * std_dev
- Interpretation: Price touching or exceeding the Upper band can suggest overbought; touching Lower can suggest oversold. Bands widen in high volatility.

## MACD — Moving Average Convergence Divergence
- Purpose: Trend-following momentum indicator showing relationship between two EMAs.
- Common parameters (notebook defaults): fast=12, slow=26, signal=9.
- Calculation:
  - EMA_fast = EMA(close, span=fast)
  - EMA_slow = EMA(close, span=slow)
  - MACD = EMA_fast - EMA_slow
  - Signal = EMA(MACD, span=signal)
  - Histogram = MACD - Signal
- Interpretation: MACD crossing above Signal is bullish; crossing below is bearish. Histogram shows divergence magnitude.

## Golden Cross / Death Cross
- Golden Cross: Short-term MA (e.g., SMA50) crosses above long-term MA (e.g., SMA200). Often considered a bullish signal.
- Death Cross: Short-term MA crosses below long-term MA. Often considered bearish.

## Overbought / Oversold
- Informal labels typically used with RSI or price vs. Bollinger Bands:
  - Overbought: indicator value is high (e.g., RSI > 70) or price > Upper Bollinger Band.
  - Oversold: indicator value is low (e.g., RSI < 30) or price < Lower Bollinger Band.

## Peak (local maxima) and Local minima
- Peak / local maxima: a data point greater than its neighbors in a local window. The notebook uses `scipy.signal.find_peaks` and `argrelextrema` to detect local extrema.
- Prominence (peak prominence): measure of how much a peak stands out from surrounding baseline — used to filter important peaks.

## Rolling window
- A contiguous subset of the time series used to compute moving statistics (mean, std, etc.). Example: a 20-period rolling mean.

## EWMA / EWM (Exponentially Weighted Moving Average)
- A moving average where weights decay exponentially. Used for EMA and also Wilder smoothing for RSI average gains/losses.

## Polynomial / Parabolic fit
- The notebook fits polynomial curves to identified peaks/dips (e.g., quadratic = parabolic) using `numpy.polyfit` and `numpy.poly1d` to visualize trend curvature.

## argrelextrema (scipy.signal)
- Finds relative extrema (maxima/minima) in an array using a neighborhood `order` parameter. Useful for identifying local tops and bottoms.

## find_peaks (scipy.signal)
- Detects peaks with optional filters like `prominence` and `distance`. The notebook uses `prominence` to ignore small or noisy peaks.

---

Usage notes
- Default periods used in the notebook: RSI=14, Bollinger period=20 (std_dev=2), SMA/EMA periods [20, 50, 200], MACD (12,26,9).
- Indicators are tools, not guarantees. Combine multiple indicators and consider price context and volume (not included here) for more robust decisions.

If you prefer this glossary embedded in `README.md` instead, I can merge it into the README or add a short README section and keep the full glossary here.
