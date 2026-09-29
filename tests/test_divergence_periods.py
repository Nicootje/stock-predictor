import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calc_indicators import calc_rsi
from src.divergence import plot_rsi_divergence


class DivergencePeriodTests(unittest.TestCase):
    def setUp(self):
        x = np.arange(1000)
        self.data = pd.DataFrame({'Close': 100 + .01*x + 5*np.sin(x/13)},
                                index=pd.bdate_range('2020-01-01', periods=len(x)))

    def tearDown(self):
        plt.close('all')

    def test_latest_bar_signals_without_future_bars_in_both_directions(self):
        values = np.array([15, 14, 12, 10, 12, 14, 13, 12, 11, 9, 11, 12], dtype=float)
        for close, direction in [(values, 'bullish'), (30-values, 'bearish')]:
            frame = pd.DataFrame({'Close': close}, index=pd.bdate_range('2024-01-01', periods=12))
            original = frame.copy(deep=True)
            current = plot_rsi_divergence(frame.iloc[:10], 'TEST', trend='short', rsi_period=2, show=False)
            signals = current['current_divergences']
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals.iloc[0].direction, direction)
            self.assertEqual(signals.iloc[0].status, 'PRELIMINARY')
            self.assertTrue(pd.isna(signals.iloc[0].confirmed_on))
            self.assertEqual(signals.iloc[0].detected_on, frame.index[9])
            self.assertTrue(current['all_divergences'].empty)
            self.assertEqual(len(current['figure'].axes[0].lines[0].get_xdata()), 10)
            current['figure'].canvas.draw()
            confirmed_only = plot_rsi_divergence(frame.iloc[:10], 'TEST', trend='short', rsi_period=2,
                                                 include_current=False, show=False)
            self.assertTrue(confirmed_only['divergences'].empty)
            self.assertEqual(len(confirmed_only['figure'].axes[0].lines[0].get_xdata()), 9)
            later = plot_rsi_divergence(frame, 'TEST', trend='short', rsi_period=2,
                                        last_bar_complete=True, show=False)
            self.assertEqual(later['all_divergences'].iloc[0].confirmed_on, frame.index[11])
            self.assertTrue(later['current_divergences'].empty)
            pd.testing.assert_frame_equal(frame, original)

    def test_latest_signal_can_disappear_and_start_date_does_not_change_snapshot(self):
        frame = pd.DataFrame({'Close': [15.,14,12,10,12,14,13,12,11,9]},
                             index=pd.bdate_range('2024-01-01', periods=10))
        result = plot_rsi_divergence(frame, 'TEST', trend='short', rsi_period=2,
                                     start_plot_date=frame.index[5], show=False)
        self.assertEqual(len(result['current_divergences']), 1)
        self.assertTrue(result['divergences'].empty)  # Anchor outside visible range.
        frame.iloc[-1, 0] = 10.5
        changed = plot_rsi_divergence(frame, 'TEST', trend='short', rsi_period=2, show=False)
        self.assertTrue(changed['current_divergences'].empty)

    def test_same_start_date_and_full_rsi_history_for_all_modes(self):
        start = self.data.index[100]
        expected_rsi = calc_rsi(self.data.copy()).RSI.loc[start:]
        for trend in ('short', 'medium', 'long'):
            result = plot_rsi_divergence(self.data, 'TEST', trend=trend,
                start_plot_date=start, last_bar_complete=True, show=False)
            line = result['figure'].axes[0].lines[0]
            pd.testing.assert_index_equal(pd.DatetimeIndex(line.get_xdata()), self.data.index[100:])
            np.testing.assert_allclose(result['figure'].axes[1].lines[0].get_ydata(), expected_rsi)
            whole = plot_rsi_divergence(self.data, 'TEST', trend=trend,
                                       last_bar_complete=True, show=False)
            self.assertEqual(len(whole['figure'].axes[0].lines[0].get_xdata()), 1000)
            pd.testing.assert_frame_equal(result['all_divergences'], whole['all_divergences'])

    def test_long_mode_skips_too_close_pivots_and_waits_for_confirmation(self):
        data = self.data.iloc[:230].copy()
        data.iloc[[20, 90, 100, 190], 0] = [100, 99, 95, 90]
        def rsi(frame, period):
            out = frame.copy()
            out['RSI'] = 50.
            for position, value in [(20, 20), (90, 10), (100, 30), (190, 40)]:
                if position < len(out):
                    out.iloc[position, out.columns.get_loc('RSI')] = value
            return out
        def pivots(values, order, low):
            return np.array([p for p in (20, 90, 100, 190)
                             if p + order < len(values)], dtype=int) if low else np.array([], dtype=int)
        with patch('src.divergence.calc_rsi', side_effect=rsi), patch('src.divergence._pivots', side_effect=pivots):
            result = plot_rsi_divergence(data, 'TEST', trend='long', last_bar_complete=True, show=False)
            events = result['all_divergences']
            self.assertEqual(events.span_bars.tolist(), [80, 90])
            self.assertEqual(events.iloc[0].first_pivot, data.index[20])
            self.assertEqual(events.iloc[0].confirmed_on, data.index[110])
            early = plot_rsi_divergence(data.iloc[:110], 'TEST', trend='long',
                                        last_bar_complete=True, show=False)
            self.assertTrue(early['all_divergences'].empty)
            filtered = plot_rsi_divergence(data, 'TEST', trend='long', start_plot_date=data.index[50],
                                           last_bar_complete=True, show=False)
            self.assertEqual(len(filtered['divergences']), 1)
            pd.testing.assert_frame_equal(filtered['all_divergences'], events)


if __name__ == '__main__':
    unittest.main()
