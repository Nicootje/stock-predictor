import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.divergence import _current_divergences, _line_points, _clear_divergence


class DivergenceContinuationTests(unittest.TestCase):
    def test_pullback_is_not_a_clear_preliminary_extremum(self):
        for sign, direction in [(1, 'bearish'), (-1, 'bullish')]:
            data = pd.DataFrame({'Close': 100 + sign*np.array([0, 5, 0, 2, 4, 8, 7]),
                                 'RSI': 50 + sign*np.array([0, 20, 0, 5, 8, 10, 9])},
                                index=pd.date_range('2024-01-01', periods=7))
            settings = dict(order=1, min_distance=4, max_distance=10)
            signals = _current_divergences(data, data.iloc[:-1], settings)
            self.assertTrue(signals.empty)
            data.iloc[-1, 0] = 100 + sign * 9
            signals = _current_divergences(data, data.iloc[:-1], settings)
            self.assertEqual(signals.direction.tolist(), [direction])
            self.assertEqual(signals.status.tolist(), ['PRELIMINARY'])
            data.iloc[-1, 0] = data.Close.iloc[1]
            self.assertTrue(_current_divergences(data, data.iloc[:-1], settings).empty)

    def test_small_and_wrong_direction_changes_are_rejected(self):
        for low, sign in [(False, 1), (True, -1)]:
            for price_change, rsi_change, expected in [
                    (.99, 10, False), (2, 4.99, False), (1, 5, True),
                    (2, 10, True), (-2, 10, False), (2, -10, False)]:
                with self.subTest(low=low, price=price_change, rsi=rsi_change):
                    self.assertEqual(_clear_divergence(
                        100, 100 + sign * price_change, 50,
                        50 - sign * rsi_change, low, {}), expected)

    def test_multiple_aligned_pivots_and_reject_off_line_rsi(self):
        dates = pd.date_range('2024-01-01', periods=9)
        data = pd.DataFrame({'Close': np.linspace(100, 108, 9),
                             'RSI': np.linspace(70, 50, 9)}, index=dates)
        event = SimpleNamespace(first_pivot=dates[0], second_pivot=dates[8],
                                first_price=100, second_price=108,
                                first_rsi=70, second_rsi=50, direction='bearish')
        with patch('src.divergence._pivots', return_value=np.array([2, 4, 6])):
            self.assertEqual(_line_points(event, data, 1), list(dates[[0, 2, 4, 6, 8]]))
            data.loc[dates[4], 'RSI'] = 80
            self.assertEqual(_line_points(event, data, 1), list(dates[[0, 2, 6, 8]]))
