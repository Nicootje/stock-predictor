import unittest
import numpy as np
import pandas as pd
from src.calc_indicators import calc_rsi
from src.RSI_predictor import price_for_target_rsi


def prices(values):
    return pd.DataFrame({'Close': values}, index=pd.bdate_range('2020-01-01', periods=len(values)))


class TargetRsiTests(unittest.TestCase):
    def test_recalculated_rsi_matches_target(self):
        frame = prices(100 + np.cumsum(np.random.default_rng(42).normal(0, 1, 300)))
        frame['RSI'] = 99.
        original = frame.copy(deep=True)
        for period in (7, 14, 21):
            for target in (20, 50, 70, 90):
                price = price_for_target_rsi(frame, target, period)
                self.assertIsNotNone(price)
                extended = prices([*frame.Close, price])
                self.assertAlmostEqual(calc_rsi(extended, period).RSI.iloc[-1], target, places=10)
        pd.testing.assert_frame_equal(frame, original)

    def test_same_rsi_and_one_sided_history(self):
        frame = prices(np.arange(10., 60.))
        self.assertEqual(price_for_target_rsi(frame, 100), 59.)
        target = price_for_target_rsi(frame, 70)
        self.assertAlmostEqual(calc_rsi(prices([*frame.Close, target])).RSI.iloc[-1], 70)

    def test_flat_unreachable_and_invalid_inputs(self):
        flat = prices([30.]*30)
        self.assertEqual(price_for_target_rsi(flat, 50), 30.)
        for target in (0, 70, 100):
            self.assertIsNone(price_for_target_rsi(flat, target))
        for kwargs in ({'target_rsi': -1}, {'target_rsi': np.nan}, {'rsi_period': 1}):
            with self.assertRaises(ValueError):
                price_for_target_rsi(flat, **kwargs)
        with self.assertRaises(ValueError):
            price_for_target_rsi(flat.iloc[:5])
        with self.assertRaises(ValueError):
            price_for_target_rsi(flat.assign(Close=np.nan))
        self.assertIsNone(price_for_target_rsi(prices([10, 20]*20), 1))

    def test_yfinance_ticker_selection(self):
        frame = prices(100 + np.sin(np.arange(100)))
        multi = pd.concat({'TEST': frame, 'OTHER': frame*2}, axis=1).swaplevel(0, 1, axis=1)
        self.assertEqual(price_for_target_rsi(multi, 70, ticker='test'), price_for_target_rsi(frame, 70))


if __name__ == '__main__':
    unittest.main()
