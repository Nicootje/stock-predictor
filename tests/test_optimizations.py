import contextlib
import io
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calc_indicators import _wilder_mean
from src.technische_indicatoren import technische_indicatoren
from src.summary_analysis import summary_technical_indicators
from src.plot_indicators import plot_full_chart


class OptimizationTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_wilder_matches_reference_with_gaps_and_short_runs(self):
        def reference(values, period):
            result = np.full(len(values), np.nan)
            seed, average = [], np.nan
            for i, value in enumerate(values):
                if not np.isfinite(value):
                    seed, average = [], np.nan
                elif np.isnan(average):
                    seed.append(value)
                    if len(seed) == period:
                        average = np.mean(seed)
                        result[i] = average
                else:
                    average = (average * (period - 1) + value) / period
                    result[i] = average
            return result

        values = np.random.default_rng(42).normal(size=1000)
        values[[0, 1, 14, 30, 31, 45, 900, 999]] = np.nan
        values[500] = np.inf
        values[600] = -np.inf
        for period in (1, 2, 14, 200, 2000):
            with self.subTest(period=period):
                np.testing.assert_allclose(_wilder_mean(pd.Series(values), period),
                    reference(values, period), rtol=1e-12, atol=1e-12, equal_nan=True)
        self.assertTrue(_wilder_mean(pd.Series(dtype=float), 14).empty)

    def test_period_iterators_work_for_table_and_summary(self):
        data = pd.DataFrame({'Close': 100 + np.arange(250)*.1},
                            index=pd.date_range('2025-01-01', periods=250))
        original = data.copy()
        pd.testing.assert_frame_equal(technische_indicatoren(data, iter([20, 50])),
                                      technische_indicatoren(data, [20, 50]))
        with contextlib.redirect_stdout(io.StringIO()):
            a = summary_technical_indicators('TEST', data, 14, iter([20, 50]), None)
            b = summary_technical_indicators('TEST', data, 14, [20, 50], None)
        self.assertEqual(a, b)
        pd.testing.assert_frame_equal(data, original)

    def test_full_chart_skips_disabled_averages(self):
        data = pd.DataFrame({'Close': 100 + np.arange(60)*.1},
                            index=pd.date_range('2025-01-01', periods=60))
        with patch('src.plot_indicators.calc_sma_ema') as averages, \
                patch('matplotlib.pyplot.show'), contextlib.redirect_stdout(io.StringIO()):
            plot_full_chart(data, 'TEST', sma=False, ema=False)
        averages.assert_not_called()
        self.assertEqual(len(plt.gcf().axes), 3)


if __name__ == '__main__':
    unittest.main()
