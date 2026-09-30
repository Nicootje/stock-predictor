import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent, PickEvent
import numpy as np
import pandas as pd

from src.manual_divergence import plot_manual_rsi_divergence


class ManualDivergenceTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_click_selection_connects_same_dates_and_can_remove_points(self):
        df = pd.DataFrame({'Close': 100 + 10 * np.sin(np.arange(250) / 5)},
                          index=pd.date_range('2025-01-01', periods=250))
        original = df.copy()
        result = plot_manual_rsi_divergence(df, 'TEST', show=False, plot_level=True, future_days=85)
        fig = result['figure']
        fig.canvas.draw()
        price_ax, rsi_ax = fig.axes
        dots = rsi_ax.collections[0]

        def click(index):
            x, y = dots.get_offset_transform().transform(dots.get_offsets()[index])
            mouse = MouseEvent('button_press_event', fig.canvas, x, y, button=1)
            fig.canvas.callbacks.process('pick_event',
                PickEvent('pick_event', fig.canvas, mouse, dots, ind=[index]))

        for index in (3, 0, 2, 1):
            click(index)
        selected = result['selected_points']
        self.assertEqual(selected.id.tolist(), [0, 1, 2, 3])
        self.assertAlmostEqual(result['rsi_level'], selected.rsi.mean())
        self.assertEqual(result['projection'].date.iloc[-1], selected.date.iloc[-1] + pd.Timedelta(days=85))
        np.testing.assert_allclose(price_ax.lines[1].get_ydata(),
                                   df.loc[selected.date, 'Close'])
        np.testing.assert_allclose(rsi_ax.lines[1].get_ydata(), selected.rsi)
        self.assertEqual(rsi_ax.lines[1].get_linestyle(), '-')
        np.testing.assert_array_equal(price_ax.lines[1].get_xdata(),
                                      rsi_ax.lines[1].get_xdata())
        self.assertGreater(price_ax.get_position().y0, rsi_ax.get_position().y0)
        for index in (0, 1, 2, 3):
            click(index)
        self.assertTrue(result['selected_points'].empty)
        self.assertTrue(result['fit'].empty)
        self.assertTrue(result['projection'].empty)
        self.assertIsNone(result['rsi_level'])
        self.assertEqual(len(price_ax.lines[1].get_xdata()), 0)
        pd.testing.assert_frame_equal(df, original)



    def test_polynomial_and_level_on_known_extrema(self):
        dates = pd.date_range('2025-01-01', periods=11)
        frame = pd.DataFrame({'Close': np.arange(100., 111.),
                              'RSI': [20, 41, 20, 49, 20, 65, 20, 70, 20, 75, 20]}, index=dates)
        def select(indices, **kwargs):
            result = plot_manual_rsi_divergence(frame, 'TEST', prominence=0, distance=1, show=False, **kwargs)
            fig = result['figure']
            fig.canvas.draw()
            dots = fig.axes[1].collections[0]
            for index in indices:
                x, y = dots.get_offset_transform().transform(dots.get_offsets()[index])
                mouse = MouseEvent('button_press_event', fig.canvas, x, y, button=1)
                fig.canvas.callbacks.process('pick_event',
                    PickEvent('pick_event', fig.canvas, mouse, dots, ind=[index]))
            return result

        for degree, expected_end in [(0, (41+49+65)/3), (1, 39+2/3+6*6), (2, 89)]:
            with self.subTest(degree=degree), patch('src.manual_divergence.calc_rsi', return_value=frame):
                result = select([4, 0, 2], dof=degree, future_days=2, plot_level=True)
                self.assertAlmostEqual(result['projection'].rsi.iloc[-1], expected_end)
                self.assertEqual(result['projection'].date.iloc[-1], dates[7])
                self.assertAlmostEqual(result['fit'].price.iloc[-1], result['projection'].price.iloc[0])
                if degree > 0:
                    self.assertAlmostEqual(result['projection'].price.iloc[-1], 107.)
                price_lines = {l.get_label(): l for l in result['figure'].axes[0].lines}
                self.assertEqual(price_lines['Doortrekking'].get_linestyle(), ':')
                self.assertFalse(any('Koers-regressie' in label for label in price_lines))
                self.assertAlmostEqual(result['rsi_level'], (41+49+65)/3)
                self.assertEqual(result['fit'].date.iloc[-1], dates[5])
                self.assertEqual(result['projection'].date.iloc[0], dates[5])
                self.assertEqual(result['fit'].rsi.iloc[-1], result['projection'].rsi.iloc[0])
                lines = {line.get_label(): line for line in result['figure'].axes[1].lines}
                self.assertEqual(lines[f'RSI-regressie (dof={degree})'].get_linestyle(), '-')
                self.assertEqual(lines['Doortrekking'].get_linestyle(), ':')
                self.assertEqual(result['extrema'].date.tolist(), dates[1:10].tolist())
                for ax in result['figure'].axes:
                    self.assertFalse(any(isinstance(t, matplotlib.text.Annotation) for t in ax.texts))
        with patch('src.manual_divergence.calc_rsi', return_value=frame):
            result = select([0, 1, 2, 3, 4, 5], dof=2, future_days=20)
            selection = result['selected_points'].copy()
            original_fit = result['fit'].copy()
            figure = result['figure']
            with patch.object(np.polynomial.Polynomial, 'fit', wraps=np.polynomial.Polynomial.fit) as fit:
                result['update_settings'](future_days=10)
                result['update_settings'](future_days=30)
                fit.assert_not_called()
                result['update_settings'](dof=5, future_days=100)
                self.assertEqual(fit.call_count, 2)
            self.assertIs(result['figure'], figure)
            pd.testing.assert_frame_equal(result['selected_points'], selection)
            self.assertFalse(result['fit'].equals(original_fit))
            self.assertEqual(result['settings']['dof'], 5)
            self.assertEqual(result['projection'].date.iloc[-1], selection.date.iloc[-1] + pd.Timedelta(days=100))
            result['update_settings'](future_days=0)
            self.assertTrue(result['projection'].empty)
            result['update_settings'](dof=6)
            self.assertFalse(result['fit'].empty)
            self.assertEqual(result['effective_dof'], 5)
            result['update_settings'](dof=2, future_days=10)
            self.assertFalse(result['fit'].empty)
            self.assertFalse(result['projection'].empty)

        for degree in (3, 4):
            with self.subTest(degree=degree), patch('src.manual_divergence.calc_rsi', return_value=frame):
                result = select(list(range(0, 2*(degree+1), 2)), dof=degree, future_days=1)
                self.assertFalse(result['fit'].empty)
                self.assertFalse(result['projection'].empty)
                self.assertAlmostEqual(result['fit'].rsi.iloc[0], 41)
                self.assertAlmostEqual(result['fit'].rsi.iloc[-1], frame.RSI.iloc[2*degree+1])
        with patch('src.manual_divergence.calc_rsi', return_value=frame):
            result = select([0, 2, 4], dof=2, future_days=20)
            self.assertTrue(result['projection'].rsi.isna().any())
            self.assertTrue(result['fit'].rsi.dropna().between(0, 100).all())
            self.assertIsNone(result['rsi_level'])
            result = select([0, 2], dof=2, plot_level=True)
            self.assertFalse(result['fit'].empty)
            self.assertEqual(result['effective_dof'], 1)
            for degree in range(1, 6):
                result['update_settings'](dof=degree, future_days=50)
                self.assertTrue(result['projection'][['rsi', 'price']].notna().any().all())
                self.assertEqual(result['effective_dof'], 1)
            self.assertEqual(result['rsi_level'], 45)
            result = select([0, 2], future_days=0)
            self.assertEqual(result['fit'].date.iloc[-1], dates[3])
            self.assertTrue(result['projection'].empty)

    def test_filters_reduce_tops_and_bottoms(self):
        dates = pd.date_range('2025-01-01', periods=13)
        frame = pd.DataFrame({'Close': 100.,
            'RSI': [50, 60, 50, 52, 50, 65, 50, 40, 50, 48, 50, 35, 50]}, index=dates)
        with patch('src.manual_divergence.calc_rsi', return_value=frame):
            all_points = plot_manual_rsi_divergence(frame, 'TEST', prominence=0, distance=1, show=False)
            prominent = plot_manual_rsi_divergence(frame, 'TEST', prominence=6, distance=1, show=False)
            spaced = plot_manual_rsi_divergence(frame, 'TEST', prominence=0, distance=5, show=False)
        self.assertLess(len(prominent['extrema']), len(all_points['extrema']))
        self.assertNotIn(dates[3], prominent['extrema'].date.tolist())
        self.assertNotIn(dates[9], prominent['extrema'].date.tolist())
        self.assertIn(dates[5], prominent['extrema'].date.tolist())
        self.assertIn(dates[11], prominent['extrema'].date.tolist())
        self.assertLess(len(spaced['extrema']), len(all_points['extrema']))
        self.assertEqual(spaced['settings']['distance'], 5)

    def test_invalid_projection_parameters(self):
        frame = pd.DataFrame({'Close': np.arange(100., 150.)},
                             index=pd.date_range('2025-01-01', periods=50))
        for kwargs in ({'dof': -1}, {'dof': 1.5}, {'dof': True},
                       {'future_days': -1}, {'future_days': np.nan},
                       {'future_days': True}, {'plot_level': 'yes'},
                       {'prominence': -1}, {'prominence': np.nan},
                       {'distance': 0}, {'distance': 1.5}, {'distance': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                plot_manual_rsi_divergence(frame, 'TEST', show=False, **kwargs)


if __name__ == '__main__':
    unittest.main()
