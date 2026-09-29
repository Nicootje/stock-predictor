import unittest

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
        result = plot_manual_rsi_divergence(df, 'TEST', prominence=1, distance=2,
                                            show=False)
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
        np.testing.assert_allclose(price_ax.lines[1].get_ydata(),
                                   df.loc[selected.date, 'Close'])
        np.testing.assert_allclose(rsi_ax.lines[1].get_ydata(), selected.rsi)
        np.testing.assert_array_equal(price_ax.lines[1].get_xdata(),
                                      rsi_ax.lines[1].get_xdata())
        self.assertGreater(price_ax.get_position().y0, rsi_ax.get_position().y0)
        for index in (0, 1, 2, 3):
            click(index)
        self.assertTrue(result['selected_points'].empty)
        self.assertEqual(len(price_ax.lines[1].get_xdata()), 0)
        pd.testing.assert_frame_equal(df, original)


if __name__ == '__main__':
    unittest.main()
