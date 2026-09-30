import contextlib
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from IPython.terminal.interactiveshell import TerminalInteractiveShell
from traitlets.config import Config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calc_indicators import calc_rsi, calc_sma_ema, calc_macd, calc_bollinger_bands, calc_stochastic
from src.technische_indicatoren import technische_indicatoren
from src.plot_indicators import plot_rsi, plot_bollinger_bands, plot_monthly_candles


def prices(values):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame(dict(Open=values-.1, High=values+1, Low=values-1,
                             Close=values, Volume=1000.),
                        index=pd.bdate_range('2019-01-01', periods=len(values)))


class IndicatorTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_rsi_wilder_reference(self):
        values = [44.34,44.09,44.15,43.61,44.33,44.83,45.10,45.42,
                  45.84,46.08,45.89,46.03,45.61,46.28,46.28,46.00]
        rsi = calc_rsi(prices(values)).RSI
        self.assertTrue(rsi.iloc[:14].isna().all())
        self.assertAlmostEqual(rsi.iloc[14], 70.46413502109705)
        self.assertAlmostEqual(rsi.iloc[15], 66.24961855355505)

    def test_rsi_flat_monotonic_and_gap(self):
        for values, expected in [(np.ones(40)*20,50), (np.arange(10.,50.),100),
                                  (np.arange(50.,10.,-1),0)]:
            self.assertEqual(calc_rsi(prices(values)).RSI.iloc[-1],expected)
        frame = prices(np.arange(10.,70.))
        frame.loc[frame.index[30],'Close'] = np.nan
        self.assertTrue(calc_rsi(frame).RSI.iloc[30:45].isna().all())

    def test_sma_ema_macd_recurrences(self):
        frame = prices([10,12,11,15,14,17,13])
        averages = calc_sma_ema(frame.copy(),[3])
        ema = [10.]
        for value in frame.Close.iloc[1:]:
            ema.append(.5*value+.5*ema[-1])
        np.testing.assert_allclose(averages.EMA3,ema)
        self.assertAlmostEqual(averages.SMA3.iloc[-1],44/3)
        result = calc_macd(frame.copy(),2,3,2)
        fast = [10.]
        for value in frame.Close.iloc[1:]:
            fast.append(2/3*value+1/3*fast[-1])
        expected = np.array(fast)-ema
        signal = [expected[0]]
        for value in expected[1:]:
            signal.append(2/3*value+1/3*signal[-1])
        np.testing.assert_allclose(result.MACD,expected,atol=1e-12)
        np.testing.assert_allclose(result.Signal,signal,atol=1e-12)
        np.testing.assert_allclose(result.Histogram,expected-signal,atol=1e-12)

    def test_bands_and_table_use_same_population_deviation(self):
        frame = prices([10,12,14,16])
        bands = calc_bollinger_bands(frame.copy(),3)
        self.assertAlmostEqual(bands.BB_Std.iloc[-1],np.sqrt(8/3))
        table = technische_indicatoren(frame,[3],bb_period=3)
        self.assertEqual(table.loc['BB_Upper'].iloc[0],round(bands.BB_Upper.iloc[-1],3))

    def test_stochastic_smoothing_and_zero_range(self):
        frame = prices([10,11,14,12,13,17,15,12,14])
        result = calc_stochastic(frame.copy(),3,2,2)
        raw = pd.Series([np.nan,np.nan]+[
            100*(frame.Close.iloc[i]-frame.Low.iloc[i-2:i+1].min())/
            (frame.High.iloc[i-2:i+1].max()-frame.Low.iloc[i-2:i+1].min())
            for i in range(2,len(frame))],index=frame.index)
        np.testing.assert_allclose(result['%K'],raw.rolling(2).mean(),equal_nan=True)
        np.testing.assert_allclose(result['%D'],raw.rolling(2).mean().rolling(2).mean(),equal_nan=True)
        flat = prices([10]*30).assign(Open=10,High=10,Low=10)
        self.assertTrue(calc_stochastic(flat)['%K'].isna().all())

    def test_multiindex_and_changed_plot_parameters(self):
        frame = prices(100+np.sin(np.arange(300)/5)*10)
        multi = pd.concat({'TEST':frame},axis=1).swaplevel(0,1,axis=1)
        pd.testing.assert_series_equal(calc_rsi(multi).RSI,calc_rsi(frame.copy()).RSI)
        original = multi.copy(deep=True)
        with patch('matplotlib.pyplot.show'):
            plot_bollinger_bands(multi,'test',period=7)
            plot_rsi(calc_rsi(frame.copy(),14),'test',period=7)
        np.testing.assert_allclose(plt.gcf().axes[0].lines[0].get_ydata(),
                                   calc_rsi(frame.copy(),7).RSI,equal_nan=True)
        pd.testing.assert_frame_equal(multi,original)

    def test_monthly_plot_without_valid_candles(self):
        frame = prices([100.] * 40)
        frame.loc[:, ['Open', 'High', 'Low', 'Close']] = np.nan
        with patch('matplotlib.pyplot.show') as show:
            plot_monthly_candles(frame, 'TEST')
        show.assert_not_called()

    def test_notebook_unchanged_cells_run(self):
        path = Path('notebooks/Beurs.ipynb')
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        notebook = json.loads(path.read_text(encoding='utf-8'))
        x = np.arange(1800)
        frame = prices(100+.03*x+5*np.sin(x/8)+3*np.sin(x/31))
        def download(ticker,**kwargs):
            return pd.concat({ticker.upper():frame},axis=1).swaplevel(0,1,axis=1)
        config = Config()
        config.HistoryManager.hist_file = ":memory:"
        shell = TerminalInteractiveShell.instance(config=config)
        scope = {"get_ipython": lambda: shell}
        with patch('yfinance.download',side_effect=download), patch('matplotlib.pyplot.show'), \
                contextlib.redirect_stdout(io.StringIO()):
            for i,cell in enumerate(notebook['cells']):
                if cell['cell_type']=='code':
                    source = shell.input_transformer_manager.transform_cell(''.join(cell['source']))
                    exec(compile(source,f'cell-{i}','exec'),scope)
        figure = scope['manual_result']['figure']
        self.assertEqual(scope['manual_view'].children[1:3],
                         (scope['dof_slider'], scope['future_slider']))
        for name, low, high in [('dof_slider', 1, 5), ('future_slider', 0, 100)]:
            self.assertEqual((scope[name].min, scope[name].max), (low, high))
        scope['dof_slider'].value = 5
        scope['future_slider'].value = 100
        self.assertEqual(scope['manual_result']['settings']['dof'], 5)
        self.assertEqual(scope['manual_result']['settings']['future_days'], 100)
        self.assertIs(scope['manual_result']['figure'], figure)
        self.assertIn('Minimaal 2 punten nodig', scope['manual_status'].value)
        from matplotlib.backend_bases import MouseEvent
        canvas = figure.canvas
        dots = figure.axes[1].collections[0]
        scope['dof_slider'].value = 2
        for index in (0, 2, 4):
            canvas.draw()
            x, y = dots.get_offset_transform().transform(dots.get_offsets()[index])
            canvas.callbacks.process('button_press_event',
                MouseEvent('button_press_event', canvas, x, y, button=1))
        self.assertEqual(len(scope['manual_result']['selected_points']), 3)
        self.assertFalse(scope['manual_result']['fit'].empty)
        self.assertFalse(scope['manual_result']['projection'].empty)
        self.assertIn('RSI-regressie met DOF 2', scope['manual_status'].value)
        scope['future_slider'].value = 0
        self.assertIn('geen stippellijn', scope['manual_status'].value)
        scope['future_slider'].value = 50
        self.assertFalse(scope['manual_result']['projection'].empty)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)


if __name__ == '__main__':
    unittest.main()
