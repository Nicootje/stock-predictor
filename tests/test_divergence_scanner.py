import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.divergence import plot_rsi_divergence
from src.divergence_scanner import scan_rsi_divergence


def frame(values):
    return pd.DataFrame({'Close': values}, index=pd.bdate_range('2024-01-01', periods=len(values)))


class DivergenceScannerTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_current_bullish_bearish_and_all_three_terms(self):
        # A sharp first trough followed by a lower, gradual second trough.
        for term, order, gap in [('short',2,10), ('medium',5,40), ('long',10,90)]:
            values = np.r_[np.linspace(150,100,21), np.linspace(102,140,order+2),
                           np.linspace(139,90,gap-order-2)]
            bullish = frame(values)
            bearish = frame(300-values)
            with patch('yfinance.download', side_effect=AssertionError('No download')):
                result = scan_rsi_divergence(['UP','DOWN'], rsi_period=14,
                    data_by_ticker={'UP':bullish,'DOWN':bearish}, show=False).set_index('ticker')
            self.assertEqual(result.loc['UP',term], 'Bullish')
            self.assertEqual(result.loc['DOWN',term], 'Bearish')
            self.assertFalse(plt.get_fignums())
            for name, data in [('UP',bullish), ('DOWN',bearish)]:
                plot = plot_rsi_divergence(data, name, trend=term, show=False)
                self.assertEqual(result.loc[name,term], plot['current_divergences'].iloc[0].direction.capitalize())
                self.assertEqual(result.loc[name,'as_of'], data.index[-1])
                plt.close(plot['figure'])

    def test_all_tickers_returned_but_only_matches_displayed(self):
        values = np.r_[np.linspace(150,100,21),np.linspace(102,140,4),np.linspace(139,90,6)]
        with patch('src.divergence_scanner.display') as display:
            result = scan_rsi_divergence(['UP','FLAT','MISSING','SHORT'],data_by_ticker={
                'UP':frame(values),'FLAT':frame([100.]*200),'SHORT':frame([100.]*5)})
        self.assertEqual(len(result),4)
        tables = [call.args[0] for call in display.call_args_list
                  if isinstance(call.args[0], pd.DataFrame)]
        self.assertEqual(tables[0].ticker.tolist(), ['UP'])
        self.assertEqual(set(tables[1].ticker), {'UP', 'MISSING', 'SHORT'})
        self.assertFalse(tables[1].status.eq('OK').any())
        self.assertEqual(result.set_index('ticker').loc['MISSING','status'],'ERROR')
        self.assertEqual(result.set_index('ticker').loc['SHORT','short'],'ONVOLDOENDE DATA')

    def test_batch_one_retry_and_duplicate_tickers(self):
        data = frame([100.]*200)
        raw = pd.concat({'A':data,'B':data*np.nan},axis=1).swaplevel(0,1,axis=1)
        with patch('yfinance.download',side_effect=[raw,data]) as download:
            result = scan_rsi_divergence(['a','B','A'],show=False)
            self.assertEqual(download.call_count,2)
            self.assertEqual(download.call_args.args[0],'B')
        self.assertEqual(len(result),2)
        self.assertTrue(result.status.eq('OK').all())

    def test_no_mutation_empty_input_and_invalid_data(self):
        data = frame([100.]*200)
        original = data.copy(deep=True)
        scan_rsi_divergence('A',data_by_ticker={'A':data},show=False)
        pd.testing.assert_frame_equal(data,original)
        self.assertTrue(scan_rsi_divergence([],show=False).empty)
        bad = data.copy()
        bad.iloc[-1,0] = np.nan
        result = scan_rsi_divergence('BAD',data_by_ticker={'BAD':bad},show=False)
        self.assertEqual(result.iloc[0].status,'ERROR')


if __name__ == '__main__':
    unittest.main()
