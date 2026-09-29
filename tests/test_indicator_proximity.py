import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.indicator_proximity import scan_indicator_proximity


def frame(values):
    return pd.DataFrame({'Close': values}, index=pd.bdate_range('2020-01-01', periods=len(values)))


class ProximityTests(unittest.TestCase):
    def test_flat_history_and_no_duplicate_middle_band(self):
        data = frame(np.ones(260)*30)
        original = data.copy(deep=True)
        result = scan_indicator_proximity(['flat'], data_by_ticker={'FLAT': data})
        row = result.iloc[0]
        self.assertTrue(row['match'])
        self.assertEqual(row.n_values, 9)
        self.assertEqual(row.cluster_width_pct, 0)
        self.assertEqual(row.bb_width_pct, 0)
        self.assertEqual(row.as_of, data.index[-2])
        self.assertIn('SMA20/BB_Middle', row.cluster)
        pd.testing.assert_frame_equal(data, original)

    def test_scale_invariance_and_group_total_width(self):
        data = frame(100+np.arange(300)*.1+np.sin(np.arange(300)))
        result = scan_indicator_proximity(['A', 'B'], max_distance_pct=2,
                                          data_by_ticker={'A':data, 'B':data*100})
        self.assertTrue(result.cluster_width_pct.le(2+1e-10).all())
        self.assertAlmostEqual(*result.cluster_width_pct.to_list())
        self.assertEqual(*result.cluster.to_list())

    def test_errors_visible_and_no_network_for_supplied_data(self):
        with patch('yfinance.download', side_effect=AssertionError('No downloads')):
            result = scan_indicator_proximity(['OK','SHORT','BAD','MISSING'], data_by_ticker={
                'OK':frame([30.]*250), 'SHORT':frame([30.]*10), 'BAD':frame([np.nan]*250)})
        status = result.set_index('ticker').status
        self.assertEqual(status['OK'],'OK')
        self.assertEqual(status['SHORT'],'INSUFFICIENT_DATA')
        self.assertEqual(status['MISSING'],'ERROR')
        self.assertFalse(result.loc[result.status.ne('OK'),'match'].any())

    def test_batch_download_and_empty_ticker_list(self):
        raw = pd.concat({'A':frame([30.]*250),'B':frame([60.]*250)},axis=1).swaplevel(0,1,axis=1)
        with patch('yfinance.download',return_value=raw) as download:
            result = scan_indicator_proximity(['a','B','A'])
            self.assertEqual(download.call_count,1)
            self.assertEqual(len(result),2)
            self.assertTrue(result.status.eq('OK').all())
        self.assertTrue(scan_indicator_proximity([]).empty)


if __name__ == '__main__':
    unittest.main()
