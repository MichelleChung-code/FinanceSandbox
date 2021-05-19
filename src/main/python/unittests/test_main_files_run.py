# For testing that all the main files run successfully
import unittest
from dateutil.relativedelta import relativedelta
import datetime as dt


class MainFilesRun(unittest.TestCase):

    def test_dummy(self):
        # TODO delete
        self.assertTrue(True)

    # def test_backtesting(self):
    #     """ Tests successful run of backtesting.py """
    #     from backtesting import backtesting
    #
    #     benchmark_index_ticker = '^DJI'
    #     backtesting(benchmark_index_ticker, 50, start_date='2000-01-01')
    #
    # def test_black_scholes_merton(self):
    #     """ Tests successful run of black_scholes_merton.py and BSM_MonteCarlo.py """
    #     from black_scholes_merton import BSM_pricing_value
    #     from BSM_MonteCarlo import BSM_monte_carlo
    #
    #     BSM_monte_carlo(500000, 100, 100, 200, 0.05, 2, 0.2)
    #     BSM_pricing_value(100, 200, 0.05, 2, 0.2)
    #
    # def test_candlestick_chart(self):
    #     """ Tests successful run of candlestick_chart.py """
    #     from candlestick_chart import CandleStickPlot
    #     ticker = '^GSPC'
    #     x = CandleStickPlot(ticker, start_date='2020-12-01')
    #     x()
    #
    # def test_correlation_matrix(self):
    #     """ Tests successful run of correlation_matrix.py """
    #
    #     from correlation_matrix import plot_corr_mat
    #
    #     ls_tickers = ['AAL', 'AA', 'AAPL', 'BBY', 'C', 'CVS', 'HD', 'IBM']
    #
    #     # Get 10 years worth of historical data
    #     start_date = (dt.datetime.today() - relativedelta(years=10)).strftime('%Y-%m-%d')
    #     end_date = dt.datetime.today().strftime('%Y-%m-%d')
    #     plot_corr_mat(ls_tickers, start_date, end_date, res_path=False)
    #
    # def test_simple_regression(self):
    #     """ Tests successful run of simple_regression.py file """
    #
    #     from simple_regression import SimpleRegression
    #
    #     x = SimpleRegression('^STOXX50E', '1999-01-01')
    #     x()
    #
    # def test_simple_stock_data_plot(self):
    #     """ Tests successful run of SimpleStockDataPlot.py """
    #
    #     from SimpleStockDataPlot import plot_bar_volume, PlotStockData, extract_data
    #
    #     ticker = 'AAPL'
    #     start_date = '2014-01-01'
    #     end_date = '2020-03-01'
    #
    #     plot_bar_volume(ticker, start_date, end_date)
    #
    #     stock_data = PlotStockData(ticker, start_date, end_date)
    #     stock_data()
    #
    #     df = extract_data(ticker, start_date, end_date)
