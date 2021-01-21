from src.main.python.SimpleStockDataPlot import extract_data
import pandas as pd
from src.main.python.common.simple_line_plot import show_line_plot
from src.main.python.common.common_functions import log_return
from datetime import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# constants
CLOSE = 'Close'
NUM_TRADE_DAYS_PER_YR = 252
NUM_TRADE_DAYS_PER_MONTH = 21
COL_NUM_TRADE_DAYS_YR_TREND = '{}_days'.format(NUM_TRADE_DAYS_PER_YR)
COL_NUM_TRADE_DAYS_MONTH_TREND = '{}_days'.format(NUM_TRADE_DAYS_PER_MONTH)
BUY = 'BUY'
SELL = 'SELL'
HOLD = 'HOLD'

SIGNAL_VAL = 'signal_val'
SIGNAL_NAME = 'signal_name'
SIGNALS_DICT = {1: BUY, -1: SELL, 0: HOLD}


def backtesting(benchmark_index, signal_tolerance, start_date, end_date=datetime.today().strftime('%Y-%m-%d')):
    """
    Backtest performance of a stock if following a signal based strategy to buy if the monthly rolling average exceeds
    a signal tolerance over the annual rolling average.  Rules for sell and hold follow simularly.

    Args:
        benchmark_index: <str> ticker for the stock of interest
        signal_tolerance: <int> tolerance controlling the buy, sell, hold actions
        start_date: <str> YYYY-MM-DD start date for data
        end_date: <str> YYYY-MM-DD end date for data.  Defaults to today if not provided.
    """
    benchmark_historic_data = extract_data(ticker=benchmark_index, start_date=start_date, end_date=end_date)

    # Get the rolling average annual and per month
    benchmark_historic_data[COL_NUM_TRADE_DAYS_YR_TREND] = benchmark_historic_data[CLOSE].rolling(
        NUM_TRADE_DAYS_PER_YR).mean()
    benchmark_historic_data[COL_NUM_TRADE_DAYS_MONTH_TREND] = benchmark_historic_data[CLOSE].rolling(
        NUM_TRADE_DAYS_PER_MONTH).mean()

    # work around the annual trend for standard deviation todo think about this, instead of using signal tolerance
    # standard_dev = np.floor(benchmark_historic_data[COL_NUM_TRADE_DAYS_YR_TREND].std())

    # trend difference
    COL_NAME_DIFF = '{}-{}_difference'.format(NUM_TRADE_DAYS_PER_MONTH, NUM_TRADE_DAYS_PER_YR)
    benchmark_historic_data[COL_NAME_DIFF] = \
        benchmark_historic_data[COL_NUM_TRADE_DAYS_MONTH_TREND] - benchmark_historic_data[COL_NUM_TRADE_DAYS_YR_TREND]

    # assuming that we neglect transaction costs and market liquidity
    benchmark_historic_data.loc[abs(benchmark_historic_data[COL_NAME_DIFF]) <= signal_tolerance, SIGNAL_VAL] = 0
    benchmark_historic_data.loc[benchmark_historic_data[COL_NAME_DIFF] > signal_tolerance, SIGNAL_VAL] = 1
    benchmark_historic_data.loc[benchmark_historic_data[COL_NAME_DIFF] < -signal_tolerance, SIGNAL_VAL] = -1

    # exhaustive mapping for signal name
    benchmark_historic_data[SIGNAL_NAME] = benchmark_historic_data[SIGNAL_VAL].map(SIGNALS_DICT)
    print(benchmark_historic_data[SIGNAL_NAME].value_counts())

    show_line_plot(benchmark_historic_data[SIGNAL_VAL], title='Trading Day Signals', x_label='Date', y_label='Signal',
                   plot_arr=True)

    # Compare the returns
    benchmark_historic_data, original_returns_col_name = log_return(benchmark_historic_data, CLOSE)

    # apply proposed trading signals, shift up one day, returns based on t-1
    benchmark_historic_data['Custom'] = benchmark_historic_data[original_returns_col_name] * benchmark_historic_data[
        SIGNAL_VAL].shift(1)

    benchmark_historic_data[[original_returns_col_name, 'Custom']] = benchmark_historic_data[
        [original_returns_col_name, 'Custom']].cumsum()

    benchmark_historic_data.rename(columns={original_returns_col_name: 'Market'}, inplace=True)

    show_line_plot(benchmark_historic_data[['Market', 'Custom']],
                   title='Backtesting of {}'.format(benchmark_index_ticker), x_label='Date',
                   y_label='Cumulative Returns',
                   plot_arr=True)


if __name__ == '__main__':
    benchmark_index_ticker = '^DJI'

    backtesting(benchmark_index_ticker, 50, start_date='2000-01-01')
