from SimpleStockDataPlot import extract_data
import pandas as pd
from common.simple_line_plot import show_line_plot
from common.common_functions import log_return
from datetime import datetime

from common.constants import CLOSE, NUM_TRADE_DAYS_PER_YR, NUM_TRADE_DAYS_PER_MONTH, \
    COL_NUM_TRADE_DAYS_YR_TREND, COL_NUM_TRADE_DAYS_MONTH_TREND, SIGNAL_VAL, SIGNAL_NAME, SIGNALS_DICT

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


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
                   title='Backtesting of {}'.format(benchmark_index), x_label='Date',
                   y_label='Cumulative Returns',
                   plot_arr=True)