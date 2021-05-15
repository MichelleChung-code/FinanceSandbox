import numpy as np
import src.main.python.common.constants as const
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path


def log_return(df, col_name):
    """
    Compute the log returns

    Args:
        df: <pd.DataFrame> containing data
        col_name: <str> name of column to compute returns on

    Returns:
        df: <pd.DataFrame> with return column appended
        ret_col_name: <str> resulting column name containing the returns.  Format is {col_name}_log_ret
    """
    ret_col_name = col_name + '_log_ret'

    df[ret_col_name] = np.log(df[col_name] / df[col_name].shift(1))

    return df, ret_col_name


def rolling_average(df, col_name, window=const.NUM_TRADE_DAYS_PER_YR):
    """
    Compute the rolling average

    Args:
        df: <pd.DataFrame> to calculate on
        col_name: <str> name of column to calculate mean on.
        window: <int> window size in number of days

    Returns:
        df: <pd.DataFrame> with new column for the rolling average.  Defaults to annual.
        res_col_name: <str> column name of column containing the rolling average
    """
    res_col_name = '{}_days_avg'.format(window)
    df[res_col_name] = df[col_name].rolling(window).mean()

    return df, res_col_name


def rolling_standard_deviation(df, col_name, window=const.NUM_TRADE_DAYS_PER_YR):
    """
    Compute the moving volatility

    Args:
        df: <pd.DataFrame> to calculate on
        col_name: <str> name of column to calculate the std on.  Should be a return type column.
        window: <int> window size in number of days

    Returns:
        df: <pd.DataFrame> with new column for the moving volatility.  Defaults to moving annualized volatility.
        res_col_name: <str> column name of column containing the moving volatlity 
    """
    res_col_name = '{}_days_moving_std'.format(window)
    df[res_col_name] = np.sqrt(window) * df[col_name].rolling(window).std()

    return df, res_col_name


def rolling_correlation(df, col_name1, col_name2, window=const.NUM_TRADE_DAYS_PER_YR):
    """
    Compute the rolling correlation (Pearson's) between two columns

    Args:
        df: <pd.DataFrame> to calculate on
        col_name1: <str> name of first column
        col_name2: <str> name of second column
        window: <int> window size in number of days

    Returns:
        <pd.Series> containing the rolling correlation
    """
    s1 = df[col_name1]
    s2 = df[col_name2]

    return s1.rolling(window).corr(s2)


def write_to_disk(data, output_pkl_path):
    """
    Write data to disk as pickle file

    Args:
        data: to write
        output_pkl_path: <str> path to write to.  Must be a .pkl file

    """

    # Needs to be a pickle file
    assert output_pkl_path.endswith('.pkl')

    pkl = open(output_pkl_path, 'wb')

    pickle.dump(data, pkl)
    pkl.close()


def read_from_disk(input_pkl_path):
    """
    Loads the data from disk pickle file into the system.
    NOTE: pickle stores according to first in, first out (FIFO) principle.

    Args:
        input_pkl_path: <str> path to data pickle file to read from

    Returns: the data that was stored

    """
    # needs to be a pickle file
    assert input_pkl_path.endswith('.pkl')

    pkl = open(input_pkl_path, 'rb')
    data = pickle.load(pkl)

    pkl.close()
    return data


if __name__ == '__main__':
    # from src.main.python.SimpleStockDataPlot import extract_data
    #
    # ticker = '^GDAXI'
    # start_date = '2000-01-01'
    # end_date = '2014-09-26'
    # df = extract_data(ticker, start_date, end_date)
    # df, ret_col_name = log_return(df, const.CLOSE)
    # df, avg_col_name = rolling_average(df, const.CLOSE)
    # df, std_col_name = rolling_standard_deviation(df, ret_col_name)
    #
    # print(df.tail())
    #
    # df[[const.CLOSE, std_col_name, ret_col_name]].plot(subplots=True)
    # plt.show()

    import random

    data = [random.gauss(1.5, 2) for i in range(1000000)]
    res_path = os.path.join(str(Path(__file__).parents[1]), 'data', 'data.pkl')

    write_to_disk(data, res_path)
    print(read_from_disk(res_path))
