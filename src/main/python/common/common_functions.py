import numpy as np
import common.constants as const
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path
from nltk.corpus import stopwords
from dateutil.relativedelta import relativedelta
import pandas as pd
from SimpleStockDataPlot import extract_data
import datetime as dt

TOL = 1e-10


def daily_risk_free_rate(days=30, tres_rate=0.05 / 100):
    """
    Use compounding interest formula to get the daily risk free rate from the yield to maturity (for a zero-coupon bond)

    Args:
        days: <int> number of days to compound by/ number of days the rate is for
        tres_rate: <float> treasury rate

    Returns:
        <float> daily risk free rate

    """
    # Default uses https://ycharts.com/indicators/1_month_treasury_rate for the 1 month treasury rate as of Jun 18 2021
    return ((tres_rate + 1) ** (1 / days)) - 1


def portfolio_return(df_rets, w):
    """
    Return a series of portfolio returns

    Args:
        df_rets: <pd.DataFrame> of individual asset returns.  Columns are asset names, index is dates
        w: <np.ndarray> of shape len(df_rets.columns), 1

    Returns:
        <pd.DataFrame> of len(df_rets), 1 of portfolio returns
    """
    assert abs(sum(w) - 1) <= TOL
    df = df_rets.dot(w)
    df.columns = ['port_rets']

    return df


def active_return_and_risk(df_port_rets, df_benchmark_rets):
    """
    Compute and return active returns and risk

    r_a = r_p - r_b
    sigma_a = std(r_a)

    Args:
        df_port_rets: <pd.DataFrame> containing portfolio returns
        df_benchmark_rets: <pd.DataFrame> containing benchmark returns

    Returns:
        <dict> containing <pd.DataFrame> and <float> of active return series and the active risk number
    """
    assert len(df_port_rets.columns) == 1 and len(df_benchmark_rets.columns) == 1
    col = ['active_return']
    df_port_rets.columns, df_benchmark_rets.columns = col, col

    df_active_return = df_port_rets.subtract(df_benchmark_rets)

    # we define tracking error to be the same as active risk

    return {col[0]: df_active_return,
            'active_risk_tracking_error': df_active_return.std()[0]}


def get_price_data(ticker_ls, end_date, look_back_mths):
    """
    Return a dataframe of daily price data (adjusted close price), sourced from Yahoo Finance

    Args:
        ticker_ls: <list> of tickers
        end_date: <dt.datetime> of end date to apply look back months to
        look_back_mths: <int> number of months from end date to define the starting date to pull data from

    Returns:
        <pd.DataFrame> containing the price data
    """
    start_date = (end_date - relativedelta(months=look_back_mths))
    df_price_data = pd.DataFrame(
        index=pd.date_range(start=start_date.strftime(const.DATE_STR_FORMAT),
                            end=end_date.strftime(const.DATE_STR_FORMAT), freq='B'),
        columns=ticker_ls)

    for asset in ticker_ls:
        df_price_data[asset] = extract_data(asset, start_date.strftime(const.DATE_STR_FORMAT),
                                            end_date.strftime(const.DATE_STR_FORMAT))[const.ADJ_CLOSE]

    df_price_data.fillna(method='ffill', inplace=True)

    return df_price_data


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


def clean_text(input_text, additional_stopwords_ls=False, text_str=False):
    """
    Initial cleaning of text data

    Args:
        input_text: <str> .txt path to corpus file or text string to clean
        additional_stopwords_ls: <list> optional parameter to include custom list on top of common stop words from nltk
        stop words
        text_str: <bool> defaults as False, whether input_text is a text string or a path to a text file.  Default is
        that it is a path to a file

    Returns:
        <list> of cleaned words
    """
    assert isinstance(text_str, bool)

    if not text_str:
        assert input_text.endswith('.txt')
        with open(input_text) as file:
            file_text = file.read()
    else:
        file_text = input_text

    words = file_text.split()
    lower_case_alphabetic_only = [word.lower() for word in words if
                                  word.isalpha()]  # consistent case, convert everything to lowercase if isalpha()

    stopwords_nltk = list(set(stopwords.words('english')))  # stop words for the english language
    if additional_stopwords_ls:
        stopwords_nltk = stopwords_nltk + additional_stopwords_ls

    return [word for word in lower_case_alphabetic_only if word not in stopwords_nltk]


if __name__ == '__main__':
    ls_assets = ['AAPL', 'NKE', 'GOOGL', 'AMZN']
    df_rets = get_price_data(ls_assets, end_date=dt.datetime.today(), look_back_mths=12).pct_change().fillna(0)

    w = np.random.rand(len(ls_assets), 1)
    w = w / sum(w)

    df_port_rets = portfolio_return(df_rets, w)

    df_benchmark_rets = get_price_data(['^GSPC'], end_date=dt.datetime.today(), look_back_mths=12).pct_change().fillna(
        0)

    print(active_return_and_risk(df_port_rets, df_benchmark_rets))

    # from SimpleStockDataPlot import extract_data
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

    # import random
    #
    # data = [random.gauss(1.5, 2) for i in range(1000000)]
    # res_path = os.path.join(str(Path(__file__).parents[1]), 'data', 'data.pkl')
    #
    # write_to_disk(data, res_path)
    # print(read_from_disk(res_path))
