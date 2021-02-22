import numpy as np
import src.main.python.common.constants as const
import matplotlib.pyplot as plt


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


# TODO add docstrings
def rolling_correlation(df, col_name1, col_name2, window=const.NUM_TRADE_DAYS_PER_YR):
    s1 = df[col_name1]
    s2 = df[col_name2]

    return s1.rolling(window).corr(s2)


if __name__ == '__main__':
    from src.main.python.SimpleStockDataPlot import extract_data

    ticker = '^GDAXI'
    start_date = '2000-01-01'
    end_date = '2014-09-26'
    df = extract_data(ticker, start_date, end_date)
    df, ret_col_name = log_return(df, const.CLOSE)
    df, avg_col_name = rolling_average(df, const.CLOSE)
    df, std_col_name = rolling_standard_deviation(df, ret_col_name)

    print(df.tail())

    df[[const.CLOSE, std_col_name, ret_col_name]].plot(subplots=True)
    plt.show()
