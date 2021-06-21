from common.common_functions import get_price_data
import datetime as dt
from scipy.stats import norm
import common.constants as const


def volatility(ret_df):
    """
    Standard Deviation

    Args:
        ret_df: <pd.DataFrame> containing return data.  Date index with asset names as columns.

    Returns:
        <pd.Series> containing standard deviation per asset
    """
    return ret_df.std()


def semi_deviation(ret_df, target=0):
    """
    Standard deviation of the values that fall below a target return

    Args:
        ret_df: <pd.DataFrame> containing return data.  Date index with asset names as columns.
        target: <float> defaults to 0

    Returns:
        <pd.Series> containing semi-deviation per asset
    """
    return ret_df[ret_df < target].std()


def gaussian_VaR(ret_df):
    """
    Calculate the value at risk for the 95th percentile - i.e. the worst possible outcome in 95% of the cases

    Args:
        ret_df: <pd.DataFrame> containing return data.  Date index with asset names as columns.

    Returns:
        <pd.Series> containing the value at risk per asset
    """
    # assumes that the returns distribution is normal/gaussian
    # calculate the z-score with the percent point function/ inverse of CDF
    # given a probability, calc the x value for which the variable is less than or equal to x

    # for 5% (area to the left of z-score)
    z_score = norm.ppf(0.05)

    return ret_df.mean() + z_score * ret_df.std()


def maximum_drawdowns(df_price_data, window_days=const.NUM_TRADE_DAYS_PER_YR):
    """
    Compute rolling maximum drawdowns from daily price data

    Args:
        df_price_data: <pd.DataFrame> for an individual stock's price history
        window_days: <int> number of days for rolling calculation, defaults to number of annual trading days

    Returns:
        <pd.DataFrame> of rolling max drawdowns, <pd.Series> of the minimum of these i.e. the worst rolling max drawdown
        value

    """
    # maximum prior peaks in the last window_days number of days per day
    # allow first 252 days to have an expanding window
    max_peaks = df_price_data.rolling(window_days, min_periods=1).max()

    # will be negative if current date price is lower than the prev window peak i.e. this is a trough
    daily_drawdowns = (df_price_data - max_peaks) / max_peaks

    # of these daily values, get the rolling minimum which is the maximum drawdown in the previous window_dats
    max_drawdowns = daily_drawdowns.rolling(window_days, min_periods=1).min()

    return max_drawdowns, max_drawdowns.min()


if __name__ == '__main__':
    ls_assets = ['AAPL', 'NKE', 'GOOGL', 'AMZN']
    df = get_price_data(ls_assets, end_date=dt.datetime.today(), look_back_mths=24).pct_change().fillna(0)

    print(volatility(df))
    print(semi_deviation(df))
    print(gaussian_VaR(df))

    df_price_data = get_price_data(['AAPL'], end_date=dt.datetime.today(), look_back_mths=48)
    print(maximum_drawdowns(df_price_data=df_price_data))
