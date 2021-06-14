from common.common_functions import get_price_data
import datetime as dt


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


if __name__ == '__main__':
    ls_assets = ['AAPL', 'NKE', 'GOOGL', 'AMZN']
    df = get_price_data(ls_assets, end_date=dt.datetime.today(), look_back_mths=24).pct_change().fillna(0)

    print(volatility(df))
    print(semi_deviation(df))
