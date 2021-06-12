from common.common_functions import get_price_data
import datetime as dt
import numpy as np


def risk_adversion_coefficient(benchmark_ticker, risk_free_rate=0.02, lookback_period_months=48):
    """
    Calculate the risk adversion coefficient

    Args:
        benchmark_ticker: <str> ticker of the benchmark market asset
        risk_free_rate: <float> the annual risk free rate
        lookback_period_months: <int> number of months from today

    Returns:
        <float> risk adversion coefficient

    """
    end_date = dt.datetime.today()
    df_price_data = get_price_data([benchmark_ticker], end_date, look_back_mths=lookback_period_months)

    # resample to annual data
    rets = df_price_data.resample('Y').last().pct_change().apply(lambda x: np.log(1 + x))

    exp_ret = rets.mean()
    var = rets.var()
    return ((exp_ret - risk_free_rate) / var)[0]


if __name__ == '__main__':
    gamma = risk_adversion_coefficient('^GSPC')
    print(gamma)
