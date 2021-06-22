from capm import calc_beta, excess_return_daily, residual_return_risk
from common.common_functions import get_price_data
import datetime as dt
import common.constants as const

def information_ratio(expected_residual_return, residual_risk):
    """
    Calculates the IR given the expected residual return and residual risk

    Args:
        expected_residual_return: <float> expected residual return
        residual_risk: <float> residual risk

    Returns:
        <float> the information ratio

    """
    return expected_residual_return / residual_risk


if __name__ == '__main__':
    # Apple
    df_stock_rets = get_price_data(['AAPL'], end_date=dt.datetime.today(), look_back_mths=12).pct_change().fillna(0)
    # S&P 500
    df_market_rets = get_price_data(['^GSPC'], end_date=dt.datetime.today(), look_back_mths=12).pct_change().fillna(
        0)
    # iShares U.S. Technology ETF
    df_benchmark_rets = get_price_data(['IYW'], end_date=dt.datetime.today(), look_back_mths=12).pct_change().fillna(
        0)

    # compute beta over the market
    beta_stock = calc_beta(df_stock_rets, df_market_rets)
    beta_benchmark = calc_beta(df_benchmark_rets, df_market_rets)

    # compute beta of the stock over the benchmark
    beta_stock_over_benchmark = calc_beta(df_stock_rets, df_benchmark_rets)

    stock_excess_rets = excess_return_daily(df_stock_rets, df_market_rets, beta_stock)
    benchmark_excess_rets = excess_return_daily(df_benchmark_rets, df_market_rets, beta_benchmark)

    # residual returns of the stock over the benchmark
    res_dict = residual_return_risk(stock_excess_rets, benchmark_excess_rets, beta_stock_over_benchmark)

    print(information_ratio(expected_residual_return=res_dict[const.EXP_RESIDUAL_RETURN][0],
                            residual_risk=res_dict[const.RESIDUAL_RISK][0]))
