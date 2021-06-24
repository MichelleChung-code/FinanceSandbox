from capm import calc_beta, excess_return_daily, residual_return_risk
from common.common_functions import get_price_data
import datetime as dt
import common.constants as const
import numpy as np
from functools import reduce


def information_ratio_br_ic(BR, IC):
    """
    Calculate the information ratio from the breadth and information coefficient

    Args:
        BR: <int> breadth, # of independent forecasts per year
        IC: <float> information coefficient, correlation of each forecast with realized outcomes

    Returns:
        <float> the information ratio

    """
    return IC * np.sqrt(BR)


def value_added_br_ic(BR, IC, risk_adversion):
    """
    Calculates the manager's value added in terms of skill and breadth

    Args:
        BR: <int> breadth, # of independent forecasts per year
        IC: <float> information coefficient, correlation of each forecast with realized outcomes
        risk_adversion: <float> risk adversion parameter

    Returns:
        <float> value added
    """
    return (IC ** 2 * BR) / (4 * risk_adversion)


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


def IR_additivity(ic_br_ls: list):
    """
    Compute the combined information ratio given a list of individual skill and breath information

    IR^2 = (BR1*IC1^2) + (BR2*IC2^2) + etc.
    Information ratios are additive in their squares

    Args:
        ic_br_ls: <list> of tuples containing IC and BR information e.g. [(IC_1, BR_1), (IC_2, BR_2)]

    Returns:
        <float> the combined information ratio

    """
    ls_indiv_ir_squared = [x[0] ** 2 * x[1] for x in ic_br_ls]
    # IR is now the square root of the sum
    return np.sqrt(reduce(lambda x, y: x + y, ls_indiv_ir_squared))


def optimal_residual_risk(info_ratio, risk_adversion):
    """
    Get the optimal level of residual risk which maximizes the value added.

    Here:
    VA[ω_p] = ω_p * IR - λ_r * ω_p^2

    Therefore, the optimal ω_p to maximize VA[ω_p] would be:
    ω_p = IR / (2 * λ_r)

    Args:
        info_ratio: <float> information ratio
        risk_adversion: <float> risk adversion parameter

    Returns:
        <float> optimal level of residual risk

    """
    return info_ratio / (2 * risk_adversion)


if __name__ == '__main__':
    # # Apple
    # df_stock_rets = get_price_data(['AAPL'], end_date=dt.datetime.today(), look_back_mths=12).pct_change().fillna(0)
    # # S&P 500
    # df_market_rets = get_price_data(['^GSPC'], end_date=dt.datetime.today(), look_back_mths=12).pct_change().fillna(
    #     0)
    # # iShares U.S. Technology ETF
    # df_benchmark_rets = get_price_data(['IYW'], end_date=dt.datetime.today(), look_back_mths=12).pct_change().fillna(
    #     0)
    #
    # # compute beta over the market
    # beta_stock = calc_beta(df_stock_rets, df_market_rets)
    # beta_benchmark = calc_beta(df_benchmark_rets, df_market_rets)
    #
    # # compute beta of the stock over the benchmark
    # beta_stock_over_benchmark = calc_beta(df_stock_rets, df_benchmark_rets)
    #
    # stock_excess_rets = excess_return_daily(df_stock_rets, df_market_rets, beta_stock)
    # benchmark_excess_rets = excess_return_daily(df_benchmark_rets, df_market_rets, beta_benchmark)
    #
    # # residual returns of the stock over the benchmark
    # res_dict = residual_return_risk(stock_excess_rets, benchmark_excess_rets, beta_stock_over_benchmark)
    #
    # IR = information_ratio(expected_residual_return=res_dict[const.EXP_RESIDUAL_RETURN][0],
    #                        residual_risk=res_dict[const.RESIDUAL_RISK][0])
    #
    # print(optimal_residual_risk(info_ratio=IR, risk_adversion=0.15))

    # print(information_ratio_br_ic(BR=4, IC=0.25))

    # Each tuple is (IC, BR)
    print(IR_additivity([(0.02, 12), (0.04, 24), (0.075, 100)]))
