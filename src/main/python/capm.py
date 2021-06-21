from common.common_functions import get_price_data
import datetime as dt


# Dealing all with daily data

def calc_beta(df_stock_rets, df_benchmark):
    """
    Calculate beta

    Args:
        df_stock_rets: <pd.DataFrame> of the returns of the stock to evaluate
        df_benchmark: <pd.DataFrame> of the benchmark returns

    Returns:
        <float> beta value
    """
    # combine the dataframes on the date index
    df = df_stock_rets.merge(df_benchmark, how='inner', left_index=True, right_index=True)

    cov_matrix = df.cov()
    covar = cov_matrix.iloc[0, 1]
    bench_var = df_benchmark.var()[0]

    # beta is covar(a,b)/ var(b); where a is the individual stock and b is the benchmark
    return covar / bench_var


def excess_return_daily(df_rets, df_market_rets, beta, rf_rate_annual=0.02):
    """
    Compute an investment's excess return over the market or another benchmark calculated under CAPM

    Args:
        df_rets: <pd.DataFrame> of the returns of the stock to evaluate
        df_market_rets: <pd.DataFrame> of the market returns or could also be another benchmark
        beta: <float> the investment's beta value against df_market_rets
        rf_rate_annual: <float> annual risk free rate

    Returns:
        <pd.DataFrame> of daily excess returns
    """
    # convert annual risk free rate to daily with the compounding interest formula
    rf_rate_daily = ((1 + rf_rate_annual) ** (1 / 365)) - 1

    # set up column name for output dataframe
    df_rets.columns, df_market_rets.columns = ['excess_return'], ['excess_return']

    # Excess return = RF + β(MR – RF) – TR
    return rf_rate_daily + beta * (df_market_rets - rf_rate_daily) - df_rets


def residual_return_risk(stock_excess_rets, benchmark_excess_rets, beta_stock_over_benchmark):
    """
    Calculate the residual return of an investment over a benchmark.  This is the return independent of a benchmark.

    Args:
        stock_excess_rets: <pd.DataFrame> of the investment's excess returns
        benchmark_excess_rets: <pd.DataFrame> of the benchmark's excess returns
        beta_stock_over_benchmark: <float> beta value of the stock relative to this benchmark

    Returns:
        <dict> of the residual return series, residual risk, and expected residual return of types:
        <pd.DataFrame>, <float>, and <float>; respectively
    """
    assert benchmark_excess_rets.columns == ['excess_return'] and stock_excess_rets.columns == ['excess_return']

    # Residual return = Excess return - (Benchmark's excess return * beta).
    df = stock_excess_rets - (benchmark_excess_rets * beta_stock_over_benchmark)
    df.columns = ['residual_return']
    return {'residual_return': df,
            'residual_risk': df.std(),
            'expected_residual_return': df.mean()}


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
    print(residual_return_risk(stock_excess_rets, benchmark_excess_rets, beta_stock_over_benchmark))
