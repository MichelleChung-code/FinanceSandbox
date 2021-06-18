from common.common_functions import get_price_data
import datetime as dt


def beta(df_stock_rets, df_benchmark):
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


if __name__ == '__main__':
    ls_asset = ['AAPL']
    df_stock_rets = get_price_data(ls_asset, end_date=dt.datetime.today(), look_back_mths=12).pct_change().fillna(0)

    df_benchmark_rets = get_price_data(['^GSPC'], end_date=dt.datetime.today(), look_back_mths=12).pct_change().fillna(
        0)

    print(beta(df_stock_rets, df_benchmark_rets))
