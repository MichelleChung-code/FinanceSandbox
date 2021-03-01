from src.main.python.SimpleStockDataPlot import extract_data
import src.main.python.common.constants as consts
from dateutil.relativedelta import relativedelta
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import os
from pathlib import Path


def plot_corr_mat(ls_tickers, start_date, end_date, res_path):
    """
    Plot heatmap showing pearson's correlation matrix between inputted stocks

    Args:
        ls_tickers: <list> of tickers to produce correlation matrix
        start_date: <str> YYYY-MM-DD start date for data
        end_date: <str> YYYY-MM-DD end date for data
        res_path: <str> output path to save plot to

    """
    df_res_ls = []

    # pull data
    for ticker in ls_tickers:
        df = extract_data(ticker, start_date, end_date)
        df[consts.TICKER] = ticker
        df_res_ls.append(df)

    df = pd.concat(df_res_ls)
    df.reset_index(inplace=True)

    # pivot to reformat data to have date as index, tickers as the columns, and adjusted close as the values
    ls_pivot = [consts.DATE, consts.TICKER, consts.ADJ_CLOSE]
    df = df[ls_pivot].pivot(*ls_pivot)

    # calculate pearson correlation
    df = df.corr(method='pearson')

    # plotting
    fig = plt.figure()
    seaborn.heatmap(df, cmap='RdYlGn', annot=True)
    plt.savefig(os.path.join(res_path, 'corr_mat_{}_{}'.format(start_date, end_date)))

    # .show() must go after .savefig() or else saved figure will be blank
    plt.show()


if __name__ == '__main__':
    ls_tickers = ['AAL', 'AA', 'AAPL', 'BBY', 'C', 'CVS', 'HD', 'IBM']

    # Get 10 years worth of historical data
    start_date = (dt.datetime.today() - relativedelta(years=10)).strftime('%Y-%m-%d')
    end_date = dt.datetime.today().strftime('%Y-%m-%d')
    res_path = os.path.join(str(Path(__file__).parents[3]), 'results')
    plot_corr_mat(ls_tickers, start_date, end_date, res_path)
