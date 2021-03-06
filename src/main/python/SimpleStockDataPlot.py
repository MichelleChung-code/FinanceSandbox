import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
from common import constants as const
import yfinance as yf

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class PlotStockData:

    def __init__(self, ticker, start_date, end_date):
        self.stock_data = data.DataReader(ticker, start=start_date, end=end_date, data_source='yahoo')
        self.ticker = ticker

    def compute_log_return(self):
        self.stock_data[const.LOG_RET_DLY] = np.log(
            self.stock_data[const.ADJ_CLOSE_PRC] / self.stock_data[const.ADJ_CLOSE_PRC].shift(1))

    def compute_annual_vol(self):
        self.stock_data[const.VOLATILITY] = self.stock_data[const.LOG_RET_DLY].rolling(
            const.NUM_TRADE_DAYS_PER_YR).std() * np.log(
            const.NUM_TRADE_DAYS_PER_YR)

    def plot_results(self):
        plt.plot(self.stock_data[const.LOG_RET_DLY], label=const.LOG_RET_DLY)
        plt.plot(self.stock_data[const.VOLATILITY], label=const.VOLATILITY)
        plt.legend()
        plt.title('{} Log Returns and Annual Volatility Timeseries'.format(self.ticker))
        plt.xlabel('Date')

        plt.show()

    def __call__(self):
        self.compute_log_return()
        self.compute_annual_vol()
        self.plot_results()


def extract_data(ticker, start_date, end_date):
    """
    Extracts the stock data from yahoo given the ticker and desired timeframe

    Args:
        ticker: <str> stock ticker
        start_date: <str> YYYY-MM-DD start date for data, if False - get all available history
        end_date: <str> YYYY-MM-DD end date for data

    Returns: <pd.DataFrame> containing data for 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'

    """

    name = yf.Ticker(ticker)
    try:
        print('Data extracted for: {}'.format(name.info['longName']))
    except:
        print('No long name for {} available'.format(ticker))

    if start_date == False: # get all available history
        return data.DataReader(ticker, end=end_date, data_source='yahoo')

    return data.DataReader(ticker, start=start_date, end=end_date, data_source='yahoo')


def plot_bar_volume(ticker, start_date, end_date):
    data = extract_data(ticker, start_date, end_date)

    plt.bar(data.index, data[const.VOLUME], width=0.5)
    plt.xticks(rotation=30)
    plt.title('{vol} of {tick}'.format(vol=const.VOLUME, tick=ticker))
    plt.ylabel(const.VOLUME)
    plt.show()
