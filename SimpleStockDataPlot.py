import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

LOG_RET_DLY = 'log_ret_dly'
ADJ_CLOSE_PRC = 'Adj Close'
VOLATILITY = 'volatility'

num_trading_days = 252


class PlotStockData:

    def __init__(self, ticker, start_date, end_date):
        self.stock_data = data.DataReader(ticker, start=start_date, end=end_date, data_source='yahoo')
        self.ticker = ticker

    def compute_log_return(self):
        self.stock_data[LOG_RET_DLY] = np.log(self.stock_data[ADJ_CLOSE_PRC] / self.stock_data[ADJ_CLOSE_PRC].shift(1))

    def compute_annual_vol(self):
        self.stock_data[VOLATILITY] = self.stock_data[LOG_RET_DLY].rolling(num_trading_days).std() * np.log(
            num_trading_days)

    def plot_results(self):
        plt.plot(self.stock_data[LOG_RET_DLY], label=LOG_RET_DLY)
        plt.plot(self.stock_data[VOLATILITY], label=VOLATILITY)
        plt.legend()
        plt.title('{} Log Returns and Annual Volatility Timeseries'.format(self.ticker))
        plt.xlabel('Date')

        plt.show()

    def __call__(self):
        self.compute_log_return()
        self.compute_annual_vol()
        self.plot_results()


def extract_data(ticker, start_date, end_date):
    return data.DataReader(ticker, start=start_date, end=end_date, data_source='yahoo')


if __name__ == '__main__':
    ticker = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2020-12-31'

    stock_data = PlotStockData(ticker, start_date, end_date)
    stock_data()

    df = extract_data(ticker, start_date, end_date)
    print(df.head())
