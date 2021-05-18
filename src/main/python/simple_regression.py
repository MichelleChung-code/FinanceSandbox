import src.main.python.common.constants as const
from src.main.python.common.common_functions import log_return, rolling_correlation
from src.main.python.SimpleStockDataPlot import extract_data
import pandas as pd
import datetime as dt
from urllib.request import urlretrieve
import os
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np


class SimpleRegression:
    def __init__(self, ticker1, start_date, end_date=dt.datetime.today().strftime('%Y-%m-%d')):
        """
        Args:
            ticker1: <str> ticker of stock to run regression against VSTOXX on
            start_date: <str> YYYY-MM-DD format
            end_date: <str> YYYY-MM-DD format, defaults to today
        """
        self.ticker1_data = extract_data(ticker1, start_date, end_date)

        vstoxx_url = r'http://www.stoxx.com/download/historical_values/h_vstoxx.txt'

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        vstoxx_file = os.path.join(cur_dir, 'data/vstoxx.txt')
        urlretrieve(vstoxx_url, vstoxx_file)
        self.ticker2_data = pd.read_csv(vstoxx_file, index_col=0, header=2, parse_dates=True, sep=',',
                                        dayfirst=True)

        self.ticker2_data = self.ticker2_data['V2TX'].to_frame()

        self.ticker1_nme = ticker1
        self.ticker2_nme = 'V2TX'

    def run_reg(self, df):
        """
        Runs the linear regression and generates plots

        Args:
            df: <pd.DataFrame> containing the cleaned data to directly regress
        """
        Y = df[self.ticker2_nme]
        X = df[self.ticker1_nme]
        X = sm.add_constant(X)
        reg_mod = OLS(endog=Y, exog=X, missing='drop')
        res = reg_mod.fit()
        intercept, m = res.params

        print(m, intercept)

        # plot the data
        plt.plot(df[self.ticker1_nme], df[self.ticker2_nme], 'r.')

        # plot the linear line
        ax = plt.axis()
        x = np.linspace(ax[0], ax[1])
        y = m * x + intercept

        plt.plot(x, y, 'b')
        plt.grid(True)
        plt.xlabel('{} Returns'.format(self.ticker1_nme))
        plt.ylabel('{} Returns'.format(self.ticker2_nme))

        plt.show()

        # correlation for sanity check
        print(df.corr())

        # rolling correlation
        df_roll_corr = rolling_correlation(df, self.ticker1_nme, self.ticker2_nme)
        df_roll_corr.plot(grid=True)
        plt.show()

    def __call__(self, *args, **kwargs):
        df = self.prep_data()
        self.run_reg(df)

    def prep_data(self):
        """
        Cleans and prepares data to regress. Truncate dates to available data, process missing middle data, calculate
        log returns.

        Returns:
            df: <pd.DataFrame> containing the cleaned data.
        """
        # truncate both to the same time frame for min and max available data
        min_date = max(self.ticker1_data.index.min(), self.ticker2_data.index.min())
        max_date = min(self.ticker1_data.index.max(), self.ticker2_data.index.max())
        self.ticker1_data = self.ticker1_data.loc[
            (self.ticker1_data.index >= min_date) & (self.ticker1_data.index <= max_date)]
        self.ticker2_data = self.ticker2_data.loc[
            (self.ticker2_data.index >= min_date) & (self.ticker2_data.index <= max_date)]

        # forward fill any missing data
        self.ticker1_data.fillna(method='ffill', inplace=True)
        self.ticker2_data.fillna(method='ffill', inplace=True)

        # log returns
        self.ticker1_data, ticker1_log_ret_col = log_return(self.ticker1_data, const.ADJ_CLOSE)
        self.ticker2_data, ticker2_log_ret_col = log_return(self.ticker2_data, self.ticker2_nme)
        self.ticker2_data.drop(self.ticker2_nme, axis=1, inplace=True)

        # make one dataframe that is only the log return columns
        df = pd.concat(
            [self.ticker1_data[ticker1_log_ret_col].to_frame().rename(columns={ticker1_log_ret_col: self.ticker1_nme}),
             self.ticker2_data[ticker2_log_ret_col].to_frame().rename(columns={ticker2_log_ret_col: self.ticker2_nme})],
            axis=1)

        return df



