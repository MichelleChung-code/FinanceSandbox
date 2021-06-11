from SimpleStockDataPlot import extract_data
import common.constants as const
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from common.timeit import timeit
import numpy as np
import pprint


class PortfolioOptPreprocess:
    """ Class to get the returns and covariance matrix needed for the portfolio optimization """

    def __init__(self, asset_name_ls, rets_hist_length_yrs=10,
                 end_date=dt.datetime.today()):
        start_date = (end_date - relativedelta(years=rets_hist_length_yrs))
        self.n = len(asset_name_ls)  # number of assets
        self.df_price_data = pd.DataFrame(
            index=pd.date_range(start=start_date.strftime(const.DATE_STR_FORMAT),
                                end=end_date.strftime(const.DATE_STR_FORMAT), freq='B'),
            columns=asset_name_ls)

        for asset in asset_name_ls:
            self.df_price_data[asset] = extract_data(asset, start_date.strftime(const.DATE_STR_FORMAT),
                                                     end_date.strftime(const.DATE_STR_FORMAT))[const.ADJ_CLOSE]

        self.df_price_data.fillna(method='ffill', inplace=True)

    def expected_returns(self):
        return self.df_price_data.pct_change().mean()

    def covar_matrix(self):
        # just want to see the volatility
        pprint.pprint({'annual_std':
            self.df_price_data.pct_change().apply(lambda x: np.log(1 + x)).std().apply(
                lambda x: x * np.sqrt(250))})

        return self.df_price_data.pct_change().apply(lambda x: np.log(1 + x)).cov()

    @timeit
    def __call__(self, *args, **kwargs):
        exp_ret = self.expected_returns()
        cov_matrix = self.covar_matrix()

        print(exp_ret)
        print(cov_matrix)

        return {'expected_returns': exp_ret.to_numpy().reshape(self.n, 1),
                'covariance_matrix': cov_matrix.to_numpy()}


if __name__ == '__main__':
    ls_assets = ['AAPL', 'NKE', 'GOOGL', 'AMZN']
    x = PortfolioOptPreprocess(ls_assets, rets_hist_length_yrs=3)
    res = x()

    print(res)
