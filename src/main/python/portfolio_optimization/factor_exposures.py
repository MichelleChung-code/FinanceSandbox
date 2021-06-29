import pandas as pd
from urllib import request
import zipfile
import numpy as np
# todo modify the get_price_data common function to allow for start_date = False (all available history)
# from common.common_functions import get_price_data
import common.constants as const
from SimpleStockDataPlot import extract_data
import statsmodels.api as sm
import seaborn as sn
import matplotlib.pyplot as plt

MKT_EXCESS = 'Mkt-RF'  # excess return on the market (market minus risk free rate)
MKT_EXCESS_RENAME = 'MKT_EXCESS'
SIZE = 'SMB'  # size of firms (small minus big)
BK_TO_MKT = 'HML'  # book to market values (high minus low)
RISK_FREE_RATE = 'RF'


class FactorExposuresFamaFrench3Factor:
    def __init__(self, ticker):
        """
        Class to regress a stock's returns over the fama french 3 factors to obtain the stock's factor exposures

        Args:
            ticker: <str> stock ticker
        """
        self.ticker = ticker
        self.factor_data = self.scrape_fama_french_factor_data()

    def scrape_fama_french_factor_data(self):
        # Get 3 factor fama-french model data from Dartmouth
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
        fname = 'factor_info.zip'
        csv_fname = 'F-F_Research_Data_Factors.csv'

        request.urlretrieve(url=url, filename=fname)

        assert zipfile.is_zipfile(fname)

        # Open the zip folder and read in the CSV
        z_file = zipfile.ZipFile(fname, 'r')
        z_file.extractall()
        z_file.close()

        factor_data = pd.read_csv(csv_fname, skiprows=3, index_col=0)

        # remove row if any column has a null value
        factor_data = factor_data.loc[~factor_data.isnull().any(axis=1)]

        return factor_data

    def factor_data_pre_process(self):
        """ Clean up the data """
        # remove the annual data
        # only get data up to the end of monthly.  There is a blank index between monthly & annual data

        self.factor_data = self.factor_data[:self.factor_data.index.get_loc(np.NaN)]

        # values in decimal form
        self.factor_data = self.factor_data.apply(lambda x: pd.to_numeric(x) / 100)
        # datetime index and stated in month ends
        self.factor_data.index = pd.to_datetime(self.factor_data.index, format='%Y%m')
        self.factor_data.index = self.factor_data.index.to_period('M').to_timestamp('M')

    def get_stock_return_data(self):
        """ Get the return data of the self.ticker stock from Yahoo Finance """
        str_date_ls = self.factor_data.index.strftime(const.DATE_STR_FORMAT)
        end_date = str_date_ls[-1]

        df = extract_data(self.ticker, start_date=False, end_date=end_date)[const.ADJ_CLOSE].to_frame()
        df.fillna(method='ffill', inplace=True)

        df.rename(columns={const.ADJ_CLOSE: self.ticker}, inplace=True)

        # monthly data only and last calendar date per month & calculate returns
        df = df.resample('M').last().pct_change()[1:]

        # only keep factor_data for dates following the earliest ticker return data date available
        self.factor_data = self.factor_data[self.factor_data.index.get_loc(df.index[0]):]

        return df

    def __call__(self, *args, **kwargs):
        self.factor_data_pre_process()
        stock_data = self.get_stock_return_data()

        # merge and run the regression to get the exposures
        df = pd.merge(stock_data, self.factor_data, how='inner', left_index=True, right_index=True)

        # excess returns
        df[self.ticker] = df[self.ticker] - df[RISK_FREE_RATE]

        # to avoid issues with the dash symbol during the regression
        df.rename(columns={MKT_EXCESS: MKT_EXCESS_RENAME}, inplace=True)

        # run regression
        model = sm.formula.ols(formula=f'{self.ticker} ~ {MKT_EXCESS_RENAME} + {SIZE} + {BK_TO_MKT}', data=df).fit()

        # also get the factor covariance matrix
        fact_cov_df = pd.DataFrame.cov(self.factor_data[[MKT_EXCESS, SIZE, BK_TO_MKT]])
        sn.heatmap(fact_cov_df, annot=True, fmt='g')
        plt.show()

        # return the params, beta vals are the factor exposures
        return {'model_params': model.params,
                'fact_exposures': np.array(model.params)[1:],  # of order MKT_EXCESS_RENAME, SIZE, BK_TO_MKT factors
                'fact_cov': fact_cov_df}

    @staticmethod
    def factor_exposures_matrix(ls_assets):
        """
        Consolidate individual stock's factor exposures to the fama french 3 factors into a factor exposure matrix

        Args:
            ls_assets: <list> of assets to obtain the factor loadings for

        Returns:
            <np.ndarray> for the factor exposures matrix
        """
        # create a num_assets x num_factors matrix
        num_factors = 3
        fact_exp_mat = np.zeros((len(ls_assets), num_factors))

        for i, item in enumerate(ls_assets):
            exposures = FactorExposuresFamaFrench3Factor(ticker=item)()['fact_exposures']
            fact_exp_mat[i] = exposures

        return fact_exp_mat


if __name__ == '__main__':
    x = FactorExposuresFamaFrench3Factor(ticker="AAPL")
    print(x())

    res = FactorExposuresFamaFrench3Factor.factor_exposures_matrix(['AAPL', 'GOOG'])
    print(res)
