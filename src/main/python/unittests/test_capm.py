import unittest
from common.common_functions import get_price_data
import datetime as dt
import capm
from sklearn.linear_model import LinearRegression
import numpy as np
import cvxpy as cp


class CAPMFunctionsTest(unittest.TestCase):

    def setUp(self):
        ls_asset = ['AAPL']
        self.df_stock_rets = get_price_data(ls_asset, end_date=dt.datetime.today(),
                                            look_back_mths=12).pct_change().fillna(0)

        self.df_benchmark_rets = get_price_data(['^GSPC'], end_date=dt.datetime.today(),
                                                look_back_mths=12).pct_change().fillna(0)

    def test_beta(self):
        """ Check that calculating beta via linear regression and covar/var formula yields the same value """
        beta_calc = capm.calc_beta(self.df_stock_rets, self.df_benchmark_rets)

        # compare to calculating beta through a linear regression
        # r_i = alpha + beta * r_m
        # r_i is the stock's return, alpha is the intercept, beta is the stock's beta, r_m is market or benchmark return
        x = np.array(self.df_benchmark_rets)
        y = np.array(self.df_stock_rets)

        lin_model = LinearRegression().fit(x, y)

        # For fun, also do the regression with cvxpy
        beta = cp.Variable()
        intercept = cp.Variable()

        # objective function is to minimize least squares
        cost = cp.sum_squares(x * beta - y + intercept)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve()

        print({'optimal_val': prob.value,
               'optimal_beta': beta.value,
               'norm_of_residual': cp.norm(cost, p=2).value})

        # check that all three ways produce the same result!  Yay! 
        self.assertTrue(np.allclose(beta_calc, lin_model.coef_, beta.value))
