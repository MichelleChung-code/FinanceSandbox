# For testing that all the main files run successfully
import unittest
from portfolio_optimization.portfolio_opt import MarkowitzOptimizePortfolio
import numpy as np
import cvxpy as cp
import scipy.sparse as sp


class PortfolioOptTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

        self.n = 10

        self.mu = np.abs(np.random.randn(self.n, 1))
        self.sigma = np.random.randn(self.n, self.n)
        self.sigma = self.sigma.T.dot(self.sigma)

    def test_long_only(self):
        """ Tests expected results from the optimization """

        x = MarkowitzOptimizePortfolio(num_assets=self.n, mu=self.mu, sigma=self.sigma,
                                       constraints=['sum_to_one', 'long_only'])
        results_dict = x()

        self.assertEqual(results_dict['status'], cp.OPTIMAL)
        self.assertTrue(np.allclose(results_dict['portfolio_ret'], [2.3015387]))
        self.assertTrue(np.allclose(results_dict['portfolio_variance'], 6.571138993238623))
        self.assertTrue(
            np.allclose(results_dict['w'], [-1.79067574e-28, -1.78626150e-28, -1.78590713e-28, -1.78826447e-28,
                                            -1.78737084e-28, 1.00000000e+00, -1.79119959e-28, -1.78690862e-28,
                                            -1.78499809e-28, -1.78468995e-28]))

    def test_leverage_limit(self):
        """ Tests expected results from the optimization """

        x = MarkowitzOptimizePortfolio(num_assets=self.n, mu=self.mu, sigma=self.sigma,
                                       constraints=['sum_to_one', 'leverage_limit'])
        results_dict = x()

        self.assertEqual(results_dict['status'], cp.OPTIMAL)
        self.assertTrue(np.allclose(results_dict['portfolio_ret'], [2.30156091]))
        self.assertTrue(np.allclose(results_dict['portfolio_variance'], 6.571354165718808))
        self.assertTrue(
            np.allclose(results_dict['w'], [-1.71642509e-06, -1.71642509e-06, -1.71642509e-06, -1.71642509e-06,
                                            -1.71642509e-06, 1.00001545e+00, -1.71642509e-06, -1.71642509e-06,
                                            -1.71642509e-06, -1.71642509e-06]))

    def test_factor_covariance(self):
        n_assets = 3000
        m_factors = 50
        np.random.seed(1)
        mu = np.abs(np.random.randn(n_assets, 1))
        factor_covar_matrix = np.random.randn(m_factors, m_factors)
        factor_covar_matrix = factor_covar_matrix.T.dot(factor_covar_matrix)
        D = sp.diags(np.random.uniform(0, 0.9, size=n_assets))  # idiosyncratic risk
        factor_loading_matrix = np.random.randn(n_assets, m_factors)

        gamma = 0.1
        lev_lim = 2

        x = MarkowitzOptimizePortfolio(num_assets=n_assets, num_factors=m_factors, mu=mu, sigma=factor_covar_matrix,
                                       constraints=['sum_to_one', 'leverage_limit'], D=D, F=factor_loading_matrix,
                                       factor_covariance=True, gamma=gamma, lev_limit=lev_lim)
        results_dict = x()

        self.assertTrue(np.allclose(results_dict['factor_exposures'],
                                    [0.00491028, -0.09214816, -0.02936822, -0.05284681, -0.06428898,
                                     -0.11684546, 0.23464628, -0.07258659, -0.12408075, -0.09640972,
                                     -0.01826205, -0.11523898, -0.13453988, -0.02224825, -0.14078008,
                                     0.08384735, -0.04538662, 0.06547044, -0.03378125, -0.06824131,
                                     0.09654797, 0.16621212, 0.20579988, -0.00536281, 0.03167332,
                                     -0.1230295, 0.01012323, -0.04549572, -0.11745823, -0.05834112,
                                     -0.10583129, -0.12166005, -0.02708789, -0.04602297, 0.04033764,
                                     0.03234207, 0.07263502, -0.12036399, 0.01905779, 0.08672913,
                                     0.04191737, -0.17216785, -0.04667414, 0.14450931, 0.13509516,
                                     0.14385637, 0.03316754, 0.07468672, 0.10282204, -0.00737077]))
