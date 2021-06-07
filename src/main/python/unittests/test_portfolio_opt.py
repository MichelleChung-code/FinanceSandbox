# For testing that all the main files run successfully
import unittest
from portfolio_optimization.portfolio_opt import MarkowitzOptimizePortfolio
import numpy as np
import cvxpy as cp


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
