# For testing that all the main files run successfully
import unittest
from portfolio_optimization.portfolio_opt import MarkowitzOptimizePortfolio
import numpy as np
import cvxpy as cp


class PortfolioOptTest(unittest.TestCase):

    def test_results(self):
        """ Tests expected results from the optimization """

        np.random.seed(1)
        n = 10

        mu = np.abs(np.random.randn(n, 1))
        sigma = np.random.randn(n, n)
        sigma = sigma.T.dot(sigma)

        x = MarkowitzOptimizePortfolio(num_assets=n, mu=mu, sigma=sigma)
        results_dict = x()

        self.assertEqual(results_dict['status'], cp.OPTIMAL)
        self.assertTrue(np.allclose(results_dict['portfolio_ret'], [2.3015387]))
        self.assertTrue(np.allclose(results_dict['portfolio_variance'], 6.571138993238623))
        self.assertTrue(
            np.allclose(results_dict['w'], [-1.79067574e-28, -1.78626150e-28, -1.78590713e-28, -1.78826447e-28,
                                            -1.78737084e-28, 1.00000000e+00, -1.79119959e-28, -1.78690862e-28,
                                            -1.78499809e-28, -1.78468995e-28]))
