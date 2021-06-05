import numpy as np
import cvxpy as cp
import pprint


# Atomic functions documentation: https://www.cvxpy.org/tutorial/functions/index.html

class MarkowitzOptimizePortfolio:
    def __init__(self, num_assets, mu, sigma, gamma=0):
        """

        Args:
            num_assets: number of assets involved in the optimization
            mu: vector containing the mean returns of the assets
            sigma: covariance matrix
            gamma: risk adversion parameter - higher the number, the more risk adverse
        """

        self.n = num_assets
        self.mu = mu
        self.sigma = sigma

        assert gamma >= 0
        self.gamma = gamma

    def get_objective_function(self, portfolio_ret, portfolio_variance):
        return cp.Maximize(portfolio_ret - self.gamma * portfolio_variance)

    def get_contraints(self, w):
        sum_to_one = cp.sum(w) == 1
        long_only = w >= 0
        return [sum_to_one, long_only]

    def __call__(self):
        # set up the optimization variables
        # long only portfolio
        w = cp.Variable(self.n)  # weight vector to solve for
        # gamma = cp.Parameter(nonneg=True)
        portfolio_ret = self.mu.T @ w
        portfolio_variance = cp.quad_form(w, self.sigma)  # cp.quad form is the same as w.T @ self.sigma @ w

        obj_func = self.get_objective_function(portfolio_ret, portfolio_variance)
        constraints = self.get_contraints(w)

        problem = cp.Problem(obj_func, constraints)
        problem.solve()

        return {'w': w.value,
                'portfolio_ret': portfolio_ret.value,
                'portfolio_variance': portfolio_variance.value,
                'status': problem.status}


if __name__ == '__main__':
    np.random.seed(1)
    n = 10

    # todo use actual data from yahoo finance
    mu = np.abs(np.random.randn(n, 1))
    sigma = np.random.randn(n, n)
    sigma = sigma.T.dot(sigma)

    x = MarkowitzOptimizePortfolio(num_assets=n, mu=mu, sigma=sigma)
    results = x()

    pprint.pprint(results)
