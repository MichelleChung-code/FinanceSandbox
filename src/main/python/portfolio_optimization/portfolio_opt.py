import numpy as np
import cvxpy as cp
import pprint


# Atomic functions documentation: https://www.cvxpy.org/tutorial/functions/index.html

class MarkowitzOptimizePortfolio:
    def __init__(self, num_assets, mu, sigma, gamma=0):
        """
        Run a classical markowitz portfolio optimization

        Args:
            num_assets: number of assets involved in the optimization
            mu: vector containing the mean returns of the assets
            sigma: covariance matrix
            gamma: risk adversion parameter - higher the number, the more risk adverse
        """

        self.n = num_assets
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma

        # quick checks on dimensions
        assert self.gamma >= 0
        assert len(self.mu) == self.n
        assert self.sigma.shape == (self.n, self.n)

    def get_objective_function(self, portfolio_ret, portfolio_variance):
        """
        Get the cvxpy objective function

        Args:
            portfolio_ret: <cvxpy expression> for the portfolio returns
            portfolio_variance: <cvxpy expression> for the portfolio variance

        Returns:
            <cvxpy problem objective> objective function
        """
        return cp.Maximize(portfolio_ret - self.gamma * portfolio_variance)

    def get_contraints(self, w):
        """
        Generate and get the contraints

        Args:
            w: <cvxpy variable> weight vector variable

        Returns:
            <list> of contraints
        """
        sum_to_one = cp.sum(w) == 1
        long_only = w >= 0
        return [sum_to_one, long_only]

    def __call__(self):
        # set up the optimization variables
        # long only portfolio
        w = cp.Variable(self.n)  # weight vector to solve for
        portfolio_ret = self.mu.T @ w
        portfolio_variance = cp.quad_form(w, self.sigma)  # cp.quad form is the same as w.T @ self.sigma @ w

        # set up and solve the optimization problem
        obj_func = self.get_objective_function(portfolio_ret, portfolio_variance)
        constraints = self.get_contraints(w)

        problem = cp.Problem(obj_func, constraints)
        problem.solve()

        return {'w': w.value,
                'portfolio_ret': portfolio_ret.value,
                'portfolio_variance': portfolio_variance.value,
                'status': problem.status}
