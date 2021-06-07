import numpy as np
import cvxpy as cp
import pprint


# Atomic functions documentation: https://www.cvxpy.org/tutorial/functions/index.html

class MarkowitzOptimizePortfolio:
    def __init__(self, num_assets, mu, sigma, constraints: list, gamma=0, lev_limit=1):
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
        self.constrs_ls = constraints
        self.lev_limit = lev_limit

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

    def get_contraints(self, w, constr_ls, lev_lim=1):
        """
        Generate and get the contraints

        Args:
            w: <cvxpy variable> weight vector variable
            constr_ls: <list> of constraints to apply
            lev_lim: <int> or <float> max leverage allowed

        Returns:
            <list> of contraints
        """
        assert lev_lim >= 1  # norm constraint
        assert not set(['long_only', 'leverage_limit']).issubset(
            constr_ls)  # cannot have both long only and leverage constraint

        dict_constraints = {'sum_to_one': cp.sum(w) == 1,
                            'long_only': w >= 0,
                            'leverage_limit': cp.norm(w, 1) == lev_lim}

        if not set(constr_ls).issubset(dict_constraints.keys()):
            missing_items = set(constr_ls) - set(dict_constraints.keys())
            raise NotImplementedError('{} constraints have not been implemented'.format(missing_items))

        return [c for k, c in dict_constraints.items() if k in constr_ls]

    def __call__(self):
        # set up the optimization variables
        # long only portfolio
        w = cp.Variable(self.n)  # weight vector to solve for
        portfolio_ret = self.mu.T @ w
        portfolio_variance = cp.quad_form(w, self.sigma)  # cp.quad form is the same as w.T @ self.sigma @ w

        # set up and solve the optimization problem
        obj_func = self.get_objective_function(portfolio_ret, portfolio_variance)
        constraints = self.get_contraints(w, lev_lim=self.lev_limit, constr_ls=self.constrs_ls)

        problem = cp.Problem(obj_func, constraints)
        problem.solve()

        return {'w': w.value,
                'portfolio_ret': portfolio_ret.value,
                'portfolio_variance': portfolio_variance.value,
                'status': problem.status}
