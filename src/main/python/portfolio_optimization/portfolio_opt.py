import numpy as np
import cvxpy as cp
import pprint
import scipy.sparse as sp
from portfolio_optimization.portfolio_opt_preprocess import PortfolioOptPreprocess


# Atomic functions documentation: https://www.cvxpy.org/tutorial/functions/index.html

class MarkowitzOptimizePortfolio:
    def __init__(self, num_assets, mu, sigma, constraints: list, gamma=0, lev_limit=1, factor_covariance=False,
                 **kwargs):
        """
        Run a classical markowitz portfolio optimization

        Args:
            num_assets: number of assets involved in the optimization
            mu: vector containing the mean returns of the assets
            sigma: covariance matrix
            gamma: risk adversion parameter - higher the number, the more risk adverse
            constraints: <list> of string elements for constraints to apply
            lev_limit: <int> or <float> to represent how much to allow to leverage
            factor_covariance: <bool> if True, run as factor covariance model.  Additionally params required.

            For factor_covariance == True:
                D: <scipy matrix> diagonal matrix  for idiosyncratic risk
                F: <np.ndarray> of factor loadings of shape: num_assets x num_factors
                num_factors: <int> number of factors
        """

        self.n = num_assets
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.constrs_ls = constraints
        self.lev_limit = lev_limit
        self.factor_covar_model_bool = factor_covariance

        # for factor covariance model
        if self.factor_covar_model_bool:
            self.D = kwargs.get('D', False)
            self.factor_loadings = kwargs.get('F', False)
            self.m = kwargs.get('num_factors', False)
            if any(isinstance(x, bool) for x in [self.D, self.factor_loadings, self.m]):
                raise Exception('Factor Covariance Model selected, but required input parameters are missing')

        # quick checks on dimensions
        assert self.gamma >= 0
        assert len(self.mu) == self.n

        if self.factor_covar_model_bool:
            assert self.sigma.shape == (self.m, self.m)
        else:
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

    def get_contraints(self, w, constr_ls, f=False, F=False, lev_lim=1):
        """
        Generate and get the contraints

        Args:
            w: <cvxpy variable> weight vector variable
            constr_ls: <list> of constraints to apply
            lev_lim: <int> or <float> max leverage allowed
            f: <cp.Variable> for factor exposures vector.  If False, model is not factor covariance one
            F: <np.ndarray> of factor loadings per asset.  Shape: num_assets x num_factors

        Returns:
            <list> of contraints
        """
        assert lev_lim >= 1  # norm constraint
        assert not set(['long_only', 'leverage_limit']).issubset(
            constr_ls)  # cannot have both long only and leverage constraint

        dict_constraints = {'sum_to_one': cp.sum(w) == 1,
                            'long_only': w >= 0,
                            'leverage_limit': cp.norm(w, 1) <= lev_lim}

        if not set(constr_ls).issubset(dict_constraints.keys()):
            missing_items = set(constr_ls) - set(dict_constraints.keys())
            raise NotImplementedError('{} constraints have not been implemented'.format(missing_items))

        model_constraints = [c for k, c in dict_constraints.items() if k in constr_ls]

        if self.factor_covar_model_bool:
            if any([isinstance(x, bool) for x in [f, F]]):
                raise Exception(
                    'Factor Exposures CVXPY variable and factor loadings must be provided for factor covariance model')
            model_constraints = model_constraints + [f == F.T @ w]

        return model_constraints

    def __call__(self):
        # set up the optimization variables
        # long only portfolio
        w = cp.Variable(self.n)  # weight vector to solve for

        if self.factor_covar_model_bool:
            f = cp.Variable(self.m)  # factor exposures
            portfolio_variance = cp.quad_form(f, self.sigma) + cp.quad_form(w, self.D)
        else:
            portfolio_variance = cp.quad_form(w, self.sigma)  # cp.quad form is the same as w.T @ self.sigma @ w

        portfolio_ret = self.mu.T @ w

        # set up and solve the optimization problem
        obj_func = self.get_objective_function(portfolio_ret, portfolio_variance)

        if self.factor_covar_model_bool:
            constraints = self.get_contraints(w, lev_lim=self.lev_limit, constr_ls=self.constrs_ls, f=f,
                                              F=self.factor_loadings)
        else:
            constraints = self.get_contraints(w, lev_lim=self.lev_limit, constr_ls=self.constrs_ls)

        problem = cp.Problem(obj_func, constraints)
        problem.solve(verbose=True)

        res_dict = {'w': w.value,
                    'portfolio_ret': portfolio_ret.value,
                    'portfolio_variance': portfolio_variance.value,
                    'status': problem.status}

        if self.factor_covar_model_bool:
            res_dict.update({'factor_exposures': f.value})

        return res_dict


class MinRiskOptimizePortfolio:
    def __init__(self, num_assets: int, sigma, mu, min_ret=0.1):
        self.n = num_assets
        self.sigma = sigma
        self.mu = mu
        self.min_ret = min_ret

    def run_opt(self):
        """ Run optimization for the minimum risk portfolio that exceeds a required minimum return """
        w = cp.Variable(self.n)
        rets = self.mu.T @ w
        risk = cp.quad_form(w, self.sigma)

        # cp.norm(w, 1) <= 2, sum of |x_i| must be <= 2
        # requires return greater than min_ret
        # fully invested portfolio
        constraints_ls = [cp.sum(w) == 1, rets >= self.min_ret, cp.norm(w, 1) <= 2]

        prob = cp.Problem(cp.Minimize(risk), constraints_ls)
        prob.solve()
        return w.value


class WorstCaseRiskPortfolio:
    def __init__(self, num_assets: int, sigma, w):
        self.n = num_assets
        self.sigma = sigma
        self.w = w

    def run_opt(self):
        """ Run optimization for the worst case risk over possible covariance matrices """
        sigma_opt = cp.Variable((self.n, self.n), PSD=True)  # positive semi-definite
        delta = cp.Variable((self.n, self.n), symmetric=True)  # difference between the input covar and the testing ones
        risk = cp.quad_form(self.w, sigma_opt)

        # elementwise delta constrained and must be zero on diagonals
        constraints_ls = [sigma_opt == self.sigma + delta, cp.diag(delta) == 0, cp.abs(delta) <= 0.2]

        prob = cp.Problem(cp.Maximize(risk), constraints_ls)
        prob.solve()
        return {'actual_std': cp.sqrt(cp.quad_form(self.w, self.sigma)).value,
                'worst_case_std': cp.sqrt(risk).value,
                'delta': delta.value}


if __name__ == '__main__':
    ls_assets = ['AAPL', 'NKE', 'GOOGL', 'AMZN']

    preprocess = PortfolioOptPreprocess(ls_assets, rets_hist_length_yrs=3)
    preprocess_res = preprocess()
    x = MarkowitzOptimizePortfolio(num_assets=preprocess.n, mu=preprocess_res['expected_returns'],
                                   sigma=preprocess_res['covariance_matrix'],
                                   constraints=['sum_to_one', 'long_only'])
    results_dict = x()

    print(results_dict)

    y = MinRiskOptimizePortfolio(num_assets=preprocess.n, mu=preprocess_res['expected_returns'],
                                 sigma=preprocess_res['covariance_matrix'])

    w = y.run_opt()

    z = WorstCaseRiskPortfolio(num_assets=preprocess.n, sigma=preprocess_res['covariance_matrix'], w=w)
    print(z.run_opt())
