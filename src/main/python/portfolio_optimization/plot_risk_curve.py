from portfolio_optimization.portfolio_opt import MarkowitzOptimizePortfolio
import pprint
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


class RiskCurve(MarkowitzOptimizePortfolio):

    def __init__(self, num_samples=10000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # overwrite gamma
        self.gamma = cp.Parameter(nonneg=True)
        self.num_samples = num_samples

    def compute_risk_curve_values(self, problem, portfolio_ret, portfolio_variance):

        # initialize the sizes of result arrays
        variance_data_ls = np.zeros(self.num_samples)
        return_data_ls = np.zeros(self.num_samples)
        opt_status_ls = [False] * self.num_samples
        gamma = np.logspace(-2, 2, base=10, num=self.num_samples)

        for i in range(self.num_samples):
            # Update the gamma value and then re-solve
            self.gamma.value = gamma[i]
            problem.solve()
            variance_data_ls[i] = portfolio_variance.value
            return_data_ls[i] = portfolio_ret.value[0]
            opt_status_ls[i] = problem.status

        if all(x for x in opt_status_ls if x == cp.OPTIMAL):
            print('All problems solved as {}'.format(cp.OPTIMAL))

        return {'variance_data': variance_data_ls,
                'return_data': return_data_ls,
                'opt_status_data': opt_status_ls}

    def __call__(self, *args, **kwargs):
        # set up the optimization variables
        # long only portfolio
        w = cp.Variable(self.n)  # weight vector to solve for
        portfolio_ret = self.mu.T @ w
        portfolio_variance = cp.quad_form(w, self.sigma)  # cp.quad form is the same as w.T @ self.sigma @ w

        # set up and solve the optimization problem
        obj_func = self.get_objective_function(portfolio_ret, portfolio_variance)
        constraints = self.get_contraints(w)

        problem = cp.Problem(obj_func, constraints)
        res_dict = self.compute_risk_curve_values(problem, portfolio_ret, portfolio_variance)

        return res_dict


if __name__ == '__main__':
    np.random.seed(1)
    n = 10

    # todo use actual data from yahoo finance
    mu = np.abs(np.random.randn(n, 1))
    sigma = np.random.randn(n, n)
    sigma = sigma.T.dot(sigma)

    x = RiskCurve(num_assets=n, mu=mu, sigma=sigma)
    results = x()

    pprint.pprint(results)
