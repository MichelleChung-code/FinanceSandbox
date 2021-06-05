from portfolio_optimization.portfolio_opt import MarkowitzOptimizePortfolio
import pprint
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import math

VAR_DATA = 'variance_data'
RET_DATA = 'return_data'
OPT_STATUS_DATA = 'opt_status_data'
GAMMA_DATA = 'gamma_data'


class RiskCurve(MarkowitzOptimizePortfolio):

    def __init__(self, num_samples=10000, *args, **kwargs):
        """
        Args:
            num_samples: <int> number of times to run the optimization problem with varying gamma values
        """
        super().__init__(*args, **kwargs)

        # overwrite gamma
        self.gamma = cp.Parameter(nonneg=True)
        self.num_samples = num_samples

    def compute_risk_curve_values(self, problem, portfolio_ret, portfolio_variance):
        """
        Loop through varying gamma values to produce optimization results

        Args:
            problem: <cvxpy Problem> with objective and constraints already set up
            portfolio_ret: <cvxpy Expression> computing portfolio returns
            portfolio_variance: <cvxpy Expression> computing portfolio variance

        Returns:
            <dict> of portfolio optimization results:
                <list> calculated variances
                <list> calculated returns
                <list> cvxpy optimization statuses per run
                <list> of gamma values (produced using base 10 logspace (base**start, base**stop) between
                start: -2 and stop: 2
        """

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

        return {VAR_DATA: variance_data_ls,
                RET_DATA: return_data_ls,
                OPT_STATUS_DATA: opt_status_ls,
                GAMMA_DATA: gamma}

    def plot_risk_curve(self, optimization_results):
        """
        Plots the risk curve

        Args:
            optimization_results: <dict> from the output of the compute_risk_curve_values() function
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(optimization_results[VAR_DATA], optimization_results[RET_DATA])

        # show a couple gamma values
        marker_pos = math.floor(self.num_samples / 3)
        marker_pos_ls = [marker_pos, self.num_samples - marker_pos]
        for marker_position in marker_pos_ls:
            plt.plot(optimization_results[VAR_DATA][marker_position], optimization_results[RET_DATA][marker_position],
                     'bs')
            ax.annotate('gamma = {:.2f}'.format(optimization_results[GAMMA_DATA][marker_position]), xy=(
                optimization_results[VAR_DATA][marker_position], optimization_results[RET_DATA][marker_position]))

        plt.xlabel('Portfolio Variance')
        plt.ylabel('Portfolio Return')
        plt.show()

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

        self.plot_risk_curve(res_dict)
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
