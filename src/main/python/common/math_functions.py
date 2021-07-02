import numpy as np

def square_root_diffusion(time_steps, iter_num, t, x0, kappa, theta, sigma):
    """
    Return discretization scheme for general square-root diffusion

    Args:
        time_steps: <int> number of time steps
        iter_num: <int> number of iterations/ simulated paths
        t: <float> time length/period
        x0: <float> process level at time 0
        kappa: <float> mean-reversion factor
        theta: <float> long-term process mean
        sigma: <float> constant volatility

    Returns:
        <np.array> of shape (time_steps + 1, iter_num) where rows are the simulated paths for a given time step
        i.e. diff rows == diff time steps, and diff cols == diff iteration/ simulated path
    """
    dt = t / time_steps
    deg_freedom = 4 * theta * kappa / (sigma ** 2)
    xt_first_term = (sigma ** 2) * (1 - np.exp(-kappa * dt)) / (4 * kappa)

    # xt is chi-squared distributed
    # set up array for populating, rows are each time step and cols are the diff iterations
    x_next = np.zeros((time_steps + 1, iter_num))
    x_next[0] = x0

    for t_step in range(1, time_steps + 1):
        # populate each time step
        non_centrality_param = x_next[t_step - 1] * (4 * kappa * np.exp(-kappa * dt)) / (
                sigma ** 2 * (1 - np.exp(-kappa * dt)))
        x_next[t_step] = xt_first_term * np.random.noncentral_chisquare(df=deg_freedom, nonc=non_centrality_param,
                                                                        size=iter_num)

    return x_next


if __name__ == '__main__':
    x0 = 0.1
    kappa = 3.0
    theta = 0.05
    sigma = 0.2

    I = 10000
    M = 100

    x = square_root_diffusion(M, I, 2, x0, kappa, theta, sigma)
    print(x)
