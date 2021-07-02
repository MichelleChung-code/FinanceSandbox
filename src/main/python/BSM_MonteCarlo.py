import numpy as np
from common.timeit import timeit
from common.math_functions import standard_normal_with_moment_matching
from black_scholes_merton import BSM_pricing_value
from common.simple_line_plot import show_line_plot
import matplotlib.pyplot as plt


@timeit
def BSM_monte_carlo(iter_num, step_num, S, K, r, t, sigma):
    """
    Monte carlo computation of the call option price

    Following the Markov property that tomorrow's process value only depends on today's process state

    Args:
        iter_num: <int> number of iterations of pseudo random numbers
        step_num: <int> number of time intervals to cover
        S: <float> underlying asset price
        K: <float> strike price
        r: <float> annualized risk-free interest rate
        t: <float> years to maturity
        sigma: <float> standard deviation of asset returns

    Returns:
        C: <float> monte carlo estimator for the call option price
    """
    dt = t / step_num

    # rows each represent a different time interval
    # columns are the pseudorandom generated numbers, of length iter_num
    arr_size = (step_num + 1, iter_num)

    # set up array storing
    S_arr = np.zeros(arr_size)

    # benchmark for estimators to follow is the given S value
    S_arr[0] = S

    # apply pseudo-random standard normally distributed numbers per timestep to discretization scheme
    # get the S value at the given time step
    for time_step in range(1, step_num + 1):
        S_arr[time_step] = S_arr[time_step - 1] * np.exp(
            (r - (0.5) * (sigma ** 2)) * dt + sigma * (dt ** 0.5) * np.random.standard_normal(iter_num))

    # get inner values
    inner_vals_ht = np.maximum(S_arr[-1] - K, 0)

    # plot the log-normally distributed resulting end values
    plt.hist(S_arr[-1], bins=50, alpha=0.5, histtype='bar', ec='k')
    plt.ylabel('Frequency')
    plt.xlabel('level')
    plt.grid(True)
    plt.show()

    # plot S_arr for the simulation of the changing price over the time intervals
    # Just plot the first 20 time intervals
    show_line_plot(S_arr[:, :20], 'BSM Monte Carlo', 'time steps', 'price', plot_arr=True)

    # return the monte carlo estimator, C
    return np.exp(-r * t) * (1 / iter_num) * sum(inner_vals_ht)


@timeit
def BSM_monte_carlo_at_maturity_only(iter_num, S, K, r, t, sigma, option_type='call'):
    """
    Monte carlo computation of the call or put option price

    At maturity only, no need for storing the whole path

    Args:
        iter_num: <int> number of iterations of pseudo random numbers
        S: <float> underlying asset price
        K: <float> strike price
        r: <float> annualized risk-free interest rate
        t: <float> years to maturity
        sigma: <float> standard deviation of asset returns
        option_type: <str> can be 'call' or 'put'

    Returns:
        C: <float> monte carlo estimator for the call option price
    """

    # get array of random numbers
    rand_nums = standard_normal_with_moment_matching((2, iter_num))

    # get the S value at maturity for various simulations
    S_arr = S * np.exp((r - (0.5) * (sigma ** 2)) * t + sigma * (t ** 0.5) * rand_nums[1])

    # get inner values
    if option_type == 'call':
        inner_vals_ht = np.maximum(S_arr - K, 0)
    elif option_type == 'put':
        inner_vals_ht = np.maximum(K - S_arr, 0)
    else:
        raise NotImplementedError

    # return the monte carlo estimator, C
    return np.exp(-r * t) * (1 / iter_num) * sum(inner_vals_ht)


if __name__ == '__main__':
    print(BSM_monte_carlo_at_maturity_only(50000, 100, 110, 0.05, 1, 0.25, option_type='call'))
    print(BSM_monte_carlo_at_maturity_only(50000, 100, 110, 0.05, 1, 0.25, option_type='put'))
