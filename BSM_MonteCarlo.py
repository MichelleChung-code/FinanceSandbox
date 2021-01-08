import numpy as np
from common.timeit import timeit
from black_scholes_merton import BSM_pricing_value

@timeit
def BSM_monte_carlo(iter_num, step_num, S, K, r, t, sigma):
    """
    Monte carlo computation of the call option price

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

    # plot S_arr for the simulation of the changing price over the time intervals

    # return the monte carlo estimator, C
    return np.exp(-r * t) * (1 / iter_num) * sum(inner_vals_ht)


if __name__ == '__main__':
    print(BSM_monte_carlo(500000, 100, 100, 200, 0.05, 2, 0.2))
    print(BSM_pricing_value(100, 200, 0.05, 2, 0.2))