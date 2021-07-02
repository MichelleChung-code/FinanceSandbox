import numpy as np
import scipy.stats as scs
from common.timeit import timeit

# For pricing European options - can only be executed at maturity/ expiration date

def BSM_pricing_value(S, K, r, t, sigma):
    """
    Assumptions and applicability in https://www.investopedia.com/terms/b/blackscholes.asp
    Args:
        S: <float> underlying asset price
        K: <float> strike price
        r: <float> annualized risk-free interest rate
        t: <float> years to maturity
        sigma: <float> standard deviation of asset returns

    Returns:
        C: <float> call option price
    """

    # N = standard normal cumulative distribution function
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * t) / (sigma * t ** 0.5)
    d2 = d1 - sigma * (t ** 0.5)

    return S * scs.norm.cdf(d1) - K * np.exp(-r * t) * scs.norm.cdf(d2)


@timeit
def BSM_index_level_standard_normal(S, r, t, sigma, iter_num=1000):
    """
    Simulate future index level possibilities

    Args:
        S: <float> index level at t0
        r: <float> annualized risk-free interest rate
        t: <float> years to future date
        sigma: <float> standard deviation of asset returns
        iter_num: <int> number of random standard normal variables to use

    Returns:
        <np.ndarray> simulated potential future index levels
    """
    # z is a standard normally distributed random variable
    z = np.random.standard_normal(iter_num)

    return S * np.exp((r - 0.5 * sigma ** 2) * t + sigma * t ** 0.5 * z)


@timeit
def BSM_index_level_log_normal(S, r, t, sigma, iter_num=1000):
    """
    Simulate future index level possibilities.
    Directly simulate as lognormal distribution.

    Args:
        S: <float> index level at t0
        r: <float> annualized risk-free interest rate
        t: <float> years to future date
        sigma: <float> standard deviation of index returns
        iter_num: <int> number of random standard normal variables to use

    Returns:
        <np.ndarray> simulated potential future index levels
    """

    return S * np.random.lognormal(mean=(r - 0.5 * sigma ** 2) * t, sigma=sigma * t ** 0.5, size=iter_num)


if __name__ == '__main__':
    S0 = 100
    r = 0.06
    sigma = 0.23
    t = 4
    I = 5000
    res1 = BSM_index_level_standard_normal(S0, r, t, sigma, I)
    res2 = BSM_index_level_log_normal(S0, r, t, sigma, I)

    # fairly similar, descrepancies mainly from sampling error
    print(scs.describe(res1))
    print(scs.describe(res2))
