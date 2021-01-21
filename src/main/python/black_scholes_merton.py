import numpy as np
from scipy.stats import norm


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

    return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
