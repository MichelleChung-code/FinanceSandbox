import numpy as np


def expected_excess_returns_apt(factor_exposures, factor_forecasts):
    """
    Expected excess returns as defined by the arbitrage pricing theory (APT) where the expected excess returns are
    determined by the factor exposures and factor forecasts

    Args:
        factor_exposures: <np.Array> of the stock's factor exposures
        factor_forecasts: <np.Array> of factor return forecasts (in decimal)

    Returns:
        <float> the expected excess return
    """
    return factor_exposures.T.dot(factor_forecasts)


if __name__ == '__main__':
    factor_exposures = np.array([0.17, -0.05, 0.19, -0.28])
    factor_forecasts = np.array([0.02, 0.025, -0.015, 0])
    print(f'{expected_excess_returns_apt(factor_exposures, factor_forecasts):.2%}')
