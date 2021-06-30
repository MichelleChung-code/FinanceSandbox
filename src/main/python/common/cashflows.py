import numpy as np


def discounted_expected_uncertain_cashflows(probabilities, uncertain_cashflows, t, rf_rate=0.06):
    """

    * Note, this does not account for risk

    Args:
        probabilities:
        uncertain_cashflows:
        t:
        rf_rate:

    Returns:

    """
    return (probabilities.T.dot(uncertain_cashflows)) / (1 + rf_rate) ** t


if __name__ == '__main__':
    print(discounted_expected_uncertain_cashflows(np.array([0.5, 0.5]), np.array([49, 53]), t=1 / 12))
