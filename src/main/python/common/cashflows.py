import numpy as np


def discounted_expected_uncertain_cashflows(probabilities, uncertain_cashflows, t, rf_rate=0.06):
    """
    Present value using weighted average as the expected cashflow

    * Note, this does not account for risk

    Args:
        probabilities: <np.Array> of cashflow probabilities of cashflows at time t
        uncertain_cashflows: <np.Array> possible cashflows at time t
        t: <float> period to discount by
        rf_rate: <float> annual risk free rate

    Returns:
        <float> value of discounted expected cash flows
    """
    return (probabilities.T.dot(uncertain_cashflows)) / (1 + rf_rate) ** t


if __name__ == '__main__':
    print(discounted_expected_uncertain_cashflows(np.array([0.5, 0.5]), np.array([49, 53]), t=1 / 12))
