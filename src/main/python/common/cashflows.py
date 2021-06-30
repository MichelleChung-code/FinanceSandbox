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


def discounted_expected_uncertain_cashflows_risk_adjusted(probabilities, uncertain_cashflows, valuation_multipliers, t,
                                                          rf_rate=0.06):
    """
    Present value using weighted average as the expected cashflow with valuation multipliers to adjust for risk

    Args:
        probabilities: <np.Array> of cashflow probabilities of cashflows at time t
        uncertain_cashflows: <np.Array> possible cashflows at time t
        valuation_multipliers: <np.Array> of the valuation multipliers - factors to adjust probabilties to adust for
        risk.  The risk adjustment is done by placing lower weight on good-time cashflows and higher weights on bad-
        time cashflows.  i.e. the investor is more significantly negatively impacted by lower flows :(
        t: <float> period to discount by
        rf_rate: <float> annual risk free rate

    Returns:
        <float> value of discounted expected cash flows
    """

    # multiply the 3 lists, element-wise
    risk_adjusted_expected_cashflows = list(
        map(lambda x, y, z: x * y * z, probabilities, uncertain_cashflows, valuation_multipliers))

    return sum(risk_adjusted_expected_cashflows) / (1 + rf_rate) ** t


if __name__ == '__main__':
    print(discounted_expected_uncertain_cashflows(np.array([0.5, 0.5]), np.array([49, 53]), t=1 / 12))

    print(discounted_expected_uncertain_cashflows_risk_adjusted(np.array([0.5, 0.5]), np.array([49, 53]),
                                                                np.array([1.38, 0.62]), t=1 / 12))
