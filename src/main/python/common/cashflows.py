import numpy as np
import math


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


# todo look into intertemporal valuation formula for the valuation multipliers 

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


def misvaluation_methods(mkt_price_0, mkt_price_1, mdl_price_0, mdl_price_1, d, rf_rate=0.06):
    """
    Compute some misvaluation measures for a stock that pays a dividend at the end of a certain time period

    Args:
        mkt_price_0: <float> market price at the start of the time period
        mkt_price_1: <float> market price at the end of the time period
        mdl_price_0: <float> model price at the start of the time period
        mdl_price_1: <float> model uncertain price at the end of the time period
        d: <float> dividend amount
        rf_rate: <float> annual risk free rate

    Returns:
        <dict> containing the following:
            kappa: <float> decimal, extent of misvaluation (percent diff between model and market prices at time 0
            gamma: <float> decimal, persistance of misvaluation, estimate of the time length for the market to correct
            misvaluation_half_life: <float> number of years for half the misvaluation to be addressed in the market
            alpha: <float> decimal, the alpha

    """
    kappa = (mdl_price_0 - mkt_price_0) / mkt_price_0
    gamma = ((mdl_price_1 - mkt_price_1) / (mkt_price_1 + d)) / kappa

    assert 0 <= gamma <= 1

    # half-life of the misvaluation is -0.69/ln(gamma)
    half_life_misval = -0.69 / math.log(gamma)
    alpha = (1 + rf_rate) * (kappa * (1 - gamma) / (1 + kappa * gamma))

    return {'kappa': kappa,
            'gamma': gamma,
            'misvaluation_half_life': half_life_misval,
            'alpha': alpha}


if __name__ == '__main__':
    print(discounted_expected_uncertain_cashflows(np.array([0.5, 0.5]), np.array([49, 53]), t=1 / 12))

    print(discounted_expected_uncertain_cashflows_risk_adjusted(np.array([0.5, 0.5]), np.array([49, 53]),
                                                                np.array([1.38, 0.62]), t=1 / 12))

    print(misvaluation_methods(mkt_price_0=50, mdl_price_0=51, mkt_price_1=53, mdl_price_1=54, d=1))
