import numpy as np
import math
from common.timeit import timeit

#todo add comments and doc strings

class BinomialOptionPricing:
    def __init__(self, S_0, maturity, short_rate, vol_factor, strike_price, time_steps):
        # for call option
        self.S_0 = S_0  # initial index level
        self.maturity = maturity  # in years
        self.short_rate = short_rate
        self.vol_factor = vol_factor
        self.strike_price = strike_price
        self.time_steps = time_steps

    @timeit
    def compute_pricing(self):
        delta_t = self.maturity / self.time_steps
        delta_discount_rate = math.exp(-self.short_rate * delta_t)
        up_movement = math.exp(self.vol_factor * math.sqrt(delta_t))
        down_movement = 1 / up_movement
        risk_neutral_prob = (math.exp(self.short_rate * delta_t) - down_movement) / (
                up_movement - down_movement)  # martingale probability

        # construct index levels per time step and then inner values at self.maturity
        lvls = np.arange(self.time_steps + 1)
        mu, md = np.resize(lvls, (self.time_steps + 1, self.time_steps + 1)), np.resize(lvls, (
            self.time_steps + 1, self.time_steps + 1)).T
        mu, md = up_movement ** (mu - md), down_movement ** md
        S = self.S_0 * mu * md  # S_t = S_s * m

        # loop to get present value
        val = np.maximum(S - self.strike_price, 0)
        i = 0

        for t in range(self.time_steps - 1, -1, -1):  # backwards loop to discount expected inner values
            val[0:self.time_steps - i, t] = (risk_neutral_prob * val[0:self.time_steps - i, t + 1] + (
                    1 - risk_neutral_prob)
                                             * val[1:self.time_steps - i + 1, t + 1]) * delta_discount_rate
            i += 1

        return val[0, 0]


if __name__ == '__main__':
    S_0 = 100  # initial index level
    maturity = 1  # in years
    short_rate = 0.05
    vol_factor = 0.20
    strike_price = 100
    time_steps = 1000

    x = BinomialOptionPricing(S_0, maturity, short_rate, vol_factor, strike_price, time_steps)

    print(x.compute_pricing())
