from sympy import Eq, Symbol, solve


class RetirementSavings:
    def __init__(self, age, current_amount=0, retirement_age=60, average_annual_ret=0.04, inflation_rate=1.5 / 100,
                 required_amt_override=False):
        """
        Class to give an idea of the contribution amounts required to retire with a certain amount of money, assuming
        an average annual rate of return for the contributions and current savings.

        Args:
            age: <int> current age
            current_amount: <float> current amount invested/ in savings
            retirement_age: <int> age to retire at
            average_annual_ret: <float> decimal value for the annual return rate, conservatively defaults to 4%
            inflation_rate: <float> decimal value for annual inflation rate, defaults to 1.5%
            required_amt_override: <bool> or <float> if False then use default of 100k per year (100-retirement_age)
            total required for retirement, if <float> then manually enter the total required to retire
        """
        self.current_savings = current_amount
        self.contribution_years = retirement_age - age
        required_years = 100 - retirement_age  # assuming 100y life expectancy
        self.required_amount = required_amt_override if required_amt_override else 100000 * required_years

        # adjust for inflation :(
        self.avg_yearly_return_inf_adjusted = ((1 + average_annual_ret) / (1 + inflation_rate)) - 1

    def __call__(self, *args, **kwargs):
        print(self)
        sol = self.calculate_annual_contributions()

        # return the annual and monthly contribution amounts required
        return {'annual_contributions': sol,
                'monthly_contributions': sol / 12}

    def calculate_annual_contributions(self):
        # make contributions at the beginning of the month
        pmt = Symbol('pmt')

        current_savings_compounded = self.current_savings * (
                1 + self.avg_yearly_return_inf_adjusted) ** self.contribution_years

        # assume that we contribute at the beginning of the month
        future_value_of_series_begin_mth = pmt * (
                ((1 + self.avg_yearly_return_inf_adjusted) ** self.contribution_years - 1)
                / self.avg_yearly_return_inf_adjusted) * (1 + self.avg_yearly_return_inf_adjusted)

        expr = current_savings_compounded + future_value_of_series_begin_mth

        # solve for when the target amount is equal to the compounded value of current savings and payments
        eqn = Eq(expr, self.required_amount)
        return solve(eqn)[0]

    def __repr__(self):
        return f'{self.__class__.__name__}(contribution_years: {self.contribution_years!r}, required_amount: {self.required_amount!r})'


if __name__ == '__main__':
    x = RetirementSavings(age=22, current_amount=0)
    print(x())
