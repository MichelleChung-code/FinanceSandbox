from sympy import Eq, Symbol, solve


class RetirementSavings:
    def __init__(self, age, current_amount=0, retirement_age=60, average_annual_ret=0.04, inflation_rate=1.5 / 100):
        self.current_savings = current_amount
        self.contribution_years = retirement_age - age
        required_years = 100 - retirement_age  # assuming 100y life expectancy
        self.required_amount = 100000 * required_years
        self.avg_yearly_return_inf_adjusted = ((1 + average_annual_ret) / (1 + inflation_rate)) - 1

    def __call__(self, *args, **kwargs):
        print(self)
        sol = self.calculate_annual_contributions()
        return {'annual_contributions': sol,
                'monthly_contributions': sol / 12}

    def calculate_annual_contributions(self):
        # make contributions at the beginning of the month
        pmt = Symbol('pmt')

        current_savings_compounded = self.current_savings * (
                1 + self.avg_yearly_return_inf_adjusted) ** self.contribution_years

        future_value_of_series_begin_mth = pmt * (
                ((1 + self.avg_yearly_return_inf_adjusted) ** self.contribution_years - 1)
                / self.avg_yearly_return_inf_adjusted) * (1 + self.avg_yearly_return_inf_adjusted)

        expr = current_savings_compounded + future_value_of_series_begin_mth

        eqn = Eq(expr, self.required_amount)
        return solve(eqn)[0]

    def __repr__(self):
        return f'{self.__class__.__name__}(contribution_years: {self.contribution_years!r}, required_amount: {self.required_amount!r})'


if __name__ == '__main__':
    x = RetirementSavings(age=22, current_amount=0)
    print(x())
