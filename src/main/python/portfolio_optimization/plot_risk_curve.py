from src.main.python.portfolio_optimization.portfolio_opt import MarkowitzOptimizePortfolio


class RiskCurve(MarkowitzOptimizePortfolio):

    def __init__(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


if __name__ == '__main__':
    x = RiskCurve()
    x()
