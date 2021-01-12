from mplfinance.original_flavor import candlestick_ochl
import os
from pathlib import Path
from SimpleStockDataPlot import extract_data
import common.constants as const
import datetime as dt
import matplotlib.dates as dates
import matplotlib.pyplot as plt


class CandleStickPlot:
    def __init__(self, ticker, start_date, end_date=dt.datetime.today().strftime('%Y-%m-%d'), res_path=False):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.res_path = res_path
        self.data = extract_data(ticker, start_date, end_date)

    def preprocess(self):
        # need to get in format for candlestick_ochl function
        self.data.index.name = const.DATE
        self.data.reset_index(inplace=True)

        req_col_order = [const.DATE, const.OPEN, const.CLOSE, const.HIGH, const.LOW]
        self.data = self.data[req_col_order]

        # Date column must be in float day format
        self.data[const.DATE] = dates.date2num(self.data[const.DATE].to_numpy())

    def plot_candlestick(self):
        fig, ax = plt.subplots()

        candlestick_ochl(ax, self.data.to_numpy(), width=0.5)

        plt.grid(True)
        ax.xaxis_date()
        plt.xticks(rotation=30)
        plt.title('Candlestick Plot for {}'.format(self.ticker))

        plt.tight_layout()

        if self.res_path:
            plt.savefig(os.path.join(self.res_path, '{}_{}_{}.png'.format(self.ticker, self.start_date, self.end_date)))

        plt.show()

    def __call__(self, *args, **kwargs):
        self.preprocess()
        self.plot_candlestick()


if __name__ == '__main__':
    ticker = '^GSPC'
    results_folder_path = os.path.join(str(Path(__file__).parents[0]), 'results')
    x = CandleStickPlot(ticker, start_date='2020-12-01', res_path=results_folder_path)
    x()
