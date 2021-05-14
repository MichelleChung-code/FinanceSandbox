import requests
import json
import ast
import pandas as pd
import matplotlib.pyplot as plt

# password stored locally
pass_key_path = r'C:\Users\tkdmc\Documents\GitHub\mchung_pass\mchung_pass.json'

with open(pass_key_path) as f:
    data = json.load(f)

mchung_quotient_key = data['quotient_api']


# Note: 15/month quota
class GetTickData:
    def __init__(self, ticker):
        # todo get access to larger intraday datasets to play around with this more

        url = "https://quotient.p.rapidapi.com/intraday"

        querystring = {"end": "2021-05-13 12:00", "interval": "5", "symbol": ticker, "start": "2021-05-01 10:00"}

        headers = {
            'x-rapidapi-key': mchung_quotient_key,
            'x-rapidapi-host': "quotient.p.rapidapi.com"
        }

        response = requests.request("GET", url, headers=headers, params=querystring)

        # convert to dataframe format
        x = ast.literal_eval(response.text)
        self.df = pd.DataFrame(x)
        self.df.drop("message", axis=1, inplace=True)
        self.df.drop(self.df.tail(1).index, inplace=True)

    def __call__(self, *args, **kwargs):
        # convert data types of what to plot
        self.df["Close"] = pd.to_numeric(self.df["Close"])
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df.set_index("Date", inplace=True)
        self.df['Close'].plot()
        plt.show()

        # if the DatetimeIndex was highly irregular (time intervals between points are heterogeneous), then might want to resample

        df_resamp = self.df.resample(rule='5min').mean()
        df_resamp['Close'].plot()
        plt.show()


if __name__ == '__main__':
    x = GetTickData('AAPL')
    x()
