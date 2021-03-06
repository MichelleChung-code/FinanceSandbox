import requests
import json
import ast
import pandas as pd
import matplotlib.pyplot as plt
import tpqoa

# password stored locally
pass_key_path = r'C:\Users\tkdmc\Documents\GitHub\mchung_pass\mchung_pass.json'

with open(pass_key_path) as f:
    data = json.load(f)

mchung_quotient_key = data['quotient_api']

mchung_oanda_config_path = r'C:\Users\tkdmc\Documents\GitHub\mchung_pass\oanda.cfg'


class OANDAData:
    def __init__(self):
        # create connection to API
        self.api = tpqoa.tpqoa(mchung_oanda_config_path)
        assert self.api.account_type == 'practice'  # make sure we're not using a live account zzz

    def _check_valid_technical_instrument_name(self, instrument_name):
        """
        Check that instrument_name is a valid instrument on OANDA for data

        Args:
            instrument_name: <str> technical instrument name

        Returns:
            <bool> True if instrument_name is valid
        """
        ls_valid_instruments_tups = self.api.get_instruments()
        return instrument_name in (x[1] for x in ls_valid_instruments_tups)

    def get_historic_data(self, instrument_name, start_date, end_date, freq, price_type):
        """
        Get historic data from OANDA

        Args:
            instrument_name: <str> technical instrument name on OANDA
            start_date: <str> YYYY-MM-DD to start getting data
            end_date: <str> YYYY-MM-DD to end getting data
            freq: <str> data frequency i.e. 'H1' for 1 hour, 'D' for daily, etc.
            price_type: <str> 'A' for ask and 'B' for bid

        Returns:
            <pd.DataFrame> of data returned, in form of candlestick data i.e. open, close, high, low prices
        """
        # first check that this is valid
        assert self._check_valid_technical_instrument_name(instrument_name)
        df = self.api.get_history(instrument=instrument_name, start=start_date, end=end_date, granularity=freq,
                                  price=price_type)
        return df

    def stream_current_data(self, instrument_name, max_iter: int):
        """
        Stream data real time - print results to console output

        Args:
            instrument_name: <str> technical OANDA instrument name
            max_iter: <int> max data points to stream

        """
        assert self._check_valid_technical_instrument_name(instrument_name)
        print(self.api.stream_data(instrument=instrument_name, stop=max_iter))


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
    x = OANDAData()
    # test = x.get_historic_data("SPX500_USD", "2021-07-01", "2021-07-05", "H1", "B")
    # print(test)
    x.stream_current_data("SPX500_USD", 10)
