import requests
import json

# password stored locally
pass_key_path = r'C:\Users\tkdmc\Documents\GitHub\mchung_pass\mchung_pass.json'

with open(pass_key_path) as f:
    data = json.load(f)

mchung_quotient_key = data['quotient_api']

# Note: 15/month quota
class GetTickData:
    def __init__(self, ticker):
        url = "https://quotient.p.rapidapi.com/intraday"

        querystring = {"end": "2021-02-26 12:00", "interval": "5", "symbol": ticker, "start": "2021-02-26 10:00"}

        headers = {
            'x-rapidapi-key': mchung_quotient_key,
            'x-rapidapi-host': "quotient.p.rapidapi.com"
        }

        response = requests.request("GET", url, headers=headers, params=querystring)

        print(response.text)

    def __call__(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    x = GetTickData('AAPL')
    x()
