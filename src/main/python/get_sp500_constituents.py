import requests
import pandas as pd
from bs4 import BeautifulSoup
import datetime as dt
from pathlib import Path
import os

pd.set_option('display.max_columns', None)


class SP500ContituentData:
    def __init__(self, overwrite_results_path=False):
        """
        Args:
            overwrite_results_path: <str> optional argument on whether to overwrite stored csv results
        """
        self.url = r'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        self.overwrite_results_path = overwrite_results_path

    def get_wikipedia_data(self):
        """
        Function that interacts with the webpage and formats the data into a pandas dataframe

        Returns:
            df: <pd.DataFrame> dataframe containing the SP500 Constituents from table on
            https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
        """
        webpage = requests.get(self.url)

        if webpage.status_code != 200:
            raise Exception('Connection to webpage UNSUCCESSFUL')

        soup = BeautifulSoup(webpage.text, 'html.parser')
        wiki_table = soup.find('table', {'class': 'wikitable'})

        df = pd.read_html(str(wiki_table))
        return pd.DataFrame(df[0])

    def __call__(self):
        """
        Calls the webscraper and writes results to csv based on self.overwrite_results_path
        """
        df = self.get_wikipedia_data()
        print(df.head())

        if self.overwrite_results_path:
            df['Update_Time'] = dt.datetime.now()

            assert self.overwrite_results_path.endswith('.csv')
            df.to_csv(self.overwrite_results_path)


if __name__ == '__main__':
    res_path = os.path.join(str(Path(__file__).parents[0]), 'data')
    overwrite_results_path = os.path.join(res_path, 'sp500_constituents.csv')

    x = SP500ContituentData(overwrite_results_path=overwrite_results_path)
    x()
