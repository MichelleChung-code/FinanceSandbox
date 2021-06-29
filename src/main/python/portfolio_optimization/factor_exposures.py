import pandas as pd
from urllib import request
import zipfile


class FactorExposuresFamaFrench3Factor:
    def __init__(self):
        pass

    def scrape_fama_french_factor_data(self):
        # Get 3 factor fama-french model data from Dartmouth
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
        fname = 'factor_info.zip'
        csv_fname = 'F-F_Research_Data_Factors.csv'

        request.urlretrieve(url=url, filename=fname)

        assert zipfile.is_zipfile(fname)

        # Open the zip folder and read in the CSV
        z_file = zipfile.ZipFile(fname, 'r')
        z_file.extractall()
        z_file.close()

        factor_data = pd.read_csv(csv_fname, skiprows=3, index_col=0)

        # remove row if any column has a null value
        factor_data = factor_data.loc[~factor_data.isnull().any(axis=1)]

        return factor_data

    def __call__(self, *args, **kwargs):
        factor_data = self.scrape_fama_french_factor_data()

        return factor_data


if __name__ == '__main__':
    x = FactorExposuresFamaFrench3Factor()
    print(x())
