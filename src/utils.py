import pandas as pd
import os

dirname = os.path.dirname(__file__)
data_path = os.path.join(dirname, 'data/')


def read_as_pdf(path: str = data_path, file_name: str = 'sample.csv') -> pd.DataFrame:
    '''Reads a csv file from the data folder and returns it as a pandas data frame.'''
    data = pd.read_csv(path + file_name)
    return data
