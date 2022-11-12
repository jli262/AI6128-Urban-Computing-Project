import logging
logger = logging.getLogger(__name__)

import sys
import os
import pandas as pd


if __name__ == '__main__':
    cwd = os.getcwd()
    data_directory = os.path.join(os.path.dirname(cwd), 'data')
    data_filepath = os.path.join(data_directory, 'train.csv')

    df = pd.read_csv(data_filepath)
    print(df.head())
