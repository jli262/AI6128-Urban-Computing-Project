import logging
logger = logging.getLogger(__name__)

import os
import pandas as pd

if __name__ == '__main__':
    data_directory = os.path.join(os.getcwd(), 'data')
    datapath = os.path.join(data_directory, 'train.csv')
    df = pd.read_csv(datapath)
    df.loc[:999,:].to_csv(os.path.join(data_directory, 'train_1000.csv'))
