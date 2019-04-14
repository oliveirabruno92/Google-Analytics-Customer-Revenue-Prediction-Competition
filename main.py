from utils.feature_engineering import feature_engineering
from utils.data_wrangling import data_wrangling
from utils.flatten_json import load_df
import pandas as pd
import numpy as np
from time import time

start = time()

flatten = False

if flatten:

    print('Loading train data..')
    train = load_df()
    print('Loading test data..')
    test = load_df('data/raw/test_v2.csv')

    train.to_csv('data/flatten_data/train_flattened.csv', index=False)
    test.to_csv('data/flatten_data/test_flattened.csv', index=False)

else:

    print('Reading data..')
    train = pd.read_csv('data/flatten_data/train_flattened.csv', low_memory=False)
    test = pd.read_csv('data/flatten_data/test_flattened.csv', dtype={'fullVisitorId': np.str}, low_memory=False)

print('Train dimensions before feature engineering: {} rows | {} features'.format(train.shape[0], train.shape[1]))
print('Test dimensions before feature engineering: {} rows | {} features'.format(test.shape[0], test.shape[1]))

print('Cleaning data..')
train, test = data_wrangling(train, test)
print('Making some feature engineering..')
train, test = feature_engineering(train), feature_engineering(test)

print('Train dimensions after feature engineering: {} rows | {} features'.format(train.shape[0], train.shape[1]))
print('Test dimensions after feature engineering: {} rows | {} features'.format(test.shape[0], test.shape[1]))

print('Application execution time: {} seconds'.format(round(time() - start, 2)))










