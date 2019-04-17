from utils.feature_engineering import feature_engineering
from utils.configurations import Configurations
from utils.data_wrangling import data_wrangling, drop_nulls
from utils.flatten_json import load_df
from model.model import Model
from model.model_preparation import ModelPreparation
import pandas as pd
import numpy as np
from time import time
import sys

start = time()

print('='*40)
print('INITIALIZING GOOGLE ANALYTICS COMPETITION')
print('='*40)

config = Configurations()
config_params = config.get_params()

flatten = config_params['flatten']
num_cols = config_params['num_cols']
no_use = config_params['no_use']
xgb_params = config_params['xgb_params']
n_folds = config_params['n_folds']
target = config_params['target']

if not flatten:

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

train, test = data_wrangling(train, test)
train, test = feature_engineering(train), feature_engineering(test)

model_preparation = ModelPreparation(num_cols, no_use, n_folds, target)
train, test = model_preparation.encoder(train, test)

print(train.isnull().any().any())
print(test.isnull().any().any())

train, test = drop_nulls(train, test)

print(train.isnull().any().any())
print(test.isnull().any().any())
sys.exit(0)

print('Train dimensions after feature engineering: {} rows | {} features'.format(train.shape[0], train.shape[1]))
print('Test dimensions after feature engineering: {} rows | {} features'.format(test.shape[0], test.shape[1]))

xgb = Model(xgb_params)
score = model_preparation.rmse_cv(xgb, train)
print("RMSE: {} (+/- {})".format(score.mean(), score.std()))

print('Application execution time: {} seconds'.format(round(time() - start, 2)))










