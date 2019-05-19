from modules.feature_engineering import feature_engineering
from modules.data_wrangling import data_wrangling
from utils.configurations import Configurations
from utils.helper import timer, submission_file
from utils.flatten_json import load_df
from model.model_preparation import ModelPreparation
from model.model import Model
import pandas as pd
import numpy as np


@timer
def main():

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
    test_size = config_params['test_size']
    seed = config_params['seed']

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

    model_preparation = ModelPreparation(num_cols, no_use, n_folds, target, test_size, seed)

    print('Train dimensions before feature engineering: {} rows | {} features'.format(train.shape[0], train.shape[1]))
    print('Test dimensions before feature engineering: {} rows | {} features'.format(test.shape[0], test.shape[1]))

    train, test = data_wrangling(train, test)
    train, test = feature_engineering(train, test)
    train, test = model_preparation.encoder(train, test)

    x_test = model_preparation.test_data(test)

    print('Train dimensions after feature engineering: {} rows | {} features'.format(train.shape[0], train.shape[1]))
    print('Test dimensions after feature engineering: {} rows | {} features'.format(test.shape[0], test.shape[1]))

    print('='*40)

    x, y = model_preparation.features_and_target(train)

    xgb = Model(xgb_params)
    cv_results = xgb.cv_xgb(x, y, n_folds=n_folds)

    print(cv_results)

    print('='*40)

    xgb.fit_xgb(x, y)
    predictions = xgb.predict(x_test)

    sub_csv = submission_file(test, predictions)


if __name__ == '__main__':

    main()










