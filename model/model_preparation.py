from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from time import time


class ModelPreparation:

    def __init__(self, num_cols, no_use, n_folds, target, test_size, seed):
        self.num_cols = num_cols
        self.no_use = no_use
        self.target = target
        self.n_folds = n_folds
        self.test_size = test_size
        self.seed = seed

    def cat_cols(self, data):

        return [col for col in data.columns if col not in self.num_cols and col not in self.no_use]

    def cols_to_use(self, data):

        return [col for col in self.no_use if col in data.columns]

    def features_and_target(self, train):

        train = train.sort_values(by='date')
        x = train.drop(self.cols_to_use(train), axis=1)
        y = train[self.target]

        return x, y

    def split_train_data(self, train):

        x, y = self.features_and_target(train)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.test_size, random_state=self.seed)

        return x_train, x_val, y_train, y_val

    def test_data(self, test):

        x_test = test.drop(self.cols_to_use(test), axis=1)

        return x_test

    def encoder(self, train, test):

        print('Enconding categorical features..')
        cat = self.cat_cols(train)
        for col in cat:
            if col != 'trafficSource_campaignCode':
                lbl = LabelEncoder()
                lbl.fit(list(train[col].values.astype(str)) + list(test[col].values.astype(str)))
                train[col] = lbl.transform(list(train[col].values.astype(str)))
                test[col] = lbl.transform(list(test[col].values.astype(str)))

            return train, test

    def rmse_cv(self, estimator, train):

        start = time()
        x, y = self.features_and_target(train)
        kf = KFold(self.n_folds, shuffle=True).get_n_splits(x.values)
        score = np.sqrt(-cross_val_score(estimator, x.values, y.values.reshape(-1, 1),
                                         scoring='neg_mean_squared_error', cv=kf))
        print('Cross Validation execution time: {} seconds'.format(round(time() - start, 2)))
        return score










