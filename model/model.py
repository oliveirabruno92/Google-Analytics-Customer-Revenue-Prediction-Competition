import xgboost as xgb
from time import time
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd

class Model:

    def __init__(self, xgb_params):
        print('Instantiating XGBoost Model..')
        self.xgb = xgb.XGBClassifier(**xgb_params)

    def cv_xgb(self, x, y, n_folds):

        start = time()
        print('Performing cross validation on training data..')
        kf = KFold(n_folds, shuffle=True).get_n_splits(x.values)
        score = np.sqrt(-cross_val_score(self.xgb, x.values, y.values.reshape(-1, 1),
                                         scoring='neg_mean_squared_error', cv=kf))
        cv_results = pd.DataFrame({'mean': score.mean(), 'std': score.std()})
        print('Cross Validation execution time: {} seconds'.format(round(time() - start, 2)))
        return cv_results

    def fit_xgb(self, x, y):

        return self.xgb.fit(x, y, early_stopping_rounds=100, eval_metric='rmse')

    def predict(self, x_test):

        return self.xgb.predict(x_test)





