import xgboost as xgb


class Model:

    def __init__(self, xgb_params):
        print('Instantiating XGBoost Model..')
        self.xgb = xgb.XGBClassifier(**xgb_params)

    def fit(self, x_train, y_train):

        return self.xgb.fit(x_train, y_train, early_stopping_rounds=10, eval_metric='rmse')

    def predict(self, x_test):

        return self.xgb.predict(x_test)





