from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb


xgb_params = {
    'colsample_bytree': 0.9,
    'gamma': 0.0468,
    'learning_rate': 0.01,
    'max_depth': 3,
    'min_child_weight': 1.7817,
    'n_estimators': 3000,
    'reg_alpha': 0.2,
    'reg_lambda': 0.2,
    'subsample': 0.5213,
    'silent': 1,
    'nthread': -1,
    'n_jobs': -1,
    'random_state': 7
}


class Model:

    def __init__(self, xgb_params):
        self.xgb = xgb.XGBClassifier(**xgb_params)
