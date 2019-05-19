from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd


def rmse(true, pred): return mean_squared_error(true, pred)**0.5


def metrics(y_train, pred_train, y_val, pred_val):

    metrics = pd.DataFrame([], columns=['Train_Metrics', 'Validation_Metrics'])
    metrics.loc['MAE'] = [mean_absolute_error(y_train, pred_train), mean_absolute_error(y_val, pred_val)]
    metrics.loc['MSE'] = [mean_squared_error(y_train, pred_train), mean_squared_error(y_val, pred_val)]
    metrics.loc['RMSE'] = [rmse(y_train, pred_train), rmse(y_val, pred_val)]
    metrics.loc['R2'] = [r2_score(y_train, pred_train), r2_score(y_val, pred_val)]

    return metrics

