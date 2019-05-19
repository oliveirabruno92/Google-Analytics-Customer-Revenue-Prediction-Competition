import numpy as np
from time import time


def submission_file(test, predictions):

    test['PredictedLogRevenue'] = predictions
    sub = test[['PredictedLogRevenue', 'fullVisitorId']].groupby('fullVisitorId')['PredictedLogRevenue'].sum().\
        reset_index()
    sub['PredictedLogRevenue'] = np.log1p(sub['PredictedLogRevenue'])
    sub['PredictedLogRevenue'] = sub['PredictedLogRevenue'].apply(lambda x: 0.0 if x < 0 else x)
    sub['PredictedLogRevenue'] = sub['PredictedLogRevenue'].fillna(0.0)

    return sub.to_csv('data/submissions/sub.csv', index=False)


def timer(method):
    def timed(*args, **kwargs):
        ts = time()
        result = method(*args, **kwargs)
        te = time()

        if 'log_time' in kwargs:
            name = kwargs.get('log_time', method.__name__.upper())
            kwargs['log_time'][name] = int((te - ts)*1000)
        else:
            print('{} {:.2f} ms'.format(method.__name__, (te - ts)*1000))
        return result
    return timed

