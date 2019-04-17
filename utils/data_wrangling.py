import pandas as pd


def fill_nulls(train, test):

    train['trafficSource_adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
    test['trafficSource_adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
    train['trafficSource_isTrueDirect'].fillna(False, inplace=True)
    test['trafficSource_isTrueDirect'].fillna(False, inplace=True)

    return train, test


def convert_dates(train, test):

    train['date'] = pd.to_datetime(train['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
    test['date'] = pd.to_datetime(test['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))

    return train, test


def drop_cols(train, test):

    cols_to_drop = [col for col in train.columns if train[col].nunique(dropna=False) == 1]
    train.drop(cols_to_drop, axis=1, inplace=True)
    test.drop([col for col in cols_to_drop if col in test.columns], axis=1, inplace=True)

    return train, test


def drop_nulls(train, test):

    if train.isnull().any().any() and test.isnull().any().any():
        train = train.dropna(axis=0)
        test = test.dropna(axis=0)
    return train, test


def convert_features(train, test):

    for col in ['visitNumber', 'totals_hits', 'totals_pageviews', 'totals_transactionRevenue']:
        train[col] = train[col].astype(float)
        test[col] = test[col].astype(float)

        return train, test


def data_wrangling(train, test):

    print('Cleaning data..')
    train, test = fill_nulls(train, test)
    train, test = convert_dates(train, test)
    train, test = drop_cols(train, test)
    train, test = convert_features(train, test)

    return train, test





