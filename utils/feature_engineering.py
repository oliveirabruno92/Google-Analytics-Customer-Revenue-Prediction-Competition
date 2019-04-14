import numpy as np


def features_to_log(data):

    data['visitNumber'] = np.log1p(data['visitNumber'])
    data['totals_hits'] = np.log1p(data['totals_hits'])
    data['totals_pageviews'] = np.log1p(data['totals_pageviews'].astype(float).fillna(0))

    return data


def date_features(data):

    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['weekday'] = data['date'].dt.weekday
    data['weekofyear'] = data['date'].dt.weekofyear

    data['month_unique_user_count'] = data.groupby('month')['fullVisitorId'].transform('nunique')
    data['day_unique_user_count'] = data.groupby('day')['fullVisitorId'].transform('nunique')
    data['weekday_unique_user_count'] = data.groupby('weekday')['fullVisitorId'].transform('nunique')
    data['weekofyear_unique_user_count'] = data.groupby('weekofyear')['fullVisitorId'].transform('nunique')

    return data


def joined_features(data):

    data['browser_category'] = data['device_browser'] + '_' + data['device_deviceCategory']
    data['browser_operatingSystem'] = data['device_browser'] + '_' + data['device_operatingSystem']

    return data


def grouped_features(data):

    data['sum_pageviews_per_network_domain'] = data.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform(
        'sum')
    data['count_pageviews_per_network_domain'] = data.groupby('geoNetwork_networkDomain')[
        'totals_pageviews'].transform('count')
    data['mean_pageviews_per_network_domain'] = data.groupby('geoNetwork_networkDomain')[
        'totals_pageviews'].transform('mean')
    data['sum_hits_per_network_domain'] = data.groupby('geoNetwork_networkDomain')['totals_hits'].transform('sum')
    data['count_hits_per_network_domain'] = data.groupby('geoNetwork_networkDomain')['totals_hits'].transform('count')
    data['mean_hits_per_network_domain'] = data.groupby('geoNetwork_networkDomain')['totals_hits'].transform('mean')
    data['mean_hits_per_day'] = data.groupby('day')['totals_hits'].transform('mean')
    data['sum_hits_per_day'] = data.groupby('day')['totals_hits'].transform('sum')
    data['sum_pageviews_per_network_domain'] = data.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform(
        'sum')
    data['count_pageviews_per_network_domain'] = data.groupby('geoNetwork_networkDomain')[
        'totals_pageviews'].transform('count')
    data['mean_pageviews_per_network_domain'] = data.groupby('geoNetwork_networkDomain')[
        'totals_pageviews'].transform('mean')
    data['sum_pageviews_per_region'] = data.groupby('geoNetwork_region')['totals_pageviews'].transform('sum')
    data['count_pageviews_per_region'] = data.groupby('geoNetwork_region')['totals_pageviews'].transform('count')
    data['mean_pageviews_per_region'] = data.groupby('geoNetwork_region')['totals_pageviews'].transform('mean')
    data['sum_hits_per_network_domain'] = data.groupby('geoNetwork_networkDomain')['totals_hits'].transform('sum')
    data['count_hits_per_network_domain'] = data.groupby('geoNetwork_networkDomain')['totals_hits'].transform('count')
    data['mean_hits_per_network_domain'] = data.groupby('geoNetwork_networkDomain')['totals_hits'].transform('mean')
    data['sum_hits_per_region'] = data.groupby('geoNetwork_region')['totals_hits'].transform('sum')
    data['count_hits_per_region'] = data.groupby('geoNetwork_region')['totals_hits'].transform('count')
    data['mean_hits_per_region'] = data.groupby('geoNetwork_region')['totals_hits'].transform('mean')
    data['sum_hits_per_country'] = data.groupby('geoNetwork_country')['totals_hits'].transform('sum')
    data['count_hits_per_country'] = data.groupby('geoNetwork_country')['totals_hits'].transform('count')
    data['mean_hits_per_country'] = data.groupby('geoNetwork_country')['totals_hits'].transform('mean')
    data['user_pageviews_sum'] = data.groupby('fullVisitorId')['totals_pageviews'].transform('sum')
    data['user_hits_sum'] = data.groupby('fullVisitorId')['totals_hits'].transform('sum')
    data['user_pageviews_count'] = data.groupby('fullVisitorId')['totals_pageviews'].transform('count')
    data['user_hits_count'] = data.groupby('fullVisitorId')['totals_hits'].transform('count')
    data['user_pageviews_sum_to_mean'] = data['user_pageviews_sum'] / data['user_pageviews_sum'].mean()
    data['user_hits_sum_to_mean'] = data['user_hits_sum'] / data['user_hits_sum'].mean()

    return data


def rate_features(data):

    data['user_pageviews_to_region'] = data['user_pageviews_sum'] / data['mean_pageviews_per_region']
    data['user_hits_to_region'] = data['user_hits_sum'] / data['mean_hits_per_region']

    return data


def feature_engineering(data):

    data = features_to_log(data)
    data = joined_features(data)
    data = grouped_features(data)
    data = rate_features(data)

    return data


