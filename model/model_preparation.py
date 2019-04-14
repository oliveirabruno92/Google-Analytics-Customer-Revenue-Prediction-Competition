from sklearn.model_selection import train_test_split

num_cols = ['visitNumber', 'totals.hits', 'totals.pageviews', 'month_unique_user_count', 'day_unique_user_count', 'mean_hits_per_day'
           'sum_pageviews_per_network_domain', 'sum_hits_per_network_domain', 'count_hits_per_network_domain', 'sum_hits_per_region',
           'sum_hits_per_day', 'count_pageviews_per_network_domain', 'mean_pageviews_per_network_domain', 'weekday_unique_user_count',
           'sum_pageviews_per_region', 'count_pageviews_per_region', 'mean_pageviews_per_region', 'user_pageviews_count', 'user_hits_count',
           'count_hits_per_region', 'mean_hits_per_region', 'user_pageviews_sum', 'user_hits_sum', 'user_pageviews_sum_to_mean',
            'user_hits_sum_to_mean', 'user_pageviews_to_region', 'user_hits_to_region', 'mean_pageviews_per_network_domain',
           'mean_hits_per_network_domain']

no_use = ["visitNumber", "date", "fullVisitorId", "sessionId", "visitId", "visitStartTime", 'totals_transactionRevenue', 'trafficSource_referralPath']


def cat_cols(data):

    return [col for col in data.columns if col not in num_cols and col not in no_use]


