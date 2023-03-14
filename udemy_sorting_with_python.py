############################################
# Sorting Products
############################################

# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR Score + Other Factors

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import scipy.stats as st

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df_ = pd.read_csv('datasets/product_sorting.csv')
df = df_.copy()

df.head()
df.shape
df.info()
df.isnull().sum()
df.describe().T

df.groupby('instructor_name').agg({'purchase_count':'sum'}).sort_values('purchase_count', ascending=False)
df.groupby('instructor_name').agg({'purchase_count':'count'}).sort_values('purchase_count', ascending=False)
df.groupby('course_name').agg({'purchase_count':'sum'}).sort_values('purchase_count', ascending=False)

############################################
# Sorting by Rating
############################################

df.sort_values('rating', ascending=False)

############################################
# Sorting by Comment Count or Purchase Count
############################################

df.sort_values('purchase_count', ascending=False)
df.sort_values('comment_count', ascending=False)

############################################
# Sorting by Rating, Comment and Purchase
############################################

df['purchase_count_scaler'] = MinMaxScaler(feature_range=(1, 5)).fit(df[['purchase_count']]).transform(df[['purchase_count']])
df['comment_count_scaler'] = MinMaxScaler(feature_range=(1, 5)).fit(df[['comment_count']]).transform(df[['comment_count']])


def weighted_sorting_score(dataframe, w1=.42, w2=.32, w3=.26):
    weighted_sorting_score = dataframe['rating'] * w1 + \
                             dataframe['comment_count_scaler'] * w2 + \
                             dataframe['purchase_count_scaler'] * w3
    return weighted_sorting_score


df['weighted_sorting_score'] = weighted_sorting_score(df)

df.sort_values('weighted_sorting_score', ascending=False)

############################################
# Sorting by Bayesian Average Rating Score
############################################

# Sorting Products According to Distribution of 5 Star Rating / Sorting Products with 5 Star Rated

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df['bar_score'] = df.apply(lambda x: bayesian_average_rating(x[['1_point', '2_point', '3_point', '4_point', '5_point']]), axis=1)

df.sort_values('weighted_sorting_score', ascending=False)
df.sort_values('bar_score', ascending=False)

df[df['course_name'].index.isin([1, 5])] # df.iloc[[1, 5]]
df[df['course_name'].index.isin([1, 5])].sort_values('bar_score', ascending=False)

############################################
# Hybrid Sorting: BAR Score + Other Factors
############################################

def hybrid_sorting_score(dataframe, bar_w=.6, wss_w=.4):
    bar_score = df.apply(lambda x: bayesian_average_rating(x[['1_point', '2_point', '3_point', '4_point', '5_point']]), axis=1)
    wss_score = weighted_sorting_score(dataframe)
    hybrid_sorting_score = bar_score * bar_w + wss_score * wss_w
    return hybrid_sorting_score

df['hybrid_sorting_score'] = hybrid_sorting_score(df)

df.sort_values('hybrid_sorting_score', ascending=False)
