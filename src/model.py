import pandas as pd
import lightgbm as lgb

from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from src.data import download_and_load_nyc_taxi_zone_data


# function that averages rides from previous 7, 14, 21, 28 days
def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    '''Adds a feature column calculated by averaging rides from
    - previous 7 days
    - previous 14 days
    - previous 21 days
    - previous 28 days
    '''

    X_ = X.copy()
    X_['average_rides_last_4_weeks'] = X_[[f'rides_previous_{7*24}_hour', f'rides_previous_{14*24}_hour', f'rides_previous_{21*24}_hour', f'rides_previous_{28*24}_hour']].mean(axis=1)

    return X_

# convert function to sklearn transformer
add_feature_average_rides_last_4_weeks = FunctionTransformer(average_rides_last_4_weeks, validate=False)


def extract_temporal_features(X: pd.DataFrame) -> pd.DataFrame:
    '''Ex_tracts temporal features from the datetime index_'''

    X_ = X.copy()
    X_['hour'] = X_.pickup_hour.dt.hour
    X_['day'] = X_.pickup_hour.dt.day
    X_['month'] = X_.pickup_hour.dt.month
    X_['weekday'] = X_.pickup_hour.dt.weekday
    X_['weekend'] = X_['weekday'].isin([5, 6]).astype(int)
    X_.drop('pickup_hour', axis=1, inplace=True)
    
    return X_

# convert function to sklearn transformer
add_temporal_features = FunctionTransformer(extract_temporal_features, validate=False)


# function to extract latitude & longitude from new york city location id data
def extract_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    '''Extracts latitude and longitude from the_geom column'''
    
    df_ = df.copy()
    # extract latitude and longitude data from the_geom column
    df_['the_geom'] = df_['the_geom'].str.replace('MULTIPOLYGON \\(\\(\\(', '', regex=True).str.replace('\\)\\)\\)', '', regex=True).str.replace(',', '', regex=True).str.replace(' ', ',', regex=True).str.replace('\\)\\)', '', regex=True).str.replace('\\(\\(', '', regex=True)

    # convert string to float
    df_['the_geom'] = df_['the_geom'].apply(lambda x: [float(i) for i in x.split(',')])

    # convert list of floats to list of tuples
    df_['the_geom'] = df_['the_geom'].apply(lambda x: list(zip(x[::2], x[1::2])))

    # take average of tuples to get center of zone
    df_['the_geom'] = df_['the_geom'].apply(lambda x: (sum([i[0] for i in x])/len(x), sum([i[1] for i in x])/len(x)))

    df_['latitude'] = df_['the_geom'].apply(lambda x: x[0])
    df_['longitude'] = df_['the_geom'].apply(lambda x: x[1])
    df_.drop(['the_geom', 'OBJECTID', 'Shape_Leng', 'Shape_Area', 'zone', 'borough'], axis=1, inplace=True)
    
    return df_


# convert function to sklearn transformer
class add_latitude_and_longitude_features(BaseEstimator, TransformerMixin):
    """add latitude and longitude features to the dataframe"""

    def __init__(self, df_lat_lon: pd.DataFrame):
        self.df_lat_lon = df_lat_lon

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X_ = X.copy()
        
        # merge latitude and longitude data with X
        X_ = X_.merge(self.df_lat_lon, how='left', left_on='pickup_location_id', right_on='LocationID')

        # drop LocationID column
        X_.drop('LocationID', axis=1, inplace=True)

        # rename columns
        X_.rename(columns={'latitude': 'pickup_latitude', 'longitude': 'pickup_longitude'}, inplace=True)

        return X_


# download and load new york city location id data
nyc_zone_data = download_and_load_nyc_taxi_zone_data()
nyc_zone_data = extract_lat_lon(nyc_zone_data)


def get_pipeline(**hyperparameters) -> Pipeline:
    '''Returns a pipeline with the following steps:
    - add_feature_average_rides_last_4_weeks
    - add_temporal_features
    - add_latitude_and_longitude_features
    - lgb.LGBMRegressor
    '''

    pipeline = make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        add_latitude_and_longitude_features(nyc_zone_data),
        lgb.LGBMRegressor(**hyperparameters, verbose_eval=None)
    )

    return pipeline 

