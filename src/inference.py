from datetime import datetime, timedelta

import warnings
import tqdm
import hopsworks
from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
import src.config as config


def get_hopsworks_project() -> hopsworks.project.Project:
    """
    Get the Hopsworks project object
    """
    
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY,
    )


def get_feature_store() -> FeatureStore:
    """
    Get the feature store object
    """
    
    project = get_hopsworks_project()
    return project.get_feature_store()


def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    '''
    Get model predictions using model
    ''' 

    predictions = model.predict(features)

    results = pd.DataFrame()
    results['pickup_location_id'] = features['pickup_location_id'].values
    results['predicted_demand'] = predictions.round(0)

    return results


def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    '''
    Load a batch of features from the feature store
    '''
    
    feature_store = get_feature_store()
    n_features = config.N_FEATURES

    # get time series data from feature store
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=28)
    print(f'Fetching data from {fetch_data_from} to {fetch_data_to}')

    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION)
    
    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1)))
    
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
    
    # sort data by location and hour
    ts_data = ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'])
    print(f'{ts_data=}')

    # transpose ts data as a feature vector for each location
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    df_features = pd.DataFrame()
    for i, location_id in enumerate(ts_data['pickup_location_id'].unique()):
        data_i = ts_data[ts_data['pickup_location_id'] == location_id].reset_index(drop=True)
    
        df_features_i = pd.DataFrame()
        # take the last n_features rows and add them as features
        for i in range(n_features):
            df_features_i['pickup_hour'] = current_date
            df_features_i['pickup_location_id'] = data_i['pickup_location_id']
            df_features_i[f'rides_previous_{n_features-i}_hour'] = data_i['rides'].shift(-i)
    
        df_features = pd.concat([df_features, df_features_i])

    return df_features


def load_model_from_registry():
    '''
    Load model from the hopsworks model registry
    '''
    
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION)
    
    model_dir = model.download()
    model = joblib.load(Path(model_dir) / 'model.pkl')

    return model