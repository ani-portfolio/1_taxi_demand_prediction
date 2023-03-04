# set current working directory to root
import os
os.chdir('..')

import zipfile
from datetime import datetime, timedelta
import pytz

import requests
import pandas as pd
import numpy as np

# plotting libraries
import geopandas as gpd
import pydeck as pdk
import streamlit as st

from src.inference import get_model_predictions, load_batch_of_features_from_store, load_model_from_registry
from src.paths import DATA_DIR
from src.plot import plot_one_sample

st.set_page_config(layout="wide")

# get current data, convert to eastern timezone and pd.datetime
current_date = pd.to_datetime(datetime.utcnow()).floor('H')
current_date = current_date.tz_localize('UTC').tz_convert('US/Eastern').replace(tzinfo=None)

current_date_title = pd.to_datetime(datetime.now(pytz.timezone('US/Eastern'))).floor('H')
current_date_title = current_date_title.strftime('%B %d %Y, %H:%M')
# convert current date to 12 hour format
current_date_title = datetime.strptime(current_date_title, '%B %d %Y, %H:%M')
current_date_title = current_date_title.strftime('%B %d %Y, %I:%M %p')

# title
st.title(f'NYC Taxi Demand Prediction')
st.header(f'{current_date_title} EST')

progress_bar = st.sidebar.header('Work in Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 7


def load_shape_data():
    URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
    response = requests.get(URL)
    path = DATA_DIR / 'taxi_zones.zip'
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception('Could not download taxi_zones.zip')
    
    # extract zip file
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR / 'taxi_zones')
    
    return gpd.read_file(DATA_DIR / 'taxi_zones' / 'taxi_zones.shp').to_crs('EPSG:4326')

with st.spinner(text='Downloading shape file to plot taxis zones'):
    geo_df = load_shape_data()
    st.sidebar.write('Shape file downloaded')
    progress_bar.progress(1/N_STEPS)

with st.spinner(text='Fetching inference data'):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write('Inference data fetched')
    progress_bar.progress(2/N_STEPS)
    print(f'{features}')

with st.spinner(text='Loading ML model from registry'):
    model = load_model_from_registry()
    st.sidebar.write('Model loaded from Registry')
    progress_bar.progress(3/N_STEPS)

with st.spinner(test='Computing predictions'):
    results = get_model_predictions(model, features)
    st.sidebar.write('Predictions computed')
    progress_bar.progress(4/N_STEPS)