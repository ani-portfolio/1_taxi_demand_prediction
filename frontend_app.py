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
st.title(f'NYC Taxi Demand Prediction - Top 10 Busiest Zones')
st.header(f'{current_date_title} EST\nDisclaimer: This app was created following a tutorial made by [Pau Labarto Bajo](https://datamachines.xyz/)')

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

with st.spinner(text='Downloading Shape File to Plot Taxi Zones (ETA 10 secs)'):
    geo_df = load_shape_data()
    st.sidebar.write('(Step 1/5) Shape File Downloaded')
    progress_bar.progress(1/N_STEPS)

with st.spinner(text='Fetching Inference Data (ETA 30 secs)'):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write('(Step 2/5) Inference Data Fetched')
    progress_bar.progress(2/N_STEPS)
    print(f'{features}')

with st.spinner(text='Loading ML Model from Registry (ETA 15 secs)'):
    model = load_model_from_registry()
    st.sidebar.write('(Step 3/5) Model Loaded from Registry')
    progress_bar.progress(3/N_STEPS)

with st.spinner(text='Computing Predictions (ETA 5 secs))'):
    results = get_model_predictions(model, features)
    st.sidebar.write('(Step 4/5) Predictions Computed')
    progress_bar.progress(4/N_STEPS)

# with st.spinner(text='Preparing Data for Plotting (ETA 15 secs)'):
    
#     # define function that changes colour intensity based on predictions
#     def get_color_intensity(value, minvalue, maxvalue, startcolour, stopcolour):
#         f = float(value - minvalue) / float(maxvalue - minvalue)
#         return tuple(f*(b-a)+a for (a, b) in zip(startcolour, stopcolour))
        
#     df = pd.merge(geo_df, results, right_on='pickup_location_id', left_on='LocationID', how='inner')
    
#     BLACK, GREEN = (0, 0, 0), (0, 255, 0)
#     df['color_scaling'] = df['predicted_demand']
#     max_pred, min_pred = df['color_scaling'].max(), df['color_scaling'].min()
#     df['fill_color'] = df['color_scaling'].apply(lambda x: get_color_intensity(x, min_pred, max_pred, BLACK, GREEN))
#     st.sidebar.write('Data Prepared')
#     progress_bar.progress(5/N_STEPS)

# with st.spinner(text='Generating NYC Map (ETA 3 mins)'):
#     INITIAL_VIEW_STATE = pdk.ViewState(
#         latitude=40.7831,
#         longitude=-73.9712,
#         zoom=11,
#         max_zoom=16,
#         pitch=45,
#         bearing=0
#     )

#     geojson = pdk.Layer(
#         "GeoJsonLayer",
#         df,
#         opacity=0.25,
#         stroked=False,
#         filled=True,
#         extruded=False,
#         wireframe=True,
#         get_elevation=10,
#         get_fill_color="fill_color",
#         get_line_color=[255, 255, 255],
#         auto_highlight=True,
#         pickable=True,
#     )

#     tooltip = {"html": "<b>Zone:</b> [{LocationID}]{zone} <br /> <b>Predicted rides:</b> {predicted_demand}"}

#     r = pdk.Deck(
#         layers=[geojson],
#         initial_view_state=INITIAL_VIEW_STATE,
#         tooltip=tooltip
#     )

#     st.pydeck_chart(r)
#     st.sidebar.write('Map Generated')
#     progress_bar.progress(6/N_STEPS)


with st.spinner(text="Plotting Time-Series for 10 Busiest Zones"):
   
    row_indices = np.argsort(results['predicted_demand'].values)[::-1]
    n_to_plot = 10

    # plot each time-series with the prediction
    for row_id in row_indices[:n_to_plot]:
        fig = plot_one_sample(
            features=features,
            targets=results['predicted_demand'],
            example_id=row_id,
            predictions=pd.Series(results['predicted_demand'])
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)
    st.sidebar.write('(Step 5/5) Time-Series Plotted for 10 Busiest Zones')
    progress_bar.progress(5/N_STEPS)