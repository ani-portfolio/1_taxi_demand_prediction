# set current working directory to root
import os
os.chdir(os.path.dirname(os.getcwd()))

from dotenv import load_dotenv

from src.paths import PARENT_DIR

# load key-value pairs from .env file
load_dotenv(PARENT_DIR /'.env')

HOPSWORKS_PROJECT_NAME = 'taxi_demand_1'
try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception('HOPSWORKS_API_KEY not found in .env file. Create a .env file in the root directory of the project and add HOPSWORKS_API_KEY=<your_api_key>')

FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = 'time_series_hourly_feature_view'
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "taxi_demand_forecaster_next_hour"
MODEL_VERSION = 1

N_FEATURES = 24 * 28