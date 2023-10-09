import json
import logging
import os
import math
import pandas as pd
from flask import Flask, request
from clients.ai_platform import AIPlatformClient

app = Flask(__name__)

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
ai_platform_model_name = os.getenv("GCP_AI_PLATFORM_MODEL_NAME")
ai_platform_model_version = os.getenv("GCP_AI_PLATFORM_MODEL_VERSION")

ai_platform_client = AIPlatformClient(project_id, ai_platform_model_name, ai_platform_model_version)


def haversine_distance(origin, destination):
    """
    Calculate the spherical distance from coordinates

    :param origin: tuple (lat, lng)
    :param destination: tuple (lat, lng)
    :return: Distance in km
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


def get_dist(target_coor, trip):
    dropoff_coor = (trip['dropoff_latitude'], trip['dropoff_longitude'])
    distance = round(haversine_distance(target_coor, dropoff_coor), 2)
    return distance

def cal_dist(df):
    jfk_coor = (40.6413, -73.7781)
    lga_coor = (40.7769, -73.8740)
    ewr_coor = (40.6895, -74.1745)
    tsq_coor = (40.7580, -73.9855)
    cpk_coor = (40.7812, -73.9665)
    lib_coor = (40.6892, -74.0445)
    gct_coor = (40.7527, -73.9772)
    met_coor = (40.7794, -73.9632)
    wtc_coor = (40.7126, -74.0099)
    df['jfk'] = df.apply(lambda x: get_dist(jfk_coor, x), axis = 1)
    df['lga'] = df.apply(lambda x: get_dist(lga_coor, x), axis = 1)
    df['ewr'] = df.apply(lambda x: get_dist(ewr_coor, x), axis = 1)
    df['tsq'] = df.apply(lambda x: get_dist(tsq_coor, x), axis = 1)
    df['cpk'] = df.apply(lambda x: get_dist(cpk_coor, x), axis = 1)
    df['lib'] = df.apply(lambda x: get_dist(lib_coor, x), axis = 1)
    df['gct'] = df.apply(lambda x: get_dist(gct_coor, x), axis = 1)
    df['met'] = df.apply(lambda x: get_dist(met_coor, x), axis = 1)
    df['wtc'] = df.apply(lambda x: get_dist(wtc_coor, x), axis = 1)
    return df

def optimize_floats(df):
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df


def optimize_ints(df):
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df

def optimize(df):
    return optimize_floats(optimize_ints(df))


def process_test_data(df):
    """
    TODO: Implement this method.
    
    You should NOT drop any rows.

    :param raw_df: the DataFrame of the raw test data
    :return: a DataFrame with the predictors created
    """
    df['year'] = pd.DatetimeIndex(df['pickup_datetime']).year
    df['month'] = pd.DatetimeIndex(df['pickup_datetime']).month
    df['hour'] = pd.DatetimeIndex(df['pickup_datetime']).hour
    df['weekday'] = pd.DatetimeIndex(df['pickup_datetime']).weekday
    df = optimize(df)
    df = cal_dist(df)
    df = df.drop(['pickup_datetime'], axis=1)
    return df


@app.route('/')
def index():
    return "Hello"


@app.route('/predict', methods=['POST'])
def predict():
    raw_data_df = pd.read_json(request.data.decode('utf-8'),
                               convert_dates=["pickup_datetime"])
    predictors_df = process_test_data(raw_data_df)
    return json.dumps(ai_platform_client.predict(predictors_df.values.tolist()))


@app.route('/farePrediction', methods=['POST'])
def fare_prediction():
    pass


@app.route('/speechToText', methods=['POST'])
def speech_to_text():
    pass


@app.route('/textToSpeech', methods=['GET'])
def text_to_speech():
    pass


@app.route('/farePredictionVision', methods=['POST'])
def fare_prediction_vision():
    pass


@app.route('/namedEntities', methods=['GET'])
def named_entities():
    pass


@app.route('/directions', methods=['GET'])
def directions():
    pass


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run()
