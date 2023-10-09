from google.cloud import storage
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hypertune import HyperTune
import argparse
import os

# ==========================
# ==== Define Variables ====
# ==========================
# When dealing with a large dataset, it is practical to randomly sample
# a smaller proportion of the data to reduce the time and money cost per iteration.
#
# When you are testing, start with 0.2. You need to change it to 1.0 when you make submissions.
# TODO: Set SAMPLE_PROB to 1.0 when you make submissions
SAMPLE_PROB = 0.2   # Sample 20% of the whole dataset
random.seed(15619)  # Set the random seed to get deterministic sampling results

# TODO: Update the value using the ID of the GS bucket
# For example, if the GS path of the bucket is gs://my-bucket the OUTPUT_BUCKET_ID will be "my-bucket"
OUTPUT_BUCKET_ID = 'ml-fare-prediction-371004'

# DO NOT CHANGE IT
DATA_BUCKET_ID = 'cmucc-public'
# DO NOT CHANGE IT
TRAIN_FILE = 'dataset/nyc-taxi-fare/cc_nyc_fare_train_small.csv'


# =========================
# ==== Utility Methods ====
# =========================
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



# =====================================
# ==== Define data transformations ====
# =====================================

def process_train_data(df):
    """
    TODO: Copy your feature engineering code from task 1 here

    :param raw_df: the DataFrame of the raw training data
    :return:  a DataFrame with the predictors created
    """
        
    df.drop(df[df['pickup_longitude'] == 0].index, axis=0, inplace = True)
    df.drop(df[df['pickup_latitude'] == 0].index, axis=0, inplace = True)
    df.drop(df[df['dropoff_longitude'] == 0].index, axis=0, inplace = True)
    df.drop(df[df['dropoff_latitude'] == 0].index, axis=0, inplace = True)
    df.drop(df[df['passenger_count'] > 5].index, axis=0, inplace = True)
    df.drop(df[df['passenger_count'] == 0].index, axis=0, inplace = True)

    df['year'] = pd.DatetimeIndex(df['pickup_datetime']).year
    df['month'] = pd.DatetimeIndex(df['pickup_datetime']).month
    df['hour'] = pd.DatetimeIndex(df['pickup_datetime']).hour
    df['weekday'] = pd.DatetimeIndex(df['pickup_datetime']).weekday
    df.dropna(inplace=True)

    df.drop(df.index[(df.pickup_longitude < -75) | 
               (df.pickup_longitude > -72) | 
               (df.pickup_latitude < 40) | 
               (df.pickup_latitude > 42)],inplace=True)
    df.drop(df[df['fare_amount'] < 2.5].index, axis=0, inplace = True)
    df.drop(df[df['fare_amount'] > df['fare_amount'].quantile(.999)].index, axis=0, inplace = True)
    df = optimize(df)
    df = cal_dist(df)
    return df


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
    return df


if __name__ == '__main__':
    # ===========================================
    # ==== Download data from Google Storage ====
    # ===========================================
    print('Downloading data from google storage')
    print('Sampling {} of the full dataset'.format(SAMPLE_PROB))
    input_bucket = storage.Client().bucket(DATA_BUCKET_ID)
    output_bucket = storage.Client().bucket(OUTPUT_BUCKET_ID)
    input_bucket.blob(TRAIN_FILE).download_to_filename('train.csv')

    raw_train = pd.read_csv('train.csv', parse_dates=["pickup_datetime"],
                            skiprows=lambda i: i > 0 and random.random() > SAMPLE_PROB)

    print('Read data: {}'.format(raw_train.shape))

    # =============================
    # ==== Data Transformation ====
    # =============================
    df_train = process_train_data(raw_train)

    # Prepare feature matrix X and labels Y
    X = df_train.drop(['key', 'fare_amount', 'pickup_datetime'], axis=1)
    Y = df_train['fare_amount']
    X_train, X_eval, y_train, y_eval = train_test_split(X, Y, test_size=0.33)
    print('Shape of feature matrix: {}'.format(X_train.shape))

    # ======================================================================
    # ==== Improve model performance with hyperparameter tuning ============
    # ======================================================================
    # You are provided with the code that creates an argparse.ArgumentParser
    # to parse the command line arguments and pass these parameters to Google AI Platform
    # to be tuned by HyperTune.
    # TODO: Your task is to add at least 3 more arguments.
    # You need to update both the code below and config.yaml.

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',  # AI Platform passes this in by default
        required=True
    )

    # the 5 lines of code below parse the --max_depth option from the command line
    # and will convert the value into "args.max_depth"
    # "args.max_depth" will be passed to XGBoost training through the `params` variables
    # i.e., xgb.train(params, ...)
    #
    # the 5 lines match the following YAML entry in `config.yaml`:
    # - parameterName: max_depth
    #   type: INTEGER
    #   minValue: 4
    #   maxValue: 10
    # "- parameterName: max_depth" matches "--max_depth"
    # "type: INTEGER" matches "type=int""
    # "minValue: 4" and "maxValue: 10" match "default=6"
    parser.add_argument(
        '--max_depth',
        default=6,
        type=int
    )
    
    parser.add_argument(
        '--learning_rate',
        default=0.05,
        type=float
    )
    
    parser.add_argument(
        '--subsample',
        default=0.7,
        type=float
    )

    parser.add_argument(
        '--colsample_bytree',
        default=0.7,
        type=float
    )

    # TODO: Create more arguments here, similar to the "max_depth" example
    # parser.add_argument(
    #     '--param2',
    #     default=...,
    #     type=...
    # )

    args = parser.parse_args()
    params = {
        'max_depth': args.max_depth,
        # TODO: Add the new parameters to this params dict, e.g.,
        # 'param2': args.param2
    }

    """
    DO NOT CHANGE THE CODE BELOW
    """
    # ===============================================
    # ==== Evaluate performance against test set ====
    # ===============================================
    # Create DMatrix for XGBoost from DataFrames
    d_matrix_train = xgb.DMatrix(X_train, y_train)
    d_matrix_eval = xgb.DMatrix(X_eval)
    model = xgb.train(params, d_matrix_train)
    y_pred = model.predict(d_matrix_eval)
    rmse = math.sqrt(mean_squared_error(y_eval, y_pred))
    print('RMSE: {:.3f}'.format(rmse))

    # Return the score back to HyperTune to inform the next iteration
    # of hyperparameter search
    hpt = HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='nyc_fare',
        metric_value=rmse)

    # ============================================
    # ==== Upload the model to Google Storage ====
    # ============================================
    JOB_NAME = os.environ['CLOUD_ML_JOB_ID']
    TRIAL_ID = os.environ['CLOUD_ML_TRIAL_ID']
    model_name = 'model.bst'
    model.save_model(model_name)
    blob = output_bucket.blob('{}/{}_rmse{:.3f}_{}'.format(
        JOB_NAME,
        TRIAL_ID,
        rmse,
        model_name))
    blob.upload_from_filename(model_name)
