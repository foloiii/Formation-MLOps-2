import os
import datetime as datetime

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
GENERATED_DATA_FOLDER = os.path.join(DATA_FOLDER, "generated_data")
PREDICTIONS_FOLDER = os.path.join(DATA_FOLDER, 'predictions')

TRAIN_DATA_PATH = os.path.join(DATA_FOLDER, 'la-haute-borne-data-2017-2020.csv')
# TODO change, should point to generator
GENERATED_DATA_PATH = os.path.join(DATA_FOLDER, 'la-haute-borne-data-2017-2020.csv')

FEATURES_PATH = os.path.join(DATA_FOLDER, 'prepared_features.parquet')

MODEL_REGISTRY_FOLDER = os.path.join(PROJECT_FOLDER, 'models')
MODEL_PATH = os.path.join(MODEL_REGISTRY_FOLDER, 'model.joblib')  # Default value

def get_model_path(model_name):
    model_name_with_tstp = concat(str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"), '_model.joblib'))
    return os.path.join(MODEL_REGISTRY_FOLDER, model_name, '.joblib')  # Default value
