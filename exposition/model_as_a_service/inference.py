from flask import Flask, jsonify
import requests
import pandas as pd

from config import GENERATED_DATA_PATH, MODEL_PATH, PREDICTIONS_FOLDER
from formation_indus_ds_avancee.feature_engineering import prepare_features
from formation_indus_ds_avancee.train_and_predict import predict

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({
        "status": "ok"
    })

@app.route('/predict')
def predict_endpoint():
    #separator = request.args.get('separator')      # default: ";"
    nr = requests.args.get('nrows')          # default: 10
    # training_mode = request.args.get('training_mode')  # training_mode: False

    ## 1. Prepare features
    generated_features_df = pd.read_csv(GENERATED_DATA_PATH, sep=';', nrows=nr)
    prepared_features_df = prepare_features(generated_features_df, training_mode=False)

    ## 2. Predict 
    predictions = predict(prepared_features_df, MODEL_PATH)

    ## 3. Return prediction outputs
    prediction_json = jsonify(predictions)
    return prediction_json

# curl localhost:5000/predict?nrows=10
