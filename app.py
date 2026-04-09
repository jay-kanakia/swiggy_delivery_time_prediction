import unicorn
import mlflow
import joblib
import json
import dagshub

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn import set_config
from pathlib import Path
from mlflow import MlflowClient

class Data(BaseModel):

    age: float
    ratings: float
    weather: str
    traffic: str
    vehicle_condition: int
    type_of_order: str
    type_of_vehicle: str
    multiple_deliveries: float
    festival: str
    city_type: str
    is_weekend: int
    pickup_time_minutes: float
    order_time_of_day: str
    distance: float
    distance_type: str

def load_model_infomation(file_path:Path):

    with open(file_path) as file:
        run_info = json.load(file)

    return run_info

def load_model(model_path:Path):

    model = joblib.load(model_path)

    return model

# load model name from run_info
run_info = load_model_infomation("run_information.json")
model_name = run_info['model_name']

# stage of the model
stage = 'Staging'

# get the latest model version
client = MlflowClient()
latest_model_version = client.get_latest_versions(name=model_name,stages=[stage])

# load model path
model_path = f"models:/{model_name}/{stage}"

# load the latest model from model registry
model = mlflow.sklearn.load_model(model_path)

# load the preprocessor
model = load_model