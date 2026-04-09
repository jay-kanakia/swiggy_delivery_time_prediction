import uvicorn
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

# dagshub initialization
dagshub.init(repo_owner='jay-kanakia', repo_name='swiggy_delivery_time_prediction', mlflow=True)
# set tracking uri
mlflow.set_tracking_uri('https://dagshub.com/jay-kanakia/swiggy_delivery_time_prediction.mlflow')
# set experiment

# get the latest model version
client = MlflowClient()
latest_model_version = client.get_latest_versions(name=model_name,stages=[stage])

# load model path
model_path = f"models:/{model_name}/{stage}"

# load the latest model from model registry
model = mlflow.sklearn.load_model(model_path)

# load the preprocessor
preprocessor_path = "models\preprocessor.joblib"
preprocessor = load_model(preprocessor_path)

# build the model pipeline
model_pipe = Pipeline(
    [
        ('preprocessor',preprocessor),
        ('regressor',model)
    ]
)

# create the app
app = FastAPI()

# create the home endpoint
@app.get(path='/')
def home():
    return "Welcome to the Swiggy Food Delivery Time Prediction App"

# create the predict endpoint
@app.post(path='/predict')
def do_prediction(data:Data):

    pred_data = pd.DataFrame(
        {
        'age': data.age,
        'ratings': data.ratings,
        'weather': data.weather,
        'traffic': data.traffic,
        'vehicle_condition': data.vehicle_condition,
        'type_of_order': data.type_of_order,
        'type_of_vehicle': data.type_of_vehicle,
        'multiple_deliveries': data.multiple_deliveries,
        'festival': data.festival,
        'city_type': data.city_type,
        'is_weekend': data.is_weekend,
        'pickup_time_minutes': data.pickup_time_minutes,
        'order_time_of_day': data.order_time_of_day,
        'distance': data.distance,
        'distance_type': data.distance_type
        },
        index=[0]
    )

    prediction = model_pipe.predict(pred_data)[0]

    return prediction

if __name__ == "__main__":
    uvicorn.run(app="app:app")