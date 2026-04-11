import uvicorn
import mlflow
import joblib
import json
import dagshub
import os

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn import set_config
from pathlib import Path
from mlflow import MlflowClient
from scripts.data_clean_utils import perform_data_cleaning
from dotenv import load_dotenv

load_dotenv()

dagshub_token=os.getenv('DAGSHUB_TOKEN')
dagshub.auth.add_app_token(dagshub_token)

# dagshub initialization
dagshub.init(repo_owner='jay-kanakia', repo_name='swiggy_delivery_time_prediction', mlflow=True)
# set tracking uri
mlflow.set_tracking_uri('https://dagshub.com/jay-kanakia/swiggy_delivery_time_prediction.mlflow')

class Data(BaseModel):

    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str

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
stage = 'Production'

# get the latest model version
client = MlflowClient()
latest_model_version = client.get_latest_versions(name=model_name,stages=[stage])

# load model path
model_path = f"models:/{model_name}/{stage}"

# load the latest model from model registry
model = mlflow.sklearn.load_model(model_path)

# load the preprocessor
preprocessor_path = "models/preprocessor.joblib"
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
        'ID': data.ID,
        'Delivery_person_ID': data.Delivery_person_ID,
        'Delivery_person_Age': data.Delivery_person_Age,
        'Delivery_person_Ratings': data.Delivery_person_Ratings,
        'Restaurant_latitude': data.Restaurant_latitude,
        'Restaurant_longitude': data.Restaurant_longitude,
        'Delivery_location_latitude': data.Delivery_location_latitude,
        'Delivery_location_longitude': data.Delivery_location_longitude,
        'Order_Date': data.Order_Date,
        'Time_Orderd': data.Time_Orderd,
        'Time_Order_picked': data.Time_Order_picked,
        'Weatherconditions': data.Weatherconditions,
        'Road_traffic_density': data.Road_traffic_density,
        'Vehicle_condition': data.Vehicle_condition,
        'Type_of_order': data.Type_of_order,
        'Type_of_vehicle': data.Type_of_vehicle,
        'multiple_deliveries': data.multiple_deliveries,
        'Festival': data.Festival,
        'City': data.City
        },
        index=[0]
    )

    # clean the raw input data
    cleaned_data = perform_data_cleaning(pred_data)

    # get the prediction
    prediction = model_pipe.predict(cleaned_data)[0]

    return prediction

if __name__ == "__main__":
    uvicorn.run(app="app:app",host="0.0.0.0",port=8000)