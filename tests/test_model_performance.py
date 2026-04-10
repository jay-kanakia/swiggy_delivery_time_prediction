import mlflow
import dagshub
import pytest
import json
import joblib

import pandas as pd

from pathlib import Path
from sklearn.metrics import mean_absolute_error
from pathlib import Path
from sklearn.pipeline  import Pipeline

def load_model_information(file_path:Path):

    with open(file_path,'r') as file:
        run_info = json.load(file)

    return run_info

def load_model(model_path : Path):
    model = joblib.load(model_path)

    return model

# dagshub initialization
dagshub.init(repo_owner='jay-kanakia', repo_name='swiggy_delivery_time_prediction', mlflow=True)
# set tracking uri
mlflow.set_tracking_uri('https://dagshub.com/jay-kanakia/swiggy_delivery_time_prediction.mlflow')


run_info = load_model_information("run_information.json")
model_name = run_info['model_name']
stage = "Staging"

model_path = f"models:/{model_name}/{stage}"

model = mlflow.sklearn.load_model(model_path)

root_path = Path(__file__).parent.parent

preprocessor_path = root_path/"models"/"preprocessor.joblib"
preprocessor = load_model(preprocessor_path)

model_pipe = Pipeline(
    [
        ('preprocessor',preprocessor),
        ('model',model)
    ]
)

test_data_path = root_path/"data"/"external"/"test.csv"

@pytest.mark.parametrize(argnames="model_pipe,test_data_path,threshold_error",argvalues=[(model_pipe,test_data_path,5)])
def test_model_performance(model_pipe,test_data_path,threshold_error):

    df = pd.read_csv(test_data_path)
    print(df.shape)

    df.dropna(inplace=True)

    X = df.drop(columns=['time_taken'])
    y = df['time_taken']

    y_pred = model_pipe.predict(X)
    print(y_pred)

    mean_error =mean_absolute_error(y_pred,y)

    assert mean_error <= threshold_error, f"The model does not pass the performance test"

    print(f"The {model_name} model passed the performance test")


