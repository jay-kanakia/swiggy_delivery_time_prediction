import mlflow
import pytest
import dagshub
import json

from mlflow import MlflowClient
from pathlib import Path

def load_model_information(file_path:Path):
    with open(file_path,'r') as file:
        run_info = json.load(file)

    return run_info

# set model name
run_info = load_model_information('run_information.json')

# fetching model name
model_name = run_info['model_name']

# dagshub initialization
dagshub.init(repo_owner='jay-kanakia', repo_name='swiggy_delivery_time_prediction', mlflow=True)
# set tracking uri
mlflow.set_tracking_uri('https://dagshub.com/jay-kanakia/swiggy_delivery_time_prediction.mlflow')

client = MlflowClient()

@pytest.mark.parametrize(argnames="model_name,stage",argvalues=[(model_name,"Staging")])
def test_load_model_from_registry(model_name,stage):
    latest_version = client.get_latest_versions(name=model_name,stages=[stage])
    if latest_version:
        latest_version = latest_version[0].version
    else:
        None

    assert latest_version is not None, f"No model at {stage} stage"

    # load the model
    model_path = f"models:/{model_name}/{stage}"

    # load the latest model from model registry
    model = mlflow.sklearn.load_model(model_path)

    assert model is not None, "Failed to load model from registry"
    print(f"The {model_name} model with version {latest_version} was loaded successfully")

