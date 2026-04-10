import mlflow
import dagshub
import json

from pathlib import Path
from mlflow.client import MlflowClient

def load_run_information(file_path:Path):

    with open(file_path,'r') as file:
        run_info = json.load(file)

    return run_info

# dagshub initialization
dagshub.init(repo_owner='jay-kanakia', repo_name='swiggy_delivery_time_prediction', mlflow=True)
# set tracking uri
mlflow.set_tracking_uri('https://dagshub.com/jay-kanakia/swiggy_delivery_time_prediction.mlflow')

# get model name
run_info = load_run_information("run_information.json")
model_name = run_info['model_name']
stage = "Staging"

client = MlflowClient()

latest_version = client.get_latest_versions(name=model_name,stages=[stage])

lates_model_version_staging = latest_version[0].version

promotion_stage = "Production"

client.transition_model_version_stage(
    name=model_name,
    version=lates_model_version_staging,
    stage=promotion_stage,
    archive_existing_versions=True
)