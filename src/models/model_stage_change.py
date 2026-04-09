from pathlib import Path
from mlflow import MlflowClient

import mlflow
import dagshub
import logging
import json

logger = logging.getLogger('model_registration')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# dagshub initialization
dagshub.init(repo_owner='jay-kanakia', repo_name='swiggy_delivery_time_prediction', mlflow=True)

# set tracking uri
mlflow.set_tracking_uri('https://dagshub.com/jay-kanakia/swiggy_delivery_time_prediction.mlflow')

def load_model_information(file_path):

    with open(file_path,'r') as file:
        run_info = json.load(file)

    return run_info

if __name__ == "__main__":

    root_path = Path(__file__).parent.parent.parent

    # run_info file path
    run_file_path = root_path/'run_information.json'

    # load model info
    run_info = load_model_information(run_file_path)

    # get the run id
    run_id = run_info['run_id']
    model_name  = run_info['model_name']
    model_registry_path = f"runs:/{run_id}/{model_name}"

    # print()
    # print('*'*100)

    # print(model_registry_path)
    # print("Artifacts in run:")
    # client = MlflowClient()
    # for f in client.list_artifacts(run_id):
    #     print(f.path)
    # for f in client.list_artifacts(run_id, path=None):
    #     print(f"path={f.path}, is_dir={f.is_dir}")
    # print('*'*100)
    # print()

    # register the model
    # model_version = mlflow.register_model(model_uri=model_registry_path,name=model_name)

    # # get the model version
    # registered_model_version = model_version.version
    # registered_model_name = model_version.name
    # logger.info(f"The latest model version in model registry is {registered_model_version}")

    # Get the latest version
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["None"])
    latest_version = versions[0].version

    # Transition to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Staging"
    )
    logger.info("Model pushed to Staging stage")