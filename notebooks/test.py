from mlflow.tracking import MlflowClient

client = MlflowClient()

run_id = "8de5ffe7e703447c861d39819049994b"

artifacts = client.list_artifacts(run_id)

for a in artifacts:
    print(a.path)