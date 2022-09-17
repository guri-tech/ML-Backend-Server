import os
import mlflow
from mlflow.tracking import MlflowClient
import environment


def predict_preprocessor(df):
    model_name = "xgboost"
    preprocessor = mlflow.sklearn.load_model(f"models:/preprocessor/Production")

    # # get model metrics
    client = MlflowClient()
    run_id = client.get_latest_versions(model_name, ["Production"])[0].run_id
    metrics = client.get_run(run_id).data.metrics

    return preprocessor, metrics
