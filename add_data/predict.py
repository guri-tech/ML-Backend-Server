import os
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.getenv("URI"))
mlflow.set_experiment("Default")


def predict_preprocessor(df):
    model_name = "xgboost"

    preprocessor = mlflow.sklearn.load_model(f"models:/preprocessor/Production")
    model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")

    # get model metrics
    client = MlflowClient()
    run_id = client.get_latest_versions(model_name, ["Production"])[0].run_id
    metrics = client.get_run(run_id).data.metrics

    df_new = preprocessor.transform(df)
    pred = model.predict(df_new)
    proba = model.predict_proba(df_new)

    proba = [round(y, 2) for x in proba for y in x]  # 2 dim -> 1 dim
    return pred, proba[1], metrics


def predict_pipeline(df):
    model_name = "xgboost_pipeline"
    model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    print(model.predict_proba)

    # get model metrics
    client = MlflowClient()
    run_id = client.get_latest_versions(model_name, ["Production"])[0].run_id
    metrics = client.get_run(run_id).data.metrics

    pred = model.predict(df)
    proba = model.predict_proba(df)
    proba = [round(y, 2) for x in proba for y in x]  # 2 dim -> 1 dim

    return pred, proba[1], metrics
