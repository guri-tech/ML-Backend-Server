import os
import mlflow
import utils.environment as environment

client = environment.mlflow_c()


def predict_preprocessor():
    model_name = "RandomForestClassifier"
    preprocessor = mlflow.sklearn.load_model(f"models:/preprocessor/Production")

    # # get model metrics
    run_id = client.get_latest_versions(model_name, ["Production"])[0].run_id
    metrics = client.get_run(run_id).data.metrics

    return preprocessor, metrics