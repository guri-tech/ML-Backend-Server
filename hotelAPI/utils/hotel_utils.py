import mlflow.sklearn


def load_mlflow_hotel():

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("HOTEL-EXPERIMENT")

    model_name = "xgboost"
    # preprocessor = mlflow.sklearn.load_model(f"models:/preprocessor/Production")
    preprocessor = mlflow.sklearn.load_model(
        f"s3://mlflow/1/0c3dc7ce6b064a03b29032b737a87711/artifacts/preprocessor"
    )
    hotel_clf = mlflow.sklearn.load_model(f"models:/{model_name}/Production")

    return hotel_clf, preprocessor
