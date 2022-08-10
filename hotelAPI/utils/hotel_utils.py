import mlflow.pyfunc


def load_mlflow_hotel():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("HOTEL-EXPERIMENT")
    model_name = "sklearn.pipeline"

    hotel_clf = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

    return hotel_clf