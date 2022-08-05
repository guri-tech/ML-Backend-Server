import mlflow.pyfunc

def load_mlflow_hotel():
    model_name = "sklearn.tree._classes"
    model_version = 1

    hotel_clf = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
    )
    return hotel_clf