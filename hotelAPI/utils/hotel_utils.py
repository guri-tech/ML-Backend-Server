import mlflow.pyfunc

def load_mlflow_hotel():

    hotel_clf = mlflow.pyfunc.load_model(
    model_uri="s3://mlflow/1/177778c21ec54229bf3845720960bfc6/artifacts/sklearn.pipeline"
    )
    return hotel_clf

    
