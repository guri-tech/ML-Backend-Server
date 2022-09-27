import redisai as rai
from mlflow.tracking import MlflowClient
import mlflow
import os


def mlflow_c():

    client = MlflowClient()
    return client

def redis_r():
    redisai_client = rai.Client(host="localhost", port=6379)
    return redisai_client

mlflow.set_tracking_uri(os.getenv("URI"))
mlflow.set_experiment("Default")