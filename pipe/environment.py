import redisai as rai
from mlflow.tracking import MlflowClient
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()


def mlflow_c():

    client = MlflowClient()
    return client


def redis_r():
    redisai_client = rai.Client(host="localhost", port=6379)
    return redisai_client


# client = MlflowClient()
# redisai_client = rai.Client(host="localhost", port=6379)

mlflow.set_tracking_uri(os.getenv("MLFLOW_URI"))
mlflow.set_experiment("Default")
