import redisai as rai
import mlflow
from dotenv import load_dotenv
import os

load_dotenv()

redisai_client = rai.Client(host="localhost", port=6379)
mlflow.set_tracking_uri(os.getenv("URI"))
mlflow.set_experiment("Default")
