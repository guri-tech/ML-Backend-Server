from fastapi import APIRouter

from utils.hotel_utils import load_mlflow_hotel
from utils.dataframe import hotel_df
from schemas.hotel_schema import hotel_features

import redisai as rai
import numpy as np

router = APIRouter()

@router.post("/predict")

def predict(hotelvariable: hotel_features):
    hotel_clf,preprocessor = load_mlflow_hotel()
    redisai_client = rai.Client(host = 'localhost', port = 6379)
    df = hotel_df(hotelvariable)

    redisai_client.tensorset('hotel_input',preprocessor.transform(df).astype(np.float32))
    redisai_client.expire('hotel_input',60)
    model_name = redisai_client.get('new_model_name')
    redisai_client.modelexecute(key = model_name, inputs = ['hotel_input'],
                                outputs = ['output_tensor_class','output_tensor_proba'])


    return redisai_client.tensorget('output_tensor_class').tolist()

