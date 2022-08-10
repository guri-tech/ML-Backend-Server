from fastapi import APIRouter

from utils.hotel_utils import load_mlflow_hotel
from utils.dataframe import hotel_df
from schemas.hotel_schema import hotel_features

router = APIRouter()


@router.post("/predict")
def predict(hotelvariable: hotel_features):

    hotel_clf = load_mlflow_hotel()

    df = hotel_df(hotelvariable)

    return hotel_clf.predict(df).tolist()
