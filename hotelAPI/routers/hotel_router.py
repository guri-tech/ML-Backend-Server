from fastapi import APIRouter

from utils.hotel_utils import load_hotel
from utils.dataframe import hotel_df
from schemas.hotel_schema import IndependentVariable
router = APIRouter()

@router.post("/predict")
def predict(hotelvariable:IndependentVariable):

    hotel_clf = load_hotel()
    df = hotel_df(hotelvariable)
    return hotel_clf.predict(df).tolist()
