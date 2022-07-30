from fastapi import APIRouter
import numpy as np
from model import model
from IrisVariable import IrisVariable
router = APIRouter()

@router.post("/predict")
def predict(prediction: IrisVariable):
    iris_clf = model()

    return iris_clf.predict(np.array([[prediction.petal_length,
                                       prediction.petal_width,
                                       prediction.fetal_length,
                                       prediction.fetal_width]])).tolist()
