import sys, os
import pandas as pd
import numpy as np
from uuid import uuid4
from models.schemas import HotelForm
import mlflow
from fastapi import FastAPI, APIRouter, Request, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn

sys.path.append(os.pardir)
from backend.utils.util import predict_preprocessor
import backend.utils.environment as environment

app = FastAPI()


@app.on_event("startup")
def startup_event():

    train_models = {}

    train_models["redisai_client"] = environment.redis_r()
    train_models["model"] = mlflow.sklearn.load_model(
        f"models:/RandomForestClassifier/Production"
    )
    preprocessor, metrics = predict_preprocessor()
    train_models["metrics"] = metrics
    train_models["preprocessor"] = preprocessor

    return train_models


train_models = startup_event()

router = APIRouter()


@router.post("/predict", response_class=HTMLResponse)
def addData(
    request: Request,
    form_data: HotelForm = Depends(HotelForm.as_form),
    templates=Jinja2Templates(directory="../frontend/templates"),
):

    id = str(uuid4())
    data = form_data.dict()
    df = pd.DataFrame([data])

    train_models["redisai_client"].tensorset(
        f"{id}_hotel_input",
        train_models["preprocessor"].transform(df).astype(np.float32),
    )
    train_models["redisai_client"].expire(f"{id}_hotel_input", 60)

    model_name = train_models["redisai_client"].get("new_model_name")
    train_models["redisai_client"].modelexecute(
        key=model_name,
        inputs=[f"{id}_hotel_input"],
        outputs=[f"{id}_output_tensor_class", f"{id}_output_tensor_proba"],
    )
    pred = train_models["redisai_client"].tensorget(f"{id}_output_tensor_class")

    result = {"predict": pred}
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "form_data": data,
            "result": result,
            "metrics": train_models["metrics"],
        },
    )
