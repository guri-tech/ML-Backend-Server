import sys, os
import pandas as pd
import numpy as np
from uuid import uuid4
from schemas import HotelForm
import mlflow
from fastapi import FastAPI, APIRouter, Request, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn

sys.path.append(os.pardir)
from utils.utils import predict_preprocessor
import environment as environment

app = FastAPI()


@app.on_event("startup")
def startup_event():

    train_models = {}

    train_models['redisai_client'] = environment.redis_r()
    train_models['model'] = mlflow.sklearn.load_model(f"models:/RandomForestClassifier/Production")
    preprocessor, metrics = predict_preprocessor()
    train_models['metrics'] = metrics
    train_models['preprocessor'] =preprocessor 

    return train_models

train_models = startup_event()

router = APIRouter()

@router.post("/predict", response_class=HTMLResponse)
def addData(request: Request,
            form_data: HotelForm = Depends(HotelForm.as_form),
            templates = Jinja2Templates(directory="templates")):

    id = str(uuid4())
    data = form_data.dict()
    df = pd.DataFrame([data])

    train_models['redisai_client'].tensorset(
        f"{id}_hotel_input", train_models['preprocessor'].transform(df).astype(np.float32)
    )
    train_models['redisai_client'].expire(f"{id}_hotel_input", 60)

    model_name = train_models['redisai_client'].get("new_model_name")
    train_models['redisai_client'].modelexecute(
        key=model_name,
        inputs=[f"{id}_hotel_input"],
        outputs=[f"{id}_output_tensor_class", f"{id}_output_tensor_proba"],
    )
    pred = train_models['redisai_client'].tensorget(f"{id}_output_tensor_class")

    result = {"predict": pred}
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "form_data": data, "result": result, "metrics": train_models['metrics']},
    )


@router.post("/predict2", response_class=HTMLResponse)
def addData(request: Request,
             form_data: HotelForm = Depends(HotelForm.as_form),
             templates = Jinja2Templates(directory="templates")):

    data = form_data.dict()
    df = pd.DataFrame([data])
    df_new = train_models['preprocessor'].transform(df)
    pred = train_models['model'].predict(df_new)

    result = {"predict": pred}
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "form_data": data, "result": result, "metrics": train_models['metrics']},
    )
@router.post("/predict3", response_class=HTMLResponse)
def addData(request: Request,
             form_data: HotelForm = Depends(HotelForm.as_form),
             templates = Jinja2Templates(directory="templates")):

    id = str(uuid4())
    data = form_data.dict()
    df = pd.DataFrame([data])

    train_models['redisai_client'].expire(f"{id}_hotel_input", 60)
    if train_models['redisai_client'].exists('model_name_2') == True:
        model_name = train_models['redisai_client'].get("model_name_2")
        train_models['redisai_client'].tensorset(f"{id}_hotel_input", train_models['preprocessor'].transform(df).astype(np.float32))
        train_models['redisai_client'].modelexecute(
            key=model_name,
            inputs=[f"{id}_hotel_input"],
            outputs=[f"{id}_output_tensor_class", f"{id}_output_tensor_proba"],
        )
    else:
        train_model = mlflow.sklearn.load_model(f"models:/RandomForestClassifier/Production")
        train_models['redisai_client'].set("model_name_2", str(train_model.__hash__()))

        initial_inputs = [("float_input", FloatTensorType([None, 19]))]
        onnx_model = convert_sklearn(
            train_model, initial_types=initial_inputs, target_opset=12
        )

        convert_model_name = train_models['redisai_client'].get("model_name_2")
        train_models['redisai_client'].modelstore(
            key=convert_model_name,
            backend="onnx",
            device="cpu",
            data=onnx_model.SerializeToString(),
        )
        train_models['redisai_client'].tensorset(f"{id}_hotel_input", train_models['preprocessor'].transform(df).astype(np.float32))
        model_name = train_models['redisai_client'].get("model_name_2")
        train_models['redisai_client'].modelexecute(
            key=model_name,
            inputs=[f"{id}_hotel_input"],
            outputs=[f"{id}_output_tensor_class", f"{id}_output_tensor_proba"],
        )


    pred = train_models['redisai_client'].tensorget(f"{id}_output_tensor_class")

    result = {"predict": pred}
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "form_data": data, "result": result, "metrics": train_models['metrics']},
    )