import sys, os
import pandas as pd
import numpy as np
from uuid import uuid4
from schemas import HotelForm

from fastapi import FastAPI,APIRouter, Request, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

sys.path.append(os.pardir)
from utils.utils import predict_preprocessor
import environment

redisai_client = environment.redis_r()
client = environment.mlflow_c()
preprocessors = {}
app = FastAPI()

@app.on_event('startup')
def startup_event():
    print('test')
    global preprocessor, metrics
    preprocessor, metrics = predict_preprocessor()
    return preprocessor, metrics
    
preprocessor, metrics = startup_event()

# router
router = APIRouter()
templates = Jinja2Templates(directory="templates")
id = str(uuid4())


@router.post("/predict", response_class=HTMLResponse)
def addData(request: Request, form_data: HotelForm = Depends(HotelForm.as_form)):

    data = dict(form_data)
    df = pd.DataFrame([data])

    redisai_client.tensorset(
        f"{id}_hotel_input", preprocessor.transform(df).astype(np.float32)
    )
    redisai_client.expire(f"{id}_hotel_input", 60)

    model_name = redisai_client.get("new_model_name")
    redisai_client.modelexecute(
        key=model_name,
        inputs=[f"{id}_hotel_input"],
        outputs=[f"{id}_output_tensor_class", f"{id}_output_tensor_proba"],
    )
    pred = redisai_client.tensorget(f"{id}_output_tensor_class")

    result = {"predict": pred}
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "form_data": data, "result": result, "metrics": metrics},
    )

@router.post("/predict2", response_class=HTMLResponse)
def addData(request: Request, form_data: HotelForm = Depends(HotelForm.as_form)):

    data = dict(form_data)
    df = pd.DataFrame([data])
    model_name = "RandomForestClassifier"
    model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    preprocessor, metrics = predict_preprocessor()
    df_new = preprocessor.transform(df)
    pred = model.predict(df_new)

    result = {"predict": pred}
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "form_data": data, "result": result, "metrics": metrics},
    )
