import sys, os
import pandas as pd
import numpy as np
from uuid import uuid4
from schemas import HotelForm
from fastapi import FastAPI,APIRouter, Request, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from datetime import datetime,timedelta,timezone
from time import time


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
    preprocessor, metrics = predict_preprocessor()
    return preprocessor, metrics
# @app.on_event('startup')
# def startup_event():
#     print('test')
#     global preprocessor, metrics
#     preprocessor, metrics = predict_preprocessor()

# router
router = APIRouter()
templates = Jinja2Templates(directory="templates")
id = str(uuid4())

# def time_check():
#     while True:
#         for _,_, save_time in preprocessors.itmes():
#             if save_time <= datetime.now(tz = timezone.utc) - timedelta(hours = 1):
#                 preprocessors.pop()
        # time.sleep(60)
# background task를 활용한 방법
# def time_check(task_name:str):
#     -, -, save_time = preprocessors[task_name]
#     if save_time <= datetime.now(tz = timezone.utc) - timedelta(hours = 1):
#         preprocessors.pop()
        # time.sleep(60)
  
#backgroundtasks test
@router.post("/predict/{task_name}", response_class=HTMLResponse)
def addData(request: Request,task_name : str,background_tasks : BackgroundTasks, form_data: HotelForm = Depends(HotelForm.as_form)):
    background_tasks.add_task(time_check,task_name)

    data = dict(form_data)
    df = pd.DataFrame([data])
    if 'hotel' in preprocessors.keys() :
        preprocessor,metrics,_ = preprocessors['hotel'] 
    else :
        preprocessor, metrics = predict_preprocessor()
        preprocessors[f'hotel-{task_name}'] = [preprocessor,metrics, datetime.now(tz = timezone.utc)]

@router.post("/predict", response_class=HTMLResponse)
def addData(request: Request, form_data: HotelForm = Depends(HotelForm.as_form)):

    data = dict(form_data)
    df = pd.DataFrame([data])
    # if 'hotel' in preprocessors.keys() :
    #     preprocessor,metrics,_ = preprocessors['hotel'] 
    # else :
    #     preprocessor, metrics = predict_preprocessor()
    #     preprocessors['hotel'] = [preprocessor,metrics, datetime.now(tz = timezone.utc)]

    preprocessor,metrics=startup_event()

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
