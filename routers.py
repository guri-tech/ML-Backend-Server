import sys, os
import pandas as pd
import redisai as rai
import numpy as np
from uuid import uuid4
from schemas import HotelForm
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

sys.path.append(os.pardir)
from utils import predict_preprocessor
import environment

# router
router = APIRouter()
templates = Jinja2Templates(directory="templates")
redisai_client = rai.Client(host="localhost", port=6379)
id = str(uuid4())


@router.post("/predict", response_class=HTMLResponse)
def addData(request: Request, form_data: HotelForm = Depends(HotelForm.as_form)):

    data = dict(form_data)
    df = pd.DataFrame([data])
    preprocessor, metrics = predict_preprocessor(df)

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
