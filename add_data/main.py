import uvicorn
from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
import pandas as pd
import sys, os

sys.path.append(os.pardir)
from schemas import HotelForm
from predict import predict


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def addData(request: Request, form_data: HotelForm = Depends(HotelForm.as_form)):
    data = dict(form_data)
    df = pd.DataFrame([data])
    pred, proba = predict(df)
    result = {"predict": pred, "proba": round(proba * 100, 1)}
    return templates.TemplateResponse(
        "result.html", {"request": request, "form_data": data, "result": result}
    )


if __name__ == "__main__":
    uvicorn.run("main:app", port=8003, reload=True)
