import sys,os
sys.path.append(os.pardir)
from fastapi import FastAPI
import routers.hotel_router
import uvicorn

app = FastAPI()
@app.get("/")
def root():
    return "호텔 예약 취소 여부 예측 분류 모델을 이용한 예측 API 생성 "

app.include_router(routers.hotel_router.router)

if __name__ == "__main__":
    uvicorn.run("hotelmain:app", port=8000, reload=True)