from fastapi import FastAPI
from hotelAPI.routers import hotel_router

app = FastAPI()
@app.get("/")
def root():
    return "호텔 예약 취소 여부 예측 분류 모델을 이용한 예측 API 생성 "

app.include_router(hotel_router.router)
