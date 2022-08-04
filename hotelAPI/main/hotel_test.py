from hotelAPI.main.hotelmain import app
from fastapi.testclient import TestClient



client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "호텔 예약 취소 여부 예측 분류 모델을 이용한 예측 API 생성 "
