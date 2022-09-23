import json
import pytest

from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_root():
    print('test1')
    async with AsyncClient(app = app, base_url="http://127.0.0.1:8000") as ac:
        response = await ac.get('/')
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_read_item():
    print('test2')

    async with AsyncClient(app = app,base_url="http://127.0.0.1:8000") as ac:
        response = await ac.get("/predict",params ={
            'days_in_waiting_list': 1,
            'booking_changes': 1,
            'total_of_special_requests': 1,
            'lead_time': 1,
            'previous_cancellations': 1,
            'previous_bookings_not_canceled': 1,
            'market_segment': 'Direct',
            'deposit_type': 'No Deposit',
            'hotel': 'Resort Hotel',
        },
        )
        assert response.status_code == 200
        # assert response.json()에 작성해야 하는 결과 메세지는 html과 연동되어 있어서 확인하기 어려움.

