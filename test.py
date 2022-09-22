import json
import pytest

from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root():
    print('test1')
    async with AsyncClient(base_url="http://127.0.0.1:8001/docs") as ac:
        response = await ac.get("/")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_read_item():
    print('test2')
    async with AsyncClient(base_url="http://127.0.0.1:8001/docs") as ac:
        response = await ac.post("/predict")
        assert response.status_code == 200
        assert response.json() == {
            'days_in_waiting_list': 1,
            'booking_changes': 1,
            'total_of_special_requests': 1,
            'lead_time': 1,
            'previous_cancellations': 1,
            'previous_bookings_not_canceled': 1,
            'market_segment': 'Direct',
            'deposit_type': 'No Deposit',
            'hotel': 'Resort Hotel',
        }

@pytest.mark.asyncio
async def test_create_item():
    async with AsyncClient(base_url="http://127.0.0.1:8001/docs") as ac:
        response = await ac.post(
            "/predict",
            content=json.dumps(
                {
            'days_in_waiting_list': 1,
            'booking_changes': 1,
            'total_of_special_requests': 1,
            'lead_time': 1,
            'previous_cancellations': 1,
            'previous_bookings_not_canceled': 1,
            'market_segment': 'Direct',
            'deposit_type': 'No Deposit',
            'hotel': 'Resort Hotel',
        }
            ),
        )
        assert response.status_code == 200
        assert response.json() == {
            'days_in_waiting_list': 1,
            'booking_changes': 1,
            'total_of_special_requests': 1,
            'lead_time': 1,
            'previous_cancellations': 1,
            'previous_bookings_not_canceled': 1,
            'market_segment': 'Direct',
            'deposit_type': 'No Deposit',
            'hotel': 'Resort Hotel',
        }