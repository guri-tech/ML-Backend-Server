from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np
import pandas as pd
from typing import Optional
app = FastAPI()

@app.get("/")
def root():
    return "호텔 예약 취소 여부 예측 분류 모델을 이용한 예측 API 생성 "

class IndependentVariable(BaseModel):
     adr: Optional[float] = 30.
     adults: Optional[int] = 0
     arrival_date_day_of_month: Optional[int] = 0
     arrival_date_month: Optional[str] = 'July'
     arrival_date_week_number: Optional[int] = 0
     arrival_date_year: Optional[int] = 2015
     assigned_room_type: Optional[str] = 'A'
     babies: Optional[int] = 0
     booking_changes: Optional[int] = 0
     children: Optional[float] = 0.
     customer_type: Optional[str] = 'Transient'
     days_in_waiting_list: Optional[int] = 0
     deposit_type: Optional[str] = 'No Deposit'
     distribution_channel: Optional[str] = 'Direct'
     hotel: Optional[str] = 'Resort Hotel'
     is_repeated_guest: Optional[int] = 0
     lead_time: Optional[int] = 0
     market_segment: Optional[str] = 'Direct'
     meal: Optional[str] = 'BB'
     previous_bookings_not_canceled: Optional[int] = 0
     previous_cancellations: Optional[int] = 0
     required_car_parking_spaces: Optional[int] = 0
     reserved_room_type: Optional[str] = 'A'
     stays_in_week_nights: Optional[int] = 0
     stays_in_weekend_nights: Optional[int] = 0
     total_of_special_requests: Optional[int] = 0

@app.post("/predict")
def predict(preidict_id: IndependentVariable):
    hotel_clf = load('./xgb.pkl')
    new_data = pd.DataFrame(data =
                            [[
                                preidict_id.adr,
                                preidict_id.adults,
                                preidict_id.arrival_date_day_of_month,
                                preidict_id.arrival_date_month,
                                preidict_id.arrival_date_week_number,
                                preidict_id.arrival_date_year,
                                preidict_id.assigned_room_type,
                                preidict_id.babies,
                                preidict_id.booking_changes,
                                preidict_id.children,
                                preidict_id.customer_type,
                                preidict_id.days_in_waiting_list,
                                preidict_id.deposit_type,
                                preidict_id.distribution_channel,
                                preidict_id.hotel,
                                preidict_id.is_repeated_guest,
                                preidict_id.lead_time,
                                preidict_id.market_segment,
                                preidict_id.meal,
                                preidict_id.previous_bookings_not_canceled,
                                preidict_id.previous_cancellations,
                                preidict_id.required_car_parking_spaces,
                                preidict_id.reserved_room_type,
                                preidict_id.stays_in_week_nights,
                                preidict_id.stays_in_weekend_nights,
                                preidict_id.total_of_special_requests
                            ]],

                            columns=
                            [
                                'adr',
                                'adults',
                                'arrival_date_day_of_month',
                                'arrival_date_month',
                                'arrival_date_week_number',
                                'arrival_date_year',
                                'assigned_room_type',
                                'babies',
                                'booking_changes',
                                'children',
                                'customer_type',
                                'days_in_waiting_list',
                                'deposit_type',
                                'distribution_channel',
                                'hotel',
                                'is_repeated_guest',
                                'lead_time',
                                'market_segment',
                                'meal',
                                'previous_bookings_not_canceled',
                                'previous_cancellations',
                                'required_car_parking_spaces',
                                'reserved_room_type',
                                'stays_in_week_nights',
                                'stays_in_weekend_nights',
                                'total_of_special_requests'
                            ]
    )
    return hotel_clf.predict(new_data).tolist()

