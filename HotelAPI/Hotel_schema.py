from pydantic import BaseModel
from typing import Optional

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