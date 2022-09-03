from pydantic import BaseModel
from typing import Optional

class hotel_features(BaseModel):
    hotel: Optional[str] = 'Resort Hotel'
    market_segment: Optional[str] = 'Direct'
    customer_type: Optional[str] = 'Transient'
    distribution_channel: Optional[str] = 'Direct'
    is_repeated_guest: Optional[int] = 0
    reserved_room_type: Optional[str] = 'A'
    assigned_room_type: Optional[str] = 'A'
    meal: Optional[str] = 'BB'
    lead_time: Optional[int] = 0
    days_in_waiting_list: Optional[int] = 0
    arrival_date_week_number: Optional[int] = 0
    stays_in_weekend_nights: Optional[int] = 0
    previous_cancellations: Optional[int] = 0
    previous_bookings_not_canceled: Optional[int] = 0
    booking_changes: Optional[int] = 0
    required_car_parking_spaces: Optional[int] = 0
    total_of_special_requests: Optional[int] = 0
    adults: Optional[int] = 0
    kids: Optional[int] = 0
    adr: Optional[float] = 30.
    

