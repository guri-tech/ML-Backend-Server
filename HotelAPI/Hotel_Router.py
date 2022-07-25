from fastapi import APIRouter
from Hotel_schema import IndependentVariable
from Hotel_utils import model
import pandas as pd

router = APIRouter()

@router.post("/predict")
def predict(hotelvariable: IndependentVariable):

    hotel_clf = model()
    new_data = pd.DataFrame(data=
                            [[
                                hotelvariable.adr,
                                hotelvariable.adults,
                                hotelvariable.arrival_date_day_of_month,
                                hotelvariable.arrival_date_month,
                                hotelvariable.arrival_date_week_number,
                                hotelvariable.arrival_date_year,
                                hotelvariable.assigned_room_type,
                                hotelvariable.babies,
                                hotelvariable.booking_changes,
                                hotelvariable.children,
                                hotelvariable.customer_type,
                                hotelvariable.days_in_waiting_list,
                                hotelvariable.deposit_type,
                                hotelvariable.distribution_channel,
                                hotelvariable.hotel,
                                hotelvariable.is_repeated_guest,
                                hotelvariable.lead_time,
                                hotelvariable.market_segment,
                                hotelvariable.meal,
                                hotelvariable.previous_bookings_not_canceled,
                                hotelvariable.previous_cancellations,
                                hotelvariable.required_car_parking_spaces,
                                hotelvariable.reserved_room_type,
                                hotelvariable.stays_in_week_nights,
                                hotelvariable.stays_in_weekend_nights,
                                hotelvariable.total_of_special_requests
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