import pandas as pd
from schemas.hotel_schema import hotel_features


def hotel_df(hotelvariable=hotel_features):

    hotel_df = pd.DataFrame(
        data=[
            [
                hotelvariable.hotel,
                hotelvariable.market_segment,
                hotelvariable.customer_type,
                hotelvariable.distribution_channel,
                hotelvariable.is_repeated_guest,
                hotelvariable.reserved_room_type,
                hotelvariable.assigned_room_type,
                hotelvariable.meal,
                hotelvariable.lead_time,
                hotelvariable.days_in_waiting_list,
                hotelvariable.arrival_date_week_number,
                hotelvariable.stays_in_weekend_nights,
                hotelvariable.previous_cancellations,
                hotelvariable.previous_bookings_not_canceled,
                hotelvariable.booking_changes,
                hotelvariable.required_car_parking_spaces,
                hotelvariable.total_of_special_requests,
                hotelvariable.adults,
                hotelvariable.kids,
                hotelvariable.adr,
            ]
        ],
        columns=[
            "adr",
            "adults",
            "arrival_date_day_of_month",
            "arrival_date_month",
            "arrival_date_week_number",
            "arrival_date_year",
            "assigned_room_type",
            "babies",
            "booking_changes",
            "children",
            "customer_type",
            "days_in_waiting_list",
            "deposit_type",
            "distribution_channel",
            "hotel",
            "is_repeated_guest",
            "lead_time",
            "market_segment",
            "meal",
            "previous_bookings_not_canceled",
            "previous_cancellations",
            "required_car_parking_spaces",
            "reserved_room_type",
            "stays_in_week_nights",
            "stays_in_weekend_nights",
            "total_of_special_requests",
        ],
    )

    return hotel_df
