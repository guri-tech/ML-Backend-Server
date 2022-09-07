import inspect
from typing import Type

from fastapi import Form
from pydantic import BaseModel
from pydantic.fields import ModelField


def as_form(cls: Type[BaseModel]):
    new_parameters = []

    for field_name, model_field in cls.__fields__.items():
        model_field: ModelField

        new_parameters.append(
            inspect.Parameter(
                model_field.alias,
                inspect.Parameter.POSITIONAL_ONLY,
                default=Form(...)
                if not model_field.required
                else Form(model_field.default),
                annotation=model_field.outer_type_,
            )
        )

    async def as_form_func(**data):
        return cls(**data)

    sig = inspect.signature(as_form_func)
    sig = sig.replace(parameters=new_parameters)
    as_form_func.__signature__ = sig
    setattr(cls, "as_form", as_form_func)
    return cls


@as_form
class HotelForm(BaseModel):
    hotel: str = None
    market_segment: str = None
    customer_type: str = None
    distribution_channel: str = None
    is_repeated_guest: str = None
    reserved_room_type: str = None
    assigned_room_type: str = None
    meal: str = None
    lead_time: int = None
    days_in_waiting_list: int = None
    arrival_date_week_number: int = None
    stays_in_weekend_nights: int = None
    previous_cancellations: int = None
    booking_changes: int = None
    required_car_parking_spaces: int = None
    total_of_special_requests: int = None
    adults: int = None
    children: int = None
    babies: int = None
    adr: int = None
