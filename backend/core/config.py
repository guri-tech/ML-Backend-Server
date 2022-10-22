from functools import lru_cache
from base import settings


@lru_cache
def get_settings():
    return settings
