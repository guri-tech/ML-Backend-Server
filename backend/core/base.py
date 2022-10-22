from pydantic import (
    BaseSettings,
    AnyUrl,
)
from dotenv import load_dotenv

load_dotenv()


class AppSettings(BaseSettings):
    MLFLOW_URI: AnyUrl
    resdis_host: str
    redis_port: str

    class Config:
        env_file: ".env"


settings = AppSettings()
