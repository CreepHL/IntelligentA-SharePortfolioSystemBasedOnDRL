import secrets
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings.
    """

    #: The name of the application.
    app_name: str = "FastAPI"

    #: The version of the application.
    app_version: str = "0.1.0"

    #: The database URL.
    database_url: str = "mysql+pymysql://root@127.0.0.1:3306/stu"

    #: The secret key used for JWT.
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))

    #: The algorithm used for JWT.
    algorithm: str = "HS256"

    #: The access token expiration time.


@lru_cache
def get_settings() -> Settings:
    """
    Get the application settings.
    """
    return Settings()