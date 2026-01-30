import secrets
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


# 股票筛选参数
STOCK_FILTER_CONFIG = {

    'max_pe_ratio': 50,

    'min_turnover_rate': 1.0,  # 最小换手率：1.0%

    'momentum_days': 20,

    'min_price': 1.0,

    'max_stocks': 10,  # 从5只增加到10只

    'min_strength_score': 40   # 从50降低到40

}

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