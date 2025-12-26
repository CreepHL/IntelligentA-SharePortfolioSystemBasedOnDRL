from datetime import datetime, timedelta, timezone
from jose import jwt
from passlib.context import CryptContext
from .config import get_settings

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
settings = get_settings()

def hash_password(p: str) -> str:
    return pwd_ctx.hash(p)

def verify_password(p: str, hp: str) -> bool:
    return pwd_ctx.verify(p, hp)

def create_access_token(sub: str, minutes: int = None) -> str:
    exp = datetime.now(tz=timezone.utc) + timedelta(
        minutes=minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    to_encode = {"sub": sub, "exp": exp}
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
