from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from core.config import get_settings

settings = get_settings()

connect_args = {}
if settings.database_url.startswith("mysql"):
    connect_args = {
        "charset": "utf8mb4",
        "connect_timeout": 10,
        "autocommit": True
    }

engine = create_engine(settings.database_url, echo=False, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
