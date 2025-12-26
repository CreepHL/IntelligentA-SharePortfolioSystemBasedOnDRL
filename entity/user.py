from datetime import datetime
from sqlalchemy import (
    Integer, String, Float, ForeignKey, DateTime, Text, JSON, UniqueConstraint
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    risk_aversion: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    portfolios: Mapped["Portfolio"] = relationship(back_populates="user", cascade="all, delete-orphan")
    preference: Mapped["Preference"] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")






