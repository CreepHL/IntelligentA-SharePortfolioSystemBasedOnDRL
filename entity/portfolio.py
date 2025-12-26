from datetime import datetime
from sqlalchemy import (
    Integer, String, Float, ForeignKey, DateTime, Text, JSON, UniqueConstraint
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base
from entity.holding import Holding
from entity.user import User


class Portfolio(Base):
    __tablename__ = "portfolios"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    # name: Mapped[str] = mapped_column(String(64), default="default")
    holdings_data: Mapped[list] = mapped_column(JSON, default=list)
    cash: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="portfolios")
    holdings: Mapped["Holding"] = relationship(back_populates="portfolio", cascade="all, delete-orphan")