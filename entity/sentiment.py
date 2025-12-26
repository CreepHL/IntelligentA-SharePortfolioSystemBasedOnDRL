from datetime import datetime
from sqlalchemy import (
    Integer, String, Float, ForeignKey, DateTime, Text, JSON, UniqueConstraint
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base
from entity.portfolio import Portfolio


class Sentiment(Base):
    __tablename__ = "sentiments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    level: Mapped[str] = mapped_column(String(16))      # 'market'/'industry'/'stock'
    target: Mapped[str] = mapped_column(String(64))     # '市场' / 行业名 / 证券代码
    time: Mapped[datetime] = mapped_column(DateTime, index=True)
    score: Mapped[float] = mapped_column(Float)         # [-1,1]
    confidence: Mapped[float] = mapped_column(Float, default=0.7)
    summary: Mapped[str] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=True)