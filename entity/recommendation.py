from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel
from sqlalchemy import (
    Integer, String, Float, ForeignKey, DateTime, Text, JSON, UniqueConstraint
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base


class Recommendation(BaseModel):
    # __tablename__ = "recommendations"
    # id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"))
    # ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    # target_weights: Mapped[dict] = mapped_column(JSON)  # {'600519.SH':0.1,...}
    # exp_return: Mapped[float] = mapped_column(Float, nullable=True)
    # risk: Mapped[float] = mapped_column(Float, nullable=True)
    # rationale: Mapped[dict] = mapped_column(JSON, nullable=True)
    # model_tag: Mapped[str] = mapped_column(String(64), nullable=True)
    id: int
    portfolio_id: int
    ts: datetime
    target_weights: Dict[str, float]  # e.g., {"600519.SH": 0.1}
    exp_return: Optional[float] = None
    risk: Optional[float] = None
    rationale: Optional[Dict] = None
    model_tag: Optional[str] = None
    class Config:
        from_attributes = True  # Pydantic v2 兼容