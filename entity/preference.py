from datetime import datetime
from sqlalchemy import (
    Integer, String, Float, ForeignKey, DateTime, Text, JSON, UniqueConstraint
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base

class Preference(Base):
    __tablename__ = "preferences"
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)
    industry_weights: Mapped[dict] = mapped_column(JSON, nullable=True) # {'食品饮料':0.3,...}
    blacklist: Mapped[dict] = mapped_column(JSON, nullable=True)        # {'symbols':['ST*',...]}
    turnover_limit: Mapped[float] = mapped_column(Float, default=0.3)          # 最大换手（当期）
    max_pos: Mapped[float] = mapped_column(Float, default=0.15)                # 单票上限
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # user: Mapped[User] = relationship(back_populates="preference")