from datetime import datetime
from sqlalchemy import (
    Integer, String, Float, ForeignKey, DateTime, Text, JSON, UniqueConstraint
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base



class Holding(Base):
    __tablename__ = "holdings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)      # 例：'600519.SH'
    shares: Mapped[float] = mapped_column(Float, default=0.0)
    avg_cost: Mapped[float] = mapped_column(Float, default=0.0)
    industry: Mapped[str] = mapped_column(String(64), nullable=True)

    portfolio: Mapped['Portfolio'] = relationship(back_populates="holdings")
    __table_args__ = (UniqueConstraint("portfolio_id", "symbol", name="uix_portfolio_symbol"),)