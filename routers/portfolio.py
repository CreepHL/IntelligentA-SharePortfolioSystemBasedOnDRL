from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from entity.holding import Holding
from entity.portfolio import Portfolio
from entity.user import User
from database.session import get_db
from routers.auth import get_current_user

from schemas import PortfolioIn, PortfolioOut, HoldingIn, HoldingOut

from datetime import datetime

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

@router.post("", response_model=PortfolioOut)
def create_portfolio(payload: PortfolioIn, db: Session = Depends(get_db),
                     user: User = Depends(get_current_user)):
    pf = Portfolio(user_id=user.id, name=payload.name, cash=payload.cash)
    db.add(pf); db.commit(); db.refresh(pf)
    return PortfolioOut(id=pf.id, name=pf.name, cash=pf.cash, holdings=[], updated_at=pf.updated_at)

# @router.get("", response_model=list[PortfolioOut])
# def list_portfolios(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
#     pfs = db.query(Portfolio).filter(Portfolio.user_id == user.id).all()
#     out = []
#     for pf in pfs:
#         out.append(PortfolioOut(
#             id=pf.id, name=pf.name, cash=pf.cash,
#             holdings=[HoldingOut(id=h.id, symbol=h.symbol, shares=h.shares, avg_cost=h.avg_cost, industry=h.industry)
#                       for h in pf.holdings],
#             updated_at=pf.updated_at
#         ))
#     return out

@router.post("/{pid}/holding", response_model=HoldingOut)
def upsert_holding(pid: int, payload: HoldingIn, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)):
    pf = db.get(Portfolio, pid)
    if not pf or pf.user_id != user.id:
        raise HTTPException(404, "portfolio not found")
    h = db.query(Holding).filter(Holding.portfolio_id==pid, Holding.symbol==payload.symbol).first()
    if h:
        h.shares, h.avg_cost, h.industry = payload.shares, payload.avg_cost, payload.industry
    else:
        h = Holding(portfolio_id=pid, **payload.model_dump())
        db.add(h)
    pf.updated_at = datetime.utcnow()
    db.commit(); db.refresh(h)
    return HoldingOut(id=h.id, **payload.model_dump())
