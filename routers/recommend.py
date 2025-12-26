from datetime import datetime

from fastapi import APIRouter
from sqlalchemy.orm import Session

from entity.portfolio import Portfolio
from entity.recommendation import Recommendation
from services.stockDataService import StockDataService
from tools.dataprocess import DataProcess

router = (APIRouter(prefix="/recommend", tags=["recommend"]))

@router.post("")
def simple_equal_weight(db: Session, portfolio_id: int) -> Recommendation:
    pf = db.get(Portfolio, portfolio_id)
    if not pf: raise ValueError("portfolio not found")
    #  LLM 情绪 + DRL调仓
    symbols = [h.symbol for h in pf.holdings]
    # if not symbols: return Recommendation(portfolio_id=portfolio_id, target_weights={})
    stocks = StockDataService()
    # datas, info = stocks.stock_data(symbols)
    # if not datas.empty:
    #     data, scaler, df_display = DataProcess.preprocess_data(datas)
    market_data = stocks.market_all_data(symbols, pf)
    # 获取研究员观点
    DataProcess().researcher_bear(market_data)
    DataProcess().researcher_bull(market_data)


    target = {}
    reco = Recommendation(portfolio_id=portfolio_id, target_weights=target,
                          rationale={"method": "equal_weight", "note": "stub"}, model_tag="baseline")
    db.add(reco); db.commit(); db.refresh(reco)
    return reco

