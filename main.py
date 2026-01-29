import asyncio

from fastapi import FastAPI, Request
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from database.session import engine
from database.base import Base
from routers import auth, portfolio, prefer, recommend
import uvicorn

from services.lstmProcess import lstm_stock_predict
from services.stockDataService import StockDataService

app = FastAPI(title="A-Share Reco Backend (FastAPI)")
# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 配置模板
templates = Jinja2Templates(directory="templates")

# 自动建表（生产建议用 Alembic）
Base.metadata.create_all(bind=engine)

app.include_router(auth.router)
app.include_router(portfolio.router)
app.include_router(prefer.router)
app.include_router(recommend.router)


# 添加路由
@app.get("/", response_class=HTMLResponse)
async def portfolio_dashboard(request: Request):
    return templates.TemplateResponse("portfolio.html", {"request": request})



if __name__ == "__main__":
    # print("--- Starting FastAPI(port 8000) ---")
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    stocks = StockDataService()
    market_data = stocks.market_all_data([], [])
    # 获取研究员观点
    # DataProcess().researcher_bear(market_data)
    # DataProcess().researcher_bull(market_data)
    # asyncio.run(lstm_stock_predict('600519'))

    # prices_df = stocks.history_price('600519'    300628, '2025-01-01', '2026-01-01')
    # print(market_data)


