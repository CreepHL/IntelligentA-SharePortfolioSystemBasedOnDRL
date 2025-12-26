from fastapi import FastAPI
from database.session import engine
from database.base import Base
from routers import auth, portfolio, prefer, recommend

app = FastAPI(title="A-Share Reco Backend (FastAPI)")

# 自动建表（生产建议用 Alembic）
Base.metadata.create_all(bind=engine)

app.include_router(auth.router)
app.include_router(portfolio.router)
app.include_router(prefer.router)
app.include_router(recommend.router)

@app.get("/health")
def health():
    return {"status": "ok"}

