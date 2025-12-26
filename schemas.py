from datetime import datetime
from pydantic import BaseModel, EmailStr, Field
from typing import Dict, List, Optional

# Auth
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    risk_aversion: float = 0.5

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

# Portfolio & holdings
class HoldingIn(BaseModel):
    symbol: str
    shares: float
    avg_cost: float
    industry: Optional[str] = None

class HoldingOut(HoldingIn):
    id: int

class PortfolioIn(BaseModel):
    name: str = "default"
    cash: float = 0.0

class PortfolioOut(PortfolioIn):
    id: int
    holdings: List[HoldingOut] = []
    updated_at: datetime

# Preference
class PreferenceIn(BaseModel):
    industry_weights: Optional[Dict[str, float]] = None
    blacklist: Optional[Dict[str, List[str]]] = None
    turnover_limit: float = 0.3
    max_pos: float = 0.15

class PreferenceOut(PreferenceIn):
    user_id: int

# Recommendation
class RecommendReq(BaseModel):
    top_n: int = Field(10, ge=1, le=50)

class RecommendItem(BaseModel):
    symbol: str
    weight: float
    reason: str = None

class RecommendResp(BaseModel):
    portfolio_id: int
    ts: datetime
    items: List[RecommendItem]
    target_weights: Dict[str, float]
