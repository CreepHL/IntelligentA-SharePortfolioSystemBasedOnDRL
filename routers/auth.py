from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import jwt, JWTError
from database.session import get_db
from entity.user import User
from schemas import Token, UserCreate

from core.security import hash_password, verify_password, create_access_token
from core.config import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
settings = get_settings()

@router.post("/register", response_model=Token)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(400, "email already registered")
    user = User(email=payload.email, password_hash=hash_password(payload.password),
                risk_aversion=payload.risk_aversion)
    db.add(user); db.commit(); db.refresh(user)
    token = create_access_token(str(user.id))
    return Token(access_token=token)

@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form.username).first()
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=401, detail="invalid credentials")
    token = create_access_token(str(user.id))
    return Token(access_token=token)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        uid = int(payload.get("sub"))
    except JWTError:
        raise HTTPException(401, "invalid token")
    user = db.get(User, uid)
    if not user:
        raise HTTPException(401, "user not found")
    return user
