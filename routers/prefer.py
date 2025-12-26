from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from entity.preference import Preference
from entity.user import User
from database.session import get_db
from routers.auth import get_current_user

from schemas import PreferenceIn, PreferenceOut


router = APIRouter(prefix="/preferences", tags=["preferences"])

@router.get("", response_model=PreferenceOut)
def get_pref(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    pref = db.get(Preference, user.id)
    if not pref:
        pref = Preference(user_id=user.id)
        db.add(pref); db.commit(); db.refresh(pref)
    return PreferenceOut(user_id=user.id, **{
        "industry_weights": pref.industry_weights,
        "blacklist": pref.blacklist,
        "turnover_limit": pref.turnover_limit,
        "max_pos": pref.max_pos
    })

@router.put("", response_model=PreferenceOut)
def put_pref(payload: PreferenceIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    pref = db.get(Preference, user.id) or Preference(user_id=user.id)
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(pref, k, v)
    db.add(pref); db.commit(); db.refresh(pref)
    return PreferenceOut(user_id=user.id, **payload.model_dump(exclude_unset=False))
