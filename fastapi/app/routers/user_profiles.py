from fastapi import APIRouter, Depends, HTTPException
from schemas.user_profiles import UserProfileCreate, UserProfileOut
from sqlalchemy.orm import Session
from crud import user_profiles
from database import get_db

router = APIRouter(prefix="/profiles", tags=["User Profiles"])

@router.post("/", response_model=UserProfileOut, status_code=201)
def create_profile(profile: UserProfileCreate, db: Session = Depends(get_db)):
    existing = user_profiles.get_profile_by_email(db, profile.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    return user_profiles.create_user_profile(db, profile)

@router.get("/", response_model=list[UserProfileOut])
def list_profiles(db: Session = Depends(get_db)):
    return user_profiles.get_all_profiles(db)

@router.get("/{email}", response_model=UserProfileOut)
def get_profile(email: str, db: Session = Depends(get_db)):
    profile = user_profiles.get_profile_by_email(db, email)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile