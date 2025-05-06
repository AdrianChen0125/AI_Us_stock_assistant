from fastapi import APIRouter, Depends, HTTPException
from schemas.user_profiles import UserProfileCreate, UserProfileOut
from sqlalchemy.ext.asyncio import AsyncSession
from crud import user_profiles
from async_db import get_db  

router = APIRouter(prefix="/profiles", tags=["User Profiles"])

@router.post("/", response_model=UserProfileOut, status_code=201)
async def create_profile(profile: UserProfileCreate, db: AsyncSession = Depends(get_db)):
    existing = await user_profiles.get_profile_by_email(db, profile.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    return await user_profiles.create_user_profile(db, profile)

@router.get("/", response_model=list[UserProfileOut])
async def list_profiles(db: AsyncSession = Depends(get_db)):
    return await user_profiles.get_all_profiles(db)

@router.get("/{email}", response_model=UserProfileOut)
async def get_profile(email: str, db: AsyncSession = Depends(get_db)):
    profile = await user_profiles.get_profile_by_email(db, email)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile