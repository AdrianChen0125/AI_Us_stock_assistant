# crud/user_profiles.py
from models.user_profiles import UserProfile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from schemas import user_profiles

# Create a new user profile
async def create_user_profile(db: AsyncSession, user: user_profiles.UserProfileCreate):
    db_user = UserProfile(**user.dict())
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

# Get all user profiles
async def get_all_profiles(db: AsyncSession):
    result = await db.execute(select(UserProfile))
    return result.scalars().all()

# Get a profile by email
async def get_profile_by_email(db: AsyncSession, email: str):
    stmt = select(UserProfile).where(UserProfile.email == email)
    result = await db.execute(stmt)
    return result.scalars().first()