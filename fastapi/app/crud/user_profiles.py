from models.user_profiles import UserProfile 
from sqlalchemy.orm import Session
from schemas import user_profiles

def create_user_profile(db: Session, user: user_profiles.UserProfileCreate):
    db_user = UserProfile(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_all_profiles(db: Session):
    return db.query(UserProfile).all()

def get_profile_by_email(db: Session, email: str):
    return db.query(UserProfile).filter(UserProfile.email == email).first()