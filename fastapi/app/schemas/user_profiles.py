from pydantic import BaseModel, EmailStr
from typing import List, Optional

class UserProfileBase(BaseModel):
    age: Optional[str]
    experience: Optional[str]
    interest: Optional[List[str]]
    sources: Optional[str]
    risk: Optional[str]
    language: Optional[str]
    email: EmailStr

class UserProfileCreate(UserProfileBase):
    pass

class UserProfileOut(UserProfileBase):
    id: int

    class Config:
        from_attributes = True