from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from services.auth import create_access_token
from pydantic import BaseModel

router = APIRouter()

fake_user = {"username": "admin", "password": "admin"}

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

@router.post("/auth/token", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != fake_user["username"] or form_data.password != fake_user["password"]:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    token = create_access_token(data={"sub": form_data.username})
    return {"access_token": token, "token_type": "bearer"}