from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from jose import JWTError, jwt
import os

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "secret")
ALGORITHM = "HS256"

EXCLUDE_PATHS = [
"/auth/token",
    "/docs",
    "/openapi.json",
    ]

EXCLUDE_PREFIXES = [
    "/economic_index",
    "/market_price",
    "/stock_data",
    "/sentiment",
    "/recommend",
]

def should_skip_auth(path: str) -> bool:
    return (
        path in EXCLUDE_PATHS
        or any(path.startswith(prefix) for prefix in EXCLUDE_PREFIXES)
        or path.startswith("/static")
    )

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if should_skip_auth(path):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid Authorization header"})

        token = auth_header.replace("Bearer ", "")
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            request.state.user = payload.get("sub")
        except JWTError:
            return JSONResponse(status_code=401, content={"detail": "Invalid token"})

        return await call_next(request)