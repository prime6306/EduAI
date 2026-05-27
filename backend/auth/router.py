"""
auth/router.py – Register, Login, /me endpoints.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from backend.auth.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut
from backend.auth.utils import create_access_token, decode_token, hash_password, verify_password
from backend.db.mongodb import get_async_db

router = APIRouter(prefix="/auth", tags=["Auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    db = get_async_db()
    user = await db.users.find_one({"email": payload.get("sub")})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user["id"] = str(user["_id"])
    return user


@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    db = get_async_db()
    if await db.users.find_one({"email": req.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    user_doc = {
        "name": req.name,
        "email": req.email,
        "password": hash_password(req.password),
        "role": req.role,
        "branch": req.branch,
        "year": req.year,
        "student_id": req.student_id,
        "created_at": datetime.utcnow(),
        "quiz_scores": [],
        "total_attendance": 0,
    }
    result = await db.users.insert_one(user_doc)
    user_doc["id"] = str(result.inserted_id)

    token = create_access_token({"sub": req.email})
    return TokenResponse(
        access_token=token,
        user={k: v for k, v in user_doc.items() if k not in ("password", "_id")},
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    db = get_async_db()
    user = await db.users.find_one({"email": req.email})
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user["id"] = str(user["_id"])
    token = create_access_token({"sub": req.email})
    safe_user = {k: v for k, v in user.items() if k not in ("password", "_id")}
    return TokenResponse(access_token=token, user=safe_user)


@router.get("/me", response_model=UserOut)
async def me(current_user: dict = Depends(get_current_user)):
    return UserOut(
        id=str(current_user["_id"]),
        name=current_user["name"],
        email=current_user["email"],
        role=current_user["role"],
        branch=current_user.get("branch", ""),
        year=current_user.get("year", ""),
        student_id=current_user.get("student_id", ""),
    )
