"""
auth/utils.py – Password hashing + JWT creation/verification.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from backend.config import get_settings
import bcrypt

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")




def hash_password(password: str) -> str:
    # Truncate to 72 bytes (bcrypt's limit) and encode
    password_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8")[:72], hashed.encode("utf-8"))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    s = get_settings()
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=s.jwt_expire_minutes)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str) -> Optional[dict]:
    s = get_settings()
    try:
        payload = jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
        return payload
    except JWTError:
        return None
