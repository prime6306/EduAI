"""
auth/schemas.py – Pydantic models for auth endpoints.
"""
from typing import Literal
from pydantic import BaseModel, EmailStr, field_validator


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: Literal["student", "teacher"] = "student"
    branch: str = "ECE"
    year: str = "3rd year"
    student_id: str = ""

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserOut(BaseModel):
    id: str
    name: str
    email: str
    role: str
    branch: str
    year: str
    student_id: str
