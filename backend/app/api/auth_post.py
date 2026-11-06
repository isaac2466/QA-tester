import sqlite3

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from app.core.database import (
    create_session,
    create_user,
    delete_session,
    find_user_by_email,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(credentials: Credentials):
    email = str(credentials.email).lower()
    if find_user_by_email(email):
        raise HTTPException(status_code=409, detail="Ese correo ya está registrado.")
    try:
        user = create_user(email, credentials.password)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Ese correo ya está registrado.")
    token, expires_at = create_session(user["id"])
    return {"token": token, "expires_at": expires_at, "user": user}


@router.post("/login")
def login(credentials: Credentials):
    email = str(credentials.email).lower()
    user = find_user_by_email(email)
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Correo o contraseña incorrectos.")
    token, expires_at = create_session(user["id"])
    return {
        "token": token,
        "expires_at": expires_at,
        "user": {"id": user["id"], "email": user["email"], "created_at": user["created_at"]},
    }


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(authorization: str | None = Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        delete_session(authorization[7:])
