import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, HTTPException, Request

from app.core.config import AUTH_TOKEN_DAYS
from app.db import fetch_one, execute


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(days=AUTH_TOKEN_DAYS)
    execute(
        "INSERT INTO user_sessions (user_id, token_hash, expires_at) VALUES (%s, %s, %s)",
        (user_id, _token_hash(token), expires_at.replace(tzinfo=None)),
    )
    return token


def get_current_user(request: Request) -> dict:
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Debes iniciar sesión.")
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(401, "Token de sesión inválido.")
    user = fetch_one(
        """
        SELECT u.id, u.email
        FROM user_sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.token_hash = %s AND s.expires_at > UTC_TIMESTAMP()
        """,
        (_token_hash(token),),
    )
    if not user:
        raise HTTPException(401, "La sesión expiró. Inicia sesión de nuevo.")
    return user


CurrentUser = Depends(get_current_user)
