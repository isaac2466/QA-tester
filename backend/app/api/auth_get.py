from fastapi import APIRouter, Header, HTTPException

from app.core.database import get_user_by_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/me")
def current_user(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Debes iniciar sesión.")
    user = get_user_by_token(authorization[7:])
    if not user:
        raise HTTPException(status_code=401, detail="La sesión no es válida o expiró.")
    return {"id": user["id"], "email": user["email"], "created_at": user["created_at"]}
