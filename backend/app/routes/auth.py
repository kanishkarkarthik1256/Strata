"""Authentication + user-management API (Phase 10).

POST /api/auth/register     — create an account (viewer/analyst by default;
                              elevated roles require an admin actor)
POST /api/auth/login        — credentials → bearer session token
POST /api/auth/logout       — revoke the current session
GET  /api/auth/me           — current user profile
GET  /api/auth/users        — list users (admin only)
POST /api/auth/users        — create a user with an explicit role (admin only)
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_session
from app.db.models import User
from app.logging_config import get_logger
from app.services.auth_service import (
    authenticate,
    create_user,
    get_current_user,
    get_optional_user,
    require_roles,
    revoke_token,
    user_public,
)
from app.services.metrics import metrics

log = get_logger("drone_recon.routes.auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=256)
    role: Optional[str] = None


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)


class CreateUserRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=256)
    role: str = "viewer"


@router.post("/register", status_code=201)
async def register(
    req: RegisterRequest,
    db: AsyncSession = Depends(get_session),
    actor: Optional[User] = Depends(get_optional_user),
) -> dict:
    """Self-registration. Elevated roles need an admin caller."""
    requested = (req.role or "analyst").lower()
    if requested in ("admin", "operator") and (actor is None or actor.role.lower() != "admin"):
        raise HTTPException(status_code=403, detail="Elevated roles require an admin actor")
    user = await create_user(db, req.email, req.password, role=requested, actor=actor)
    metrics.inc("drone_auth_events_total", {"event": "register"})
    return {"id": user.id, "email": user.email, "role": user.role}


@router.post("/login")
async def login(req: LoginRequest, db: AsyncSession = Depends(get_session)) -> dict:
    user, token = await authenticate(db, req.email, req.password)
    metrics.inc("drone_auth_events_total", {"event": "login"})
    return {"token": token, "user": user_public(user), "token_type": "bearer"}


@router.post("/logout")
async def logout(request: Request, db: AsyncSession = Depends(get_session)) -> dict:
    """Revoke the caller's current session token."""
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    revoked = await revoke_token(db, token) if token else False
    metrics.inc("drone_auth_events_total", {"event": "logout"})
    return {"revoked": revoked}


@router.get("/me")
async def me(user: User = Depends(get_current_user)) -> dict:
    return user_public(user)


@router.get("/users")
async def list_users(
    db: AsyncSession = Depends(get_session),
    _admin: User = Depends(require_roles("admin")),
) -> dict:
    rows = (await db.execute(select(User).order_by(User.created_at))).scalars().all()
    return {"users": [user_public(u) for u in rows], "count": len(rows)}


@router.post("/users", status_code=201)
async def create_user_route(
    req: CreateUserRequest,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_roles("admin")),
) -> dict:
    user = await create_user(db, req.email, req.password, role=req.role, actor=admin)
    return {"id": user.id, "email": user.email, "role": user.role}
