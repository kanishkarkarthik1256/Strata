"""Authentication + authorization (Phase 10).

Password hashing uses PBKDF2-HMAC-SHA256 from the standard library (per-user
random salt, 200 000 iterations — OWASP-style for a 2026 deployment) and
session tokens are random 256-bit values stored only as SHA-256 digests.

Roles (ascending): viewer < analyst < operator < admin. Authorization is
enforced server-side through :func:`require_roles` used as a FastAPI
dependency on protected routes; when ``platform.auth_mode == \"required\"`` a
middleware additionally blocks unauthenticated access to every /api route not
on the public allowlist.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config.settings import settings
from app.db.engine import get_session
from app.db.models import AuthSession, User
from app.logging_config import get_logger

log = get_logger("drone_recon.services.auth")

#: Role hierarchy — higher index = more privilege.
ROLES = ("viewer", "analyst", "operator", "admin")

_PBKDF2_ITERATIONS = 200_000


# ---------------------------------------------------------------------------
# Password hashing (stdlib only)
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Return ``pbkdf2_sha256$iterations$salt_hex$digest_hex``."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time comparison of *password* against a stored hash."""
    try:
        algo, iterations, salt_hex, digest_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _utcnow() -> datetime:
    """Naive UTC now — SQLite returns naive datetimes, so comparisons must
    use naive UTC on both sides."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def role_index(role: str) -> int:
    try:
        return ROLES.index((role or "viewer").lower())
    except ValueError:
        return 0


def role_at_least(role: str, minimum: str) -> bool:
    return role_index(role) >= role_index(minimum)


# ---------------------------------------------------------------------------
# User / session operations (async, DB-backed)
# ---------------------------------------------------------------------------


async def create_user(
    db: AsyncSession,
    email: str,
    password: str,
    role: str = "viewer",
    actor: Optional[User] = None,
) -> User:
    """Create a user. Elevated roles require an admin actor."""
    email = email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=422, detail="A valid email address is required")
    if role.lower() not in ROLES:
        raise HTTPException(status_code=422, detail=f"Unknown role '{role}'")
    existing = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Email already registered")

    if role.lower() not in ("viewer", "analyst"):
        if actor is None or actor.role.lower() != "admin":
            raise HTTPException(
                status_code=403,
                detail=f"Only an admin can create a '{role}' account",
            )
    user = User(email=email, password_hash=hash_password(password), role=role.lower())
    db.add(user)
    await db.flush()
    log.info("user_created", email=email, role=user.role)
    return user


async def authenticate(db: AsyncSession, email: str, password: str) -> tuple[User, str]:
    """Verify credentials and return (user, raw_token)."""
    user = (
        await db.execute(select(User).where(User.email == email.strip().lower()))
    ).scalar_one_or_none()
    if user is None or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    raw = secrets.token_urlsafe(settings.platform.token_bytes)
    session = AuthSession(
        user_id=user.id,
        token_hash=_token_digest(raw),
        expires_at=_utcnow() + timedelta(hours=settings.platform.session_ttl_hours),
    )
    db.add(session)
    user.last_login_at = _utcnow()
    await db.flush()
    log.info("session_issued", email=user.email, user_id=user.id)
    return user, raw


async def resolve_token(db: AsyncSession, token: str) -> Optional[User]:
    """Resolve a bearer token to its user, or None when invalid/expired."""
    if not token:
        return None
    session = (
        await db.execute(
            select(AuthSession)
            .options(selectinload(AuthSession.user))
            .where(AuthSession.token_hash == _token_digest(token))
        )
    ).scalar_one_or_none()
    if session is None or session.revoked_at is not None:
        return None
    if session.expires_at < _utcnow():
        return None
    return session.user if session.user.is_active else None


async def revoke_token(db: AsyncSession, token: str) -> bool:
    session = (
        await db.execute(
            select(AuthSession).where(AuthSession.token_hash == _token_digest(token))
        )
    ).scalar_one_or_none()
    if session is None:
        return False
    session.revoked_at = datetime.now(timezone.utc)
    await db.flush()
    return True


async def ensure_bootstrap_admin(db: AsyncSession) -> None:
    """Create the configured bootstrap admin on first startup (idempotent)."""
    email = (settings.platform.bootstrap_admin_email or "").strip().lower()
    password = settings.platform.bootstrap_admin_password or ""
    if not email or not password:
        return
    existing = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if existing is not None:
        return
    db.add(User(email=email, password_hash=hash_password(password), role="admin"))
    await db.flush()
    log.info("bootstrap_admin_created", email=email)


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


def _bearer_token(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


async def get_current_user(
    request: Request, db: AsyncSession = Depends(get_session)
) -> User:
    """Require a valid bearer token; returns the authenticated user."""
    token = _bearer_token(request)
    user = await resolve_token(db, token or "")
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_optional_user(
    request: Request, db: AsyncSession = Depends(get_session)
) -> Optional[User]:
    token = _bearer_token(request)
    return await resolve_token(db, token or "")


def require_roles(*minimum: str):
    """Dependency factory: reject callers below the given role(s).

    Usage: ``Depends(require_roles(\"operator\"))`` — caller must be at
    least an operator. If several roles are given the caller must satisfy
    the *highest* one (use :func:`require_any_role` for alternatives).
    """

    async def _checker(user: User = Depends(get_current_user)) -> User:
        if not any(role_at_least(user.role, m) for m in minimum):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role {'/'.join(minimum)}",
            )
        return user

    return _checker


def require_any_role(*acceptable: str):
    """Dependency factory: caller must hold at least one of *acceptable*."""

    async def _checker(user: User = Depends(get_current_user)) -> User:
        if not any(role_at_least(user.role, m) for m in acceptable):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles {acceptable}",
            )
        return user

    return _checker


def user_public(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
    }
