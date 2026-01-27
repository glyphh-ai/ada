from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import Body, Depends, HTTPException, Request, status

from ..core.config import get_settings
from ..core.schemas import (
    AuthTokenCreateRequest,
    AuthTokenCreateResponse,
    AuthTokenExchangeRequest,
    AuthTokenExchangeResponse,
    AuthTokenListResponse,
)
from ..services.auth_runtime import encode_jwt, load_runtime_secret, runtime_jwt_secret
from ..services.token_store import create_token, list_tokens, revoke_token, verify_token
from .router import api_router


router = api_router(tags=["auth"])


def require_runtime_secret(request: Request) -> None:
    settings = get_settings()
    secret = load_runtime_secret(settings)
    if not secret:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Runtime secret not configured")
    provided = request.headers.get("x-runtime-secret")
    if not provided or provided != secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid runtime secret")


@router.post("/auth/tokens", response_model=AuthTokenCreateResponse)
def create_auth_token(
    request: Request,
    payload: AuthTokenCreateRequest = Body(...),
) -> Any:
    require_runtime_secret(request)
    scopes = payload.scopes or []
    return create_token(payload.name, scopes)


@router.get("/auth/tokens", response_model=AuthTokenListResponse)
def list_auth_tokens(request: Request) -> AuthTokenListResponse:
    require_runtime_secret(request)
    return AuthTokenListResponse(items=list_tokens())


@router.delete("/auth/tokens/{token_id}", status_code=204)
def revoke_auth_token(request: Request, token_id: str) -> None:
    require_runtime_secret(request)
    if not revoke_token(token_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Token not found")


@router.post("/auth/exchange", response_model=AuthTokenExchangeResponse)
def exchange_token(payload: AuthTokenExchangeRequest = Body(...)) -> Any:
    settings = get_settings()
    record = verify_token(payload.token)
    if not record:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    expires_minutes = payload.expires_in_minutes or 60
    now = dt.datetime.utcnow()
    exp = now + dt.timedelta(minutes=expires_minutes)
    claims = {
        "token_type": "runtime",
        "scopes": record.get("scopes") or [],
        "org_id": payload.org_id,
        "runtime_id": payload.runtime_id,
        "model_id": payload.model_id,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    token = encode_jwt(claims, runtime_jwt_secret(settings), settings.jwt_algorithm)
    return AuthTokenExchangeResponse(
        access_token=token,
        token_type="runtime",
        expires_at=exp.isoformat(),
        scopes=record.get("scopes") or [],
    )
