from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse


def decode_jwt(token: str, secret: str, algorithm: str) -> dict:
    if algorithm != "HS256":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unsupported token algorithm")
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("invalid token")
        header_b64, payload_b64, signature_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode()
        sig = base64.urlsafe_b64decode(signature_b64 + "==")
        expected = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            raise ValueError("bad signature")
        payload_json = base64.urlsafe_b64decode(payload_b64 + "==")
        return json.loads(payload_json)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


def parse_license_expiry(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def extract_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return request.query_params.get("token")


def enforce_runtime_token(request: Request, settings) -> dict:
    token = extract_bearer_token(request)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing runtime token")
    payload = decode_jwt(token, settings.jwt_secret, settings.jwt_algorithm)
    if payload.get("token_type") != "runtime":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid runtime token")
    now_ts = int(dt.datetime.utcnow().timestamp())
    exp = payload.get("exp")
    if isinstance(exp, int) and exp < now_ts:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    license_expiry = parse_license_expiry(payload.get("license_expires_at"))
    if license_expiry and license_expiry < dt.datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="License expired")
    return payload


def build_runtime_auth_middleware(settings):
    async def runtime_auth_middleware(request: Request, call_next):
        if (
            request.url.path in ("/health", "/openapi.json")
            or request.url.path.startswith("/docs")
            or request.url.path.startswith("/redoc")
        ):
            return await call_next(request)
        try:
            enforce_runtime_token(request, settings)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        return await call_next(request)

    return runtime_auth_middleware
