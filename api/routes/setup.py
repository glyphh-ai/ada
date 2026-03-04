"""
Bootstrap endpoint for first-time runtime setup.

POST /setup creates the first admin token when the database has zero tokens.
No authentication required (that's the point — bootstrap).
Returns 403 if any tokens already exist.
"""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.db_models import Token
from infrastructure.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["setup"])


class SetupRequest(BaseModel):
    org_id: str = Field(default="default", description="Organization ID for the admin token")
    name: str = Field(default="admin", description="Token name")


class SetupResponse(BaseModel):
    token: str
    token_prefix: str
    org_id: str
    name: str
    message: str


@router.post("/setup", response_model=SetupResponse)
async def bootstrap_setup(
    request: SetupRequest,
    db: AsyncSession = Depends(get_db),
) -> SetupResponse:
    """Create the first admin token. Only works when no tokens exist in the database."""

    # Check if any tokens exist
    result = await db.execute(select(func.count()).select_from(Token))
    token_count = result.scalar()

    if token_count > 0:
        raise HTTPException(
            status_code=403,
            detail="Runtime already initialized. Use an existing admin token to create new tokens.",
        )

    # Create the admin token
    raw_token = f"glyphh_{secrets.token_urlsafe(32)}"
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    token_prefix = raw_token[:12]

    db_token = Token(
        name=request.name,
        token_hash=token_hash,
        token_prefix=token_prefix,
        org_id=request.org_id,
        permissions=["read", "write", "admin"],
        expires_at=datetime.utcnow() + timedelta(days=365),
    )
    db.add(db_token)
    await db.flush()

    logger.info(f"Bootstrap: created admin token '{request.name}' for org '{request.org_id}'")

    return SetupResponse(
        token=raw_token,
        token_prefix=token_prefix,
        org_id=request.org_id,
        name=request.name,
        message="Admin token created. Store it securely — it won't be shown again.",
    )
