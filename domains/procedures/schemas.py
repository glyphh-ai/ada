"""
Pydantic schemas for stored procedures API.

This module defines request/response schemas for the stored procedures API.
"""

import re
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# Name validation pattern
NAME_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9_]*$')


class StoredProcedureCreate(BaseModel):
    """Request schema for creating a stored procedure."""
    name: str = Field(..., min_length=1, max_length=255, description="Unique procedure name")
    gql_query: str = Field(..., min_length=1, description="GQL query string")
    lexicons: List[str] = Field(..., min_length=1, description="Keywords for NL matching")
    description: str = Field(default="", max_length=1000, description="Human-readable description")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not NAME_PATTERN.match(v):
            raise ValueError(
                "Name must start with a letter and contain only alphanumeric characters and underscores"
            )
        return v
    
    @field_validator("lexicons")
    @classmethod
    def validate_lexicons(cls, v: List[str]) -> List[str]:
        # Clean and validate lexicons
        cleaned = [lex.strip().lower() for lex in v if lex.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty lexicon is required")
        return cleaned


class StoredProcedureUpdate(BaseModel):
    """Request schema for updating a stored procedure."""
    gql_query: Optional[str] = Field(None, min_length=1, description="GQL query string")
    lexicons: Optional[List[str]] = Field(None, min_length=1, description="Keywords for NL matching")
    description: Optional[str] = Field(None, max_length=1000, description="Human-readable description")
    
    @field_validator("lexicons")
    @classmethod
    def validate_lexicons(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        # Clean and validate lexicons
        cleaned = [lex.strip().lower() for lex in v if lex.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty lexicon is required")
        return cleaned


class StoredProcedureResponse(BaseModel):
    """Response schema for a stored procedure."""
    id: UUID
    org_id: str
    model_id: str
    name: str
    gql_query: str
    lexicons: List[str]
    description: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class StoredProcedureListResponse(BaseModel):
    """Response schema for listing stored procedures."""
    procedures: List[StoredProcedureResponse]
    total: int
