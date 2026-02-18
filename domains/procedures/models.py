"""
SQLAlchemy ORM model for stored procedures.

This module defines the database model for stored similarity procedures.
"""

import uuid
from datetime import datetime
from typing import List

from sqlalchemy import Column, JSON, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, ARRAY

from infrastructure.database.connection import Base

# Use ARRAY(Text) on PostgreSQL, plain JSON on SQLite/others
ArrayType = JSON().with_variant(ARRAY(Text), "postgresql")


class StoredProcedureModel(Base):
    """
    SQLAlchemy model for stored procedures.
    
    Stored procedures are saved GQL queries with associated lexicons
    for natural language matching.
    
    Attributes:
        id: Unique identifier (UUID)
        org_id: Organization ID for scoping
        model_id: Model ID for scoping
        name: Unique procedure name within org/model scope
        gql_query: The GQL query string
        lexicons: List of keywords for NL matching (ARRAY on PG, JSON on SQLite)
        description: Human-readable description
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = "stored_procedures"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(String(255), nullable=False, index=True)
    model_id = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    gql_query = Column(Text, nullable=False)
    lexicons = Column(ArrayType, nullable=False)
    description = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self) -> str:
        return f"<StoredProcedure(name={self.name}, org_id={self.org_id}, model_id={self.model_id})>"
