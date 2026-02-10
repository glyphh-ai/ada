"""
Service layer for stored procedures.

This module provides CRUD operations for stored procedures.
"""

import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from domains.procedures.models import StoredProcedureModel
from domains.procedures.schemas import StoredProcedureCreate, StoredProcedureUpdate
from shared.exceptions import ValidationException, ConflictException, NotFoundException

logger = logging.getLogger(__name__)


class StoredProcedureService:
    """
    Service for managing stored procedures.
    
    Provides CRUD operations for stored procedures with validation
    and error handling.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize the service.
        
        Args:
            session: Async SQLAlchemy session
        """
        self._session = session
    
    async def create(
        self,
        org_id: str,
        model_id: str,
        data: StoredProcedureCreate,
    ) -> StoredProcedureModel:
        """
        Create a new stored procedure.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            data: Procedure creation data
            
        Returns:
            The created StoredProcedureModel
            
        Raises:
            ConflictException: If a procedure with the same name already exists
            ValidationException: If the data is invalid
        """
        # Validate GQL query by parsing it
        try:
            from glyphh.gql import parse
            parse(data.gql_query)
        except Exception as e:
            raise ValidationException(f"Invalid GQL query: {e}")
        
        procedure = StoredProcedureModel(
            org_id=org_id,
            model_id=model_id,
            name=data.name,
            gql_query=data.gql_query,
            lexicons=data.lexicons,
            description=data.description,
        )
        
        try:
            self._session.add(procedure)
            await self._session.commit()
            await self._session.refresh(procedure)
            logger.info(f"Created stored procedure: {procedure.name} for {org_id}/{model_id}")
            return procedure
        except IntegrityError as e:
            await self._session.rollback()
            if "uq_procedure_org_model_name" in str(e):
                raise ConflictException(f"Procedure '{data.name}' already exists for this model")
            raise ValidationException(f"Failed to create procedure: {e}")
    
    async def list(
        self,
        org_id: str,
        model_id: str,
    ) -> List[StoredProcedureModel]:
        """
        List all stored procedures for a model.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            
        Returns:
            List of StoredProcedureModel objects
        """
        result = await self._session.execute(
            select(StoredProcedureModel)
            .where(StoredProcedureModel.org_id == org_id)
            .where(StoredProcedureModel.model_id == model_id)
            .order_by(StoredProcedureModel.name)
        )
        procedures = list(result.scalars().all())
        logger.debug(f"Listed {len(procedures)} procedures for {org_id}/{model_id}")
        return procedures
    
    async def get(
        self,
        org_id: str,
        model_id: str,
        name: str,
    ) -> Optional[StoredProcedureModel]:
        """
        Get a stored procedure by name.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            name: Procedure name
            
        Returns:
            StoredProcedureModel if found, None otherwise
        """
        result = await self._session.execute(
            select(StoredProcedureModel)
            .where(StoredProcedureModel.org_id == org_id)
            .where(StoredProcedureModel.model_id == model_id)
            .where(StoredProcedureModel.name == name)
        )
        return result.scalar_one_or_none()
    
    async def get_by_id(
        self,
        procedure_id: UUID,
    ) -> Optional[StoredProcedureModel]:
        """
        Get a stored procedure by ID.
        
        Args:
            procedure_id: Procedure UUID
            
        Returns:
            StoredProcedureModel if found, None otherwise
        """
        result = await self._session.execute(
            select(StoredProcedureModel)
            .where(StoredProcedureModel.id == procedure_id)
        )
        return result.scalar_one_or_none()
    
    async def update(
        self,
        org_id: str,
        model_id: str,
        name: str,
        data: StoredProcedureUpdate,
    ) -> Optional[StoredProcedureModel]:
        """
        Update a stored procedure.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            name: Procedure name
            data: Update data
            
        Returns:
            Updated StoredProcedureModel if found, None otherwise
            
        Raises:
            ValidationException: If the update data is invalid
        """
        procedure = await self.get(org_id, model_id, name)
        if not procedure:
            return None
        
        # Validate GQL query if provided
        if data.gql_query is not None:
            try:
                from glyphh.gql import parse
                parse(data.gql_query)
            except Exception as e:
                raise ValidationException(f"Invalid GQL query: {e}")
            procedure.gql_query = data.gql_query
        
        if data.lexicons is not None:
            procedure.lexicons = data.lexicons
        
        if data.description is not None:
            procedure.description = data.description
        
        await self._session.commit()
        await self._session.refresh(procedure)
        logger.info(f"Updated stored procedure: {name} for {org_id}/{model_id}")
        return procedure
    
    async def delete(
        self,
        org_id: str,
        model_id: str,
        name: str,
    ) -> bool:
        """
        Delete a stored procedure.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            name: Procedure name
            
        Returns:
            True if deleted, False if not found
        """
        result = await self._session.execute(
            delete(StoredProcedureModel)
            .where(StoredProcedureModel.org_id == org_id)
            .where(StoredProcedureModel.model_id == model_id)
            .where(StoredProcedureModel.name == name)
        )
        await self._session.commit()
        deleted = result.rowcount > 0
        if deleted:
            logger.info(f"Deleted stored procedure: {name} for {org_id}/{model_id}")
        return deleted
    
    async def upsert(
        self,
        org_id: str,
        model_id: str,
        data: StoredProcedureCreate,
    ) -> StoredProcedureModel:
        """
        Create or update a stored procedure.
        
        If a procedure with the same name exists, it will be updated.
        Otherwise, a new procedure will be created.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            data: Procedure data
            
        Returns:
            The created or updated StoredProcedureModel
        """
        existing = await self.get(org_id, model_id, data.name)
        
        if existing:
            # Update existing procedure
            update_data = StoredProcedureUpdate(
                gql_query=data.gql_query,
                lexicons=data.lexicons,
                description=data.description,
            )
            return await self.update(org_id, model_id, data.name, update_data)
        else:
            # Create new procedure
            return await self.create(org_id, model_id, data)
