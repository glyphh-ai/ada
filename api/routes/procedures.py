"""
Stored Procedures API Routes for Glyphh Runtime.

Endpoints for creating, reading, updating, and deleting stored procedures.
All routes scoped by /{org_id}/{model_id}/procedures/...
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from domains.procedures.service import StoredProcedureService
from domains.procedures.schemas import (
    StoredProcedureCreate,
    StoredProcedureUpdate,
    StoredProcedureResponse,
    StoredProcedureListResponse,
)
from infrastructure.database import get_db
from shared.exceptions import ConflictException, NotFoundException, ValidationException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}/procedures", tags=["procedures"])


# Dependency injection
async def get_procedure_service(db: AsyncSession = Depends(get_db)) -> StoredProcedureService:
    """Get StoredProcedureService instance."""
    return StoredProcedureService(db)


# Endpoints
@router.post("", response_model=StoredProcedureResponse, status_code=201)
async def create_procedure(
    org_id: str,
    model_id: str,
    request: StoredProcedureCreate,
    service: StoredProcedureService = Depends(get_procedure_service),
) -> StoredProcedureResponse:
    """
    Create a new stored procedure.
    
    A stored procedure is a saved GQL query with associated lexicons
    for natural language matching.
    """
    try:
        procedure = await service.create(org_id, model_id, request)
        logger.info(f"Created procedure '{request.name}' for {org_id}/{model_id}")
        return StoredProcedureResponse.model_validate(procedure)
    except ConflictException as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create procedure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=StoredProcedureListResponse)
async def list_procedures(
    org_id: str,
    model_id: str,
    service: StoredProcedureService = Depends(get_procedure_service),
) -> StoredProcedureListResponse:
    """
    List all stored procedures for a model.
    
    Returns all procedures associated with the specified org_id and model_id.
    """
    try:
        procedures = await service.list(org_id, model_id)
        return StoredProcedureListResponse(
            procedures=[StoredProcedureResponse.model_validate(p) for p in procedures],
            total=len(procedures),
        )
    except Exception as e:
        logger.error(f"Failed to list procedures: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{name}", response_model=StoredProcedureResponse)
async def get_procedure(
    org_id: str,
    model_id: str,
    name: str,
    service: StoredProcedureService = Depends(get_procedure_service),
) -> StoredProcedureResponse:
    """
    Get a stored procedure by name.
    
    Returns the procedure with the specified name, or 404 if not found.
    """
    try:
        procedure = await service.get(org_id, model_id, name)
        if procedure is None:
            raise HTTPException(
                status_code=404,
                detail=f"Procedure '{name}' not found for {org_id}/{model_id}"
            )
        return StoredProcedureResponse.model_validate(procedure)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get procedure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{name}", response_model=StoredProcedureResponse)
async def update_procedure(
    org_id: str,
    model_id: str,
    name: str,
    request: StoredProcedureUpdate,
    service: StoredProcedureService = Depends(get_procedure_service),
) -> StoredProcedureResponse:
    """
    Update a stored procedure.
    
    Updates the specified fields of the procedure. Only provided fields
    will be updated; omitted fields retain their current values.
    """
    try:
        procedure = await service.update(org_id, model_id, name, request)
        if procedure is None:
            raise HTTPException(
                status_code=404,
                detail=f"Procedure '{name}' not found for {org_id}/{model_id}"
            )
        logger.info(f"Updated procedure '{name}' for {org_id}/{model_id}")
        return StoredProcedureResponse.model_validate(procedure)
    except HTTPException:
        raise
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update procedure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{name}")
async def delete_procedure(
    org_id: str,
    model_id: str,
    name: str,
    service: StoredProcedureService = Depends(get_procedure_service),
) -> dict:
    """
    Delete a stored procedure.
    
    Permanently removes the procedure with the specified name.
    """
    try:
        deleted = await service.delete(org_id, model_id, name)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Procedure '{name}' not found for {org_id}/{model_id}"
            )
        logger.info(f"Deleted procedure '{name}' for {org_id}/{model_id}")
        return {"status": "deleted", "name": name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete procedure: {e}")
        raise HTTPException(status_code=500, detail=str(e))
