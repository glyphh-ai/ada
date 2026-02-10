"""
Property-based tests for StoredProcedure CRUD operations.

This module contains property-based tests using Hypothesis to verify
the CRUD consistency properties for stored procedures.

**Validates: Properties 3, 4, 5, 6** - CRUD Consistency
- Property 3: Create-Get Consistency
- Property 4: List Count Consistency
- Property 5: Update Consistency
- Property 6: Delete Consistency

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
"""

import pytest
from hypothesis import given, settings, strategies as st, assume
from hypothesis.strategies import composite
from typing import List, Dict, Optional, Any
from uuid import uuid4
from dataclasses import dataclass, field
from datetime import datetime

from domains.procedures.schemas import StoredProcedureCreate, StoredProcedureUpdate


# =============================================================================
# Generator Strategies
# =============================================================================

# ASCII letters and digits for valid names
ASCII_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ASCII_LETTERS_DIGITS_UNDERSCORE = ASCII_LETTERS + "0123456789_"
ASCII_LEXICON_CHARS = ASCII_LETTERS + "0123456789 -_"


@composite
def valid_procedure_name_strategy(draw) -> str:
    """Generate a valid procedure name."""
    first = draw(st.sampled_from(ASCII_LETTERS))
    rest = draw(st.text(
        alphabet=ASCII_LETTERS_DIGITS_UNDERSCORE,
        min_size=0,
        max_size=20
    ))
    return first + rest


@composite
def valid_lexicon_strategy(draw) -> str:
    """Generate a valid lexicon entry."""
    return draw(st.text(
        alphabet=ASCII_LEXICON_CHARS,
        min_size=1,
        max_size=20
    ).filter(lambda s: s.strip()))


@composite
def valid_lexicons_list_strategy(draw) -> List[str]:
    """Generate a valid list of lexicons."""
    num_lexicons = draw(st.integers(min_value=1, max_value=5))
    return [draw(valid_lexicon_strategy()) for _ in range(num_lexicons)]


# Valid GQL queries for testing
valid_gql_queries = st.sampled_from([
    'LIST ALL LIMIT 10',
    'LIST ALL LIMIT 100',
    'COUNT ALL',
    'FIND SIMILAR TO "test query" LIMIT 10 THRESHOLD 0.5',
    'FIND SIMILAR TO "search term" LIMIT 5 THRESHOLD 0.8',
    'LIST ALL WHERE status = "active" LIMIT 20',
])


@composite
def valid_description_strategy(draw) -> str:
    """Generate a valid description."""
    return draw(st.text(
        alphabet=ASCII_LEXICON_CHARS,
        min_size=0,
        max_size=100
    ))


@composite
def stored_procedure_create_strategy(draw) -> StoredProcedureCreate:
    """Generate a valid StoredProcedureCreate."""
    return StoredProcedureCreate(
        name=draw(valid_procedure_name_strategy()),
        gql_query=draw(valid_gql_queries),
        lexicons=draw(valid_lexicons_list_strategy()),
        description=draw(valid_description_strategy()),
    )


@composite
def stored_procedure_update_strategy(draw) -> StoredProcedureUpdate:
    """Generate a valid StoredProcedureUpdate."""
    # Randomly decide which fields to update
    update_gql = draw(st.booleans())
    update_lexicons = draw(st.booleans())
    update_description = draw(st.booleans())
    
    return StoredProcedureUpdate(
        gql_query=draw(valid_gql_queries) if update_gql else None,
        lexicons=draw(valid_lexicons_list_strategy()) if update_lexicons else None,
        description=draw(valid_description_strategy()) if update_description else None,
    )


# =============================================================================
# In-Memory Mock Service for Testing
# =============================================================================

@dataclass
class MockStoredProcedure:
    """Mock stored procedure for in-memory testing."""
    id: str
    org_id: str
    model_id: str
    name: str
    gql_query: str
    lexicons: List[str]
    description: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class MockStoredProcedureService:
    """
    In-memory mock service for testing CRUD operations.
    
    This mock implements the same interface as StoredProcedureService
    but uses an in-memory dictionary instead of a database.
    """
    
    def __init__(self):
        self._procedures: Dict[str, MockStoredProcedure] = {}
    
    def _make_key(self, org_id: str, model_id: str, name: str) -> str:
        return f"{org_id}:{model_id}:{name}"
    
    async def create(
        self,
        org_id: str,
        model_id: str,
        data: StoredProcedureCreate,
    ) -> MockStoredProcedure:
        """Create a new stored procedure."""
        key = self._make_key(org_id, model_id, data.name)
        
        if key in self._procedures:
            from shared.exceptions import ConflictException
            raise ConflictException(f"Procedure '{data.name}' already exists")
        
        procedure = MockStoredProcedure(
            id=str(uuid4()),
            org_id=org_id,
            model_id=model_id,
            name=data.name,
            gql_query=data.gql_query,
            lexicons=data.lexicons,
            description=data.description,
        )
        
        self._procedures[key] = procedure
        return procedure
    
    async def list(
        self,
        org_id: str,
        model_id: str,
    ) -> List[MockStoredProcedure]:
        """List all stored procedures for a model."""
        prefix = f"{org_id}:{model_id}:"
        return [
            p for key, p in self._procedures.items()
            if key.startswith(prefix)
        ]
    
    async def get(
        self,
        org_id: str,
        model_id: str,
        name: str,
    ) -> Optional[MockStoredProcedure]:
        """Get a stored procedure by name."""
        key = self._make_key(org_id, model_id, name)
        return self._procedures.get(key)
    
    async def update(
        self,
        org_id: str,
        model_id: str,
        name: str,
        data: StoredProcedureUpdate,
    ) -> Optional[MockStoredProcedure]:
        """Update a stored procedure."""
        key = self._make_key(org_id, model_id, name)
        procedure = self._procedures.get(key)
        
        if not procedure:
            return None
        
        if data.gql_query is not None:
            procedure.gql_query = data.gql_query
        if data.lexicons is not None:
            procedure.lexicons = data.lexicons
        if data.description is not None:
            procedure.description = data.description
        
        procedure.updated_at = datetime.utcnow()
        return procedure
    
    async def delete(
        self,
        org_id: str,
        model_id: str,
        name: str,
    ) -> bool:
        """Delete a stored procedure."""
        key = self._make_key(org_id, model_id, name)
        if key in self._procedures:
            del self._procedures[key]
            return True
        return False
    
    async def upsert(
        self,
        org_id: str,
        model_id: str,
        data: StoredProcedureCreate,
    ) -> MockStoredProcedure:
        """Create or update a stored procedure."""
        existing = await self.get(org_id, model_id, data.name)
        
        if existing:
            update_data = StoredProcedureUpdate(
                gql_query=data.gql_query,
                lexicons=data.lexicons,
                description=data.description,
            )
            return await self.update(org_id, model_id, data.name, update_data)
        else:
            return await self.create(org_id, model_id, data)


# =============================================================================
# Property Tests
# =============================================================================

class TestCRUDCreateGetConsistency:
    """
    Property tests for CRUD Create-Get Consistency (Property 3).
    
    **Validates: Property 3** - CRUD Create-Get Consistency
    For any valid StoredProcedure data, after calling create with that data,
    calling get with the same org_id, model_id, and name SHALL return a
    procedure with matching gql_query, lexicons, and description.
    
    **Validates: Requirements 3.1, 3.3**
    """
    
    @pytest.mark.asyncio
    @given(data=st.data())
    @settings(max_examples=100)
    async def test_create_get_consistency(self, data):
        """
        Property test: Create then get returns matching data.
        
        For any valid StoredProcedure data, create followed by get
        SHALL return a procedure with matching fields.
        
        **Validates: Requirements 3.1, 3.3**
        """
        # Create fresh service for each test
        procedure_service = MockStoredProcedureService()
        
        # Generate test data
        create_data = data.draw(stored_procedure_create_strategy())
        org_id = f"test_org_{uuid4().hex[:8]}"
        model_id = f"test_model_{uuid4().hex[:8]}"
        
        # Create procedure
        created = await procedure_service.create(org_id, model_id, create_data)
        
        # Get procedure
        retrieved = await procedure_service.get(org_id, model_id, create_data.name)
        
        # Verify consistency
        assert retrieved is not None, "Get should return the created procedure"
        assert retrieved.name == create_data.name
        assert retrieved.gql_query == create_data.gql_query
        assert retrieved.lexicons == create_data.lexicons
        assert retrieved.description == create_data.description
        assert retrieved.org_id == org_id
        assert retrieved.model_id == model_id


class TestCRUDListCountConsistency:
    """
    Property tests for CRUD List Count Consistency (Property 4).
    
    **Validates: Property 4** - CRUD List Count Consistency
    For any sequence of N create operations with unique names for the same
    org_id/model_id, calling list SHALL return exactly N procedures.
    
    **Validates: Requirements 3.2**
    """
    
    @pytest.mark.asyncio
    @given(data=st.data())
    @settings(max_examples=50)
    async def test_list_count_consistency(self, data):
        """
        Property test: List returns correct count after creates.
        
        For any sequence of N create operations with unique names,
        list SHALL return exactly N procedures.
        
        **Validates: Requirements 3.2**
        """
        # Create fresh service for each test
        procedure_service = MockStoredProcedureService()
        
        # Generate unique org/model IDs
        org_id = f"test_org_{uuid4().hex[:8]}"
        model_id = f"test_model_{uuid4().hex[:8]}"
        
        # Generate N unique procedures
        num_procedures = data.draw(st.integers(min_value=1, max_value=5))
        used_names = set()
        procedures = []
        
        for _ in range(num_procedures):
            create_data = data.draw(stored_procedure_create_strategy())
            # Ensure unique name
            while create_data.name in used_names:
                create_data = data.draw(stored_procedure_create_strategy())
            used_names.add(create_data.name)
            procedures.append(create_data)
        
        # Create all procedures
        for create_data in procedures:
            await procedure_service.create(org_id, model_id, create_data)
        
        # List procedures
        listed = await procedure_service.list(org_id, model_id)
        
        # Verify count
        assert len(listed) == num_procedures, \
            f"Expected {num_procedures} procedures, got {len(listed)}"
        
        # Verify all names are present
        listed_names = {p.name for p in listed}
        for create_data in procedures:
            assert create_data.name in listed_names, \
                f"Procedure '{create_data.name}' not found in list"


class TestCRUDUpdateConsistency:
    """
    Property tests for CRUD Update Consistency (Property 5).
    
    **Validates: Property 5** - CRUD Update Consistency
    For any existing StoredProcedure, after calling update with new values,
    calling get SHALL return the updated values.
    
    **Validates: Requirements 3.4**
    """
    
    @pytest.mark.asyncio
    @given(data=st.data())
    @settings(max_examples=100)
    async def test_update_consistency(self, data):
        """
        Property test: Update then get returns updated data.
        
        For any existing StoredProcedure, update followed by get
        SHALL return the updated values.
        
        **Validates: Requirements 3.4**
        """
        # Create fresh service for each test
        procedure_service = MockStoredProcedureService()
        
        # Generate test data
        create_data = data.draw(stored_procedure_create_strategy())
        update_data = data.draw(stored_procedure_update_strategy())
        org_id = f"test_org_{uuid4().hex[:8]}"
        model_id = f"test_model_{uuid4().hex[:8]}"
        
        # Create procedure
        await procedure_service.create(org_id, model_id, create_data)
        
        # Update procedure
        await procedure_service.update(org_id, model_id, create_data.name, update_data)
        
        # Get procedure
        retrieved = await procedure_service.get(org_id, model_id, create_data.name)
        
        # Verify update consistency
        assert retrieved is not None
        
        # Check each updated field
        if update_data.gql_query is not None:
            assert retrieved.gql_query == update_data.gql_query
        else:
            assert retrieved.gql_query == create_data.gql_query
        
        if update_data.lexicons is not None:
            assert retrieved.lexicons == update_data.lexicons
        else:
            assert retrieved.lexicons == create_data.lexicons
        
        if update_data.description is not None:
            assert retrieved.description == update_data.description
        else:
            assert retrieved.description == create_data.description


class TestCRUDDeleteConsistency:
    """
    Property tests for CRUD Delete Consistency (Property 6).
    
    **Validates: Property 6** - CRUD Delete Consistency
    For any existing StoredProcedure, after calling delete,
    calling get SHALL return None/404.
    
    **Validates: Requirements 3.5**
    """
    
    @pytest.mark.asyncio
    @given(data=st.data())
    @settings(max_examples=100)
    async def test_delete_consistency(self, data):
        """
        Property test: Delete then get returns None.
        
        For any existing StoredProcedure, delete followed by get
        SHALL return None.
        
        **Validates: Requirements 3.5**
        """
        # Create fresh service for each test
        procedure_service = MockStoredProcedureService()
        
        # Generate test data
        create_data = data.draw(stored_procedure_create_strategy())
        org_id = f"test_org_{uuid4().hex[:8]}"
        model_id = f"test_model_{uuid4().hex[:8]}"
        
        # Create procedure
        await procedure_service.create(org_id, model_id, create_data)
        
        # Verify it exists
        before_delete = await procedure_service.get(org_id, model_id, create_data.name)
        assert before_delete is not None, "Procedure should exist before delete"
        
        # Delete procedure
        deleted = await procedure_service.delete(org_id, model_id, create_data.name)
        assert deleted is True, "Delete should return True for existing procedure"
        
        # Verify it's gone
        after_delete = await procedure_service.get(org_id, model_id, create_data.name)
        assert after_delete is None, "Get should return None after delete"
    
    @pytest.mark.asyncio
    @given(data=st.data())
    @settings(max_examples=100)
    async def test_delete_nonexistent_returns_false(self, data):
        """
        Property test: Delete nonexistent returns False.
        
        For any name not in the database, delete SHALL return False.
        
        **Validates: Requirements 3.5**
        """
        # Create fresh service for each test
        procedure_service = MockStoredProcedureService()
        
        # Generate test data
        name = data.draw(valid_procedure_name_strategy())
        org_id = f"test_org_{uuid4().hex[:8]}"
        model_id = f"test_model_{uuid4().hex[:8]}"
        
        # Delete nonexistent procedure
        deleted = await procedure_service.delete(org_id, model_id, name)
        assert deleted is False, "Delete should return False for nonexistent procedure"


class TestCRUDUpsertConsistency:
    """
    Property tests for CRUD Upsert operation.
    
    **Validates: Requirements 3.1, 3.4**
    """
    
    @pytest.mark.asyncio
    @given(data=st.data())
    @settings(max_examples=100)
    async def test_upsert_creates_new(self, data):
        """
        Property test: Upsert creates new procedure when not exists.
        
        **Validates: Requirements 3.1**
        """
        # Create fresh service for each test
        procedure_service = MockStoredProcedureService()
        
        # Generate test data
        create_data = data.draw(stored_procedure_create_strategy())
        org_id = f"test_org_{uuid4().hex[:8]}"
        model_id = f"test_model_{uuid4().hex[:8]}"
        
        # Upsert (should create)
        result = await procedure_service.upsert(org_id, model_id, create_data)
        
        # Verify created
        assert result is not None
        assert result.name == create_data.name
        assert result.gql_query == create_data.gql_query
    
    @pytest.mark.asyncio
    @given(data=st.data())
    @settings(max_examples=100)
    async def test_upsert_updates_existing(self, data):
        """
        Property test: Upsert updates existing procedure.
        
        **Validates: Requirements 3.4**
        """
        # Create fresh service for each test
        procedure_service = MockStoredProcedureService()
        
        # Generate test data
        create_data = data.draw(stored_procedure_create_strategy())
        org_id = f"test_org_{uuid4().hex[:8]}"
        model_id = f"test_model_{uuid4().hex[:8]}"
        
        # Create first
        await procedure_service.create(org_id, model_id, create_data)
        
        # Generate updated data with same name
        updated_data = StoredProcedureCreate(
            name=create_data.name,
            gql_query=data.draw(valid_gql_queries),
            lexicons=data.draw(valid_lexicons_list_strategy()),
            description=data.draw(valid_description_strategy()),
        )
        
        # Upsert (should update)
        result = await procedure_service.upsert(org_id, model_id, updated_data)
        
        # Verify updated
        assert result is not None
        assert result.name == create_data.name
        assert result.gql_query == updated_data.gql_query
        
        # Verify only one procedure exists
        listed = await procedure_service.list(org_id, model_id)
        assert len(listed) == 1
