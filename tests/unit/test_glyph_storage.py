"""
Unit tests for GlyphStorage.
"""

import pytest
import pytest_asyncio
from uuid import uuid4

from domains.models.storage import GlyphStorage
from shared.exceptions import GlyphNotFoundException, ValidationException


class TestGlyphStorage:
    """Tests for GlyphStorage CRUD operations."""
    
    @pytest.mark.asyncio
    async def test_create_glyph_success(self, test_db, sample_org_id, sample_model_id, sample_embedding):
        """Test successful glyph creation."""
        storage = GlyphStorage(test_db)
        
        result = await storage.create_glyph(
            org_id=sample_org_id,
            model_id=sample_model_id,
            concept_text="test concept",
            embedding=sample_embedding,
            metadata={"test": True},
        )
        
        assert result.glyph_id is not None
        assert result.org_id == sample_org_id
        assert result.model_id == sample_model_id
    
    @pytest.mark.asyncio
    async def test_create_glyph_invalid_embedding_dimension(self, test_db, sample_org_id, sample_model_id):
        """Test that invalid embedding dimension raises ValidationException."""
        storage = GlyphStorage(test_db)
        
        with pytest.raises(ValidationException) as exc_info:
            await storage.create_glyph(
                org_id=sample_org_id,
                model_id=sample_model_id,
                concept_text="test concept",
                embedding=[0.1] * 3000,  # Exceeds max dimension (2000)
            )
        
        assert "exceeds runtime limit" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_glyph_success(self, test_db, sample_org_id, sample_model_id, sample_embedding):
        """Test successful glyph retrieval."""
        storage = GlyphStorage(test_db)
        
        created = await storage.create_glyph(
            org_id=sample_org_id,
            model_id=sample_model_id,
            concept_text="test concept",
            embedding=sample_embedding,
        )
        await test_db.commit()
        
        glyph = await storage.get_glyph(sample_org_id, sample_model_id, created.glyph_id)
        
        assert glyph.id == created.glyph_id
        assert glyph.concept_text == "test concept"
    
    @pytest.mark.asyncio
    async def test_get_glyph_not_found(self, test_db, sample_org_id, sample_model_id):
        """Test that non-existent glyph raises GlyphNotFoundException."""
        storage = GlyphStorage(test_db)
        
        with pytest.raises(GlyphNotFoundException):
            await storage.get_glyph(sample_org_id, sample_model_id, uuid4())
    
    @pytest.mark.asyncio
    async def test_get_glyph_wrong_org(self, test_db, sample_embedding):
        """Test that glyph in different org is not found."""
        storage = GlyphStorage(test_db)
        
        created = await storage.create_glyph(
            org_id="org_a",
            model_id="model_x",
            concept_text="test concept",
            embedding=sample_embedding,
        )
        await test_db.commit()
        
        with pytest.raises(GlyphNotFoundException):
            await storage.get_glyph("org_b", "model_x", created.glyph_id)
    
    @pytest.mark.asyncio
    async def test_update_glyph_success(self, test_db, sample_org_id, sample_model_id, sample_embedding):
        """Test successful glyph update."""
        storage = GlyphStorage(test_db)
        
        created = await storage.create_glyph(
            org_id=sample_org_id,
            model_id=sample_model_id,
            concept_text="original concept",
            embedding=sample_embedding,
        )
        await test_db.commit()
        
        updated = await storage.update_glyph(
            org_id=sample_org_id,
            model_id=sample_model_id,
            glyph_id=created.glyph_id,
            concept_text="updated concept",
            metadata={"updated": True},
        )
        
        assert updated.concept_text == "updated concept"
        assert updated.glyph_metadata["updated"] is True
    
    @pytest.mark.asyncio
    async def test_delete_glyph_success(self, test_db, sample_org_id, sample_model_id, sample_embedding):
        """Test successful glyph deletion."""
        storage = GlyphStorage(test_db)
        
        created = await storage.create_glyph(
            org_id=sample_org_id,
            model_id=sample_model_id,
            concept_text="test concept",
            embedding=sample_embedding,
        )
        await test_db.commit()
        
        deleted = await storage.delete_glyph(sample_org_id, sample_model_id, created.glyph_id)
        assert deleted is True
        
        with pytest.raises(GlyphNotFoundException):
            await storage.get_glyph(sample_org_id, sample_model_id, created.glyph_id)
    
    @pytest.mark.asyncio
    async def test_delete_glyph_not_found(self, test_db, sample_org_id, sample_model_id):
        """Test deleting non-existent glyph returns False."""
        storage = GlyphStorage(test_db)
        
        deleted = await storage.delete_glyph(sample_org_id, sample_model_id, uuid4())
        assert deleted is False
    
    @pytest.mark.asyncio
    async def test_list_glyphs_pagination(self, test_db, sample_org_id, sample_model_id, sample_embedding):
        """Test glyph listing with pagination."""
        storage = GlyphStorage(test_db)
        
        for i in range(5):
            await storage.create_glyph(
                org_id=sample_org_id,
                model_id=sample_model_id,
                concept_text=f"concept {i}",
                embedding=sample_embedding,
            )
        await test_db.commit()
        
        page1 = await storage.list_glyphs(sample_org_id, sample_model_id, limit=2, offset=0)
        page2 = await storage.list_glyphs(sample_org_id, sample_model_id, limit=2, offset=2)
        
        assert len(page1) == 2
        assert len(page2) == 2
    
    @pytest.mark.asyncio
    async def test_count_glyphs(self, test_db, sample_org_id, sample_model_id, sample_embedding):
        """Test glyph counting."""
        storage = GlyphStorage(test_db)
        
        for i in range(3):
            await storage.create_glyph(
                org_id=sample_org_id,
                model_id=sample_model_id,
                concept_text=f"concept {i}",
                embedding=sample_embedding,
            )
        await test_db.commit()
        
        count = await storage.count_glyphs(sample_org_id, sample_model_id)
        assert count == 3


class TestGlyphStorageSerialization:
    """Tests for glyph serialization/deserialization."""
    
    def test_from_json_valid(self, sample_embedding):
        """Test valid JSON deserialization."""
        storage = GlyphStorage(None)
        
        data = {
            "concept_text": "test concept",
            "embedding": sample_embedding,
            "embedding_format": "array",
            "metadata": {"test": True},
        }
        
        concept_text, embedding, metadata = storage.from_json(data)
        
        assert concept_text == "test concept"
        assert len(embedding) == 768
        assert metadata["test"] is True
    
    def test_from_json_missing_concept(self):
        """Test that missing concept_text raises ValidationException."""
        storage = GlyphStorage(None)
        
        with pytest.raises(ValidationException) as exc_info:
            storage.from_json({"metadata": {}})
        
        assert "concept_text" in str(exc_info.value)
    
    def test_from_json_invalid_embedding_dimension(self):
        """Test that wrong embedding dimension raises ValidationException."""
        storage = GlyphStorage(None)
        
        with pytest.raises(ValidationException) as exc_info:
            storage.from_json({
                "concept_text": "test",
                "embedding": [0.1] * 100,
                "embedding_format": "array",
            })
        
        assert "768 dimensions" in str(exc_info.value)
