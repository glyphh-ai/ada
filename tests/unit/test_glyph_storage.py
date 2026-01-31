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
    async def test_create_glyph_success(self, test_db, sample_namespace, sample_embedding):
        """Test successful glyph creation."""
        storage = GlyphStorage(test_db)
        
        result = await storage.create_glyph(
            namespace=sample_namespace,
            concept_text="test concept",
            embedding=sample_embedding,
            metadata={"test": True},
        )
        
        assert result.glyph_id is not None
        assert result.namespace == sample_namespace
    
    @pytest.mark.asyncio
    async def test_create_glyph_invalid_embedding_dimension(self, test_db, sample_namespace):
        """Test that invalid embedding dimension raises ValidationException."""
        storage = GlyphStorage(test_db)
        
        with pytest.raises(ValidationException) as exc_info:
            await storage.create_glyph(
                namespace=sample_namespace,
                concept_text="test concept",
                embedding=[0.1] * 100,  # Wrong dimension
            )
        
        assert "768 dimensions" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_glyph_success(self, test_db, sample_namespace, sample_embedding):
        """Test successful glyph retrieval."""
        storage = GlyphStorage(test_db)
        
        # Create glyph
        created = await storage.create_glyph(
            namespace=sample_namespace,
            concept_text="test concept",
            embedding=sample_embedding,
        )
        await test_db.commit()
        
        # Retrieve glyph
        glyph = await storage.get_glyph(sample_namespace, created.glyph_id)
        
        assert glyph.id == created.glyph_id
        assert glyph.concept_text == "test concept"
    
    @pytest.mark.asyncio
    async def test_get_glyph_not_found(self, test_db, sample_namespace):
        """Test that non-existent glyph raises GlyphNotFoundException."""
        storage = GlyphStorage(test_db)
        
        with pytest.raises(GlyphNotFoundException):
            await storage.get_glyph(sample_namespace, uuid4())
    
    @pytest.mark.asyncio
    async def test_get_glyph_wrong_namespace(self, test_db, sample_embedding):
        """Test that glyph in different namespace is not found."""
        storage = GlyphStorage(test_db)
        
        # Create glyph in namespace A
        created = await storage.create_glyph(
            namespace="namespace_a",
            concept_text="test concept",
            embedding=sample_embedding,
        )
        await test_db.commit()
        
        # Try to retrieve from namespace B
        with pytest.raises(GlyphNotFoundException):
            await storage.get_glyph("namespace_b", created.glyph_id)
    
    @pytest.mark.asyncio
    async def test_update_glyph_success(self, test_db, sample_namespace, sample_embedding):
        """Test successful glyph update."""
        storage = GlyphStorage(test_db)
        
        # Create glyph
        created = await storage.create_glyph(
            namespace=sample_namespace,
            concept_text="original concept",
            embedding=sample_embedding,
        )
        await test_db.commit()
        
        # Update glyph
        updated = await storage.update_glyph(
            namespace=sample_namespace,
            glyph_id=created.glyph_id,
            concept_text="updated concept",
            metadata={"updated": True},
        )
        
        assert updated.concept_text == "updated concept"
        assert updated.metadata["updated"] is True
    
    @pytest.mark.asyncio
    async def test_delete_glyph_success(self, test_db, sample_namespace, sample_embedding):
        """Test successful glyph deletion."""
        storage = GlyphStorage(test_db)
        
        # Create glyph
        created = await storage.create_glyph(
            namespace=sample_namespace,
            concept_text="test concept",
            embedding=sample_embedding,
        )
        await test_db.commit()
        
        # Delete glyph
        deleted = await storage.delete_glyph(sample_namespace, created.glyph_id)
        
        assert deleted is True
        
        # Verify deletion
        with pytest.raises(GlyphNotFoundException):
            await storage.get_glyph(sample_namespace, created.glyph_id)
    
    @pytest.mark.asyncio
    async def test_delete_glyph_not_found(self, test_db, sample_namespace):
        """Test deleting non-existent glyph returns False."""
        storage = GlyphStorage(test_db)
        
        deleted = await storage.delete_glyph(sample_namespace, uuid4())
        
        assert deleted is False
    
    @pytest.mark.asyncio
    async def test_list_glyphs_pagination(self, test_db, sample_namespace, sample_embedding):
        """Test glyph listing with pagination."""
        storage = GlyphStorage(test_db)
        
        # Create multiple glyphs
        for i in range(5):
            await storage.create_glyph(
                namespace=sample_namespace,
                concept_text=f"concept {i}",
                embedding=sample_embedding,
            )
        await test_db.commit()
        
        # List with pagination
        page1 = await storage.list_glyphs(sample_namespace, limit=2, offset=0)
        page2 = await storage.list_glyphs(sample_namespace, limit=2, offset=2)
        
        assert len(page1) == 2
        assert len(page2) == 2
    
    @pytest.mark.asyncio
    async def test_count_glyphs(self, test_db, sample_namespace, sample_embedding):
        """Test glyph counting."""
        storage = GlyphStorage(test_db)
        
        # Create glyphs
        for i in range(3):
            await storage.create_glyph(
                namespace=sample_namespace,
                concept_text=f"concept {i}",
                embedding=sample_embedding,
            )
        await test_db.commit()
        
        count = await storage.count_glyphs(sample_namespace)
        
        assert count == 3


class TestGlyphStorageSerialization:
    """Tests for glyph serialization/deserialization."""
    
    def test_from_json_valid(self, sample_embedding):
        """Test valid JSON deserialization."""
        storage = GlyphStorage(None)  # Session not needed for this test
        
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
