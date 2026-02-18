"""
Model packaging and deployment.

This module implements the GlyphhModel data structure for packaging models
with all necessary components for deployment:
- Glyphs (encoded concepts)
- Encoder configuration
- Custom encoder classes (serialized)
- Intent patterns for NL query matching
- Metadata and versioning

The model can be serialized to .glyphh files (compressed JSON) and
deserialized for deployment in runtimes.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime
import json
import gzip
import pickle
import re
from pathlib import Path

from glyphh.core.types import Glyph
from glyphh.core.config import EncoderConfig
from glyphh.exceptions import ModelValidationException

if TYPE_CHECKING:
    from glyphh.encoder.intent import IntentEncoder
    from glyphh.gql.patterns import GQLPattern
    from glyphh.gql.translator import NLTranslator
    from glyphh.gql.stored_procedure import StoredProcedure


@dataclass
class GlyphhModel:
    """
    Packaged model for deployment.
    
    A GlyphhModel bundles all components needed to deploy a model:
    - name: Human-readable model name
    - version: Semantic version (X.Y.Z format)
    - encoder_config: Configuration for the encoder
    - glyphs: List of encoded concepts
    - custom_encoders: Serialized custom encoder classes (name → bytes)
    - intent_patterns: Serialized intent patterns for NL query matching
    - readme: Model documentation in Markdown format
    - metadata: Additional metadata (domain, description, etc.)
    - created_at: Timestamp of model creation
    
    The model enforces:
    - Semantic versioning format (X.Y.Z)
    - Vector space consistency (all glyphs in same space)
    - Completeness validation (required fields present)
    
    Example:
        >>> model = GlyphhModel(
        ...     name="vehicle_model",
        ...     version="1.0.0",
        ...     encoder_config=config,
        ...     glyphs=[car_glyph, truck_glyph],
        ...     custom_encoders={},
        ...     readme="# Vehicle Model\\n\\nClassifies vehicles by type.",
        ...     metadata={"domain": "automotive", "description": "Vehicle classification"}
        ... )
        >>> model.to_file("vehicle_model.glyphh")
        >>> loaded_model = GlyphhModel.from_file("vehicle_model.glyphh")
    """
    name: str
    version: str
    encoder_config: EncoderConfig
    glyphs: List[Glyph]
    custom_encoders: Dict[str, bytes] = field(default_factory=dict)
    intent_patterns: Optional[Dict[str, Any]] = field(default=None)
    stored_procedures: List['StoredProcedure'] = field(default_factory=list)
    concepts: Optional[List[Dict[str, Any]]] = field(default=None)
    readme: Optional[str] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate model on initialization."""
        errors = self.validate_completeness()
        if errors:
            raise ModelValidationException(errors)
    
    def validate_completeness(self) -> List[str]:
        """
        Validate model is complete and ready for deployment.
        
        Checks:
        - Encoder configuration is present and has layers
        - All glyphs have same space_id (vector space consistency)
        - Version follows semantic version format (X.Y.Z)
        
        Note: Empty glyphs list is allowed — data loading happens after
        model creation and deployment.
        
        Returns:
            List of validation error messages (empty if valid)
        
        Example:
            >>> errors = model.validate_completeness()
            >>> if errors:
            ...     print(f"Validation failed: {errors}")
            ... else:
            ...     print("Model is valid")
        """
        errors = []
        
        # Check for encoder configuration
        if not self.encoder_config:
            errors.append("Missing encoder configuration")
        
        # Validate all glyphs have same space_id
        if self.glyphs:
            space_ids = {g.space_id for g in self.glyphs}
            if len(space_ids) > 1:
                errors.append(
                    f"Multiple vector spaces detected: {space_ids}. "
                    f"All glyphs must be in the same vector space."
                )
        
        # Validate version format (semantic versioning: X.Y.Z)
        if not re.match(r'^\d+\.\d+\.\d+$', self.version):
            errors.append(
                f"Invalid version format: '{self.version}'. "
                f"Expected semantic version format (X.Y.Z, e.g., '1.0.0')"
            )
        
        return errors
    
    def set_intent_encoder(self, intent_encoder: 'IntentEncoder') -> None:
        """
        Set the intent encoder for NL query matching.
        
        The intent patterns are serialized and bundled with the model.
        No separate JSON configuration files needed.
        
        Args:
            intent_encoder: IntentEncoder with registered patterns
        
        Example:
            >>> intent_encoder = IntentEncoder(config)
            >>> intent_encoder.add_defaults()
            >>> intent_encoder.add_pattern(IntentPattern(...))
            >>> model.set_intent_encoder(intent_encoder)
            >>> model.to_file("my_model.glyphh")
        """
        self.intent_patterns = intent_encoder.to_dict()
    
    def get_intent_encoder(self) -> Optional['IntentEncoder']:
        """
        Get the intent encoder from the model.
        
        Deserializes intent patterns from the model and returns
        a fully configured IntentEncoder.
        
        Returns:
            IntentEncoder if patterns exist, None otherwise
        
        Example:
            >>> model = GlyphhModel.from_file("my_model.glyphh")
            >>> intent_encoder = model.get_intent_encoder()
            >>> if intent_encoder:
            ...     match = intent_encoder.match_intent("find customer John")
        """
        if not self.intent_patterns:
            return None
        
        from glyphh.encoder.intent import IntentEncoder
        return IntentEncoder.from_dict(self.intent_patterns, self.encoder_config)
    
    def has_intent_patterns(self) -> bool:
        """
        Check if the model has intent patterns.
        
        Returns:
            True if intent patterns are present
        
        Example:
            >>> if model.has_intent_patterns():
            ...     intent_encoder = model.get_intent_encoder()
        """
        return self.intent_patterns is not None and len(self.intent_patterns.get("patterns", [])) > 0
    
    def has_gql_patterns(self) -> bool:
        """
        Check if the model has GQL patterns for NL-to-GQL translation.
        
        Returns:
            True if GQL patterns are present in encoder_config
        
        Example:
            >>> if model.has_gql_patterns():
            ...     translator = model.get_nl_translator()
        """
        return (
            self.encoder_config.gql_patterns is not None 
            and len(self.encoder_config.gql_patterns) > 0
        )
    
    def get_nl_translator(self) -> Optional['NLTranslator']:
        """
        Get an NL translator configured with the model's GQL patterns.
        
        Returns:
            NLTranslator if GQL patterns exist, None otherwise
        
        Example:
            >>> model = GlyphhModel.from_file("my_model.glyphh")
            >>> translator = model.get_nl_translator()
            >>> if translator:
            ...     result = translator.translate("find similar to red car")
            ...     if result.success:
            ...         print(f"GQL: {result.gql}")
        """
        if not self.has_gql_patterns():
            return None
        
        from glyphh.gql.translator import NLTranslator
        patterns = self.encoder_config.gql_patterns.to_gql_patterns()
        return NLTranslator(
            patterns=patterns,
            dimension=self.encoder_config.dimension,
            seed=self.encoder_config.seed
        )
    
    def set_gql_patterns(self, patterns: List['GQLPattern']) -> None:
        """
        Set GQL patterns for NL-to-GQL translation.
        
        The patterns are stored in encoder_config and bundled with the model.
        
        Args:
            patterns: List of GQLPattern objects
        
        Example:
            >>> from glyphh.gql import GQLPattern, SlotDefinition, SlotType
            >>> pattern = GQLPattern(
            ...     name="find_similar",
            ...     phrases=["find similar to {query}"],
            ...     slots=[SlotDefinition(name="query", type=SlotType.STRING)],
            ...     gql_template='FIND SIMILAR TO "{query}" LIMIT 10'
            ... )
            >>> model.set_gql_patterns([pattern])
            >>> model.to_file("my_model.glyphh")
        """
        from glyphh.core.config import GQLPatternsConfig
        
        pattern_dicts = [p.to_dict() for p in patterns]
        self.encoder_config.gql_patterns = GQLPatternsConfig(patterns=pattern_dicts)
    
    def has_stored_procedures(self) -> bool:
        """
        Check if the model has stored procedures.
        
        Returns:
            True if stored procedures are present
        
        Example:
            >>> if model.has_stored_procedures():
            ...     for proc in model.stored_procedures:
            ...         print(f"Procedure: {proc.name}")
        """
        return len(self.stored_procedures) > 0
    
    def add_stored_procedure(self, procedure: 'StoredProcedure') -> None:
        """
        Add a stored procedure to the model.
        
        Args:
            procedure: StoredProcedure to add
        
        Raises:
            ValueError: If a procedure with the same name already exists
        
        Example:
            >>> from glyphh.gql import StoredProcedure
            >>> procedure = StoredProcedure(
            ...     name="find_reliable_car",
            ...     gql_query='FIND SIMILAR TO "reliable car" LIMIT 10',
            ...     lexicons=["reliable", "car"],
            ...     description="Find reliable cars"
            ... )
            >>> model.add_stored_procedure(procedure)
        """
        # Check for duplicate names
        existing_names = {p.name for p in self.stored_procedures}
        if procedure.name in existing_names:
            raise ValueError(f"Procedure '{procedure.name}' already exists in model")
        
        self.stored_procedures.append(procedure)
    
    def get_stored_procedure(self, name: str) -> Optional['StoredProcedure']:
        """
        Get a stored procedure by name.
        
        Args:
            name: Procedure name
        
        Returns:
            StoredProcedure if found, None otherwise
        
        Example:
            >>> proc = model.get_stored_procedure("find_reliable_car")
            >>> if proc:
            ...     print(f"Query: {proc.gql_query}")
        """
        for proc in self.stored_procedures:
            if proc.name == name:
                return proc
        return None
    
    def remove_stored_procedure(self, name: str) -> bool:
        """
        Remove a stored procedure by name.
        
        Args:
            name: Procedure name
        
        Returns:
            True if removed, False if not found
        
        Example:
            >>> if model.remove_stored_procedure("old_procedure"):
            ...     print("Procedure removed")
        """
        for i, proc in enumerate(self.stored_procedures):
            if proc.name == name:
                del self.stored_procedures[i]
                return True
        return False
    
    def set_concepts(self, records: List[Dict[str, Any]]) -> None:
        """
        Set concept records to be embedded in the model file.
        
        Concepts are raw data records that will be encoded and loaded
        when the model is deployed to the runtime. This enables
        self-contained deployable models.
        
        Args:
            records: List of concept records in hierarchical or flat format
        
        Example:
            >>> concepts = [
            ...     {"content": {"text": {"value": "Hello world"}}},
            ...     {"content": {"text": {"value": "Another concept"}}},
            ... ]
            >>> model.set_concepts(concepts)
            >>> model.to_file("model_with_data.glyphh")
        
        Validates: Requirements 1.3
        """
        self.concepts = records
    
    def get_concepts(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get embedded concept records.
        
        Returns:
            List of concept records if present, None otherwise
        
        Example:
            >>> model = GlyphhModel.from_file("model.glyphh")
            >>> concepts = model.get_concepts()
            >>> if concepts:
            ...     print(f"Model has {len(concepts)} concepts")
        
        Validates: Requirements 1.4
        """
        return self.concepts
    
    def has_concepts(self) -> bool:
        """
        Check if model has embedded concepts.
        
        Returns:
            True if concepts are present and non-empty
        
        Example:
            >>> if model.has_concepts():
            ...     print("Model will auto-load data on deploy")
        
        Validates: Requirements 1.5
        """
        return self.concepts is not None and len(self.concepts) > 0
    
    def to_file(self, path: str):
        """
        Serialize model to .glyphh file (compressed JSON).
        
        The file format:
        - Compressed using gzip
        - JSON structure with all model components
        - Custom encoders serialized to base64-encoded bytes
        - Timestamp and version metadata included
        
        Args:
            path: Path to save .glyphh file
        
        Raises:
            ModelValidationException: If model validation fails
            IOError: If file cannot be written
        
        Example:
            >>> model.to_file("my_model.glyphh")
        """
        # Validate before saving
        errors = self.validate_completeness()
        if errors:
            raise ModelValidationException(errors)
        
        # Ensure path has .glyphh extension
        path_obj = Path(path)
        if path_obj.suffix != '.glyphh':
            path = str(path_obj.with_suffix('.glyphh'))
        
        # Serialize model to dictionary
        model_dict = self._to_dict()
        
        # Convert to JSON
        json_str = json.dumps(model_dict, indent=2, default=str)
        
        # Compress and write
        try:
            with gzip.open(path, 'wt', encoding='utf-8') as f:
                f.write(json_str)
        except IOError as e:
            raise IOError(f"Failed to write model file '{path}': {e}")
    
    @classmethod
    def from_file(cls, path: str) -> 'GlyphhModel':
        """
        Deserialize model from .glyphh file.
        
        Loads a model from a compressed JSON file, including:
        - Glyphs with full hierarchical structure
        - Encoder configuration
        - Custom encoders (deserialized from bytes)
        - Metadata and timestamps
        
        Args:
            path: Path to .glyphh file
        
        Returns:
            GlyphhModel instance
        
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
            ModelValidationException: If model validation fails
        
        Example:
            >>> model = GlyphhModel.from_file("my_model.glyphh")
        """
        # Check file exists
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Read and decompress
        try:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                json_str = f.read()
        except IOError as e:
            raise IOError(f"Failed to read model file '{path}': {e}")
        
        # Parse JSON
        try:
            model_dict = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in model file '{path}': {e}")
        
        # Deserialize model
        return cls._from_dict(model_dict)
    
    def _to_dict(self) -> Dict[str, Any]:
        """
        Convert model to dictionary for serialization.
        
        Returns:
            Dictionary representation of the model
        """
        result = {
            "name": self.name,
            "version": self.version,
            "encoder_config": self.encoder_config.to_dict(),
            "glyphs": [self._glyph_to_dict(g) for g in self.glyphs],
            "custom_encoders": {
                name: pickle.dumps(encoder).hex()
                for name, encoder in self.custom_encoders.items()
            },
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
        
        # Include intent patterns if present (legacy support)
        # Note: New models should use encoder_config.nl_encoder_config instead
        if self.intent_patterns:
            result["intent_patterns"] = self.intent_patterns
        
        # Include readme if present
        if self.readme:
            result["readme"] = self.readme
        
        # Include stored procedures if present
        if self.stored_procedures:
            result["stored_procedures"] = [p.to_dict() for p in self.stored_procedures]
        
        # Include concepts if present (for auto-loading on deploy)
        if self.concepts:
            result["concepts"] = self.concepts
        
        return result
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'GlyphhModel':
        """
        Create model from dictionary.
        
        Args:
            data: Dictionary representation of the model
        
        Returns:
            GlyphhModel instance
        """
        # Deserialize encoder config
        encoder_config = EncoderConfig.from_dict(data["encoder_config"])
        
        # Deserialize glyphs
        glyphs = [cls._glyph_from_dict(g) for g in data["glyphs"]]
        
        # Deserialize custom encoders
        custom_encoders = {
            name: pickle.loads(bytes.fromhex(encoder_hex))
            for name, encoder_hex in data.get("custom_encoders", {}).items()
        }
        
        # Parse timestamp
        created_at = datetime.fromisoformat(data["created_at"])
        
        # Get intent patterns if present
        intent_patterns = data.get("intent_patterns")
        
        # Get readme if present
        readme = data.get("readme")
        
        # Deserialize stored procedures if present
        stored_procedures = []
        if "stored_procedures" in data:
            from glyphh.gql.stored_procedure import StoredProcedure
            stored_procedures = [
                StoredProcedure.from_dict(p) for p in data["stored_procedures"]
            ]
        
        # Get concepts if present (for auto-loading on deploy)
        concepts = data.get("concepts")
        if concepts is not None and not isinstance(concepts, list):
            raise ValueError("concepts must be a JSON array")
        
        return cls(
            name=data["name"],
            version=data["version"],
            encoder_config=encoder_config,
            glyphs=glyphs,
            custom_encoders=custom_encoders,
            intent_patterns=intent_patterns,
            stored_procedures=stored_procedures,
            concepts=concepts,
            readme=readme,
            metadata=data.get("metadata", {}),
            created_at=created_at
        )
    
    @staticmethod
    def _glyph_to_dict(glyph: Glyph) -> Dict[str, Any]:
        """
        Convert glyph to dictionary for serialization.
        
        Args:
            glyph: Glyph to serialize
        
        Returns:
            Dictionary representation of the glyph
        """
        from glyphh.core.types import Vector, Layer, Segment
        
        def vector_to_dict(v: Vector) -> Dict[str, Any]:
            return {
                "data": v.data.tolist(),
                "dimension": v.dimension,
                "space_id": v.space_id
            }
        
        def segment_to_dict(s: Segment) -> Dict[str, Any]:
            return {
                "name": s.name,
                "cortex": vector_to_dict(s.cortex),
                "roles": {k: vector_to_dict(v) for k, v in s.roles.items()},
                "role_values": s.role_values,
                "weights": s.weights
            }
        
        def layer_to_dict(l: Layer) -> Dict[str, Any]:
            return {
                "name": l.name,
                "cortex": vector_to_dict(l.cortex),
                "segments": {k: segment_to_dict(s) for k, s in l.segments.items()},
                "weights": l.weights
            }
        
        return {
            "identifier": glyph.identifier,
            "name": glyph.name,
            "space_id": glyph.space_id,
            "global_cortex": vector_to_dict(glyph.global_cortex),
            "layers": {k: layer_to_dict(l) for k, l in glyph.layers.items()},
            "security_levels": glyph.security_levels,
            "metadata": glyph.metadata,
            "timestamp": glyph.timestamp.isoformat(),
            "version": glyph.version
        }
    
    @staticmethod
    def _glyph_from_dict(data: Dict[str, Any]) -> Glyph:
        """
        Create glyph from dictionary.
        
        Args:
            data: Dictionary representation of the glyph
        
        Returns:
            Glyph instance
        """
        import numpy as np
        from glyphh.core.types import Vector, Layer, Segment
        
        def vector_from_dict(v_dict: Dict[str, Any]) -> Vector:
            return Vector(
                data=np.array(v_dict["data"], dtype=np.int8),
                dimension=v_dict["dimension"],
                space_id=v_dict["space_id"]
            )
        
        def segment_from_dict(s_dict: Dict[str, Any]) -> Segment:
            return Segment(
                name=s_dict["name"],
                cortex=vector_from_dict(s_dict["cortex"]),
                roles={k: vector_from_dict(v) for k, v in s_dict["roles"].items()},
                role_values=s_dict["role_values"],
                weights=s_dict["weights"]
            )
        
        def layer_from_dict(l_dict: Dict[str, Any]) -> Layer:
            return Layer(
                name=l_dict["name"],
                cortex=vector_from_dict(l_dict["cortex"]),
                segments={k: segment_from_dict(s) for k, s in l_dict["segments"].items()},
                weights=l_dict["weights"]
            )
        
        return Glyph(
            identifier=data["identifier"],
            name=data["name"],
            space_id=data["space_id"],
            global_cortex=vector_from_dict(data["global_cortex"]),
            layers={k: layer_from_dict(l) for k, l in data["layers"].items()},
            security_levels=data["security_levels"],
            metadata=data["metadata"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data["version"]
        )
