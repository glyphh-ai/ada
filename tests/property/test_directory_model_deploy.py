"""
Property-based tests for directory-based model deployment.

Feature: directory-model-deploy

Tests manifest loading, model discovery, deployment filtering,
metadata extraction, and query encoding dispatch.
"""

import os
import tempfile
import pytest
import yaml
from hypothesis import given, settings, strategies as st, assume
from hypothesis.strategies import composite
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domains.models.loader import (
    ModelManifest,
    load_manifest,
    discover_models,
    load_model,
)


# =============================================================================
# Strategies
# =============================================================================

SAFE_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789_-"


@composite
def manifest_strategy(draw) -> dict:
    """Generate a random manifest dict with all fields."""
    name = draw(st.text(min_size=1, max_size=30, alphabet=SAFE_CHARS))
    return {
        "name": name,
        "description": draw(st.text(min_size=0, max_size=100)),
        "version": draw(st.from_regex(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", fullmatch=True)),
        "author": draw(st.text(min_size=1, max_size=30, alphabet=SAFE_CHARS)),
        "license": draw(st.text(min_size=0, max_size=20, alphabet=SAFE_CHARS)),
        "icon": draw(st.sampled_from(["", "🤖", "📦", "🔍"])),
        "category": draw(st.text(min_size=0, max_size=20, alphabet=SAFE_CHARS)),
        "public": draw(st.booleans()),
        "load_on_startup": draw(st.booleans()),
        "tags": draw(st.lists(st.text(min_size=1, max_size=15, alphabet=SAFE_CHARS), max_size=5)),
    }


@composite
def dir_name_strategy(draw) -> str:
    """Generate a valid non-hidden directory name."""
    first = draw(st.sampled_from(list("abcdefghijklmnopqrstuvwxyz")))
    rest = draw(st.text(min_size=0, max_size=15, alphabet=SAFE_CHARS))
    return first + rest


# =============================================================================
# Property 2: Manifest Round-Trip
# Feature: directory-model-deploy, Property 2: Manifest Round-Trip
# Validates: Requirements 1.2
# =============================================================================

class TestManifestRoundTrip:
    """For any valid ModelManifest, writing to YAML and parsing back
    via load_manifest should produce an equivalent manifest."""

    @given(data=manifest_strategy())
    @settings(max_examples=100)
    def test_manifest_round_trip(self, data):
        """**Validates: Requirements 1.2**"""
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "test_model"
            model_dir.mkdir()

            manifest_path = model_dir / "manifest.yaml"
            manifest_path.write_text(yaml.dump(data))

            result = load_manifest(model_dir)

            assert result.model_id == "test_model"
            assert result.name == data["name"]
            assert result.description == data["description"]
            assert result.version == data["version"]
            assert result.author == data["author"]
            assert result.license == data["license"]
            assert result.icon == data["icon"]
            assert result.category == data["category"]
            assert result.public == data["public"]
            assert result.load_on_startup == data["load_on_startup"]
            assert result.tags == data["tags"]

    def test_manifest_fallback_to_config_yaml(self, tmp_path):
        """When manifest.yaml is missing, fall back to config.yaml
        with load_on_startup defaulting to False."""
        model_dir = tmp_path / "legacy_model"
        model_dir.mkdir()

        config = {"name": "Legacy", "version": "1.0.0", "description": "old"}
        (model_dir / "config.yaml").write_text(yaml.dump(config))

        result = load_manifest(model_dir)
        assert result.model_id == "legacy_model"
        assert result.name == "Legacy"
        assert result.load_on_startup is False

    def test_manifest_no_files(self, tmp_path):
        """When neither manifest.yaml nor config.yaml exists,
        use directory name and default all fields."""
        model_dir = tmp_path / "bare_model"
        model_dir.mkdir()

        result = load_manifest(model_dir)
        assert result.model_id == "bare_model"
        assert result.name == "bare_model"
        assert result.load_on_startup is False


# =============================================================================
# Property 1: Model Discovery Correctness
# Feature: directory-model-deploy, Property 1: Model Discovery Correctness
# Validates: Requirements 1.1
# =============================================================================

class TestModelDiscoveryCorrectness:
    """For any filesystem layout, discover_models should return exactly
    the set of non-hidden directories with correct manifests."""

    @given(
        visible_names=st.lists(dir_name_strategy(), min_size=1, max_size=5, unique=True),
        hidden_count=st.integers(min_value=0, max_value=3),
    )
    @settings(max_examples=100)
    def test_discover_returns_non_hidden_dirs(self, visible_names, hidden_count):
        """**Validates: Requirements 1.1**"""
        with tempfile.TemporaryDirectory() as tmp:
            core_dir = Path(tmp) / "models"
            core_dir.mkdir()

            for name in visible_names:
                (core_dir / name).mkdir(exist_ok=True)

            for i in range(hidden_count):
                (core_dir / f".hidden_{i}").mkdir(exist_ok=True)

            # Regular file should be excluded
            (core_dir / "not_a_dir.txt").write_text("ignore me")

            models = discover_models(core_dir)

            model_ids = {m.model_id for m in models}
            assert model_ids == set(visible_names)

            for m in models:
                assert m.model_id == m.manifest.model_id



# =============================================================================
# Dead Code Removal Tests
# Feature: directory-model-deploy
# Validates: Requirements 10.1, 10.2, 10.3, 10.4
# =============================================================================

class TestDeadCodeRemoval:
    """Verify obsolete .glyphh-based code paths are removed."""

    def test_query_embedded_concepts_removed(self):
        """_query_embedded_concepts should not exist on ModelChatService.
        **Validates: Requirements 10.1**"""
        from domains.chat.service import ModelChatService
        assert not hasattr(ModelChatService, "_query_embedded_concepts")

    def test_deploy_bundled_assistant_deleted(self):
        """deploy_bundled_assistant.py should not exist.
        **Validates: Requirements 10.2**"""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "deploy_bundled_assistant.py"
        assert not script_path.exists()

    def test_main_no_deploy_bundled_import(self):
        """main.py should not import deploy_bundled_assistant.
        **Validates: Requirements 10.3**"""
        main_path = Path(__file__).parent.parent.parent / "main.py"
        content = main_path.read_text()
        assert "deploy_bundled_assistant" not in content

    def test_chat_service_no_ask_offline(self):
        """Chat service should not reference Assistant._ask_offline.
        **Validates: Requirements 10.4**"""
        service_path = Path(__file__).parent.parent.parent / "domains" / "chat" / "service.py"
        content = service_path.read_text()
        assert "_ask_offline" not in content

    def test_main_calls_deploy_all_models(self):
        """main.py should call deploy_all_models from deploy_models.
        Models are lazy-loaded from DB on first query — no eager
        register_model_encoders call needed.
        **Validates: Requirements 5.1**"""
        main_path = Path(__file__).parent.parent.parent / "main.py"
        content = main_path.read_text()
        assert "deploy_all_models" in content



# =============================================================================
# Property 10: Metadata Extraction in ChatResult
# Feature: directory-model-deploy, Property 10: Metadata Extraction
# Validates: Requirements 8.3, 8.4
# =============================================================================

class TestMetadataExtraction:
    """For any glyph metadata dict, the extraction should produce correct
    results without raising exceptions."""

    @given(
        response=st.one_of(st.none(), st.text(min_size=0, max_size=100)),
        command=st.one_of(st.none(), st.text(min_size=0, max_size=50)),
        code=st.one_of(st.none(), st.text(min_size=0, max_size=200)),
        extra_keys=st.dictionaries(
            st.text(min_size=1, max_size=10, alphabet=SAFE_CHARS),
            st.text(min_size=0, max_size=20),
            max_size=3,
        ),
    )
    @settings(max_examples=100)
    def test_metadata_extraction_never_raises(self, response, command, code, extra_keys):
        """**Validates: Requirements 8.3, 8.4**"""
        from domains.chat.service import _extract_glyph_metadata

        metadata = {**extra_keys}
        if response is not None:
            metadata["response"] = response
        if command is not None:
            metadata["command"] = command
        if code is not None:
            metadata["code"] = code

        response_data = {
            "fact_tree": {
                "text": "",
                "children": [{"data_context": metadata}],
            }
        }

        result = _extract_glyph_metadata(response_data)

        # Should never raise
        assert isinstance(result, dict)

        # If response was in metadata and fact_tree text was empty, it should be set
        if response:
            assert result["fact_tree"]["text"] == response

    def test_empty_metadata(self):
        """Empty metadata should not raise."""
        from domains.chat.service import _extract_glyph_metadata

        result = _extract_glyph_metadata({
            "fact_tree": {"text": "existing", "children": [{"data_context": {}}]}
        })
        assert result["fact_tree"]["text"] == "existing"

    def test_no_fact_tree(self):
        """Missing fact_tree should return data unchanged."""
        from domains.chat.service import _extract_glyph_metadata

        result = _extract_glyph_metadata({"confidence": 0.5})
        assert result == {"confidence": 0.5}

    def test_no_children(self):
        """Empty children should return data unchanged."""
        from domains.chat.service import _extract_glyph_metadata

        result = _extract_glyph_metadata({
            "fact_tree": {"text": "hello", "children": []}
        })
        assert result["fact_tree"]["text"] == "hello"



# =============================================================================
# Property 3: Deployment Eligibility Filtering
# Feature: directory-model-deploy, Property 3: Deployment Eligibility Filtering
# Validates: Requirements 2.1, 2.2, 2.3
# =============================================================================


class TestDeploymentEligibilityFiltering:
    """Property 3: Deployment Eligibility Filtering.

    For any set of models with varying load_on_startup flags and sources,
    the deployed set should match the filtering rule:
    - models/ (core): deploy only if load_on_startup is True
    - custom_models/: always deploy regardless of load_on_startup flag
    """

    @given(
        core_models=st.lists(
            st.tuples(
                dir_name_strategy(),
                st.booleans(),  # load_on_startup
            ),
            min_size=1,
            max_size=5,
            unique_by=lambda x: x[0],
        ),
        custom_models=st.lists(
            st.tuples(
                dir_name_strategy(),
                st.booleans(),  # load_on_startup (ignored for custom)
            ),
            min_size=0,
            max_size=3,
            unique_by=lambda x: x[0],
        ),
    )
    @settings(max_examples=100)
    def test_filtering_rule(self, core_models, custom_models):
        """**Validates: Requirements 2.1, 2.2, 2.3**

        Core models: only deploy if load_on_startup is True.
        Custom models: always deploy regardless of flag.
        Tests the actual deploy_all_models function with real filesystem
        directories and mocked deploy_model_to_db.
        """
        import asyncio
        from unittest.mock import AsyncMock, patch, MagicMock

        # Ensure no name collisions between core and custom
        core_names = {name for name, _ in core_models}
        custom_names = {name for name, _ in custom_models}
        assume(not core_names & custom_names)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            core_dir = tmp / "models"
            custom_dir = tmp / "custom_models"
            core_dir.mkdir()
            custom_dir.mkdir()

            # Create core model directories with manifest.yaml
            for name, flag in core_models:
                model_dir = core_dir / name
                model_dir.mkdir()
                manifest = {"name": name, "load_on_startup": flag, "version": "1.0.0"}
                (model_dir / "manifest.yaml").write_text(yaml.dump(manifest))

            # Create custom model directories with manifest.yaml
            for name, flag in custom_models:
                model_dir = custom_dir / name
                model_dir.mkdir()
                manifest = {"name": name, "load_on_startup": flag, "version": "1.0.0"}
                (model_dir / "manifest.yaml").write_text(yaml.dump(manifest))

            # Track which models deploy_model_to_db is called for
            deployed_models = []

            async def fake_deploy(model_dir, org_id, model_manager, session_factory):
                deployed_models.append((model_dir.name, org_id))
                return 1  # pretend 1 glyph deployed

            mock_manager = MagicMock()
            mock_session_factory = AsyncMock()

            with patch("scripts.deploy_models.CUSTOM_MODELS_DIR", custom_dir), \
                 patch("scripts.deploy_models.deploy_model_to_db", side_effect=fake_deploy):

                from scripts.deploy_models import deploy_all_models
                results = asyncio.get_event_loop().run_until_complete(
                    deploy_all_models(
                        model_manager=mock_manager,
                        session_factory=mock_session_factory,
                    )
                )

            # Compute expected deployed set
            expected_core = {(name, "glyphh") for name, flag in core_models if flag}
            expected_custom = {(name, "custom") for name, _ in custom_models}
            expected = expected_core | expected_custom

            actual = set(deployed_models)
            assert actual == expected, (
                f"Deployed {actual} but expected {expected}. "
                f"Core models: {core_models}, Custom models: {custom_models}"
            )




# =============================================================================
# Property 7: Partial Failure Resilience
# Feature: directory-model-deploy, Property 7: Partial Failure Resilience
# Validates: Requirements 3.5
# =============================================================================


class TestPartialFailureResilience:
    """Property 7: Partial Failure Resilience.

    For any list of JSONL entries where some entries cause encoding failures,
    the number of glyphs stored should equal the number of entries that encoded
    successfully, and no exception should propagate from the deployment function.
    """

    @given(
        entries=st.lists(
            st.fixed_dictionaries({
                "question": st.text(min_size=1, max_size=50, alphabet=SAFE_CHARS + " "),
                "response": st.text(min_size=1, max_size=100),
                "command": st.one_of(st.none(), st.text(min_size=1, max_size=30)),
                "content_type": st.sampled_from(["concept", "command"]),
                "intent": st.text(min_size=1, max_size=15, alphabet=SAFE_CHARS),
                "keywords": st.lists(
                    st.text(min_size=1, max_size=10, alphabet=SAFE_CHARS),
                    max_size=3,
                ),
            }),
            min_size=2,
            max_size=15,
        ),
        fail_ratio=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=100)
    def test_partial_failure_count(self, entries, fail_ratio):
        """**Validates: Requirements 3.5**

        Deploy JSONL entries via deploy_model_to_db where some entries cause
        encoding failures. Verify:
        1. No exception propagates from deploy_model_to_db
        2. Glyph count equals the number of successfully encoded entries
        """
        import asyncio
        import json
        from unittest.mock import AsyncMock, MagicMock, patch
        from glyphh.assistant.encoder_config import entry_to_record

        # Determine which entries will fail encoding
        num_failures = max(1, int(len(entries) * fail_ratio))
        fail_indices = set(range(num_failures))
        expected_success = len(entries) - len(fail_indices)

        # Track glyphs created by the mocked storage
        created_glyphs = []

        async def fake_create_glyph(
            org_id, model_id, concept_text, embedding, metadata=None
        ):
            created_glyphs.append({
                "org_id": org_id,
                "model_id": model_id,
                "concept_text": concept_text,
                "metadata": metadata,
            })
            return MagicMock()

        # Build a temporary model directory with JSONL data
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "test_model"
            model_dir.mkdir()

            # Write manifest
            manifest = {"name": "test_model", "version": "1.0.0", "load_on_startup": True}
            (model_dir / "manifest.yaml").write_text(yaml.dump(manifest))

            # Write JSONL data
            data_dir = model_dir / "data"
            data_dir.mkdir()
            with open(data_dir / "train.jsonl", "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

            # Mock the storage layer
            mock_storage = MagicMock()
            mock_storage.count_glyphs = AsyncMock(return_value=0)
            mock_storage.delete_model_data = AsyncMock()
            mock_storage.create_glyph = AsyncMock(side_effect=fake_create_glyph)

            # Mock session with ModelConfig query returning None (fresh deploy)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.commit = AsyncMock()
            mock_session.add = MagicMock()

            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_session_factory = MagicMock(return_value=mock_session)

            mock_manager = MagicMock()

            # Build an encoder mock that fails on specific entry indices
            encode_call_count = [0]

            def failing_encode(concept):
                idx = encode_call_count[0]
                encode_call_count[0] += 1
                if idx in fail_indices:
                    raise ValueError(f"Simulated encoding failure for entry {idx}")
                mock_glyph = MagicMock()
                mock_glyph.global_cortex.data.astype.return_value.tolist.return_value = [0.1] * 768
                return mock_glyph

            mock_encoder_instance = MagicMock()
            mock_encoder_instance.encode = MagicMock(side_effect=failing_encode)
            mock_encoder_cls = MagicMock(return_value=mock_encoder_instance)

            from domains.models.loader import LoadedModel as LoaderLoadedModel, ModelManifest

            fake_loaded = LoaderLoadedModel(
                model_id="test_model",
                manifest=ModelManifest(
                    model_id="test_model",
                    name="test_model",
                    version="1.0.0",
                ),
                model_dir=model_dir,
                encoder_config=MagicMock(to_dict=MagicMock(return_value={})),
                has_custom_encoder=True,
                entry_to_record_fn=entry_to_record,
            )

            with patch("domains.models.loader.load_model", return_value=fake_loaded), \
                 patch("domains.models.storage.GlyphStorage", return_value=mock_storage), \
                 patch("glyphh.encoder.Encoder", mock_encoder_cls):

                from scripts.deploy_models import deploy_model_to_db

                # Property: no exception should propagate
                count = asyncio.get_event_loop().run_until_complete(
                    deploy_model_to_db(
                        model_dir=model_dir,
                        org_id="test_org",
                        model_manager=mock_manager,
                        session_factory=mock_session_factory,
                    )
                )

        # Property: glyph count equals number of successfully encoded entries
        assert count == expected_success, (
            f"Expected {expected_success} glyphs (from {len(entries)} entries "
            f"with {len(fail_indices)} failures), got {count}"
        )
        assert len(created_glyphs) == expected_success, (
            f"Expected {expected_success} create_glyph calls, "
            f"got {len(created_glyphs)}"
        )



# =============================================================================
# Property 4: Deployment Data Preservation (logic test)
# Feature: directory-model-deploy, Property 4: Deployment Data Preservation
# Validates: Requirements 3.1
# =============================================================================

class TestDeploymentDataPreservation:
    """Property 4: Deployment Data Preservation.

    For any model with a valid entry_to_record_fn and a non-empty list of
    JSONL entries where all entries encode successfully, after deployment
    the number of glyphs in the database for that (org_id, model_id) should
    equal the number of entries, and each glyph's glyph_metadata should
    contain the metadata produced by entry_to_record_fn for the corresponding
    entry.
    """

    @given(
        entries=st.lists(
            st.fixed_dictionaries({
                "question": st.text(min_size=1, max_size=50, alphabet=SAFE_CHARS + " "),
                "response": st.text(min_size=1, max_size=100),
                "command": st.one_of(st.none(), st.text(min_size=1, max_size=30)),
                "content_type": st.sampled_from(["concept", "command"]),
                "intent": st.text(min_size=1, max_size=15, alphabet=SAFE_CHARS),
                "keywords": st.lists(
                    st.text(min_size=1, max_size=10, alphabet=SAFE_CHARS),
                    max_size=3,
                ),
            }),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=100)
    def test_deploy_preserves_glyph_count_and_metadata(self, entries):
        """**Validates: Requirements 3.1**

        Deploy random JSONL entries via deploy_model_to_db with mocked
        storage layer, then verify:
        1. Number of glyphs stored == number of entries
        2. Each glyph's metadata matches entry_to_record_fn output
        """
        import asyncio
        import json
        from unittest.mock import AsyncMock, MagicMock, patch
        from glyphh.assistant.encoder_config import entry_to_record

        # Compute expected metadata for each entry
        expected_metadata = []
        for entry in entries:
            record = entry_to_record(entry)
            expected_metadata.append(record["metadata"])

        # Track glyphs created by the mocked storage
        created_glyphs = []

        async def fake_create_glyph(
            org_id, model_id, concept_text, embedding, metadata=None
        ):
            created_glyphs.append({
                "org_id": org_id,
                "model_id": model_id,
                "concept_text": concept_text,
                "metadata": metadata,
            })
            return MagicMock()

        # Build a temporary model directory with JSONL data
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "test_model"
            model_dir.mkdir()

            # Write manifest
            manifest = {"name": "test_model", "version": "1.0.0", "load_on_startup": True}
            (model_dir / "manifest.yaml").write_text(yaml.dump(manifest))

            # Write JSONL data
            data_dir = model_dir / "data"
            data_dir.mkdir()
            with open(data_dir / "train.jsonl", "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

            # Mock the storage layer and DB session
            mock_storage = MagicMock()
            mock_storage.count_glyphs = AsyncMock(return_value=0)
            mock_storage.delete_model_data = AsyncMock()
            mock_storage.create_glyph = AsyncMock(side_effect=fake_create_glyph)

            # Mock session with ModelConfig query returning None (fresh deploy)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.commit = AsyncMock()
            mock_session.add = MagicMock()

            # Context manager for session_factory
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_session_factory = MagicMock(return_value=mock_session)

            mock_manager = MagicMock()

            # Patch GlyphStorage to return our mock, and load_model to use
            # the real assistant encoder's entry_to_record_fn
            from domains.models.loader import LoadedModel as LoaderLoadedModel, ModelManifest

            fake_loaded = LoaderLoadedModel(
                model_id="test_model",
                manifest=ModelManifest(
                    model_id="test_model",
                    name="test_model",
                    version="1.0.0",
                ),
                model_dir=model_dir,
                encoder_config=MagicMock(to_dict=MagicMock(return_value={})),
                has_custom_encoder=True,
                entry_to_record_fn=entry_to_record,
            )

            # Mock the Encoder to return a fake glyph with embedding
            mock_glyph = MagicMock()
            mock_glyph.global_cortex.data.astype.return_value.tolist.return_value = [0.1] * 768

            mock_encoder_cls = MagicMock(return_value=MagicMock(
                encode=MagicMock(return_value=mock_glyph)
            ))

            with patch("domains.models.loader.load_model", return_value=fake_loaded), \
                 patch("domains.models.storage.GlyphStorage", return_value=mock_storage), \
                 patch("glyphh.encoder.Encoder", mock_encoder_cls):

                from scripts.deploy_models import deploy_model_to_db
                count = asyncio.get_event_loop().run_until_complete(
                    deploy_model_to_db(
                        model_dir=model_dir,
                        org_id="test_org",
                        model_manager=mock_manager,
                        session_factory=mock_session_factory,
                    )
                )

        # Property 1: glyph count == number of entries
        assert count == len(entries), (
            f"Expected {len(entries)} glyphs, got {count}"
        )
        assert len(created_glyphs) == len(entries), (
            f"Expected {len(entries)} create_glyph calls, got {len(created_glyphs)}"
        )

        # Property 2: each glyph's metadata matches entry_to_record output
        for i, (glyph, expected_meta) in enumerate(
            zip(created_glyphs, expected_metadata)
        ):
            assert glyph["metadata"] == expected_meta, (
                f"Glyph {i} metadata mismatch: "
                f"got {glyph['metadata']}, expected {expected_meta}"
            )


# =============================================================================
# Property 6: Deployment Idempotence (logic test)
# Feature: directory-model-deploy, Property 6: Deployment Idempotence
# Validates: Requirements 3.3, 3.4
# =============================================================================


class TestDeploymentIdempotence:
    """Property 6: Deployment Idempotence and Version-Based Re-deploy.

    For any model with a fixed version and JSONL data, deploying it twice
    in succession should result in the second deployment being a no-op skip
    (same glyph count, same version in DB). When the version changes between
    deployments, the old glyphs should be cleared and new glyphs created
    from the updated data.
    """

    @given(
        version=st.from_regex(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", fullmatch=True),
        entry_count=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_same_version_same_count_skips(self, version, entry_count):
        """**Validates: Requirements 3.3**

        Deploy a model, then deploy again with the same version.
        The second deployment should be a no-op skip returning the
        existing glyph count without creating new glyphs.
        """
        import asyncio
        import json
        from unittest.mock import AsyncMock, MagicMock, patch
        from glyphh.assistant.encoder_config import entry_to_record

        entries = [
            {"question": f"q{i}", "response": f"r{i}", "content_type": "concept",
             "intent": "test", "keywords": []}
            for i in range(entry_count)
        ]

        # --- First deploy: fresh (no existing data in DB) ---
        first_created = []

        async def fake_create_glyph_first(org_id, model_id, concept_text, embedding, metadata=None):
            first_created.append(concept_text)
            return MagicMock()

        def _run_deploy(model_dir, version_str, storage_count, db_version, create_side_effect):
            mock_storage = MagicMock()
            mock_storage.count_glyphs = AsyncMock(return_value=storage_count)
            mock_storage.delete_model_data = AsyncMock()
            mock_storage.create_glyph = AsyncMock(side_effect=create_side_effect)

            mock_db_config = None
            if db_version is not None:
                mock_db_config = MagicMock()
                mock_db_config.model_version = db_version

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_db_config

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.commit = AsyncMock()
            mock_session.add = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_session_factory = MagicMock(return_value=mock_session)

            from domains.models.loader import LoadedModel as LoaderLoadedModel, ModelManifest as LM

            fake_loaded = LoaderLoadedModel(
                model_id="test_model",
                manifest=LM(model_id="test_model", name="test_model", version=version_str),
                model_dir=model_dir,
                encoder_config=MagicMock(to_dict=MagicMock(return_value={})),
                has_custom_encoder=True,
                entry_to_record_fn=entry_to_record,
            )

            mock_glyph = MagicMock()
            mock_glyph.global_cortex.data.astype.return_value.tolist.return_value = [0.1] * 768
            mock_encoder_cls = MagicMock(return_value=MagicMock(
                encode=MagicMock(return_value=mock_glyph)
            ))

            with patch("domains.models.loader.load_model", return_value=fake_loaded), \
                 patch("domains.models.storage.GlyphStorage", return_value=mock_storage), \
                 patch("glyphh.encoder.Encoder", mock_encoder_cls):
                from scripts.deploy_models import deploy_model_to_db
                count = asyncio.get_event_loop().run_until_complete(
                    deploy_model_to_db(
                        model_dir=model_dir,
                        org_id="test_org",
                        model_manager=MagicMock(),
                        session_factory=mock_session_factory,
                    )
                )
            return count, mock_storage

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "test_model"
            model_dir.mkdir()
            (model_dir / "manifest.yaml").write_text(
                yaml.dump({"name": "test_model", "version": version, "load_on_startup": True})
            )
            data_dir = model_dir / "data"
            data_dir.mkdir()
            with open(data_dir / "train.jsonl", "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

            # First deploy: no existing data
            count1, _ = _run_deploy(
                model_dir, version,
                storage_count=0, db_version=None,
                create_side_effect=fake_create_glyph_first,
            )
            assert count1 == entry_count, f"First deploy: expected {entry_count}, got {count1}"

            # Second deploy: DB now has same version and same count
            second_created = []

            async def fake_create_glyph_second(org_id, model_id, concept_text, embedding, metadata=None):
                second_created.append(concept_text)
                return MagicMock()

            count2, storage2 = _run_deploy(
                model_dir, version,
                storage_count=entry_count, db_version=version,
                create_side_effect=fake_create_glyph_second,
            )

            # Second deploy should skip: return existing count, no new glyphs created
            assert count2 == entry_count, (
                f"Second deploy should return existing count {entry_count}, got {count2}"
            )
            assert len(second_created) == 0, (
                f"Second deploy should not create glyphs, but created {len(second_created)}"
            )
            # delete_model_data should NOT have been called
            storage2.delete_model_data.assert_not_called()

    @given(
        v1=st.from_regex(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", fullmatch=True),
        v2=st.from_regex(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", fullmatch=True),
        entry_count=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_different_version_redeploys(self, v1, v2, entry_count):
        """**Validates: Requirements 3.4**

        Deploy a model with version v1, then deploy again with version v2.
        The second deployment should clear old glyphs and create new ones.
        """
        assume(v1 != v2)

        import asyncio
        import json
        from unittest.mock import AsyncMock, MagicMock, patch
        from glyphh.assistant.encoder_config import entry_to_record

        entries = [
            {"question": f"q{i}", "response": f"r{i}", "content_type": "concept",
             "intent": "test", "keywords": []}
            for i in range(entry_count)
        ]

        def _run_deploy(model_dir, version_str, storage_count, db_version, create_side_effect):
            mock_storage = MagicMock()
            mock_storage.count_glyphs = AsyncMock(return_value=storage_count)
            mock_storage.delete_model_data = AsyncMock()
            mock_storage.create_glyph = AsyncMock(side_effect=create_side_effect)

            mock_db_config = None
            if db_version is not None:
                mock_db_config = MagicMock()
                mock_db_config.model_version = db_version

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_db_config

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.commit = AsyncMock()
            mock_session.add = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_session_factory = MagicMock(return_value=mock_session)

            from domains.models.loader import LoadedModel as LoaderLoadedModel, ModelManifest as LM

            fake_loaded = LoaderLoadedModel(
                model_id="test_model",
                manifest=LM(model_id="test_model", name="test_model", version=version_str),
                model_dir=model_dir,
                encoder_config=MagicMock(to_dict=MagicMock(return_value={})),
                has_custom_encoder=True,
                entry_to_record_fn=entry_to_record,
            )

            mock_glyph = MagicMock()
            mock_glyph.global_cortex.data.astype.return_value.tolist.return_value = [0.1] * 768
            mock_encoder_cls = MagicMock(return_value=MagicMock(
                encode=MagicMock(return_value=mock_glyph)
            ))

            with patch("domains.models.loader.load_model", return_value=fake_loaded), \
                 patch("domains.models.storage.GlyphStorage", return_value=mock_storage), \
                 patch("glyphh.encoder.Encoder", mock_encoder_cls):
                from scripts.deploy_models import deploy_model_to_db
                count = asyncio.get_event_loop().run_until_complete(
                    deploy_model_to_db(
                        model_dir=model_dir,
                        org_id="test_org",
                        model_manager=MagicMock(),
                        session_factory=mock_session_factory,
                    )
                )
            return count, mock_storage

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "test_model"
            model_dir.mkdir()
            data_dir = model_dir / "data"
            data_dir.mkdir()
            with open(data_dir / "train.jsonl", "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

            # First deploy with v1
            (model_dir / "manifest.yaml").write_text(
                yaml.dump({"name": "test_model", "version": v1, "load_on_startup": True})
            )
            first_created = []

            async def fake_create_first(org_id, model_id, concept_text, embedding, metadata=None):
                first_created.append(concept_text)
                return MagicMock()

            count1, _ = _run_deploy(
                model_dir, v1,
                storage_count=0, db_version=None,
                create_side_effect=fake_create_first,
            )
            assert count1 == entry_count

            # Second deploy with v2: DB has v1 with entry_count glyphs
            (model_dir / "manifest.yaml").write_text(
                yaml.dump({"name": "test_model", "version": v2, "load_on_startup": True})
            )
            second_created = []

            async def fake_create_second(org_id, model_id, concept_text, embedding, metadata=None):
                second_created.append(concept_text)
                return MagicMock()

            count2, storage2 = _run_deploy(
                model_dir, v2,
                storage_count=entry_count, db_version=v1,
                create_side_effect=fake_create_second,
            )

            # Should re-deploy: old glyphs cleared, new ones created
            assert count2 == entry_count, (
                f"Re-deploy should create {entry_count} glyphs, got {count2}"
            )
            assert len(second_created) == entry_count, (
                f"Re-deploy should create {entry_count} glyphs, created {len(second_created)}"
            )
            # delete_model_data SHOULD have been called to clear old glyphs
            storage2.delete_model_data.assert_called_once_with("test_org", "test_model")



# =============================================================================
# Property 9: Query Encoding Dispatch (logic test)
# Feature: directory-model-deploy, Property 9: Query Encoding Dispatch
# Validates: Requirements 6.1, 6.2
# =============================================================================

class TestQueryEncodingDispatch:
    """Property 9: Query Encoding Dispatch.

    For any model and query string:
    - If the model has a custom encode_query_fn, _encode_query should dispatch
      to it, call it with the query, and produce a valid embedding.
    - If the model has no custom encode_query_fn, _encode_query should produce
      an embedding via the generic lexicon-based path without error.

    **Validates: Requirements 6.1, 6.2**
    """

    @pytest.fixture(autouse=True)
    def _setup_encoders(self):
        """Create shared encoders once for all tests in this class."""
        from glyphh.assistant.encoder_config import encode_query, ENCODER_CONFIG
        from glyphh.encoder import Encoder
        from glyphh.core.config import EncoderConfig

        self._assistant_encoder = Encoder(ENCODER_CONFIG)
        self._encode_query_fn = encode_query
        # Simple encoder that can encode any concept (no strict schema)
        self._simple_encoder = Encoder(EncoderConfig(dimension=1000, seed=42))

    @given(query=st.text(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_custom_fn_produces_concept(self, query):
        """encode_query_fn always returns a valid Concept with expected attributes.

        **Validates: Requirements 6.1**
        """
        from glyphh.core.types import Concept

        result = self._encode_query_fn(query)
        assert isinstance(result, Concept)
        assert "verb" in result.attributes
        assert "object" in result.attributes
        assert "domain" in result.attributes

    @given(query=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
        min_size=1,
        max_size=80,
    ))
    @settings(max_examples=100)
    def test_custom_encode_query_fn_dispatch(self, query):
        """When a model has encode_query_fn, _encode_query dispatches to it:
        the fn is called exactly once with the query, and the result is a
        valid embedding of the correct dimension.

        **Validates: Requirements 6.1**
        """
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from glyphh.core.types import Concept
        from domains.query.service import QueryService

        encoder = self._assistant_encoder
        real_encode_query = self._encode_query_fn

        # Wrap encode_query_fn to track calls
        call_log = []

        def tracking_encode_query(q):
            result = real_encode_query(q)
            call_log.append((q, result))
            return result

        loaded_model = MagicMock()
        loaded_model.encode_query_fn = tracking_encode_query

        model_manager = MagicMock()
        model_manager.get_model = AsyncMock(return_value=loaded_model)

        service = QueryService(
            model_manager=model_manager,
            session_factory=MagicMock(),
        )

        loop = asyncio.new_event_loop()
        try:
            embedding = loop.run_until_complete(
                service._encode_query(encoder, query, org_id="test", model_id="test")
            )
        finally:
            loop.close()

        # The custom fn was called exactly once with the query
        assert len(call_log) == 1
        assert call_log[0][0] == query
        # The fn returned a valid Concept
        assert isinstance(call_log[0][1], Concept)
        # The result is a valid embedding list of correct dimension
        assert isinstance(embedding, list)
        assert len(embedding) == encoder.dimension
        assert all(isinstance(v, float) for v in embedding)

    @given(query=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
        min_size=1,
        max_size=80,
    ))
    @settings(max_examples=100)
    def test_no_encode_query_fn_uses_generic_path(self, query):
        """When a model has no encode_query_fn, _encode_query falls back to
        generic encoding and produces a valid embedding without raising.

        **Validates: Requirements 6.2**
        """
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from domains.query.service import QueryService

        # Use simple encoder so the generic fallback Concept can be encoded
        encoder = self._simple_encoder

        # LoadedModel with NO encode_query_fn
        loaded_model = MagicMock()
        loaded_model.encode_query_fn = None

        model_manager = MagicMock()
        model_manager.get_model = AsyncMock(return_value=loaded_model)

        service = QueryService(
            model_manager=model_manager,
            session_factory=MagicMock(),
        )

        loop = asyncio.new_event_loop()
        try:
            embedding = loop.run_until_complete(
                service._encode_query(encoder, query, org_id="test", model_id="test")
            )
        finally:
            loop.close()

        # Should return a valid embedding list via the generic path
        assert isinstance(embedding, list)
        assert len(embedding) == encoder.dimension
        assert all(isinstance(v, float) for v in embedding)
