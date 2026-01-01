"""Tests for SPDF v1.1 features (model checkpoint, embeddings_v2)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from spdf.reference.writer import SPDFWriter, SCHEMA_VERSION_V1, SCHEMA_VERSION_V11
from spdf.reference.reader import SPDFReader, ModelCheckpoint, EmbeddingV2
from spdf.validator import SPDFValidator


class TestV11SchemaVersion:
    """Test schema version handling."""

    def test_basic_file_uses_v1_schema(self, tmp_path):
        """Files without v1.1 features should use schema version 1."""
        output = tmp_path / "v1.spdf"

        writer = SPDFWriter()
        writer.set_metadata(
            citation_key="v1_test",
            authors=["Author"],
            year=2026,
            title="V1 Test",
            ocr_model="test",
            embedding_model="test",
            embedding_dim=768,
        )
        writer.add_page(pdf_page=1, book_page=1, text="Test", confidence=0.95)
        writer.add_chunk(page_id=0, chunk_index=0, text="Test", book_page=1, pdf_page=1)
        writer.add_embedding(np.random.randn(768).astype(np.float32))
        writer.save(output)

        reader = SPDFReader()
        data = reader.read(output)

        assert data.schema_version == 1
        assert not data.has_model_checkpoint

    def test_model_checkpoint_uses_v11_schema(self, tmp_path):
        """Files with model checkpoint should use schema version 2."""
        output = tmp_path / "v11.spdf"
        model_file = tmp_path / "fake_model.gguf"
        model_file.write_bytes(b"fake model content for testing")

        writer = SPDFWriter()
        writer.set_metadata(
            citation_key="v11_test",
            authors=["Author"],
            year=2026,
            title="V1.1 Test",
            ocr_model="test",
            embedding_model="test-model",
            embedding_dim=768,
        )
        writer.add_page(pdf_page=1, book_page=1, text="Test", confidence=0.95)
        writer.add_chunk(page_id=0, chunk_index=0, text="Test", book_page=1, pdf_page=1)
        writer.add_embedding(np.random.randn(768).astype(np.float32))

        # Add model checkpoint
        writer.embed_model_checkpoint(
            model_path=model_file,
            model_name="test-model",
            model_version="1.0",
            storage_mode="embedded",
            embedding_dim=768,
        )

        writer.save(output)

        reader = SPDFReader()
        data = reader.read(output)

        assert data.schema_version == 2
        assert data.has_model_checkpoint

    def test_embeddings_v2_uses_v11_schema(self, tmp_path):
        """Files with embeddings_v2 should use schema version 2."""
        output = tmp_path / "v11_emb.spdf"

        writer = SPDFWriter()
        writer.set_metadata(
            citation_key="v11_emb_test",
            authors=["Author"],
            year=2026,
            title="V1.1 Embeddings Test",
            ocr_model="test",
            embedding_model="test-model",
            embedding_dim=768,
        )
        writer.add_page(pdf_page=1, book_page=1, text="Test", confidence=0.95)
        writer.add_chunk(page_id=0, chunk_index=0, text="Test", book_page=1, pdf_page=1)
        writer.add_embedding(np.random.randn(768).astype(np.float32))

        # Add v2 embedding
        writer.add_embedding_v2(
            chunk_id=0,
            model_id="local:test-model",
            vector=np.random.randn(768).astype(np.float32),
        )

        writer.save(output)

        reader = SPDFReader()
        data = reader.read(output)

        assert data.schema_version == 2
        assert len(data.embeddings_v2) == 1


class TestModelCheckpoint:
    """Test model checkpoint functionality."""

    def test_embedded_model_checkpoint(self, tmp_path):
        """Test embedding a model checkpoint."""
        output = tmp_path / "embedded.spdf"
        model_file = tmp_path / "model.Q2_K.gguf"
        model_content = b"fake model bytes " * 1000
        model_file.write_bytes(model_content)

        writer = SPDFWriter()
        writer.set_metadata(
            citation_key="embedded_test",
            authors=["Author"],
            year=2026,
            title="Embedded Model Test",
            ocr_model="test",
            embedding_model="nomic-embed-text-v2-moe",
            embedding_dim=768,
        )
        writer.add_page(pdf_page=1, book_page=1, text="Test content", confidence=0.95)
        writer.add_chunk(page_id=0, chunk_index=0, text="Test content", book_page=1, pdf_page=1)
        writer.add_embedding(np.random.randn(768).astype(np.float32))

        hash_result = writer.embed_model_checkpoint(
            model_path=model_file,
            model_name="nomic-embed-text-v2-moe",
            model_version="v2.0",
            storage_mode="embedded",
            embedding_dim=768,
            source_url="https://example.com/model.gguf",
        )

        assert hash_result.startswith("sha256:")
        writer.save(output)

        # Read and verify
        reader = SPDFReader()
        data = reader.read(output)

        assert data.has_model_checkpoint
        assert data.is_reproducible
        mc = data.model_checkpoint

        assert mc.model_name == "nomic-embed-text-v2-moe"
        assert mc.model_version == "v2.0"
        assert mc.storage_mode == "embedded"
        assert mc.is_embedded
        assert mc.embedding_dim == 768
        assert mc.quantization == "Q2_K"  # Inferred from filename
        assert mc.checkpoint_blob == model_content
        assert mc.checkpoint_size == len(model_content)

    def test_external_model_checkpoint(self, tmp_path):
        """Test external model reference."""
        output = tmp_path / "external.spdf"
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"fake model bytes")

        writer = SPDFWriter()
        writer.set_metadata(
            citation_key="external_test",
            authors=["Author"],
            year=2026,
            title="External Model Test",
            ocr_model="test",
            embedding_model="test-model",
            embedding_dim=768,
        )
        writer.add_page(pdf_page=1, book_page=1, text="Test", confidence=0.95)
        writer.add_chunk(page_id=0, chunk_index=0, text="Test", book_page=1, pdf_page=1)
        writer.add_embedding(np.random.randn(768).astype(np.float32))

        writer.embed_model_checkpoint(
            model_path=model_file,
            model_name="test-model",
            storage_mode="external",
            embedding_dim=768,
        )
        writer.save(output)

        reader = SPDFReader()
        data = reader.read(output)

        assert data.has_model_checkpoint
        assert data.is_reproducible
        mc = data.model_checkpoint

        assert mc.storage_mode == "external"
        assert mc.is_external
        assert not mc.is_embedded
        assert mc.checkpoint_blob is None
        assert str(model_file) in mc.external_path

    def test_extract_embedded_model(self, tmp_path):
        """Test extracting an embedded model."""
        output = tmp_path / "with_model.spdf"
        model_file = tmp_path / "original.gguf"
        model_content = b"model content for extraction test"
        model_file.write_bytes(model_content)

        writer = SPDFWriter()
        writer.set_metadata(
            citation_key="extract_test",
            authors=["Author"],
            year=2026,
            title="Extract Test",
            ocr_model="test",
            embedding_model="test-model",
            embedding_dim=768,
        )
        writer.add_page(pdf_page=1, book_page=1, text="Test", confidence=0.95)
        writer.add_chunk(page_id=0, chunk_index=0, text="Test", book_page=1, pdf_page=1)
        writer.add_embedding(np.random.randn(768).astype(np.float32))
        writer.embed_model_checkpoint(
            model_path=model_file,
            model_name="test",
            storage_mode="embedded",
            embedding_dim=768,
        )
        writer.save(output)

        # Read and extract
        reader = SPDFReader()
        data = reader.read(output)

        extracted_path = tmp_path / "extracted.gguf"
        result = data.extract_model_checkpoint(extracted_path)

        assert result == extracted_path
        assert extracted_path.exists()
        assert extracted_path.read_bytes() == model_content


class TestEmbeddingsV2:
    """Test multi-model embedding support."""

    def test_add_multiple_model_embeddings(self, tmp_path):
        """Test adding embeddings from multiple models."""
        output = tmp_path / "multi.spdf"

        writer = SPDFWriter()
        writer.set_metadata(
            citation_key="multi_test",
            authors=["Author"],
            year=2026,
            title="Multi-Model Test",
            ocr_model="test",
            embedding_model="model-a",
            embedding_dim=768,
        )
        writer.add_page(pdf_page=1, book_page=1, text="Test content", confidence=0.95)
        writer.add_chunk(page_id=0, chunk_index=0, text="Test content", book_page=1, pdf_page=1)
        writer.add_embedding(np.random.randn(768).astype(np.float32))

        # Add embeddings from two different models
        vec_a = np.random.randn(768).astype(np.float32)
        vec_b = np.random.randn(768).astype(np.float32)

        writer.add_embedding_v2(chunk_id=0, model_id="local:model-a", vector=vec_a)
        writer.add_embedding_v2(chunk_id=0, model_id="api:model-b", vector=vec_b)

        writer.save(output)

        reader = SPDFReader()
        data = reader.read(output)

        assert len(data.embeddings_v2) == 2
        assert "local:model-a" in data.list_embedding_models()
        assert "api:model-b" in data.list_embedding_models()

        # Get embeddings for specific model
        model_a_embs = data.get_embeddings_for_model("local:model-a")
        assert len(model_a_embs) == 1
        assert model_a_embs[0].chunk_id == 0


class TestV11Validation:
    """Test validator with v1.1 files."""

    def test_validator_accepts_v11_file(self, tmp_path):
        """Validator should accept valid v1.1 files."""
        output = tmp_path / "valid_v11.spdf"
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"fake model")

        writer = SPDFWriter()
        writer.set_metadata(
            citation_key="valid_v11",
            authors=["Author"],
            year=2026,
            title="Valid V1.1",
            ocr_model="test",
            embedding_model="test",
            embedding_dim=768,
        )
        writer.add_page(pdf_page=1, book_page=1, text="Test", confidence=0.95)
        writer.add_chunk(page_id=0, chunk_index=0, text="Test", book_page=1, pdf_page=1)
        writer.add_embedding(np.random.randn(768).astype(np.float32))
        writer.embed_model_checkpoint(
            model_path=model_file,
            model_name="test",
            storage_mode="embedded",
            embedding_dim=768,
        )
        writer.save(output)

        validator = SPDFValidator()
        result = validator.validate(output)

        # Should be valid (or only have info-level issues like NO_PREVIEWS)
        errors = [e for e in result.errors if e.severity.value == "error"]
        assert len(errors) == 0, f"Unexpected errors: {errors}"


class TestV11Metadata:
    """Test v1.1 metadata additions."""

    def test_metadata_with_model_checkpoint(self, tmp_path):
        """Test that v1.1 metadata fields are set correctly."""
        output = tmp_path / "meta.spdf"
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"fake model")

        writer = SPDFWriter()
        writer.set_metadata(
            citation_key="meta_test",
            authors=["Author"],
            year=2026,
            title="Metadata Test",
            ocr_model="test",
            embedding_model="test",
            embedding_dim=768,
        )
        writer.add_page(pdf_page=1, book_page=1, text="Test", confidence=0.95)
        writer.add_chunk(page_id=0, chunk_index=0, text="Test", book_page=1, pdf_page=1)
        writer.add_embedding(np.random.randn(768).astype(np.float32))
        writer.embed_model_checkpoint(
            model_path=model_file,
            model_name="test",
            storage_mode="embedded",
            embedding_dim=768,
        )
        writer.save(output)

        reader = SPDFReader()
        data = reader.read(output)

        assert data.metadata.get("model_storage_mode") == "embedded"
        assert data.metadata.get("model_reproducible") == "true"
        assert data.metadata.get("embedding_source") == "local"
        assert "model_checkpoint_hash" in data.metadata
