"""Tests for the SPDF validator."""

import gzip
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from validator import SPDFValidator, ValidationResult, ValidationSeverity


@pytest.fixture
def validator():
    return SPDFValidator(strict=False)


@pytest.fixture
def examples_dir():
    return Path(__file__).parent.parent / "examples"


class TestValidatorWithExamples:
    """Test validator with pre-generated example files."""

    def test_validate_minimal_example(self, validator, examples_dir):
        """Test validation of minimal.spdf."""
        result = validator.validate(examples_dir / "minimal.spdf")
        assert result.valid
        assert result.metadata['citation_key'] == 'minimal2026'
        assert result.stats['pages'] == 1
        assert result.stats['chunks'] == 1

    def test_validate_full_example(self, validator, examples_dir):
        """Test validation of full.spdf."""
        result = validator.validate(examples_dir / "full.spdf")
        assert result.valid
        assert result.metadata['citation_key'] == 'beaugrande1981'
        assert result.stats['pages'] == 3
        assert result.stats['chunks'] == 6
        assert result.stats['previews'] == 3


class TestValidatorErrors:
    """Test validator error detection."""

    def test_file_not_found(self, validator):
        """Test error when file doesn't exist."""
        result = validator.validate("/nonexistent/file.spdf")
        assert not result.valid
        assert any(e.code == "FILE_NOT_FOUND" for e in result.errors)

    def test_invalid_extension(self, validator, tmp_path):
        """Test error for invalid extension."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("test")
        result = validator.validate(invalid_file)
        assert not result.valid
        assert any(e.code == "INVALID_EXTENSION" for e in result.errors)

    def test_not_gzip(self, validator, tmp_path):
        """Test error when file isn't gzip-compressed."""
        not_gzip = tmp_path / "test.spdf"
        not_gzip.write_text("not gzip content")
        result = validator.validate(not_gzip)
        assert not result.valid
        assert any(e.code == "NOT_GZIP" for e in result.errors)

    def test_not_sqlite(self, validator, tmp_path):
        """Test error when decompressed content isn't SQLite."""
        not_sqlite = tmp_path / "test.spdf"
        compressed = gzip.compress(b"not a sqlite database")
        not_sqlite.write_bytes(compressed)
        result = validator.validate(not_sqlite)
        assert not result.valid
        assert any(e.code == "NOT_SQLITE" for e in result.errors)


class TestValidatorWithCustomFiles:
    """Test validator with programmatically created files."""

    def create_spdf(self, tmp_path, metadata=None, pages=None, chunks=None, embeddings=None):
        """Helper to create test SPDF files."""
        db_path = tmp_path / "temp.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create schema
        cursor.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE pages (id INTEGER PRIMARY KEY, pdf_page INTEGER,
                book_page INTEGER, text TEXT, confidence REAL, is_landscape_half INTEGER);
            CREATE TABLE chunks (id INTEGER PRIMARY KEY, page_id INTEGER,
                chunk_index INTEGER, text TEXT, book_page INTEGER, pdf_page INTEGER);
            CREATE TABLE embeddings (chunk_id INTEGER PRIMARY KEY, vector BLOB);
            CREATE TABLE previews (pdf_page INTEGER PRIMARY KEY, thumbnail BLOB,
                width INTEGER, height INTEGER);
        """)

        # Insert metadata
        default_metadata = {
            'citation_key': 'test2026',
            'authors': '["Test"]',
            'year': '2026',
            'title': 'Test',
            'source_pdf_hash': 'sha256:' + '0' * 64,
            'source_pdf_filename': 'test.pdf',
            'processed_at': '2026-01-01T00:00:00',
            'ocr_model': 'test',
            'embedding_model': 'test',
            'embedding_dim': '768',
            'schema_version': '1',
            'total_pages': '1',
            'total_chunks': '1',
        }
        if metadata:
            default_metadata.update(metadata)

        for key, value in default_metadata.items():
            cursor.execute("INSERT INTO metadata VALUES (?, ?)", (key, value))

        # Insert pages
        if pages is None:
            pages = [{'id': 0, 'pdf_page': 1, 'book_page': 1, 'text': 'Test', 'confidence': 0.9, 'is_landscape_half': 0}]
        for p in pages:
            cursor.execute("INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?)",
                          (p['id'], p['pdf_page'], p['book_page'], p['text'], p['confidence'], p['is_landscape_half']))

        # Insert chunks
        if chunks is None:
            chunks = [{'id': 0, 'page_id': 0, 'chunk_index': 0, 'text': 'Test', 'book_page': 1, 'pdf_page': 1}]
        for c in chunks:
            cursor.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                          (c['id'], c['page_id'], c['chunk_index'], c['text'], c['book_page'], c['pdf_page']))

        # Insert embeddings
        if embeddings is None:
            embeddings = [np.random.randn(768).astype(np.float32).tobytes()]
        for i, emb in enumerate(embeddings):
            cursor.execute("INSERT INTO embeddings VALUES (?, ?)", (i, emb))

        conn.commit()
        conn.close()

        # Compress and save
        with open(db_path, 'rb') as f:
            db_bytes = f.read()
        compressed = gzip.compress(db_bytes)

        spdf_path = tmp_path / "test.spdf"
        spdf_path.write_bytes(compressed)
        db_path.unlink()

        return spdf_path

    def test_valid_custom_file(self, validator, tmp_path):
        """Test validation of a valid custom file."""
        spdf_path = self.create_spdf(tmp_path)
        result = validator.validate(spdf_path)
        assert result.valid

    def test_missing_metadata_key(self, validator, tmp_path):
        """Test error when required metadata is missing."""
        spdf_path = self.create_spdf(tmp_path, metadata={'citation_key': None})
        # We need to manually remove the key after creation
        # Actually, let's create a file missing the key entirely

        db_path = tmp_path / "temp2.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE pages (id INTEGER PRIMARY KEY, pdf_page INTEGER,
                book_page INTEGER, text TEXT, confidence REAL, is_landscape_half INTEGER);
            CREATE TABLE chunks (id INTEGER PRIMARY KEY, page_id INTEGER,
                chunk_index INTEGER, text TEXT, book_page INTEGER, pdf_page INTEGER);
            CREATE TABLE embeddings (chunk_id INTEGER PRIMARY KEY, vector BLOB);
            CREATE TABLE previews (pdf_page INTEGER PRIMARY KEY, thumbnail BLOB,
                width INTEGER, height INTEGER);
        """)

        # Missing citation_key
        for key, value in [
            ('authors', '["Test"]'),
            ('year', '2026'),
            ('title', 'Test'),
            ('source_pdf_hash', 'sha256:' + '0' * 64),
            ('source_pdf_filename', 'test.pdf'),
            ('processed_at', '2026-01-01'),
            ('ocr_model', 'test'),
            ('embedding_model', 'test'),
            ('embedding_dim', '768'),
            ('schema_version', '1'),
            ('total_pages', '1'),
            ('total_chunks', '1'),
        ]:
            cursor.execute("INSERT INTO metadata VALUES (?, ?)", (key, value))

        cursor.execute("INSERT INTO pages VALUES (0, 1, 1, 'Test', 0.9, 0)")
        cursor.execute("INSERT INTO chunks VALUES (0, 0, 0, 'Test', 1, 1)")
        cursor.execute("INSERT INTO embeddings VALUES (0, ?)", (np.random.randn(768).astype(np.float32).tobytes(),))

        conn.commit()
        conn.close()

        with open(db_path, 'rb') as f:
            db_bytes = f.read()
        compressed = gzip.compress(db_bytes)
        spdf_path = tmp_path / "missing_key.spdf"
        spdf_path.write_bytes(compressed)

        result = validator.validate(spdf_path)
        assert not result.valid
        assert any(e.code == "MISSING_METADATA" for e in result.errors)

    def test_count_mismatch(self, validator, tmp_path):
        """Test error when counts don't match."""
        spdf_path = self.create_spdf(tmp_path, metadata={'total_pages': '5'})  # Says 5, has 1
        result = validator.validate(spdf_path)
        assert not result.valid
        assert any(e.code == "PAGE_COUNT_MISMATCH" for e in result.errors)

    def test_invalid_confidence(self, validator, tmp_path):
        """Test error for invalid confidence values."""
        pages = [{'id': 0, 'pdf_page': 1, 'book_page': 1, 'text': 'Test', 'confidence': 1.5, 'is_landscape_half': 0}]
        spdf_path = self.create_spdf(tmp_path, pages=pages)
        result = validator.validate(spdf_path)
        assert not result.valid
        assert any(e.code == "INVALID_CONFIDENCE" for e in result.errors)

    def test_embedding_dimension_mismatch(self, validator, tmp_path):
        """Test error when embedding dimensions don't match metadata."""
        # Create with 512-dim embedding but metadata says 768
        embeddings = [np.random.randn(512).astype(np.float32).tobytes()]
        spdf_path = self.create_spdf(tmp_path, embeddings=embeddings)
        result = validator.validate(spdf_path)
        assert not result.valid
        assert any(e.code == "INVALID_EMBEDDING_DIM" for e in result.errors)


class TestStrictMode:
    """Test strict mode behavior."""

    def test_strict_mode_treats_warnings_as_errors(self, examples_dir):
        """Test that strict mode fails on warnings."""
        validator = SPDFValidator(strict=True)
        result = validator.validate(examples_dir / "minimal.spdf")
        # minimal.spdf has NO_PREVIEWS info - in strict mode this should still pass
        # because INFO is not treated as error even in strict mode
        # Let's check what warnings we have
        has_warnings = any(e.severity == ValidationSeverity.WARNING for e in result.errors)
        if has_warnings:
            assert not result.valid
        else:
            assert result.valid

    def test_non_strict_mode_allows_warnings(self, examples_dir):
        """Test that non-strict mode allows warnings."""
        validator = SPDFValidator(strict=False)
        result = validator.validate(examples_dir / "minimal.spdf")
        assert result.valid  # Passes despite INFO messages


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
