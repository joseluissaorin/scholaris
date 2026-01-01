"""Tests for SPDF read/write roundtrip."""

import json
from pathlib import Path

import pytest
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from reference.reader import SPDFReader, read_spdf
from reference.writer import SPDFWriter, write_spdf
from validator import SPDFValidator


@pytest.fixture
def examples_dir():
    return Path(__file__).parent.parent / "examples"


class TestRoundtrip:
    """Test write -> read roundtrip."""

    def test_minimal_roundtrip(self, tmp_path):
        """Test roundtrip with minimal data."""
        # Write
        writer = SPDFWriter()
        writer.set_metadata(
            citation_key="roundtrip_test",
            authors=["Test Author"],
            year=2026,
            title="Roundtrip Test",
            ocr_model="test",
            embedding_model="test",
            embedding_dim=768,
        )
        writer.add_page(pdf_page=1, book_page=1, text="Test content.", confidence=0.95)
        writer.add_chunk(page_id=0, chunk_index=0, text="Test content.", book_page=1, pdf_page=1)

        np.random.seed(42)
        embedding = np.random.randn(768).astype(np.float32)
        writer.add_embedding(embedding)

        output_path = tmp_path / "roundtrip.spdf"
        writer.save(output_path)

        # Validate
        validator = SPDFValidator()
        result = validator.validate(output_path)
        assert result.valid, f"Validation failed: {result.errors}"

        # Read back
        data = read_spdf(output_path)

        # Verify
        assert data.citation_key == "roundtrip_test"
        assert data.title == "Roundtrip Test"
        assert data.authors == ["Test Author"]
        assert data.year == 2026
        assert len(data.pages) == 1
        assert len(data.chunks) == 1
        assert len(data.embeddings) == 1
        assert data.pages[0]['text'] == "Test content."
        assert data.chunks[0]['text'] == "Test content."
        np.testing.assert_array_almost_equal(data.embeddings[0], embedding)

    def test_full_roundtrip(self, tmp_path):
        """Test roundtrip with full data including previews."""
        # Write
        writer = SPDFWriter()
        writer.set_metadata(
            citation_key="full_roundtrip",
            authors=["Author One", "Author Two"],
            year=2025,
            title="Full Roundtrip Test with Multiple Pages",
            ocr_model="gemini-2.0-flash-lite",
            embedding_model="gemini-embedding-exp-03-07",
            embedding_dim=768,
        )

        # Add multiple pages
        pages_data = [
            {"pdf_page": 1, "book_page": -1, "text": "Preface text here.", "confidence": 0.92},
            {"pdf_page": 2, "book_page": 1, "text": "Chapter 1 content.", "confidence": 0.95},
            {"pdf_page": 3, "book_page": 2, "text": "Chapter 1 continued.", "confidence": 0.93},
        ]

        for pd in pages_data:
            writer.add_page(**pd)

        # Add chunks
        chunks_data = [
            {"page_id": 0, "chunk_index": 0, "text": "Preface text", "book_page": -1, "pdf_page": 1},
            {"page_id": 0, "chunk_index": 1, "text": "here.", "book_page": -1, "pdf_page": 1},
            {"page_id": 1, "chunk_index": 0, "text": "Chapter 1", "book_page": 1, "pdf_page": 2},
            {"page_id": 1, "chunk_index": 1, "text": "content.", "book_page": 1, "pdf_page": 2},
            {"page_id": 2, "chunk_index": 0, "text": "Chapter 1", "book_page": 2, "pdf_page": 3},
            {"page_id": 2, "chunk_index": 1, "text": "continued.", "book_page": 2, "pdf_page": 3},
        ]

        for cd in chunks_data:
            writer.add_chunk(**cd)

        # Add embeddings
        np.random.seed(123)
        embeddings = [np.random.randn(768).astype(np.float32) for _ in range(6)]
        for emb in embeddings:
            writer.add_embedding(emb)

        # Add previews (minimal JPEG placeholder)
        minimal_jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10] + [0] * 100 + [0xFF, 0xD9])
        for pdf_page in [1, 2, 3]:
            writer.add_preview(pdf_page=pdf_page, thumbnail=minimal_jpeg, width=100, height=150)

        output_path = tmp_path / "full_roundtrip.spdf"
        writer.save(output_path)

        # Validate
        validator = SPDFValidator()
        result = validator.validate(output_path)
        assert result.valid, f"Validation failed: {result.errors}"

        # Read back
        data = read_spdf(output_path)

        # Verify metadata
        assert data.citation_key == "full_roundtrip"
        assert data.authors == ["Author One", "Author Two"]
        assert data.year == 2025

        # Verify counts
        assert len(data.pages) == 3
        assert len(data.chunks) == 6
        assert len(data.embeddings) == 6
        assert len(data.previews) == 3

        # Verify page numbers
        assert data.pages[0]['book_page'] == -1  # Roman numeral
        assert data.pages[1]['book_page'] == 1
        assert data.pages[2]['book_page'] == 2

        # Verify embeddings match
        for i, emb in enumerate(embeddings):
            np.testing.assert_array_almost_equal(data.embeddings[i], emb)

        # Verify previews
        for preview in data.previews:
            assert preview['width'] == 100
            assert preview['height'] == 150


class TestReadExamples:
    """Test reading the pre-generated examples."""

    def test_read_minimal_example(self, examples_dir):
        """Test reading minimal.spdf."""
        data = read_spdf(examples_dir / "minimal.spdf")
        assert data.citation_key == "minimal2026"
        assert data.title == "Minimal SPDF Example"
        assert len(data.pages) == 1
        assert len(data.chunks) == 1
        assert len(data.embeddings) == 1
        assert len(data.previews) == 0

    def test_read_full_example(self, examples_dir):
        """Test reading full.spdf."""
        data = read_spdf(examples_dir / "full.spdf")
        assert data.citation_key == "beaugrande1981"
        assert "Beaugrande" in data.authors[0]
        assert data.year == 1981
        assert len(data.pages) == 3
        assert len(data.chunks) == 6
        assert len(data.embeddings) == 6
        assert len(data.previews) == 3

    def test_embedding_matrix(self, examples_dir):
        """Test getting embedding matrix."""
        data = read_spdf(examples_dir / "full.spdf")
        matrix = data.get_embedding_matrix()
        assert matrix.shape == (6, 768)

    def test_get_text_for_page(self, examples_dir):
        """Test getting text for specific page."""
        data = read_spdf(examples_dir / "full.spdf")
        text = data.get_text_for_page(1)  # Book page 1
        assert "TEXT" in text or "COMMUNICATIVE" in text


class TestWriteSpdfFunction:
    """Test the convenience write_spdf function."""

    def test_write_spdf_convenience(self, tmp_path):
        """Test write_spdf convenience function."""
        metadata = {
            'citation_key': 'convenience_test',
            'authors': json.dumps(["Test"]),
            'year': '2026',
            'title': 'Convenience Test',
            'source_pdf_hash': 'sha256:' + '0' * 64,
            'source_pdf_filename': 'test.pdf',
            'processed_at': '2026-01-01T00:00:00',
            'ocr_model': 'test',
            'embedding_model': 'test',
            'embedding_dim': '768',
            'schema_version': '1',
        }

        pages = [{'id': 0, 'pdf_page': 1, 'book_page': 1, 'text': 'Test', 'confidence': 0.9, 'is_landscape_half': 0}]
        chunks = [{'id': 0, 'page_id': 0, 'chunk_index': 0, 'text': 'Test', 'book_page': 1, 'pdf_page': 1}]
        embeddings = [np.random.randn(768).astype(np.float32)]

        output_path = tmp_path / "convenience.spdf"
        write_spdf(output_path, metadata, pages, chunks, embeddings)

        # Verify
        assert output_path.exists()
        validator = SPDFValidator()
        result = validator.validate(output_path)
        assert result.valid


class TestSPDFDataMethods:
    """Test SPDFData helper methods."""

    def test_embedding_dim_property(self, examples_dir):
        """Test embedding_dim property."""
        data = read_spdf(examples_dir / "full.spdf")
        assert data.embedding_dim == 768

    def test_metadata_only_read(self, examples_dir):
        """Test reading metadata only (fast path)."""
        reader = SPDFReader()
        metadata = reader.read_metadata_only(examples_dir / "full.spdf")
        assert metadata['citation_key'] == 'beaugrande1981'

    def test_info_method(self, examples_dir):
        """Test info method."""
        reader = SPDFReader()
        info = reader.info(examples_dir / "full.spdf")
        assert info['citation_key'] == 'beaugrande1981'
        assert info['pages'] == 3
        assert info['chunks'] == 6
        assert 'size_mb' in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
