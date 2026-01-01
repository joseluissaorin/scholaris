"""SPDF Reference Writer - Minimal implementation for writing .spdf files.

This module provides a minimal, dependency-light implementation of the SPDF
writer that can be used as a reference for porting to other languages.

Dependencies:
- Python 3.9+ standard library (gzip, sqlite3, json, tempfile, hashlib)
- numpy (for embedding vectors only)

Example:
    from spdf.reference import write_spdf

    write_spdf(
        path="output.spdf",
        metadata={...},
        pages=[...],
        chunks=[...],
        embeddings=[...],
    )
"""

import gzip
import hashlib
import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np


# Supported extensions
SUPPORTED_EXTENSIONS = [".spdf", ".scholaris", ".scpdf"]
DEFAULT_EXTENSION = ".spdf"

# Current schema version
SCHEMA_VERSION = 1

# SQL schema
SCHEMA_SQL = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE pages (
    id INTEGER PRIMARY KEY,
    pdf_page INTEGER NOT NULL,
    book_page INTEGER NOT NULL,
    text TEXT NOT NULL,
    confidence REAL NOT NULL,
    is_landscape_half INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    page_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    book_page INTEGER NOT NULL,
    pdf_page INTEGER NOT NULL,
    FOREIGN KEY (page_id) REFERENCES pages(id)
);

CREATE TABLE embeddings (
    chunk_id INTEGER PRIMARY KEY,
    vector BLOB NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
);

CREATE TABLE previews (
    pdf_page INTEGER PRIMARY KEY,
    thumbnail BLOB NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL
);

CREATE INDEX idx_chunks_book_page ON chunks(book_page);
CREATE INDEX idx_pages_pdf_page ON pages(pdf_page);
"""


class SPDFWriter:
    """Writer for SPDF files.

    Example:
        writer = SPDFWriter()
        writer.set_metadata(
            citation_key="smith2023",
            authors=["John Smith"],
            year=2023,
            title="Example Paper",
            source_pdf_path="source.pdf",
            ocr_model="gemini-2.0-flash-lite",
            embedding_model="gemini-embedding-exp-03-07",
            embedding_dim=768,
        )
        writer.add_page(pdf_page=1, book_page=1, text="...", confidence=0.95)
        writer.add_chunk(page_id=0, chunk_index=0, text="...", book_page=1, pdf_page=1)
        writer.add_embedding(np.array([...]))
        writer.save("output.spdf")
    """

    def __init__(self):
        self._metadata: Dict[str, str] = {}
        self._pages: List[Dict[str, Any]] = []
        self._chunks: List[Dict[str, Any]] = []
        self._embeddings: List[np.ndarray] = []
        self._previews: List[Dict[str, Any]] = []

    def set_metadata(
        self,
        citation_key: str,
        authors: List[str],
        year: int,
        title: str,
        source_pdf_path: Optional[Union[str, Path]] = None,
        source_pdf_hash: Optional[str] = None,
        source_pdf_filename: Optional[str] = None,
        ocr_model: str = "unknown",
        embedding_model: str = "unknown",
        embedding_dim: int = 768,
    ) -> None:
        """Set document metadata.

        Args:
            citation_key: Unique identifier for this source
            authors: List of author names
            year: Publication year
            title: Document title
            source_pdf_path: Path to source PDF (for hash computation)
            source_pdf_hash: Pre-computed hash (if source_pdf_path not provided)
            source_pdf_filename: Original filename (if source_pdf_path not provided)
            ocr_model: OCR model used
            embedding_model: Embedding model used
            embedding_dim: Embedding dimensions
        """
        # Compute hash if path provided
        if source_pdf_path:
            source_pdf_path = Path(source_pdf_path)
            source_pdf_hash = self._compute_file_hash(source_pdf_path)
            source_pdf_filename = source_pdf_path.name
        elif not source_pdf_hash:
            source_pdf_hash = "sha256:0" * 64
            source_pdf_filename = source_pdf_filename or "unknown"

        self._metadata = {
            'citation_key': citation_key,
            'authors': json.dumps(authors),
            'year': str(year),
            'title': title,
            'source_pdf_hash': source_pdf_hash,
            'source_pdf_filename': source_pdf_filename or "unknown",
            'processed_at': datetime.now().isoformat(),
            'ocr_model': ocr_model,
            'embedding_model': embedding_model,
            'embedding_dim': str(embedding_dim),
            'schema_version': str(SCHEMA_VERSION),
            'total_pages': '0',  # Updated on save
            'total_chunks': '0',  # Updated on save
        }

    def add_page(
        self,
        pdf_page: int,
        book_page: int,
        text: str,
        confidence: float,
        is_landscape_half: bool = False,
    ) -> int:
        """Add a page.

        Args:
            pdf_page: PDF page number (1-indexed)
            book_page: Printed page number
            text: OCR text content
            confidence: OCR confidence (0.0-1.0)
            is_landscape_half: True if half of landscape double-page

        Returns:
            Page ID (0-indexed)
        """
        page_id = len(self._pages)
        self._pages.append({
            'id': page_id,
            'pdf_page': pdf_page,
            'book_page': book_page,
            'text': text,
            'confidence': confidence,
            'is_landscape_half': int(is_landscape_half),
        })
        return page_id

    def add_chunk(
        self,
        page_id: int,
        chunk_index: int,
        text: str,
        book_page: int,
        pdf_page: int,
    ) -> int:
        """Add a text chunk.

        Args:
            page_id: Reference to page ID
            chunk_index: Chunk index within page
            text: Chunk text
            book_page: Book page number
            pdf_page: PDF page number

        Returns:
            Chunk ID (0-indexed)
        """
        chunk_id = len(self._chunks)
        self._chunks.append({
            'id': chunk_id,
            'page_id': page_id,
            'chunk_index': chunk_index,
            'text': text,
            'book_page': book_page,
            'pdf_page': pdf_page,
        })
        return chunk_id

    def add_embedding(self, vector: np.ndarray) -> None:
        """Add an embedding vector.

        Args:
            vector: Embedding vector (must match embedding_dim)
        """
        self._embeddings.append(vector.astype(np.float32))

    def add_preview(
        self,
        pdf_page: int,
        thumbnail: bytes,
        width: int,
        height: int,
    ) -> None:
        """Add a page preview.

        Args:
            pdf_page: PDF page number
            thumbnail: JPEG image bytes
            width: Image width
            height: Image height
        """
        self._previews.append({
            'pdf_page': pdf_page,
            'thumbnail': thumbnail,
            'width': width,
            'height': height,
        })

    def save(self, path: Union[str, Path], compression_level: int = 6) -> str:
        """Save to .spdf file.

        Args:
            path: Output path
            compression_level: Gzip compression level (1-9)

        Returns:
            Path to saved file
        """
        path = Path(path)

        # Ensure valid extension
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            path = path.with_suffix(DEFAULT_EXTENSION)

        # Validate
        if not self._metadata:
            raise ValueError("Metadata not set. Call set_metadata() first.")

        if len(self._embeddings) != len(self._chunks):
            raise ValueError(
                f"Embedding count ({len(self._embeddings)}) != chunk count ({len(self._chunks)})"
            )

        # Update counts in metadata
        self._metadata['total_pages'] = str(len(self._pages))
        self._metadata['total_chunks'] = str(len(self._chunks))

        # Create temp database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            conn = sqlite3.connect(tmp_path)
            cursor = conn.cursor()

            # Create schema
            cursor.executescript(SCHEMA_SQL)

            # Insert metadata
            for key, value in self._metadata.items():
                cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", (key, value))

            # Insert pages
            for page in self._pages:
                cursor.execute(
                    "INSERT INTO pages (id, pdf_page, book_page, text, confidence, is_landscape_half) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (page['id'], page['pdf_page'], page['book_page'], page['text'],
                     page['confidence'], page['is_landscape_half'])
                )

            # Insert chunks
            for chunk in self._chunks:
                cursor.execute(
                    "INSERT INTO chunks (id, page_id, chunk_index, text, book_page, pdf_page) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (chunk['id'], chunk['page_id'], chunk['chunk_index'], chunk['text'],
                     chunk['book_page'], chunk['pdf_page'])
                )

            # Insert embeddings
            for i, embedding in enumerate(self._embeddings):
                vector_bytes = embedding.tobytes()
                cursor.execute(
                    "INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)",
                    (i, vector_bytes)
                )

            # Insert previews
            for preview in self._previews:
                cursor.execute(
                    "INSERT INTO previews (pdf_page, thumbnail, width, height) VALUES (?, ?, ?, ?)",
                    (preview['pdf_page'], preview['thumbnail'], preview['width'], preview['height'])
                )

            conn.commit()
            conn.close()

            # Read and compress
            with open(tmp_path, 'rb') as f:
                db_bytes = f.read()

            compressed = gzip.compress(db_bytes, compresslevel=compression_level)

            # Write to final path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(compressed)

            return str(path)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def clear(self) -> None:
        """Clear all data to reuse writer."""
        self._metadata = {}
        self._pages = []
        self._chunks = []
        self._embeddings = []
        self._previews = []

    @staticmethod
    def _compute_file_hash(path: Union[str, Path]) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}"


def write_spdf(
    path: Union[str, Path],
    metadata: Dict[str, str],
    pages: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    embeddings: List[np.ndarray],
    previews: Optional[List[Dict[str, Any]]] = None,
    compression_level: int = 6,
) -> str:
    """Convenience function to write an SPDF file.

    Args:
        path: Output path
        metadata: Metadata dictionary (must include all required keys)
        pages: List of page dictionaries
        chunks: List of chunk dictionaries
        embeddings: List of embedding vectors
        previews: Optional list of preview dictionaries
        compression_level: Gzip compression level (1-9)

    Returns:
        Path to saved file
    """
    path = Path(path)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        path = path.with_suffix(DEFAULT_EXTENSION)

    # Update counts
    metadata = metadata.copy()
    metadata['total_pages'] = str(len(pages))
    metadata['total_chunks'] = str(len(chunks))

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        conn = sqlite3.connect(tmp_path)
        cursor = conn.cursor()

        # Create schema
        cursor.executescript(SCHEMA_SQL)

        # Insert metadata
        for key, value in metadata.items():
            cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", (key, value))

        # Insert pages
        for page in pages:
            cursor.execute(
                "INSERT INTO pages (id, pdf_page, book_page, text, confidence, is_landscape_half) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (page['id'], page['pdf_page'], page['book_page'], page['text'],
                 page['confidence'], page.get('is_landscape_half', 0))
            )

        # Insert chunks
        for chunk in chunks:
            cursor.execute(
                "INSERT INTO chunks (id, page_id, chunk_index, text, book_page, pdf_page) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (chunk['id'], chunk['page_id'], chunk['chunk_index'], chunk['text'],
                 chunk['book_page'], chunk['pdf_page'])
            )

        # Insert embeddings
        for i, embedding in enumerate(embeddings):
            vector_bytes = embedding.astype(np.float32).tobytes()
            cursor.execute(
                "INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)",
                (i, vector_bytes)
            )

        # Insert previews
        if previews:
            for preview in previews:
                cursor.execute(
                    "INSERT INTO previews (pdf_page, thumbnail, width, height) VALUES (?, ?, ?, ?)",
                    (preview['pdf_page'], preview['thumbnail'], preview['width'], preview['height'])
                )

        conn.commit()
        conn.close()

        # Read and compress
        with open(tmp_path, 'rb') as f:
            db_bytes = f.read()

        compressed = gzip.compress(db_bytes, compresslevel=compression_level)

        # Write to final path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(compressed)

        return str(path)

    finally:
        Path(tmp_path).unlink(missing_ok=True)
