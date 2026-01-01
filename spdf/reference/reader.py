"""SPDF Reference Reader - Minimal implementation for reading .spdf files.

This module provides a minimal, dependency-light implementation of the SPDF
reader that can be used as a reference for porting to other languages.

Dependencies:
- Python 3.9+ standard library (gzip, sqlite3, json, tempfile)
- numpy (for embedding vectors only)

Example:
    from spdf.reference import read_spdf

    data = read_spdf("file.spdf")
    print(data['metadata']['title'])
    for chunk in data['chunks']:
        print(f"Page {chunk['book_page']}: {chunk['text'][:50]}...")
"""

import gzip
import json
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np


# Supported extensions
SUPPORTED_EXTENSIONS = [".spdf", ".scholaris", ".scpdf"]


@dataclass
class SPDFData:
    """Container for SPDF file contents."""
    metadata: Dict[str, str]
    pages: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    embeddings: List[np.ndarray]
    previews: List[Dict[str, Any]]

    @property
    def citation_key(self) -> str:
        return self.metadata.get('citation_key', '')

    @property
    def title(self) -> str:
        return self.metadata.get('title', '')

    @property
    def authors(self) -> List[str]:
        return json.loads(self.metadata.get('authors', '[]'))

    @property
    def year(self) -> int:
        return int(self.metadata.get('year', 0))

    @property
    def embedding_dim(self) -> int:
        return int(self.metadata.get('embedding_dim', 768))

    def get_embedding_matrix(self) -> np.ndarray:
        """Get all embeddings as a numpy matrix.

        Returns:
            Array of shape (n_chunks, embedding_dim)
        """
        if not self.embeddings:
            return np.array([])
        return np.vstack(self.embeddings)

    def get_text_for_page(self, book_page: int) -> str:
        """Get concatenated text for a specific book page."""
        page_chunks = [c for c in self.chunks if c['book_page'] == book_page]
        return ' '.join(c['text'] for c in sorted(page_chunks, key=lambda x: x['chunk_index']))


class SPDFReader:
    """Reader for SPDF files.

    Example:
        reader = SPDFReader()
        data = reader.read("file.spdf")
        print(data.title)
    """

    def __init__(self):
        pass

    def read(self, path: Union[str, Path]) -> SPDFData:
        """Read an SPDF file.

        Args:
            path: Path to .spdf file

        Returns:
            SPDFData containing all file contents

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file has invalid extension or format
        """
        path = Path(path)

        # Validate path
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Invalid extension: {path.suffix}. Expected: {SUPPORTED_EXTENSIONS}")

        # Decompress
        try:
            with gzip.open(path, 'rb') as f:
                db_bytes = f.read()
        except gzip.BadGzipFile as e:
            raise ValueError(f"File is not gzip-compressed: {e}")

        # Write to temp file (SQLite requires file path)
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.write(db_bytes)
            tmp_path = tmp.name

        try:
            conn = sqlite3.connect(tmp_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Read metadata
            cursor.execute("SELECT key, value FROM metadata")
            metadata = {row['key']: row['value'] for row in cursor.fetchall()}

            # Read pages
            cursor.execute("SELECT * FROM pages ORDER BY id")
            pages = [dict(row) for row in cursor.fetchall()]

            # Read chunks
            cursor.execute("SELECT * FROM chunks ORDER BY id")
            chunks = [dict(row) for row in cursor.fetchall()]

            # Read embeddings
            cursor.execute("SELECT chunk_id, vector FROM embeddings ORDER BY chunk_id")
            embeddings = [
                np.frombuffer(row['vector'], dtype=np.float32)
                for row in cursor.fetchall()
            ]

            # Read previews
            cursor.execute("SELECT * FROM previews ORDER BY pdf_page")
            previews = [dict(row) for row in cursor.fetchall()]

            conn.close()

            return SPDFData(
                metadata=metadata,
                pages=pages,
                chunks=chunks,
                embeddings=embeddings,
                previews=previews,
            )

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def read_metadata_only(self, path: Union[str, Path]) -> Dict[str, str]:
        """Read only metadata from an SPDF file (faster than full read).

        Args:
            path: Path to .spdf file

        Returns:
            Metadata dictionary
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with gzip.open(path, 'rb') as f:
            db_bytes = f.read()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.write(db_bytes)
            tmp_path = tmp.name

        try:
            conn = sqlite3.connect(tmp_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT key, value FROM metadata")
            metadata = {row['key']: row['value'] for row in cursor.fetchall()}

            conn.close()
            return metadata

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def info(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Get summary info about an SPDF file.

        Args:
            path: Path to .spdf file

        Returns:
            Dictionary with file info
        """
        path = Path(path)
        file_size = path.stat().st_size / 1024 / 1024  # MB

        metadata = self.read_metadata_only(path)

        return {
            "citation_key": metadata.get('citation_key', ''),
            "title": metadata.get('title', ''),
            "authors": json.loads(metadata.get('authors', '[]')),
            "year": int(metadata.get('year', 0)),
            "pages": int(metadata.get('total_pages', 0)),
            "chunks": int(metadata.get('total_chunks', 0)),
            "size_mb": round(file_size, 2),
            "embedding_dim": int(metadata.get('embedding_dim', 768)),
            "schema_version": int(metadata.get('schema_version', 1)),
        }


def read_spdf(path: Union[str, Path]) -> SPDFData:
    """Convenience function to read an SPDF file.

    Args:
        path: Path to .spdf file

    Returns:
        SPDFData containing all file contents
    """
    return SPDFReader().read(path)
