# SPDF Format Specification

**Version:** 1.1
**Status:** Stable
**Last Updated:** 2026-01-01

## Overview

SPDF (Scholaris Processed Document Format) is a portable, self-contained file format for storing processed PDF documents ready for citation matching. It encapsulates OCR text, verified page numbers, text chunks, and semantic embeddings in a single compressed file.

### Design Goals

1. **Portability** — Single file, no external dependencies
2. **Efficiency** — Gzip compression, ~1.5 MB per 15-page article
3. **Self-contained** — Contains everything needed for citation matching
4. **Recoverable** — Optional page previews for reconstruction
5. **Verifiable** — Source PDF hash for integrity checking
6. **Searchable** — FTS5 full-text search with BM25 ranking built-in
7. **Reproducible** (v1.1) — Optional embedded model checkpoint for permanent reproducibility

### File Extensions

| Extension | Status |
|-----------|--------|
| `.spdf` | Primary (recommended) |
| `.scholaris` | Alternative |
| `.scpdf` | Alternative |

## File Structure

An SPDF file is a **gzip-compressed SQLite database**.

```
┌──────────────────────────────────────┐
│          SPDF File (.spdf)           │
├──────────────────────────────────────┤
│  ┌────────────────────────────────┐  │
│  │   Gzip Compression (level 6)   │  │
│  │  ┌──────────────────────────┐  │  │
│  │  │    SQLite Database       │  │  │
│  │  │  ┌────────────────────┐  │  │  │
│  │  │  │   metadata         │  │  │  │
│  │  │  │   pages            │  │  │  │
│  │  │  │   chunks           │  │  │  │
│  │  │  │   chunks_fts (FTS5)│  │  │  │
│  │  │  │   embeddings       │  │  │  │
│  │  │  │   previews         │  │  │  │
│  │  │  │   model_checkpoint │  │  │  │
│  │  │  │   (v1.1, optional) │  │  │  │
│  │  │  └────────────────────┘  │  │  │
│  │  └──────────────────────────┘  │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

## Schema (v1.0)

### Table: `metadata`

Key-value store for document metadata.

```sql
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
```

**Required Keys:**

| Key | Type | Description |
|-----|------|-------------|
| `citation_key` | string | Unique identifier (e.g., "smith2023") |
| `authors` | JSON array | List of author names |
| `year` | integer (as string) | Publication year |
| `title` | string | Document title |
| `source_pdf_hash` | string | SHA256 hash with prefix "sha256:..." |
| `source_pdf_filename` | string | Original PDF filename |
| `processed_at` | ISO 8601 | Processing timestamp |
| `ocr_model` | string | OCR model used (e.g., "gemini-2.0-flash-lite") |
| `embedding_model` | string | Embedding model (e.g., "gemini-embedding-exp-03-07") |
| `embedding_dim` | integer (as string) | Embedding dimensions (typically 768) |
| `schema_version` | integer (as string) | Schema version (currently 1) |
| `total_pages` | integer (as string) | Number of pages |
| `total_chunks` | integer (as string) | Number of chunks |

### Table: `pages`

OCR-extracted pages with verified page numbers.

```sql
CREATE TABLE pages (
    id INTEGER PRIMARY KEY,
    pdf_page INTEGER NOT NULL,        -- 1-indexed PDF page number
    book_page INTEGER NOT NULL,       -- Printed page number (can be negative for roman numerals)
    text TEXT NOT NULL,               -- OCR text content
    confidence REAL NOT NULL,         -- OCR confidence (0.0-1.0)
    is_landscape_half INTEGER NOT NULL DEFAULT 0  -- 1 if half of landscape double-page
);

CREATE INDEX idx_pages_pdf_page ON pages(pdf_page);
```

**Page Number Conventions:**

- `pdf_page`: Physical PDF page (1-indexed)
- `book_page`: Printed page number from the document
  - Positive integers: Normal pages (1, 2, 3...)
  - Negative integers: Roman numeral front matter (-1 = i, -2 = ii, -12 = xii)
  - Zero: No page number detected

### Table: `chunks`

Text segments for semantic search.

```sql
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    page_id INTEGER NOT NULL,         -- Foreign key to pages.id
    chunk_index INTEGER NOT NULL,     -- Chunk index within page
    text TEXT NOT NULL,               -- Chunk text content
    book_page INTEGER NOT NULL,       -- Denormalized for fast lookup
    pdf_page INTEGER NOT NULL,        -- Denormalized for fast lookup
    FOREIGN KEY (page_id) REFERENCES pages(id)
);

CREATE INDEX idx_chunks_book_page ON chunks(book_page);
```

**Chunking Parameters (informational):**

- Default chunk size: 500 characters
- Default overlap: 100 characters
- Chunks break at sentence boundaries when possible

### Virtual Table: `chunks_fts`

FTS5 full-text search index for fast keyword search.

```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,                                   -- Indexed text content
    content='chunks',                       -- Shadow table (contentless FTS)
    content_rowid='id',                     -- Maps to chunks.id
    tokenize='porter unicode61 remove_diacritics 1'  -- Porter stemming + Unicode
);

-- Synchronization triggers
CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER chunks_fts_update AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
```

**FTS5 Features:**

- **BM25 Ranking**: Results ranked by relevance
- **Porter Stemming**: Matches word variants (e.g., "running" matches "run")
- **Unicode Support**: Full Unicode text handling with diacritics removal
- **Query Syntax**:
  - Simple terms: `machine learning`
  - Phrases: `"neural network"`
  - Boolean: `deep AND learning`, `NOT supervised`
  - Prefix: `optim*` (matches "optimize", "optimization")
  - NEAR: `NEAR(word1 word2, 5)` (within 5 words)

**Searching (Python):**
```python
# Basic search with BM25 ranking
cursor.execute("""
    SELECT c.*, bm25(chunks_fts) as score
    FROM chunks_fts
    JOIN chunks c ON chunks_fts.rowid = c.id
    WHERE chunks_fts MATCH 'machine learning'
    ORDER BY score
    LIMIT 10
""")

# With snippet extraction
cursor.execute("""
    SELECT c.*, snippet(chunks_fts, 0, '<b>', '</b>', '...', 64) as snippet
    FROM chunks_fts
    JOIN chunks c ON chunks_fts.rowid = c.id
    WHERE chunks_fts MATCH 'neural networks'
""")
```

**Hybrid Search (FTS5 + Vector):**

Combine keyword and semantic search for better results:

```python
# 1. Get FTS candidates
fts_results = search_fts(query_text, limit=30)

# 2. Re-rank with vector similarity
for result in fts_results:
    embedding = get_embedding(result.chunk_id)
    cosine_sim = dot(query_embedding, embedding)
    result.hybrid_score = 0.3 * result.fts_score + 0.7 * cosine_sim

# 3. Return top results
return sorted(fts_results, key=lambda x: x.hybrid_score)[:10]
```

### Table: `embeddings`

Vector embeddings for each chunk.

```sql
CREATE TABLE embeddings (
    chunk_id INTEGER PRIMARY KEY,     -- Foreign key to chunks.id
    vector BLOB NOT NULL,             -- Binary float32 array
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
);
```

**Vector Format:**

- Type: Little-endian float32 array
- Dimensions: As specified in `metadata.embedding_dim` (typically 768)
- Size: `embedding_dim * 4` bytes per vector

**Reading vectors (Python):**
```python
import numpy as np
vector = np.frombuffer(blob, dtype=np.float32)
```

**Writing vectors (Python):**
```python
blob = embedding.astype(np.float32).tobytes()
```

### Table: `previews`

Optional low-resolution page images for recovery.

```sql
CREATE TABLE previews (
    pdf_page INTEGER PRIMARY KEY,     -- PDF page number (1-indexed)
    thumbnail BLOB NOT NULL,          -- JPEG image bytes
    width INTEGER NOT NULL,           -- Image width in pixels
    height INTEGER NOT NULL           -- Image height in pixels
);
```

**Preview Parameters (informational):**

- Default DPI: 100
- Default JPEG quality: 60
- Purpose: Allow document recovery if original PDF is lost

---

## Schema Additions (v1.1)

Version 1.1 adds optional model checkpoint support for full reproducibility.

### Table: `model_checkpoint` (Optional)

Stores an embedded model checkpoint for reproducible embeddings.

```sql
CREATE TABLE model_checkpoint (
    id INTEGER PRIMARY KEY DEFAULT 1,

    -- Model identification
    model_name TEXT NOT NULL,               -- "nomic-embed-text-v2-moe"
    model_version TEXT NOT NULL,            -- "v2.0"
    model_hash TEXT NOT NULL,               -- "sha256:abc123..."

    -- Source information
    source_url TEXT,                        -- HuggingFace URL for re-download
    license TEXT,                           -- "Apache-2.0"

    -- Quantization info
    quantization TEXT,                      -- "Q2_K", "Q4_K_M", "F16", etc.
    format TEXT NOT NULL,                   -- "gguf", "onnx"

    -- Storage mode
    storage_mode TEXT NOT NULL,             -- "embedded" | "external" | "api"

    -- The actual model bytes (if storage_mode = "embedded")
    -- Stored UNCOMPRESSED (GGUF is already compressed)
    checkpoint_blob BLOB,
    checkpoint_size INTEGER,                -- Size in bytes

    -- For external mode: path or model store reference
    external_path TEXT,                     -- "~/.spdf/models/sha256_abc123/"

    -- Inference parameters
    embedding_dim INTEGER NOT NULL,         -- 768
    max_tokens INTEGER,                     -- 8192
    prefix_query TEXT,                      -- "search_query: " (for asymmetric models)
    prefix_document TEXT,                   -- "search_document: "
    normalize_embeddings INTEGER DEFAULT 1, -- 1 = L2 normalize output

    CHECK (storage_mode IN ('embedded', 'external', 'api'))
);
```

### Additional Metadata Keys (v1.1)

| Key | Type | Description |
|-----|------|-------------|
| `model_storage_mode` | string | "embedded", "external", or "api" |
| `model_checkpoint_hash` | string | SHA256 hash of the model file |
| `model_reproducible` | string | "true" if embeddings can be regenerated |
| `embedding_source` | string | "local" or "api" - how embeddings were generated |

### Embedding Sets (v1.1)

v1.1 files MAY contain multiple embedding sets for different models:

```sql
-- Extended embeddings table for multi-model support
CREATE TABLE embeddings_v2 (
    chunk_id INTEGER NOT NULL,
    model_id TEXT NOT NULL,               -- References model_checkpoint or "api:model-name"
    vector BLOB NOT NULL,
    created_at TEXT,                      -- ISO 8601 timestamp
    PRIMARY KEY (chunk_id, model_id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
);
```

For backward compatibility, the original `embeddings` table remains the primary source.
`embeddings_v2` is optional and used when multiple embedding sets are stored.

### Model Storage Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `embedded` | Model bytes stored in `checkpoint_blob` | Archival, sharing, offline |
| `external` | Model in `~/.spdf/models/` by hash | Multiple files, save space |
| `api` | No local model, API-dependent | Quick creation, smaller files |

### Recommended Models

| Model | Quantization | Size | Dimensions | License |
|-------|--------------|------|------------|---------|
| nomic-embed-text-v2-moe | Q2_K | ~280 MB | 768 | Apache 2.0 |
| nomic-embed-text-v2-moe | Q4_K_M | ~560 MB | 768 | Apache 2.0 |
| all-MiniLM-L6-v2 | Q4_0 | ~80 MB | 384 | Apache 2.0 |

**Recommended default:** `nomic-embed-text-v2-moe` with `Q2_K` quantization.

### Size Impact

| Content | v1.0 (API) | v1.1 (Embedded Q2_K) |
|---------|------------|----------------------|
| 15-page article | ~1.5 MB | ~282 MB |
| 50-page thesis | ~5 MB | ~285 MB |
| 200-page book | ~15 MB | ~295 MB |

The model is the dominant factor. For collections, use `external` mode to share one model across files.

## Validation Rules

A valid SPDF file MUST:

1. **Be gzip-compressed** — Decompresses to valid SQLite database
2. **Have all required tables** — metadata, pages, chunks, embeddings, previews
3. **Have all required metadata keys** — As listed above
4. **Have valid schema_version** — Must be 1 or 2 (for v1.1)
5. **Have consistent counts** — `total_pages` = COUNT(pages), `total_chunks` = COUNT(chunks)
6. **Have matching embeddings** — One embedding per chunk, same count
7. **Have valid embedding dimensions** — Each vector = `embedding_dim * 4` bytes
8. **Have valid foreign keys** — All chunk.page_id reference valid pages.id
9. **Have valid confidence values** — 0.0 <= confidence <= 1.0
10. **Have valid hash format** — source_pdf_hash starts with "sha256:"

### Additional Validation (v1.1)

If `model_checkpoint` table exists:

11. **Valid storage_mode** — Must be "embedded", "external", or "api"
12. **Embedded model present** — If storage_mode="embedded", checkpoint_blob must not be NULL
13. **Model hash valid** — model_hash must start with "sha256:"
14. **Matching dimensions** — model_checkpoint.embedding_dim must match metadata.embedding_dim
15. **Hash verification** — SHA256(checkpoint_blob) must match model_hash (if embedded)

## Compression

- **Algorithm:** Gzip
- **Level:** 6 (default)
- **Typical compression ratio:** 60-80% reduction

## Size Estimates

| Content | Approximate Size |
|---------|-----------------|
| 15-page journal article | ~1.5 MB |
| 30-page book chapter | ~3 MB |
| 200-page full book | ~15 MB |

Primary storage is embeddings (768 * 4 * n_chunks bytes uncompressed).

## Versioning

The schema version is stored in `metadata.schema_version`.

| Version | Status | Notes |
|---------|--------|-------|
| 1 | Stable | Initial release |
| 2 | Current | v1.1 - Model checkpoint support |

### Migration Policy

- Minor updates (1.0 → 1.1): Backward compatible, new optional tables/fields only
- Major updates (1 → 2): May require migration scripts in `schema/migrations/`

### Backward Compatibility

- v1.0 readers can read v1.1 files (ignore `model_checkpoint` table)
- v1.1 readers can read v1.0 files (no model checkpoint, API-dependent embeddings)
- Schema version 2 indicates v1.1 features are present

## Reading SPDF Files

### Algorithm

1. Open file with gzip decompression
2. Write decompressed bytes to temporary SQLite file
3. Connect to SQLite database
4. Read metadata table into key-value dict
5. Load pages, chunks, embeddings, previews as needed
6. Clean up temporary file

### Minimal Python Reader

```python
import gzip
import sqlite3
import tempfile
import numpy as np
from pathlib import Path

def read_spdf(path):
    """Read an SPDF file and return its contents."""
    with gzip.open(path, 'rb') as f:
        db_bytes = f.read()

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        tmp.write(db_bytes)
        tmp_path = tmp.name

    try:
        conn = sqlite3.connect(tmp_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Read metadata
        cursor.execute("SELECT key, value FROM metadata")
        metadata = {row['key']: row['value'] for row in cursor}

        # Read pages
        cursor.execute("SELECT * FROM pages ORDER BY id")
        pages = [dict(row) for row in cursor]

        # Read chunks
        cursor.execute("SELECT * FROM chunks ORDER BY id")
        chunks = [dict(row) for row in cursor]

        # Read embeddings
        cursor.execute("SELECT chunk_id, vector FROM embeddings ORDER BY chunk_id")
        embeddings = [np.frombuffer(row['vector'], dtype=np.float32) for row in cursor]

        # Read previews
        cursor.execute("SELECT * FROM previews ORDER BY pdf_page")
        previews = [dict(row) for row in cursor]

        conn.close()
        return {'metadata': metadata, 'pages': pages, 'chunks': chunks,
                'embeddings': embeddings, 'previews': previews}
    finally:
        Path(tmp_path).unlink(missing_ok=True)
```

## Writing SPDF Files

### Algorithm

1. Create temporary SQLite database
2. Create all tables with schema
3. Insert metadata key-value pairs
4. Insert pages, chunks, embeddings, previews
5. Create indexes
6. Read database bytes
7. Gzip compress
8. Write to output file
9. Clean up temporary file

### Minimal Python Writer

```python
import gzip
import json
import sqlite3
import tempfile
import numpy as np
from pathlib import Path

def write_spdf(path, metadata, pages, chunks, embeddings, previews=None):
    """Write an SPDF file."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        conn = sqlite3.connect(tmp_path)
        cursor = conn.cursor()

        # Create schema
        cursor.executescript('''
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE pages (id INTEGER PRIMARY KEY, pdf_page INTEGER,
                book_page INTEGER, text TEXT, confidence REAL, is_landscape_half INTEGER);
            CREATE TABLE chunks (id INTEGER PRIMARY KEY, page_id INTEGER,
                chunk_index INTEGER, text TEXT, book_page INTEGER, pdf_page INTEGER);
            CREATE TABLE embeddings (chunk_id INTEGER PRIMARY KEY, vector BLOB);
            CREATE TABLE previews (pdf_page INTEGER PRIMARY KEY, thumbnail BLOB,
                width INTEGER, height INTEGER);
            CREATE INDEX idx_chunks_book_page ON chunks(book_page);
            CREATE INDEX idx_pages_pdf_page ON pages(pdf_page);
        ''')

        # Insert metadata
        for key, value in metadata.items():
            cursor.execute("INSERT INTO metadata VALUES (?, ?)", (key, value))

        # Insert pages
        for p in pages:
            cursor.execute("INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?)",
                (p['id'], p['pdf_page'], p['book_page'], p['text'],
                 p['confidence'], p.get('is_landscape_half', 0)))

        # Insert chunks
        for c in chunks:
            cursor.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                (c['id'], c['page_id'], c['chunk_index'], c['text'],
                 c['book_page'], c['pdf_page']))

        # Insert embeddings
        for i, emb in enumerate(embeddings):
            cursor.execute("INSERT INTO embeddings VALUES (?, ?)",
                (i, emb.astype(np.float32).tobytes()))

        # Insert previews
        if previews:
            for pv in previews:
                cursor.execute("INSERT INTO previews VALUES (?, ?, ?, ?)",
                    (pv['pdf_page'], pv['thumbnail'], pv['width'], pv['height']))

        conn.commit()
        conn.close()

        # Compress and write
        with open(tmp_path, 'rb') as f:
            db_bytes = f.read()

        compressed = gzip.compress(db_bytes, compresslevel=6)
        Path(path).write_bytes(compressed)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
```

## Security Considerations

1. **Hash Verification** — Always verify `source_pdf_hash` when original PDF is available
2. **Untrusted Files** — Validate schema before processing untrusted .spdf files
3. **Size Limits** — Consider limiting decompressed size to prevent zip bombs
4. **SQL Injection** — Use parameterized queries when reading metadata

## Reference Implementation

The canonical implementation is in `scholaris/auto_cite/processed_pdf.py`.

Reference reader/writer implementations are in `spdf/reference/`.

## Examples

- `spdf/examples/minimal.spdf` — Smallest valid file (1 page, no previews)
- `spdf/examples/full.spdf` — Complete example with all features

## Changelog

### Version 1.1 (2026-01-01)

- **Model checkpoint support** — Optional embedded GGUF model for reproducibility
- New table: `model_checkpoint` with model bytes, hash, inference params
- New table: `embeddings_v2` for multi-model embedding storage
- New metadata keys: `model_storage_mode`, `model_checkpoint_hash`, `model_reproducible`
- Three storage modes: `embedded`, `external`, `api`
- Recommended model: `nomic-embed-text-v2-moe` Q2_K (~280 MB)
- Full backward compatibility with v1.0

### Version 1.0 (2026-01-01)

- Initial specification release
- Tables: metadata, pages, chunks, embeddings, previews
- Gzip compression
- Support for landscape double-page detection
- Roman numeral page number support
