# CLAUDE.md - Scholaris Project Guide

## Project Overview

**Scholaris** is an AI-powered academic citation system that automatically inserts verified citations into academic writing. Unlike reference managers (Zotero, Mendeley), Scholaris reads your text, matches claims to sources, and inserts properly formatted citations with real page numbers.

### Key Differentiators

1. **Auto-insertion** — Citations inserted automatically, not just organized
2. **Verified page numbers** — Vision OCR reads printed page numbers from PDFs
3. **Temporal logic** — Detects anachronistic citations (e.g., citing 1981 paper for BERT)
4. **Framework rewrites** — Proper attribution when applying older theories to new domains
5. **Shareable .spdf format** — Process once, share and reuse forever
6. **Cross-lingual matching** — Spanish documents cite English sources seamlessly (v1.2)
7. **Markdown-tolerant** — Handles `*italics*`, `**bold**` in claim matching (v1.2)

## Architecture

```
scholaris/
├── auto_cite/           # Core citation system
│   ├── citation_index.py    # Main entry point - CitationIndex class
│   ├── citation_engine.py   # Gemini-powered citation matching
│   ├── models.py            # Data models (ProcessedPDF, Citation, etc.)
│   ├── vision_ocr.py        # Gemini Vision OCR for scanned PDFs
│   └── page_aware_rag.py    # Embedding and retrieval
├── providers/           # External service integrations
│   ├── llm/gemini.py       # Gemini API wrapper
│   └── scholar/            # Paper search
└── utils/               # Utilities
```

## Core Components

### CitationIndex (`citation_index.py`)

The main entry point for the citation system. Manages:
- Loading .spdf files from bibliography folder
- Building embedding matrix for semantic search
- Generating citations via `cite_document()`

```python
from scholaris.auto_cite import CitationIndex, CitationStyle

index = CitationIndex.from_bibliography(
    folder="./bib/",
    gemini_api_key="your-key",
    auto_process=True
)

result = index.cite_document(
    document_text="Your paper...",
    style=CitationStyle.APA7
)
```

**Key Methods:**
- `from_bibliography()` — Load sources from folder
- `add()` — Add a ProcessedPDF to index
- `query()` — Semantic search for relevant chunks
- `cite_document()` — Generate citations for a document

### GeminiCitationEngine (`citation_engine.py`)

Handles the actual citation matching logic:
- Paragraph-by-paragraph processing with batching
- Citation type classification (DIRECT_SUPPORT, FRAMEWORK_APPLICATION, etc.)
- Temporal impossibility detection
- Framework rewrite generation
- Language-aware quotation marks

**Key Methods:**
- `analyze_with_grounded_rag()` — Main citation generation
- `_cite_paragraph_with_rag()` — Process single batch
- `_build_grounded_rag_prompt()` — Construct Gemini prompt
- `_parse_grounded_citations()` — Parse AI response

### ProcessedPDF (`models.py`)

Represents a processed academic source:
- OCR text with verified page numbers
- Chunked text (500 chars, 100 overlap)
- Embeddings (768-dim Gemini vectors)
- Optional page previews for recovery

**Key Methods:**
- `from_pdf()` — Process PDF with Vision OCR
- `save()` / `load()` — Serialize to/from .spdf format

### Citation Types

```python
class CitationType(Enum):
    DIRECT_SUPPORT = "direct_support"           # Source directly makes claim
    FRAMEWORK_APPLICATION = "framework_application"  # Applying theory to new domain
    BACKGROUND_CONTEXT = "background_context"    # General domain knowledge
    TEMPORAL_IMPOSSIBLE = "temporal_impossible"  # Source predates concept
    NOVEL_CONTRIBUTION = "novel_contribution"    # Author's original work
```

## .spdf Format

Gzip-compressed SQLite database containing:

| Table | Contents |
|-------|----------|
| `metadata` | Citation key, authors, year, title, language (ISO 639-1), hash |
| `pages` | PDF page ↔ book page mapping, OCR text |
| `chunks` | Text segments with page numbers, chunk_index |
| `embeddings` | 768-dim vectors as binary blobs |
| `previews` | Optional low-res page images (JPEG) |

**Extensions:** `.spdf`, `.scholaris`, `.scpdf`

**Language field (v1.2):** Each SPDF stores detected language code (en, es, fr, de) for cross-lingual citation matching.

## Key Algorithms

### Chunk Lookup Index

The `_chunk_lookup` list maps embedding matrix rows to (citation_key, chunk_index) pairs:

```python
# When adding a source:
for i, chunk in enumerate(processed_pdf.chunks):
    self._chunk_lookup.append((citation_key, i))  # Use index, not chunk.id

# When querying:
citation_key, chunk_idx = self._chunk_lookup[embedding_row_idx]
chunk_data = source.processed_pdf.chunks[chunk_idx]
```

**Important:** Always use enumerate index, not `chunk.id`, to avoid index errors.

### Temporal Logic

The system detects when sources cannot support claims:

```python
# Modern NLP concepts requiring recent sources:
MODERN_NLP_KEYWORDS = {
    "transformer": 2017,
    "attention mechanism": 2017,
    "BPE": 2016,
    "WordPiece": 2016,
    "BERT": 2018,
    "GPT": 2018,
    "embeddings": 2013,
}

# If source.year < concept_year → FRAMEWORK_APPLICATION
```

### Language Detection

Quotation mark style based on word frequency:

```python
spanish_indicators = ["el", "la", "los", "de", "que", "en"]
french_indicators = ["le", "la", "les", "de", "des", "du"]

# Spanish: «texto» (no space)
# French: « texte » (with space)
# English: "text"
```

### Cross-Lingual Matching (v1.2)

Documents in one language can cite sources in another:

```python
# Document language detected from text
# Source languages detected from chunk samples
# When languages differ, prompt includes semantic equivalences:
#   - Spanish "cohesión léxica" = English "lexical cohesion"
#   - Spanish "coherencia textual" = English "textual coherence"
```

### Markdown-Tolerant Claim Matching (v1.2)

The `_find_claim_position()` function uses a 5-try strategy:

1. **Exact match** — `document.find(claim)`
2. **Case-insensitive** — `document.lower().find(claim.lower())`
3. **Unicode-normalized** — Handle special characters
4. **Regex with markdown** — Match `*text*` as `text`
5. **Full markdown strip** — Strip all `*_` markers, build position map, match, then map back to original positions

**Key fix:** Returns `(position, matched_length)` tuple to handle markdown markers correctly when calculating insertion points.

## Common Tasks

### Process New PDFs

```python
from scholaris.auto_cite import ProcessedPDF

processed = ProcessedPDF.from_pdf(
    pdf_path="paper.pdf",
    citation_key="author2024",
    authors=["First Author", "Second Author"],
    year=2024,
    title="Paper Title",
    gemini_api_key="your-key"
)
processed.save("paper.spdf")
```

### Batch Process PDFs

```python
from pathlib import Path
from scholaris.auto_cite import ProcessedPDF

bib_folder = Path("./bib/")
for pdf_path in bib_folder.glob("*.pdf"):
    spdf_path = pdf_path.with_suffix(".spdf")
    if not spdf_path.exists():
        # Parse metadata from filename: Author_Year_Title.pdf
        parts = pdf_path.stem.split("_")
        processed = ProcessedPDF.from_pdf(
            pdf_path=pdf_path,
            citation_key=pdf_path.stem.lower(),
            authors=[parts[0]] if parts else [],
            year=int(parts[1]) if len(parts) > 1 else 0,
            title=" ".join(parts[2:]) if len(parts) > 2 else pdf_path.stem,
            gemini_api_key="your-key"
        )
        processed.save(spdf_path)
```

### Debug Citation Issues

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check index stats
print(f"Sources: {len(index)}")
print(f"Chunks: {index.total_chunks}")
print(f"Chunk lookup size: {len(index._chunk_lookup)}")

# Test query
chunks = index.query("test query", n_results=5)
for c in chunks:
    print(f"{c.citation_key} p.{c.book_page}: {c.text[:100]}...")
```

## Configuration

### Environment Variables

```bash
GEMINI_API_KEY=your_key_here       # Required
CROSSREF_EMAIL=your@email.com      # Optional (better metadata)
```

### Citation Engine Settings

```python
engine = GeminiCitationEngine(
    api_key="your-key",
    model="gemini-3-flash-preview",  # Default: gemini-3-flash-preview
    aggressive_mode=True,             # More citations
    paragraph_mode=True,              # Batch processing
)
```

## Testing

```bash
# Run citation test
python test_trabajo_new_logic.py

# Expected output:
# Total citations: 150+
# Framework rewrites: 20+
# Processing time: ~2 min
```

## Known Issues & Solutions

### "list index out of range" in batch processing

**Cause:** `_chunk_lookup` stores stale indices when sources are replaced.

**Solution:** The `add()` method now rebuilds `_chunk_lookup` when replacing sources:

```python
if citation_key in self._sources:
    del self._sources[citation_key]
    self._chunk_lookup = []
    for source in self._sources.values():
        for i, chunk in enumerate(source.processed_pdf.chunks):
            self._chunk_lookup.append((source.citation_key, i))
```

### Duplicate sources warning

**Cause:** Same PDF loaded from both .pdf and .spdf files.

**Solution:** Set `auto_process=False` when .spdf files already exist, or ensure PDF filenames match .spdf filenames.

### Vision OCR failures

**Cause:** API rate limits or malformed PDFs.

**Solution:** Add retry logic and check PDF validity before processing.

### Text corruption in framework rewrites (fixed in v1.2)

**Cause:** `_find_claim_position()` returned only position, but markdown markers (`*italics*`) caused length mismatch between claim and actual document text.

**Solution:** Function now returns `(position, matched_length)` tuple. Insertion logic uses `matched_length` instead of `len(claim)`.

### Cross-lingual citations not matching (fixed in v1.2)

**Cause:** Spanish document claims not matching English source evidence due to language barrier.

**Solution:** Added language detection to SPDF metadata and cross-lingual semantic matching instructions to citation engine prompt.

## API Reference

### CitationResult

```python
@dataclass
class CitationResult:
    modified_document: str          # Document with citations inserted
    citations: List[GroundedCitation]  # All citations
    metadata: Dict[str, Any]        # Stats and info
    warnings: List[str]             # Any issues encountered
```

### GroundedCitation

```python
@dataclass
class GroundedCitation:
    claim_text: str                 # Original text from document
    citation_key: str               # Source identifier
    book_page: int                  # Verified page number
    evidence_text: str              # Supporting text from source
    confidence: float               # 0.0-1.0 reliability
    citation_type: CitationType     # Classification
    suggested_rewrite: Optional[str]  # For framework applications
    year: int                       # Source publication year
```

## Development

### Adding New Citation Types

1. Add to `CitationType` enum in `models.py`
2. Update prompt in `_build_grounded_rag_prompt()` in `citation_engine.py`
3. Handle in `_parse_grounded_citations()`

### Adding New Output Formats

1. Create exporter in `citation_export.py`
2. Register format in `CitationOrchestrator`
3. Add tests

### Modifying OCR Logic

Vision OCR lives in `vision_ocr.py`:
- `_ocr_page()` — Single page OCR
- `_detect_layout()` — Single vs landscape detection
- `_extract_page_numbers()` — Header/footer parsing

## Performance Tips

1. **Use .spdf files** — Avoid re-processing PDFs
2. **Batch size 3-4** — Optimal for parallel processing
3. **Limit sources** — 30-50 most relevant works
4. **Pre-filter paragraphs** — Skip short paragraphs (<50 chars)

## License

MIT License - See LICENSE file.
