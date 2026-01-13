---
name: scholaris
description: AI-powered academic citation system. Use when working with academic papers, PDFs, bibliography management, generating in-text citations, processing research documents, or when the user mentions citations, bibliography, academic writing, SPDF files, or reference management. Automatically inserts verified page-number citations into documents.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash(python:*, python3:*, pip:*, curl:*)
  - Grep
  - Glob
  - WebSearch
  - WebFetch
---

# Scholaris - AI-Powered Academic Citation System

Scholaris automatically inserts in-text citations with verified page numbers into academic documents. It uses Vision OCR, semantic RAG, and temporal logic to ensure accurate, grounded citations.

## Quick Start

### 1. Setup Environment

```python
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Import scholaris
from scholaris.auto_cite import CitationIndex, CitationStyle, ProcessedPDF
from scholaris import Scholaris
```

### 2. Process a PDF to SPDF Format

```python
# Process a single PDF with metadata
processed = ProcessedPDF.from_pdf(
    pdf_path="paper.pdf",
    citation_key="smith2024",
    authors=["John Smith", "Jane Doe"],
    year=2024,
    title="Understanding Neural Networks",
    gemini_api_key=GEMINI_API_KEY,
    include_previews=False,
)
processed.save("paper.spdf")
```

### 3. Cite a Document

```python
# Load bibliography from folder of .spdf files
index = CitationIndex.from_bibliography(
    folder="./bibliography/",
    gemini_api_key=GEMINI_API_KEY,
    auto_process=True,
    save_processed=True,
)

# Generate citations
result = index.cite_document(
    document_text="Your academic paper text...",
    style=CitationStyle.APA7,
    batch_size=3,
    min_confidence=0.6,
    include_bibliography=True,
)

# Save result
with open("paper_cited.md", "w") as f:
    f.write(result.modified_document)

print(f"Inserted {result.metadata['total_citations']} citations")
```

## Core Concepts

### SPDF Format (Shareable Processed PDF)
- **Process once, use forever** - No re-processing needed
- Contains: OCR text, embeddings, page metadata, language detection
- Extensions: `.spdf`, `.scholaris`, `.scpdf`
- Size: ~1.5 MB per 15-page article

### Citation Types
| Type | Description |
|------|-------------|
| `DIRECT_SUPPORT` | Source directly makes this claim |
| `FRAMEWORK_APPLICATION` | Applying older theory to new domain |
| `BACKGROUND_CONTEXT` | General domain knowledge |
| `TEMPORAL_IMPOSSIBLE` | Source predates the concept |

### Citation Styles
- `CitationStyle.APA7` - APA 7th Edition
- `CitationStyle.CHICAGO17` - Chicago 17th Edition

## Common Tasks

### Search for Papers
```python
scholar = Scholaris(gemini_api_key=GEMINI_API_KEY)
papers = scholar.search_papers(
    topic="machine learning transformers",
    max_papers=10,
    min_year=2020,
)
```

### Generate BibTeX
```python
bibtex_entries = scholar.generate_bibtex(
    pdf_paths=["paper1.pdf", "paper2.pdf"],
    method="auto",  # "auto", "pdf2bib", or "llm"
)
scholar.export_bibtex(bibtex_entries, "references.bib")
```

### Batch Process PDFs
```python
pdf_metadata = {
    "paper1.pdf": {"citation_key": "smith2024", "authors": ["Smith"], "year": 2024, "title": "Title 1"},
    "paper2.pdf": {"citation_key": "jones2023", "authors": ["Jones"], "year": 2023, "title": "Title 2"},
}

for pdf_file, meta in pdf_metadata.items():
    processed = ProcessedPDF.from_pdf(
        pdf_path=pdf_file,
        gemini_api_key=GEMINI_API_KEY,
        **meta
    )
    processed.save(f"{meta['citation_key']}.spdf")
```

## Workflow Patterns

For detailed workflows, see [WORKFLOWS.md](WORKFLOWS.md).

For complete API reference, see [REFERENCE.md](REFERENCE.md).

## Requirements

- Python 3.9+
- Gemini API Key (set as `GEMINI_API_KEY` environment variable)
- Dependencies: `pip install scholaris` or from git

## File Locations

| Purpose | Typical Location |
|---------|------------------|
| Source PDFs | `./pdfs/` or `./bibliography/` |
| Processed SPDFs | `./spdf/` or same as PDFs |
| Output documents | Same directory as input |
| Environment file | `.env` in project root |

## Error Handling

```python
try:
    result = index.cite_document(document_text, style=CitationStyle.APA7)
except Exception as e:
    print(f"Citation failed: {e}")
    # Fallback: return original document
```

## Tips

1. **Always provide metadata** - citation_key, authors, year, title improve matching
2. **Use SPDF caching** - Process PDFs once, reuse the .spdf files
3. **Set appropriate confidence** - 0.6 is a good default, lower for more citations
4. **Check temporal logic** - Scholaris detects anachronistic citations automatically
5. **Review framework rewrites** - When citing old theories for new concepts
