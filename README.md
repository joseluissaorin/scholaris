<div align="center">

# Scholaris

### **AI-Powered Academic Citation System**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0.0-blue?style=for-the-badge)](https://github.com/joseluissaorin/scholaris)

**Automatically cite your academic writing with verified page numbers.**

Vision OCR · Page-Aware RAG · Grounded Citations · Shareable .spdf Format

[Get Started](#-quick-start) · [Auto-Citation](#-auto-citation-system) · [.spdf Format](#-shareable-spdf-format)

</div>

---

<br/>

## What Makes Scholaris Different

Other tools help you *organize* references. Scholaris **automatically inserts citations into your writing** with real, verified page numbers.

<table>
<tr>
<td width="50%" valign="top">

### The Problem

You write: *"Transformers revolutionized NLP through self-attention mechanisms."*

Now you need to:
- Find which paper supports this claim
- Hunt through a 50-page PDF for the right page
- Format the citation correctly
- Repeat for every claim in your document

</td>
<td width="50%" valign="top">

### The Solution

Scholaris reads your text, matches claims to sources, and inserts citations automatically:

*"Transformers revolutionized NLP through self-attention mechanisms.* **(Vaswani et al., 2017, p. 1)**"

- Page numbers verified via Vision OCR
- Every citation grounded in retrieved evidence
- Confidence scores show reliability

</td>
</tr>
</table>

<br/>

---

<br/>

## Auto-Citation in Action

<div align="center">

![Auto-Citation Example](media/example-auto-citation.png)

*Your text → AI matches claims to sources → Verified page numbers with confidence scores*

</div>

<br/>

**Before:**
> Beaugrande and Dressler identified seven standards of textuality that distinguish a text from a random collection of sentences: cohesion, coherence, intentionality, acceptability, informativity, situationality, and intertextuality.

**After:**
> Beaugrande and Dressler identified seven standards of textuality that distinguish a text from a random collection of sentences: cohesion, coherence, intentionality, acceptability, informativity, situationality, and intertextuality. **(Beaugrande et al., 1981, p. 3)**

<br/>

---

<br/>

## Core Features

<table>
<tr>
<td width="50%" valign="top">

### Vision OCR + Page-Aware RAG
- **Scanned PDF support** — OCR extracts text from images
- **Real page numbers** — reads printed page numbers, not PDF indices
- **Landscape detection** — handles double-page book scans
- **Roman numerals** — front matter (i, ii, xii) supported

### Grounded Citations
- **No guessing** — page numbers come from retrieval, not hallucination
- **Confidence scores** — 0.0-1.0 reliability rating per citation
- **Evidence tracking** — see exactly what text matched each claim
- **CSV/JSON export** — full audit trail for verification

</td>
<td width="50%" valign="top">

### Shareable .spdf Format
- **Process once, use forever** — no re-processing needed
- **Single portable file** — share with colleagues instantly
- **Contains everything** — OCR text, embeddings, page metadata
- **Recovery features** — export previews if original PDF lost

### Citation Styles & Formats
- **APA 7th & Chicago 17th** — automatic formatting
- **Multi-format I/O** — DOCX, PDF, Markdown, LaTeX, HTML, RTF, ODT
- **Page ranges** — supports `pp. 123-126` for multi-page concepts

</td>
</tr>
</table>

<br/>

---

<br/>

## Auto-Citation System

### Basic Usage

```python
from scholaris.auto_cite import CitationOrchestrator
from scholaris.auto_cite.models import CitationRequest, CitationStyle

orchestrator = CitationOrchestrator(gemini_api_key="your-key")

# Process your bibliography
bibliography = orchestrator.process_bibliography(
    pdf_paths=["paper1.pdf", "paper2.pdf"],
    citation_keys=["smith2023", "jones2024"],
    references=references,
    bib_entries=bib_entries
)

# Insert citations into your document
request = CitationRequest(
    document_text="Your research paper text here...",
    bibliography=bibliography,
    style=CitationStyle.APA7,
    preview_mode=True  # Preview before applying
)

result = orchestrator.insert_citations(request)

# Review results
for citation in result.citations:
    print(f"{citation.citation_string} (confidence: {citation.confidence})")
```

### Multi-Format Workflow

```python
# Read DOCX → Insert citations → Export to PDF
result = orchestrator.insert_citations_with_export(
    input_file="draft.docx",
    output_file="cited_paper.pdf",
    bibliography=bibliography,
    style=CitationStyle.APA7
)

print(f"Inserted {len(result.citations)} citations")
```

### Supported Formats

| Format | Input | Output |
|--------|:-----:|:------:|
| Plain Text (.txt) | Yes | Yes |
| Markdown (.md) | Yes | Yes |
| Microsoft Word (.docx) | Yes | Yes |
| PDF (.pdf) | Yes | Yes |
| LaTeX (.tex) | Yes | Yes |
| HTML (.html) | Yes | Yes |
| Rich Text (.rtf) | Yes | Yes |
| OpenDocument (.odt) | Yes | Yes |

<br/>

---

<br/>

## Shareable .spdf Format

Process PDFs once, share and reuse forever. The `.spdf` format stores everything needed for citation matching in a single compressed file.

### Create .spdf Files

```python
from scholaris.auto_cite import ProcessedPDF

# Process a PDF (Vision OCR + embeddings)
processed = ProcessedPDF.from_pdf(
    pdf_path="beaugrande1981.pdf",
    citation_key="beaugrande1981",
    authors=["R.A. de Beaugrande", "W.U. Dressler"],
    year=1981,
    title="Introduction to Text Linguistics",
    gemini_api_key="your-key",
    include_previews=True  # Store low-res pages for recovery
)

# Save as shareable file
processed.save("beaugrande1981.spdf")
```

### Use .spdf Files

```python
from scholaris.auto_cite import CitationIndex, CitationStyle

# Load from a bibliography folder
# Existing .spdf files load instantly (no API calls)
# New PDFs are auto-processed and saved as .spdf
index = CitationIndex.from_bibliography(
    folder="./bibliography/",
    gemini_api_key="your-key",
    auto_process=True,
    save_processed=True
)

# Generate citations
citations = index.cite(
    document_text="Your research paper text...",
    style=CitationStyle.APA7
)
```

### What's Inside .spdf

| Content | Description |
|---------|-------------|
| **Metadata** | Citation key, authors, year, title, source PDF hash |
| **Pages** | PDF page ↔ book page mapping, OCR confidence |
| **Chunks** | Text segments with verified page numbers |
| **Embeddings** | 768-dim vectors for semantic search |
| **Previews** | Optional low-res page images for recovery |

**File sizes:** ~1.5 MB per 15-page article, ~15 MB per 200-page book

**Supported extensions:** `.spdf`, `.scholaris`, `.scpdf`

<br/>

---

<br/>

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                    PHASE 1: VISION OCR                       │
├──────────────────────────────────────────────────────────────┤
│  PDF Page → Render Image → Gemini Vision OCR                 │
│     • Detect layout (single page vs landscape double-page)   │
│     • Extract printed page numbers from image                │
│     • OCR full text content (handles scanned books)          │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                 PHASE 2: PAGE-AWARE INDEXING                 │
├──────────────────────────────────────────────────────────────┤
│  For each extracted page:                                    │
│     1. Chunk text (500 chars, 100 overlap)                   │
│     2. Generate Gemini embedding                             │
│     3. Store in ChromaDB with verified page metadata         │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│              PHASE 3: GROUNDED CITATION MATCHING             │
├──────────────────────────────────────────────────────────────┤
│  For each claim in your document:                            │
│     1. Embed claim → Query vector DB → Retrieve top chunks   │
│     2. Each chunk has VERIFIED page from OCR metadata        │
│     3. AI matches claim to retrieved evidence ONLY           │
│     4. Page numbers come from retrieval, NOT guessing        │
│                                                              │
│  ✗ Cannot default to page 1 (no evidence = no citation)     │
│  ✓ Every citation grounded in retrieved evidence            │
└──────────────────────────────────────────────────────────────┘
```

<br/>

---

<br/>

## Scaling: Full Context vs RAG Mode

| Mode | Papers | How It Works |
|------|--------|--------------|
| **Full Context** | 1-50 | Loads entire bibliography into Gemini context |
| **RAG Mode** | 50-500+ | Vector search retrieves only relevant sources |

RAG mode automatically activates for large bibliographies:
- 80% reduction in API token usage
- 3-5x faster processing
- Same citation accuracy

```python
orchestrator = CitationOrchestrator(
    gemini_api_key="your-key",
    pdf_threshold=50  # Switch to RAG at 50+ papers
)
```

<br/>

---

<br/>

## Additional Features

Scholaris also includes tools to support the citation workflow:

### Paper Discovery & Download

```python
from scholaris import Scholaris

scholar = Scholaris(gemini_api_key="your-key")

# Search and download papers
papers = scholar.search_papers("transformer neural networks", max_papers=10)
pdf_paths = scholar.download_papers(papers, output_dir="./papers")
bibtex = scholar.generate_bibtex(pdf_paths)
```

### AI Literature Reviews

```python
review = scholar.complete_workflow(
    topic="Machine Learning in Healthcare",
    max_papers=10,
    sections=["Introduction", "Methods", "Applications", "Conclusion"],
    output_format="docx"
)
```

<br/>

---

<br/>

## Quick Start

### Installation

```bash
pip install git+https://github.com/joseluissaorin/scholaris.git
```

**Requirements:** Python 3.9+ · [Gemini API Key](https://makersuite.google.com/app/apikey) (free tier available)

### Configuration

```bash
# .env
GEMINI_API_KEY=your_key_here       # Required
CROSSREF_EMAIL=your@email.com      # Optional (better page detection)
```

<br/>

---

<br/>

## Performance

| Component | Metric |
|-----------|--------|
| Vision OCR page accuracy | 95%+ |
| Page detection (5-strategy cascade) | 97.4% |
| BibTeX extraction (pdf2bib + AI) | 75-85% |
| Grounded citations | 0% page-1 fallback |
| End-to-end (10 papers) | ~15 min |

<br/>

---

<br/>

## Comparison

| Feature | Scholaris | Zotero | Mendeley | Elicit |
|---------|:---------:|:------:|:--------:|:------:|
| **Auto-insert citations** | **Yes** | No | No | No |
| **Verified page numbers** | **Yes** | No | No | No |
| **Vision OCR (scanned PDFs)** | **Yes** | No | No | No |
| **Shareable processed format** | **Yes** | No | No | No |
| Paper search | Yes | Yes | Yes | Yes |
| BibTeX extraction | Yes | Yes | Yes | No |
| AI literature review | Yes | No | No | Limited |

<br/>

---

<br/>

## Ethics & Responsible Use

- **Sci-Hub:** Legal status varies by jurisdiction. Use responsibly.
- **AI Content:** Generated reviews are first drafts. Always review and edit.
- **Citations:** Verify accuracy before submitting academic work.
- **API Keys:** Use environment variables, never commit to git.

<br/>

---

<br/>

## Contributing

```bash
git clone https://github.com/joseluissaorin/scholaris.git
cd scholaris
pip install -e .[dev]
pytest tests/
```

PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

<br/>

---

<br/>

## Citation

```bibtex
@software{scholaris2026,
  title={Scholaris: AI-Powered Academic Citation System},
  author={Saorín Ferrer, José Luis},
  year={2026},
  url={https://github.com/joseluissaorin/scholaris}
}
```

<br/>

---

<br/>

## Acknowledgments

Built with:
- [PyPaperBot](https://github.com/ferru97/PyPaperBot) — Google Scholar integration
- [pdf2bib](https://github.com/MicheleCotrufo/pdf2bib) — PDF metadata extraction
- [Google Gemini](https://ai.google.dev/) — Vision OCR, embeddings, and language model
- [ChromaDB](https://www.trychroma.com/) — Vector database for RAG
- [python-docx](https://python-docx.readthedocs.io/) — Word document generation

<br/>

---

<div align="center">

**[Documentation](https://github.com/joseluissaorin/scholaris/wiki)** · **[Issues](https://github.com/joseluissaorin/scholaris/issues)** · **[Examples](examples/)**

MIT License · Copyright 2026 José Luis Saorín Ferrer

</div>
