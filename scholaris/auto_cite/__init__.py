"""Auto-citation system for Scholaris.

This module provides automated in-text citation insertion for academic documents,
supporting both APA 7th edition and Chicago 17th edition citation styles.

Key Features:
- Accurate page number detection (journal vs PDF pages)
- RAG-powered citation matching using ChromaDB
- Support for bibliographies of 50-1000 PDFs
- Gemini 2.0 Flash with 500k token context
- Preview mode for citation validation

Main Components:
- PageOffsetDetector: Detects journal page numbers from PDFs
- CitationOrchestrator: Main controller for citation insertion
- CitationFormatter: Formats citations in APA/Chicago styles
- BibliographyIndexer: RAG indexing for large bibliographies
"""

from .models import (
    PageOffsetResult,
    PDFPage,
    PageAwarePDF,
    CitationStyle,
    CitationRequest,
    CitationResult,
)
from .page_offset import PageOffsetDetector
from .orchestrator import CitationOrchestrator

__all__ = [
    "PageOffsetResult",
    "PDFPage",
    "PageAwarePDF",
    "CitationStyle",
    "CitationRequest",
    "CitationResult",
    "PageOffsetDetector",
    "CitationOrchestrator",
]
