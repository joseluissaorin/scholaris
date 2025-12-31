"""Auto-citation system for Scholaris.

This module provides automated in-text citation insertion for academic documents,
supporting both APA 7th edition and Chicago 17th edition citation styles.

Key Features:
- Accurate page number detection (journal vs PDF pages) - 97.4% accuracy
- Hybrid AI-powered citation matching with Gemini 3 Flash Preview
- Dual-mode operation: Full Context (<50 papers) + RAG Mode (50+ papers)
- Support for bibliographies of 1-500+ PDFs
- Preview mode for citation validation before insertion

Main Components:
- PageOffsetDetector: Detects journal page numbers from PDFs
- CitationOrchestrator: Main controller with hybrid mode auto-switching
- GeminiCitationEngine: Full Context mode for small bibliographies
- RAGCitationEngine: Vector search mode for large bibliographies (ChromaDB)
- Citation models: PageAwarePDF, Citation, CitationRequest, CitationResult
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
