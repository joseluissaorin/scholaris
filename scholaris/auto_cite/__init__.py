"""Auto-citation system for Scholaris.

This module provides automated in-text citation insertion for academic documents,
supporting both APA 7th edition and Chicago 17th edition citation styles.

Key Features:
- Vision OCR for scanned PDFs with verified page numbers
- Page-Aware RAG with grounded citation matching
- AI-powered metadata extraction when pdf2bib fails
- Persistent ChromaDB index for reuse across sessions
- Support for bibliographies of 1-500+ PDFs
- Preview mode for citation validation before insertion

Main Components:
- CitationOrchestrator: Main controller with Vision OCR + Page-Aware RAG
- VisionOCRProcessor: Gemini Vision OCR for scanned PDFs
- PageAwareRAG: ChromaDB with verified page metadata
- MetadataExtractor: AI metadata extraction from PDF first pages
- GeminiCitationEngine: Citation matching with grounded RAG mode
"""

from .models import (
    PageOffsetResult,
    PDFPage,
    PageAwarePDF,
    CitationStyle,
    CitationRequest,
    CitationResult,
    OCRPage,
    PageChunk,
    RetrievedChunk,
)
from .page_offset import PageOffsetDetector
from .orchestrator import CitationOrchestrator
from .citation_engine import GeminiCitationEngine, GroundedCitation
from .citation_export import (
    CitationExporter,
    CitationExportResult,
    CitationExportRow,
    export_citations_to_csv,
)

# Optional components (require chromadb)
try:
    from .vision_ocr import VisionOCRProcessor, process_pdf_vision
    from .page_aware_rag import PageAwareRAG, GeminiEmbedder
    from .metadata_extractor import MetadataExtractor, extract_pdf_metadata, batch_extract_metadata
    VISION_RAG_AVAILABLE = True
except ImportError:
    VISION_RAG_AVAILABLE = False
    VisionOCRProcessor = None
    PageAwareRAG = None
    MetadataExtractor = None

__all__ = [
    # Core models
    "PageOffsetResult",
    "PDFPage",
    "PageAwarePDF",
    "CitationStyle",
    "CitationRequest",
    "CitationResult",
    "OCRPage",
    "PageChunk",
    "RetrievedChunk",
    # Main orchestrator
    "CitationOrchestrator",
    "PageOffsetDetector",
    # Citation engine
    "GeminiCitationEngine",
    "GroundedCitation",
    # Citation export
    "CitationExporter",
    "CitationExportResult",
    "CitationExportRow",
    "export_citations_to_csv",
    # Vision OCR + RAG (optional)
    "VisionOCRProcessor",
    "PageAwareRAG",
    "GeminiEmbedder",
    "MetadataExtractor",
    # Convenience functions
    "process_pdf_vision",
    "extract_pdf_metadata",
    "batch_extract_metadata",
    # Availability flag
    "VISION_RAG_AVAILABLE",
]
