"""Citation orchestrator - main controller for auto-citation system.

This module handles the complete citation insertion workflow:
1. Load and process bibliography PDFs
2. Analyze user document to identify claims needing citations
3. Match claims to relevant sources using Gemini 2.0 Flash
4. Insert formatted citations (APA 7th or Chicago 17th)
5. Generate preview for user validation

Week 2 Implementation: Full Context Mode with intelligent citation matching.
Week 4 Enhancement: Multi-format input/output support.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal

from .models import (
    CitationRequest,
    CitationResult,
    CitationStyle,
    PageAwarePDF,
    Citation,
)
from .pdf_processor import PDFProcessor
from .formatters import format_citation
from .citation_engine import GeminiCitationEngine
from .document_formats import DocumentFormatHandler, DocumentFormat

# RAG engine (optional - only if ChromaDB installed)
try:
    from .rag_engine import RAGCitationEngine
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    RAGCitationEngine = None

logger = logging.getLogger(__name__)


class CitationOrchestrator:
    """Main controller for the auto-citation system.

    This class orchestrates all components to provide end-to-end
    citation insertion functionality.
    """

    def __init__(
        self,
        gemini_api_key: str,
        pdf_threshold: int = 50,
        use_rag_mode: bool = True,
        mistral_api_key: Optional[str] = None,
        crossref_email: Optional[str] = None,
    ):
        """Initialize citation orchestrator.

        Args:
            gemini_api_key: API key for Google Gemini
            pdf_threshold: Threshold for switching to RAG mode (default: 50)
            use_rag_mode: Whether to use RAG for large bibliographies
            mistral_api_key: API key for Mistral Pixtral (OCR)
            crossref_email: Email for Crossref API polite pool
        """
        self.gemini_api_key = gemini_api_key
        self.pdf_threshold = pdf_threshold
        self.use_rag_mode = use_rag_mode
        self.mistral_api_key = mistral_api_key
        self.crossref_email = crossref_email

        # Initialize PDF processor
        from .page_offset import PageOffsetDetector
        page_offset_detector = PageOffsetDetector(
            crossref_email=crossref_email,
            mistral_api_key=mistral_api_key,
            gemini_api_key=gemini_api_key,
        )
        self.pdf_processor = PDFProcessor(page_offset_detector=page_offset_detector)

        # Initialize Gemini citation engine (Full Context Mode)
        self.citation_engine = GeminiCitationEngine(api_key=gemini_api_key)

        # Initialize RAG citation engine (for large bibliographies)
        self.rag_engine = None
        if use_rag_mode:
            if RAG_AVAILABLE:
                self.rag_engine = RAGCitationEngine(api_key=gemini_api_key)
                logger.info("RAG engine initialized for large bibliographies")
            else:
                logger.warning(
                    "RAG mode requested but ChromaDB not installed. "
                    "Install with: pip install chromadb"
                )
                logger.warning("Falling back to Full Context Mode for all bibliographies")

        logger.info(
            f"CitationOrchestrator initialized: "
            f"pdf_threshold={pdf_threshold}, rag_mode={use_rag_mode and RAG_AVAILABLE}"
        )

    def insert_citations(
        self,
        request: CitationRequest,
    ) -> CitationResult:
        """Insert citations into user document.

        This is the main entry point for citation insertion.

        Workflow:
        1. Use Gemini to analyze document and suggest citations
        2. Filter citations by confidence threshold
        3. Insert citations into document (or generate preview)
        4. Return result with metadata and warnings

        Args:
            request: CitationRequest with document and bibliography

        Returns:
            CitationResult with modified document and metadata
        """
        logger.info(
            f"Citation insertion requested: "
            f"{len(request.bibliography)} sources, "
            f"style={request.style.value}, "
            f"preview={request.preview_mode}"
        )

        warnings = []

        # Step 1: Determine which engine to use (Full Context vs RAG)
        num_papers = len(request.bibliography)
        use_rag = (
            self.use_rag_mode and
            self.rag_engine is not None and
            num_papers >= self.pdf_threshold
        )

        if use_rag:
            logger.info(
                f"Using RAG mode ({num_papers} papers >= {self.pdf_threshold} threshold)"
            )
            engine = self.rag_engine
            mode_name = "RAG"
        else:
            logger.info(
                f"Using Full Context mode ({num_papers} papers < {self.pdf_threshold} threshold)"
            )
            engine = self.citation_engine
            mode_name = "Full Context"

        # Step 2: Analyze document and suggest citations
        logger.info(f"Analyzing document with {mode_name} engine...")
        try:
            suggested_citations = engine.analyze_and_cite(
                document_text=request.document_text,
                bibliography=request.bibliography,
                style=request.style,
                max_citations_per_claim=request.max_citations_per_claim,
            )
        except Exception as e:
            logger.error(f"{mode_name} citation analysis failed: {e}")
            return CitationResult(
                modified_document=request.document_text,
                citations=[],
                warnings=[f"Citation analysis failed: {str(e)}"],
                metadata={'mode': mode_name, 'error': str(e)}
            )

        # Step 3: Filter by confidence threshold
        citations = [
            c for c in suggested_citations
            if c.confidence >= request.min_confidence
        ]

        logger.info(
            f"Suggested {len(suggested_citations)} citations, "
            f"{len(citations)} meet confidence threshold (≥{request.min_confidence})"
        )

        if len(citations) < len(suggested_citations):
            warnings.append(
                f"Filtered out {len(suggested_citations) - len(citations)} "
                f"low-confidence citations"
            )

        # Step 3: Check for low-confidence page detections
        for citation in citations:
            if not citation.source_pdf.page_offset_result.is_reliable:
                warnings.append(
                    f"Low-confidence page detection for {citation.source_pdf.citation_key}"
                )
            if citation.source_pdf.page_offset_result.uses_pdf_pagination:
                warnings.append(
                    f"Using PDF pagination for {citation.source_pdf.citation_key} - "
                    f"may not match published page numbers"
                )

        # Step 4: Insert citations (or generate preview)
        if request.preview_mode:
            # Preview mode - don't modify document, just show what would change
            modified_document = request.document_text
            preview_data = self._generate_preview(request.document_text, citations, request.style)
        else:
            # Apply mode - actually insert citations
            modified_document = self._insert_citations_into_document(
                document=request.document_text,
                citations=citations,
                style=request.style,
            )
            preview_data = {}

        # Step 5: Build result
        result = CitationResult(
            modified_document=modified_document,
            citations=citations,
            preview_data=preview_data,
            warnings=warnings,
            metadata={
                'mode': mode_name,
                'num_papers': num_papers,
                'threshold': self.pdf_threshold,
                'total_suggested': len(suggested_citations),
                'total_inserted': len(citations),
                'confidence_threshold': request.min_confidence,
                'style': request.style.value,
                'preview_mode': request.preview_mode,
            }
        )

        logger.info(
            f"✓ Citation insertion complete: {len(citations)} citations, "
            f"{len(warnings)} warnings"
        )

        return result

    def insert_citations_from_file(
        self,
        input_file: str,
        bibliography: List[PageAwarePDF],
        style: CitationStyle = CitationStyle.APA7,
        preview_mode: bool = False,
        min_confidence: float = 0.7,
        max_citations_per_claim: int = 3,
        input_format: Optional[DocumentFormat] = None,
    ) -> CitationResult:
        """Insert citations from a document file (multi-format support).

        Supports: TXT, MD, DOCX, PDF, HTML, RTF, ODT, LaTeX

        Args:
            input_file: Path to input document
            bibliography: Processed bibliography (from process_bibliography)
            style: Citation style (APA7 or CHICAGO17)
            preview_mode: If True, preview without modifying
            min_confidence: Minimum confidence for citation insertion
            max_citations_per_claim: Maximum citations per claim
            input_format: Document format (auto-detected if None)

        Returns:
            CitationResult with modified document text

        Example:
            >>> orchestrator = CitationOrchestrator(gemini_api_key="...")
            >>> bibliography = orchestrator.process_bibliography(...)
            >>> result = orchestrator.insert_citations_from_file(
            ...     input_file="draft.docx",
            ...     bibliography=bibliography,
            ...     style=CitationStyle.APA7
            ... )
            >>> print(result.modified_document)
        """
        logger.info(f"Reading document from file: {input_file}")

        # Read document text
        try:
            document_text = DocumentFormatHandler.read_document(
                input_file,
                format=input_format
            )
        except Exception as e:
            logger.error(f"Failed to read document: {e}")
            return CitationResult(
                modified_document="",
                citations=[],
                warnings=[f"Failed to read document: {str(e)}"],
                metadata={'error': str(e)}
            )

        logger.info(
            f"✓ Document loaded: {len(document_text)} characters "
            f"({len(document_text.split())} words)"
        )

        # Create citation request
        request = CitationRequest(
            document_text=document_text,
            bibliography=bibliography,
            style=style,
            preview_mode=preview_mode,
            min_confidence=min_confidence,
            max_citations_per_claim=max_citations_per_claim,
        )

        # Process citations
        return self.insert_citations(request)

    def insert_citations_with_export(
        self,
        input_file: str,
        output_file: str,
        bibliography: List[PageAwarePDF],
        style: CitationStyle = CitationStyle.APA7,
        min_confidence: float = 0.7,
        max_citations_per_claim: int = 3,
        input_format: Optional[DocumentFormat] = None,
        output_format: Optional[DocumentFormat] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CitationResult:
        """Complete workflow: Read document → Insert citations → Export to file.

        Supports input/output in multiple formats:
        - Plain text (.txt)
        - Markdown (.md)
        - Microsoft Word (.docx)
        - PDF (.pdf) - input only
        - HTML (.html)
        - Rich Text Format (.rtf)
        - OpenDocument Text (.odt)
        - LaTeX (.tex)

        Args:
            input_file: Path to input document
            output_file: Path to save cited document
            bibliography: Processed bibliography (from process_bibliography)
            style: Citation style (APA7 or CHICAGO17)
            min_confidence: Minimum confidence for citation insertion
            max_citations_per_claim: Maximum citations per claim
            input_format: Input document format (auto-detected if None)
            output_format: Output document format (auto-detected if None)
            metadata: Document metadata (title, author, date)

        Returns:
            CitationResult with statistics and warnings

        Example:
            >>> orchestrator = CitationOrchestrator(gemini_api_key="...")
            >>> bibliography = orchestrator.process_bibliography(...)
            >>> result = orchestrator.insert_citations_with_export(
            ...     input_file="draft.docx",
            ...     output_file="cited_paper.docx",
            ...     bibliography=bibliography,
            ...     style=CitationStyle.APA7,
            ...     metadata={"title": "My Research", "author": "John Doe"}
            ... )
            >>> print(f"✓ Inserted {len(result.citations)} citations")
        """
        logger.info(
            f"Complete citation workflow: {input_file} → {output_file}"
        )

        # Step 1: Read and process document
        result = self.insert_citations_from_file(
            input_file=input_file,
            bibliography=bibliography,
            style=style,
            preview_mode=False,  # Actually insert citations
            min_confidence=min_confidence,
            max_citations_per_claim=max_citations_per_claim,
            input_format=input_format,
        )

        if result.warnings and any('Failed to read' in w for w in result.warnings):
            logger.error("Failed to process document - aborting export")
            return result

        # Step 2: Export cited document
        logger.info(f"Exporting cited document to: {output_file}")
        try:
            # Map citation style for format handler
            citation_style_str = "apa" if style == CitationStyle.APA7 else "chicago"

            DocumentFormatHandler.write_document(
                text=result.modified_document,
                output_path=output_file,
                format=output_format,
                citation_style=citation_style_str,
                metadata=metadata or {},
            )

            logger.info(
                f"✓ Export complete: {len(result.citations)} citations inserted"
            )

            # Add export info to metadata
            result.metadata['input_file'] = input_file
            result.metadata['output_file'] = output_file
            result.metadata['export_format'] = str(
                output_format or DocumentFormatHandler.detect_format(output_file)
            )

        except Exception as e:
            logger.error(f"Failed to export document: {e}")
            result.warnings.append(f"Export failed: {str(e)}")
            result.metadata['export_error'] = str(e)

        return result

    def process_bibliography(
        self,
        pdf_paths: List[str],
        citation_keys: List[str],
        references: List[Any],  # List[scholaris.core.models.Reference]
        bib_entries: List[Dict[str, Any]],
    ) -> List[PageAwarePDF]:
        """Process a bibliography of PDFs into PageAwarePDF objects.

        This method:
        1. Extracts text from all PDFs
        2. Detects page offsets for accurate citations
        3. Creates PageAwarePDF objects

        Args:
            pdf_paths: List of PDF file paths
            citation_keys: List of citation keys
            references: List of Reference objects
            bib_entries: List of BibTeX entries

        Returns:
            List of PageAwarePDF objects ready for citation
        """
        logger.info(f"Processing bibliography of {len(pdf_paths)} PDFs...")

        page_aware_pdfs = self.pdf_processor.batch_process_pdfs(
            pdf_paths=pdf_paths,
            citation_keys=citation_keys,
            references=references,
            bib_entries=bib_entries,
        )

        # Log page detection results
        if page_aware_pdfs:
            reliable_count = sum(
                1 for pdf in page_aware_pdfs
                if pdf.page_offset_result.is_reliable
            )
            logger.info(
                f"✓ Bibliography processed: {len(page_aware_pdfs)} PDFs, "
                f"{reliable_count} with reliable page detection "
                f"({reliable_count/len(page_aware_pdfs)*100:.1f}%)"
            )
        else:
            logger.warning("No PDFs to process")

        # Log warnings for low-confidence detections
        for pdf in page_aware_pdfs:
            if not pdf.page_offset_result.is_reliable:
                logger.warning(
                    f"Low confidence page detection for {pdf.citation_key}: "
                    f"{pdf.page_offset_result}"
                )

        return page_aware_pdfs

    def _generate_preview(
        self,
        document_text: str,
        citations: List[Citation],
        style: CitationStyle,
    ) -> Dict[str, Any]:
        """Generate preview data showing what citations will be inserted.

        Args:
            document_text: Original document text
            citations: List of citations to preview
            style: Citation style

        Returns:
            Preview data dictionary
        """
        preview_items = []

        for i, citation in enumerate(citations, 1):
            # Find claim position in document
            claim_start = document_text.find(citation.claim_text)
            claim_end = claim_start + len(citation.claim_text) if claim_start != -1 else -1

            # Build preview item
            preview_item = {
                'citation_number': i,
                'claim_text': citation.claim_text,
                'claim_position': claim_start,
                'citation_string': citation.citation_string,
                'source_key': citation.source_pdf.citation_key,
                'source_title': citation.source_pdf.reference.title,
                'page_number': citation.journal_page,
                'confidence': citation.confidence,
                'evidence': citation.evidence_text,
            }

            # Add context (50 chars before/after)
            if claim_start != -1:
                context_start = max(0, claim_start - 50)
                context_end = min(len(document_text), claim_end + 50)
                preview_item['context'] = document_text[context_start:context_end]
            else:
                preview_item['context'] = None
                preview_item['warning'] = 'Claim text not found in document'

            preview_items.append(preview_item)

        return {
            'total_citations': len(citations),
            'style': style.value,
            'citations': preview_items,
        }

    def _insert_citations_into_document(
        self,
        document: str,
        citations: List[Citation],
        style: CitationStyle,
    ) -> str:
        """Insert citation strings into document.

        Args:
            document: Original document text
            citations: List of citations to insert
            style: Citation style

        Returns:
            Modified document with citations inserted
        """
        modified_document = document

        # Sort citations by position in document (reverse order to maintain positions)
        citations_with_positions = []
        for citation in citations:
            position = modified_document.find(citation.claim_text)
            if position != -1:
                citations_with_positions.append((position, citation))

        # Sort by position (descending) to insert from end to start
        citations_with_positions.sort(key=lambda x: x[0], reverse=True)

        # Insert citations
        for position, citation in citations_with_positions:
            claim_end = position + len(citation.claim_text)

            if style == CitationStyle.APA7:
                # APA: Insert inline citation after claim
                insertion = f" {citation.citation_string}"
                modified_document = (
                    modified_document[:claim_end] +
                    insertion +
                    modified_document[claim_end:]
                )
            else:  # Chicago17
                # Chicago: Insert superscript footnote number
                # Note: Full footnotes would be added at bottom in real implementation
                insertion = citation.citation_string  # Already contains superscript
                modified_document = (
                    modified_document[:claim_end] +
                    insertion +
                    modified_document[claim_end:]
                )

        return modified_document


# ==================== Helper Functions ====================

def validate_bibliography(
    page_aware_pdfs: List[PageAwarePDF],
    min_confidence: float = 0.7,
) -> Dict[str, Any]:
    """Validate a processed bibliography.

    Args:
        page_aware_pdfs: List of PageAwarePDF objects
        min_confidence: Minimum acceptable confidence

    Returns:
        Validation report with statistics and warnings
    """
    total = len(page_aware_pdfs)
    reliable = sum(1 for pdf in page_aware_pdfs if pdf.page_offset_result.is_reliable)
    uses_pdf_pagination = sum(
        1 for pdf in page_aware_pdfs
        if pdf.page_offset_result.uses_pdf_pagination
    )

    # Calculate average confidence
    avg_confidence = sum(
        pdf.page_offset_result.confidence for pdf in page_aware_pdfs
    ) / total if total > 0 else 0

    # Identify problematic PDFs
    warnings = []
    for pdf in page_aware_pdfs:
        if pdf.page_offset_result.confidence < min_confidence:
            warnings.append(
                f"{pdf.citation_key}: Low confidence "
                f"({pdf.page_offset_result.confidence:.2f})"
            )

    report = {
        'total_pdfs': total,
        'reliable_count': reliable,
        'reliable_percentage': (reliable / total * 100) if total > 0 else 0,
        'uses_pdf_pagination': uses_pdf_pagination,
        'avg_confidence': avg_confidence,
        'warnings': warnings,
    }

    return report
