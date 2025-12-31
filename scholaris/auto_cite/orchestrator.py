"""Citation orchestrator - main controller for auto-citation system.

This module handles the complete citation insertion workflow:
1. Load and process bibliography PDFs
2. Analyze user document to identify claims needing citations
3. Match claims to relevant sources using Gemini 2.0 Flash
4. Insert formatted citations (APA 7th or Chicago 17th)
5. Generate preview for user validation

Week 2 Implementation: Full Context Mode with intelligent citation matching.
"""

import logging
import re
from typing import List, Optional, Dict, Any

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
        )
        self.pdf_processor = PDFProcessor(page_offset_detector=page_offset_detector)

        # Initialize Gemini citation engine
        self.citation_engine = GeminiCitationEngine(api_key=gemini_api_key)

        logger.info(
            f"CitationOrchestrator initialized: "
            f"pdf_threshold={pdf_threshold}, rag_mode={use_rag_mode}"
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

        # Step 1: Use Gemini to analyze and suggest citations
        logger.info("Analyzing document with Gemini citation engine...")
        try:
            suggested_citations = self.citation_engine.analyze_and_cite(
                document_text=request.document_text,
                bibliography=request.bibliography,
                style=request.style,
                max_citations_per_claim=request.max_citations_per_claim,
            )
        except Exception as e:
            logger.error(f"Gemini citation analysis failed: {e}")
            return CitationResult(
                modified_document=request.document_text,
                citations=[],
                warnings=[f"Citation analysis failed: {str(e)}"],
            )

        # Step 2: Filter by confidence threshold
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
