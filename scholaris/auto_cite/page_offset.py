"""Page offset detection for accurate citation page numbers.

This module implements the 5-strategy cascade for detecting the offset between
PDF page numbers and journal page numbers.

Strategy Priority (from fastest/most reliable to slowest/least reliable):
1. Footer/Header Parsing - Regex patterns on first/last pages (0.9 confidence)
2. BibTeX Page Range Validation - Validate against pages field (0.7 confidence)
3. Vision-Based Detection - Mistral Pixtral OCR (0.85 confidence)
4. DOI Crossref Lookup - Query Crossref API (0.95 confidence)
5. Assume PDF Pagination - Fallback for preprints (0.2-0.5 confidence)
"""

import re
import logging
import requests
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import fitz  # PyMuPDF

from .models import PageOffsetResult, PageDetectionStrategy

logger = logging.getLogger(__name__)


class PageOffsetDetector:
    """Orchestrates all 5 page offset detection strategies.

    This is the core component that ensures accurate page numbers in citations.
    It tries each strategy in order until one succeeds with sufficient confidence.
    """

    def __init__(
        self,
        min_confidence: float = 0.7,
        crossref_email: Optional[str] = None,
        mistral_api_key: Optional[str] = None,
    ):
        """Initialize page offset detector.

        Args:
            min_confidence: Minimum confidence threshold (default: 0.7)
            crossref_email: Email for Crossref API (polite pool)
            mistral_api_key: API key for Mistral Pixtral (OCR fallback)
        """
        self.min_confidence = min_confidence
        self.crossref_email = crossref_email
        self.mistral_api_key = mistral_api_key

        # Compile regex patterns for footer/header parsing
        self._compile_page_patterns()

    def _compile_page_patterns(self):
        """Compile regex patterns for page number detection."""
        # Common page number patterns in footers/headers
        self.page_patterns = [
            # "Page 337", "p. 337", "pp. 337"
            re.compile(r'\b(?:page|p\.?|pp\.?)\s*(\d+)', re.IGNORECASE),
            # Just a number at start/end of line
            re.compile(r'^\s*(\d+)\s*$', re.MULTILINE),
            # Number with separators: "| 337 |", "- 337 -"
            re.compile(r'[|\-]\s*(\d+)\s*[|\-]'),
            # Volume(Issue): Page format: "15(3): 337"
            re.compile(r'\d+\(\d+\):\s*(\d+)'),
        ]

    def detect_offset(
        self,
        pdf_path: str,
        bib_entry: Dict[str, Any],
    ) -> PageOffsetResult:
        """Detect page offset using cascade of 5 strategies.

        Args:
            pdf_path: Path to PDF file
            bib_entry: BibTeX entry dictionary with metadata

        Returns:
            PageOffsetResult with offset and confidence
        """
        logger.info(f"Detecting page offset for: {Path(pdf_path).name}")

        # Strategy 1: Footer/Header Parsing (fastest)
        result = self._try_footer_header_parsing(pdf_path, bib_entry)
        if result and result.is_reliable:
            logger.info(f"✓ Strategy 1 (Footer/Header): {result}")
            return result

        # Strategy 2: BibTeX Page Range Validation
        result = self._try_bibtex_validation(pdf_path, bib_entry)
        if result and result.is_reliable:
            logger.info(f"✓ Strategy 2 (BibTeX): {result}")
            return result

        # Strategy 3: DOI Crossref Lookup (very reliable but requires API call)
        if bib_entry.get('doi'):
            result = self._try_crossref_lookup(pdf_path, bib_entry)
            if result and result.is_reliable:
                logger.info(f"✓ Strategy 3 (Crossref): {result}")
                return result

        # Strategy 4: Vision-Based Detection (Mistral Pixtral)
        if self.mistral_api_key:
            result = self._try_vision_ocr(pdf_path, bib_entry)
            if result and result.is_reliable:
                logger.info(f"✓ Strategy 4 (Vision OCR): {result}")
                return result

        # Strategy 5: Assume PDF Pagination (fallback)
        logger.warning(f"All strategies failed, using PDF pagination fallback for {Path(pdf_path).name}")
        return self._assume_pdf_pagination(pdf_path, bib_entry)

    # ==================== Strategy 1: Footer/Header Parsing ====================

    def _try_footer_header_parsing(
        self,
        pdf_path: str,
        bib_entry: Dict[str, Any],
    ) -> Optional[PageOffsetResult]:
        """Try to detect page offset from footer/header text.

        This strategy reads the first and last pages, looks for page numbers
        in headers/footers, and calculates the offset.

        Args:
            pdf_path: Path to PDF file
            bib_entry: BibTeX entry dictionary

        Returns:
            PageOffsetResult if successful, None otherwise
        """
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            if total_pages == 0:
                return None

            # Check first page
            first_page = doc[0]
            first_page_text = first_page.get_text()

            # Extract potential page numbers from first page
            page_numbers = self._extract_page_numbers(first_page_text)

            if page_numbers:
                # Use the most common/reliable number found
                detected_page = max(page_numbers, key=page_numbers.count)

                # Calculate offset: journal_page = pdf_page + offset
                # For first page: detected_page = 1 + offset
                # Therefore: offset = detected_page - 1
                offset = detected_page - 1

                # Validate offset makes sense
                if self._validate_offset(offset, total_pages, bib_entry):
                    doc.close()
                    return PageOffsetResult(
                        offset=offset,
                        confidence=0.9,
                        strategy_used=PageDetectionStrategy.FOOTER_HEADER_PARSING,
                        metadata={
                            'detected_first_page': detected_page,
                            'pdf_total_pages': total_pages,
                        }
                    )

            # If first page didn't work, try last page
            last_page = doc[-1]
            last_page_text = last_page.get_text()
            page_numbers = self._extract_page_numbers(last_page_text)

            if page_numbers:
                detected_page = max(page_numbers, key=page_numbers.count)
                # For last page: detected_page = total_pages + offset
                offset = detected_page - total_pages

                if self._validate_offset(offset, total_pages, bib_entry):
                    doc.close()
                    return PageOffsetResult(
                        offset=offset,
                        confidence=0.85,  # Slightly lower for last page
                        strategy_used=PageDetectionStrategy.FOOTER_HEADER_PARSING,
                        metadata={
                            'detected_last_page': detected_page,
                            'pdf_total_pages': total_pages,
                        }
                    )

            doc.close()
            return None

        except Exception as e:
            logger.warning(f"Footer/header parsing failed: {e}")
            return None

    def _extract_page_numbers(self, text: str) -> List[int]:
        """Extract potential page numbers from text using regex patterns.

        Args:
            text: Text to search for page numbers

        Returns:
            List of potential page numbers found
        """
        page_numbers = []

        for pattern in self.page_patterns:
            matches = pattern.findall(text)
            for match in matches:
                try:
                    page_num = int(match)
                    # Filter out unrealistic page numbers
                    if 1 <= page_num <= 10000:
                        page_numbers.append(page_num)
                except (ValueError, TypeError):
                    continue

        return page_numbers

    def _validate_offset(
        self,
        offset: int,
        total_pages: int,
        bib_entry: Dict[str, Any],
    ) -> bool:
        """Validate that a detected offset makes sense.

        Args:
            offset: Proposed page offset
            total_pages: Total pages in PDF
            bib_entry: BibTeX entry with metadata

        Returns:
            True if offset is valid, False otherwise
        """
        # Offset should be reasonable (not negative, not huge)
        if offset < 0 or offset > 5000:
            return False

        # If BibTeX has page range, verify consistency
        pages_field = bib_entry.get('pages', '')
        if pages_field:
            page_range = self._parse_page_range(pages_field)
            if page_range:
                start_page, end_page = page_range
                expected_length = end_page - start_page + 1
                actual_length = total_pages

                # Check if lengths match (within tolerance)
                if abs(expected_length - actual_length) > 3:
                    logger.debug(
                        f"Page count mismatch: BibTeX says {expected_length} pages, "
                        f"PDF has {actual_length} pages"
                    )
                    return False

                # Check if first page matches
                expected_first = start_page
                detected_first = 1 + offset
                if abs(expected_first - detected_first) > 2:
                    logger.debug(
                        f"First page mismatch: BibTeX says {expected_first}, "
                        f"detected {detected_first}"
                    )
                    return False

        return True

    # ==================== Strategy 2: BibTeX Page Range Validation ====================

    def _try_bibtex_validation(
        self,
        pdf_path: str,
        bib_entry: Dict[str, Any],
    ) -> Optional[PageOffsetResult]:
        """Try to detect offset from BibTeX page range field.

        This strategy parses the 'pages' field from BibTeX (e.g., "337--357"),
        validates against PDF page count, and calculates offset.

        Args:
            pdf_path: Path to PDF file
            bib_entry: BibTeX entry dictionary

        Returns:
            PageOffsetResult if successful, None otherwise
        """
        try:
            pages_field = bib_entry.get('pages', '')
            if not pages_field:
                return None

            page_range = self._parse_page_range(pages_field)
            if not page_range:
                return None

            start_page, end_page = page_range

            # Open PDF to get page count
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()

            if total_pages == 0:
                return None

            # Calculate expected page count from BibTeX
            expected_length = end_page - start_page + 1

            # Validate page count matches (within small tolerance)
            if abs(expected_length - total_pages) > 3:
                logger.debug(
                    f"BibTeX page count mismatch: expected {expected_length}, "
                    f"PDF has {total_pages}"
                )
                return None

            # Calculate offset
            # First PDF page (1) should be journal page start_page
            # Therefore: start_page = 1 + offset
            offset = start_page - 1

            return PageOffsetResult(
                offset=offset,
                confidence=0.7,
                strategy_used=PageDetectionStrategy.BIBTEX_VALIDATION,
                metadata={
                    'bibtex_page_range': f"{start_page}--{end_page}",
                    'pdf_total_pages': total_pages,
                    'page_count_match': abs(expected_length - total_pages) <= 3,
                }
            )

        except Exception as e:
            logger.warning(f"BibTeX validation failed: {e}")
            return None

    def _parse_page_range(self, pages_field: str) -> Optional[Tuple[int, int]]:
        """Parse page range from BibTeX pages field.

        Handles formats like:
        - "337--357"
        - "337-357"
        - "337"

        Args:
            pages_field: Value of 'pages' field from BibTeX

        Returns:
            Tuple of (start_page, end_page) or None if parsing fails
        """
        try:
            # Remove whitespace
            pages_field = pages_field.strip()

            # Try double dash (BibTeX standard)
            if '--' in pages_field:
                parts = pages_field.split('--')
                if len(parts) == 2:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    return (start, end)

            # Try single dash
            if '-' in pages_field:
                parts = pages_field.split('-')
                if len(parts) == 2:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    return (start, end)

            # Single page number
            page = int(pages_field)
            return (page, page)

        except (ValueError, IndexError):
            return None

    # ==================== Strategy 3: DOI Crossref Lookup ====================

    def _try_crossref_lookup(
        self,
        pdf_path: str,
        bib_entry: Dict[str, Any],
    ) -> Optional[PageOffsetResult]:
        """Try to detect offset using DOI Crossref API.

        This strategy queries the Crossref API for authoritative page information.

        Args:
            pdf_path: Path to PDF file
            bib_entry: BibTeX entry dictionary

        Returns:
            PageOffsetResult if successful, None otherwise
        """
        try:
            doi = bib_entry.get('doi', '').strip()
            if not doi:
                return None

            # Query Crossref API
            headers = {}
            if self.crossref_email:
                headers['User-Agent'] = f'Scholaris/1.0 (mailto:{self.crossref_email})'

            url = f"https://api.crossref.org/works/{doi}"
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                logger.warning(f"Crossref API returned status {response.status_code}")
                return None

            data = response.json()
            message = data.get('message', {})

            # Extract page range
            page_str = message.get('page', '')
            if not page_str:
                logger.debug(f"No page information in Crossref response for DOI: {doi}")
                return None

            page_range = self._parse_page_range(page_str)
            if not page_range:
                return None

            start_page, end_page = page_range

            # Open PDF to validate
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()

            expected_length = end_page - start_page + 1

            # Validate page count
            if abs(expected_length - total_pages) > 3:
                logger.warning(
                    f"Crossref page count mismatch for {doi}: "
                    f"expected {expected_length}, PDF has {total_pages}"
                )
                # Still return result but with lower confidence
                confidence = 0.6
            else:
                confidence = 0.95  # Very high confidence for Crossref

            offset = start_page - 1

            return PageOffsetResult(
                offset=offset,
                confidence=confidence,
                strategy_used=PageDetectionStrategy.DOI_CROSSREF,
                metadata={
                    'doi': doi,
                    'crossref_page_range': page_str,
                    'pdf_total_pages': total_pages,
                }
            )

        except requests.exceptions.RequestException as e:
            logger.warning(f"Crossref API request failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Crossref lookup failed: {e}")
            return None

    # ==================== Strategy 4: Vision-Based OCR ====================

    def _try_vision_ocr(
        self,
        pdf_path: str,
        bib_entry: Dict[str, Any],
    ) -> Optional[PageOffsetResult]:
        """Try to detect offset using Mistral Pixtral vision model.

        This strategy uses OCR on the first page for complex layouts where
        text extraction fails or is unreliable.

        Args:
            pdf_path: Path to PDF file
            bib_entry: BibTeX entry dictionary

        Returns:
            PageOffsetResult if successful, None otherwise
        """
        # TODO: Implement Mistral Pixtral OCR
        # This requires:
        # 1. Convert first PDF page to image
        # 2. Send to Mistral Pixtral API
        # 3. Extract page number from OCR response
        # 4. Calculate offset

        logger.debug("Vision OCR strategy not yet implemented")
        return None

    # ==================== Strategy 5: Assume PDF Pagination ====================

    def _assume_pdf_pagination(
        self,
        pdf_path: str,
        bib_entry: Dict[str, Any],
    ) -> PageOffsetResult:
        """Fallback strategy: assume PDF uses its own pagination.

        This is common for preprints (arXiv, bioRxiv) where PDF pages
        match the actual page numbers.

        Args:
            pdf_path: Path to PDF file
            bib_entry: BibTeX entry dictionary

        Returns:
            PageOffsetResult with offset=0 and low confidence
        """
        # Check if this is likely a preprint
        entry_type = bib_entry.get('ENTRYTYPE', '').lower()
        url = bib_entry.get('url', '').lower()
        source = bib_entry.get('journal', bib_entry.get('booktitle', '')).lower()

        preprint_indicators = ['arxiv', 'biorxiv', 'medrxiv', 'preprint', 'ssrn']
        is_preprint = any(indicator in url or indicator in source for indicator in preprint_indicators)

        if is_preprint:
            confidence = 0.5  # Medium-low confidence for preprints
            warning = None
        else:
            confidence = 0.2  # Very low confidence for published papers
            warning = (
                "Could not detect journal page numbers. Using PDF page numbers. "
                "Citations may be inaccurate - please verify manually."
            )

        return PageOffsetResult(
            offset=0,
            confidence=confidence,
            strategy_used=PageDetectionStrategy.ASSUME_PDF_PAGINATION,
            uses_pdf_pagination=True,
            warning_message=warning,
            metadata={
                'is_preprint': is_preprint,
                'entry_type': entry_type,
            }
        )


# ==================== Helper Functions ====================

def batch_detect_offsets(
    pdf_paths: List[str],
    bib_entries: List[Dict[str, Any]],
    min_confidence: float = 0.7,
    crossref_email: Optional[str] = None,
    mistral_api_key: Optional[str] = None,
) -> List[PageOffsetResult]:
    """Detect page offsets for multiple PDFs in batch.

    Args:
        pdf_paths: List of PDF file paths
        bib_entries: List of corresponding BibTeX entries
        min_confidence: Minimum confidence threshold
        crossref_email: Email for Crossref API
        mistral_api_key: API key for Mistral Pixtral

    Returns:
        List of PageOffsetResult objects

    Raises:
        ValueError: If pdf_paths and bib_entries lengths don't match
    """
    if len(pdf_paths) != len(bib_entries):
        raise ValueError(
            f"Number of PDFs ({len(pdf_paths)}) must match number of BibTeX entries ({len(bib_entries)})"
        )

    detector = PageOffsetDetector(
        min_confidence=min_confidence,
        crossref_email=crossref_email,
        mistral_api_key=mistral_api_key,
    )

    results = []
    for pdf_path, bib_entry in zip(pdf_paths, bib_entries):
        result = detector.detect_offset(pdf_path, bib_entry)
        results.append(result)

    return results
