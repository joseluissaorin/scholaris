"""Vision OCR processor using Gemini Vision for scanned PDFs.

Extracts text and page numbers directly from PDF page images using
Gemini 2.0 Flash's vision capabilities. Handles landscape scans
where one PDF page contains two book pages.
"""

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union

import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image

from .models import OCRPage


class VisionOCRProcessor:
    """OCR processor using Gemini Vision for scanned PDFs.

    Features:
    - Detects single vs landscape (double-page) layouts
    - Extracts printed page numbers from images
    - Full text OCR for scanned documents
    - Handles both scanned and text-layer PDFs
    """

    def __init__(
        self,
        gemini_api_key: str,
        model_name: str = "gemini-flash-lite-latest",
        dpi: int = 150,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize Vision OCR processor.

        Args:
            gemini_api_key: Gemini API key
            model_name: Gemini model to use for vision
            dpi: DPI for rendering PDF pages to images
            max_retries: Maximum retries on API errors
            retry_delay: Delay between retries in seconds
        """
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)
        self.dpi = dpi
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def process_pdf(
        self,
        pdf_path: str,
        progress_callback: Optional[callable] = None,
    ) -> List[OCRPage]:
        """Process entire PDF with Vision OCR.

        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of OCRPage objects with verified page numbers
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        ocr_pages: List[OCRPage] = []

        for pdf_page_num in range(total_pages):
            if progress_callback:
                progress_callback(pdf_page_num + 1, total_pages)

            # Render page to image
            page = doc[pdf_page_num]
            image_bytes = self._render_page_to_image(page)

            # OCR with Gemini Vision
            try:
                page_results = self._ocr_page(image_bytes, pdf_page_num + 1)
                ocr_pages.extend(page_results)
            except Exception as e:
                print(f"Warning: OCR failed for page {pdf_page_num + 1}: {e}")
                # Create fallback with unknown page number
                ocr_pages.append(OCRPage(
                    pdf_page=pdf_page_num + 1,
                    book_page=pdf_page_num + 1,  # Fallback to PDF page
                    text="[OCR FAILED]",
                    confidence=0.0,
                    is_landscape_half=False,
                    layout_type="single",
                ))

        doc.close()
        return ocr_pages

    def _render_page_to_image(self, page: fitz.Page) -> bytes:
        """Render PDF page to PNG bytes.

        Args:
            page: PyMuPDF page object

        Returns:
            PNG image as bytes
        """
        # Render at specified DPI
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PNG bytes
        return pix.tobytes("png")

    def _ocr_page(self, image_bytes: bytes, pdf_page_num: int) -> List[OCRPage]:
        """OCR single page with layout detection.

        Args:
            image_bytes: PNG image bytes
            pdf_page_num: Page number in PDF (1-indexed)

        Returns:
            List of OCRPage objects (1 for single, 2 for landscape double)
        """
        # Build the OCR prompt
        prompt = """Analyze this scanned book/document page image.

TASK 1 - LAYOUT DETECTION:
Is this a SINGLE page or a LANDSCAPE scan with TWO book pages side by side?
Look for:
- Clear vertical division in the middle
- Two separate text columns with their own margins
- Two different page numbers visible

TASK 2 - PAGE NUMBER EXTRACTION:
Find the printed page number(s) on the page. Look in:
- Page headers or footers
- Top/bottom corners
- Running headers with chapter title + page number
If you see TWO page numbers (landscape scan), identify LEFT and RIGHT page numbers.
If no page number is visible, set to null.

TASK 3 - TEXT EXTRACTION:
Extract ALL text content from the page.
- Preserve paragraph structure
- Include headers, footnotes, and marginalia
- If landscape with two pages:
  * Extract LEFT page text completely
  * Then RIGHT page text completely
  * Keep them separate

OUTPUT FORMAT (JSON only, no markdown):
{
  "layout": "single" or "landscape_double",
  "pages": [
    {
      "position": "single" or "left" or "right",
      "printed_page": <number or null if not visible>,
      "text": "full extracted text..."
    }
  ]
}

For landscape_double, include TWO page objects (left first, then right).
For single, include ONE page object.

IMPORTANT: Output ONLY valid JSON, no markdown code blocks."""

        # Call Gemini Vision with retry logic
        for attempt in range(self.max_retries):
            try:
                # Create image part for Gemini
                image_part = {
                    "mime_type": "image/png",
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                }

                response = self.model.generate_content(
                    [prompt, image_part],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,  # Low temperature for accuracy
                        max_output_tokens=8192,
                    ),
                )

                # Parse response
                return self._parse_ocr_response(response.text, pdf_page_num)

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"OCR failed after {self.max_retries} attempts: {e}")

        return []  # Should not reach here

    def _parse_ocr_response(self, response_text: str, pdf_page_num: int) -> List[OCRPage]:
        """Parse Gemini OCR response into OCRPage objects.

        Args:
            response_text: Raw response from Gemini
            pdf_page_num: PDF page number (1-indexed)

        Returns:
            List of OCRPage objects
        """
        # Clean response - remove markdown code blocks if present
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            # Remove ```json or ``` prefix and ``` suffix
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    raise ValueError(f"Could not parse OCR response as JSON: {e}")
            else:
                raise ValueError(f"No JSON found in OCR response: {e}")

        layout = data.get("layout", "single")
        pages_data = data.get("pages", [])

        if not pages_data:
            # Fallback: create single page with no text
            return [OCRPage(
                pdf_page=pdf_page_num,
                book_page=pdf_page_num,
                text="",
                confidence=0.5,
                is_landscape_half=False,
                layout_type="single",
            )]

        ocr_pages = []
        for page_data in pages_data:
            position = page_data.get("position", "single")
            printed_page = page_data.get("printed_page")
            text = page_data.get("text", "")

            # Determine book page number
            if printed_page is not None:
                book_page = self._parse_page_number(printed_page, pdf_page_num)
                confidence = 0.95 if book_page != pdf_page_num else 0.5
            else:
                # Fallback: estimate from PDF page
                book_page = pdf_page_num
                confidence = 0.3  # Low confidence without page number

            is_landscape = layout == "landscape_double"

            ocr_pages.append(OCRPage(
                pdf_page=pdf_page_num,
                book_page=book_page,
                text=text.strip(),
                confidence=confidence,
                is_landscape_half=is_landscape,
                layout_type=layout,
            ))

        return ocr_pages

    def _parse_page_number(self, printed_page: Any, fallback: int) -> int:
        """Parse page number from OCR output, handling various formats.

        Handles:
        - Integer: 42
        - String integer: "42"
        - Roman numerals: "xii", "XII", "iv"
        - Invalid values: falls back to PDF page number

        Args:
            printed_page: Page number from OCR (int, str, or other)
            fallback: Fallback page number (PDF page)

        Returns:
            Integer page number
        """
        if printed_page is None:
            return fallback

        # If already an integer, return it
        if isinstance(printed_page, int):
            return printed_page

        # Convert to string and clean
        page_str = str(printed_page).strip().lower()

        # Try direct integer conversion
        try:
            return int(page_str)
        except ValueError:
            pass

        # Try Roman numeral conversion
        roman_page = self._roman_to_int(page_str)
        if roman_page is not None:
            # Return negative for front matter (Roman numerals)
            # This helps distinguish from regular page numbers
            return -roman_page

        # Fallback to PDF page
        return fallback

    def _roman_to_int(self, s: str) -> Optional[int]:
        """Convert Roman numeral string to integer.

        Args:
            s: Roman numeral string (case-insensitive)

        Returns:
            Integer value, or None if not a valid Roman numeral
        """
        roman_values = {
            'i': 1, 'v': 5, 'x': 10, 'l': 50,
            'c': 100, 'd': 500, 'm': 1000
        }

        s = s.lower().strip()

        # Validate all characters are Roman numerals
        if not all(c in roman_values for c in s):
            return None

        if not s:
            return None

        result = 0
        prev_value = 0

        for char in reversed(s):
            value = roman_values[char]
            if value < prev_value:
                result -= value
            else:
                result += value
            prev_value = value

        return result if result > 0 else None

    def check_has_text_layer(self, pdf_path: str, sample_pages: int = 3) -> bool:
        """Check if PDF has a usable text layer.

        Args:
            pdf_path: Path to PDF file
            sample_pages: Number of pages to sample

        Returns:
            True if PDF has sufficient text layer
        """
        doc = fitz.open(pdf_path)
        total_chars = 0
        pages_checked = min(sample_pages, len(doc))

        for i in range(pages_checked):
            page = doc[i]
            text = page.get_text()
            total_chars += len(text.strip())

        doc.close()

        # Threshold: at least 100 chars per page on average
        avg_chars = total_chars / pages_checked if pages_checked > 0 else 0
        return avg_chars >= 100

    def process_pdf_with_fallback(
        self,
        pdf_path: str,
        force_ocr: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[List[OCRPage], str]:
        """Process PDF with intelligent fallback.

        Uses text extraction if available, falls back to OCR if needed.

        Args:
            pdf_path: Path to PDF file
            force_ocr: If True, always use OCR even with text layer
            progress_callback: Optional progress callback

        Returns:
            Tuple of (list of OCRPage, extraction_method string)
        """
        if not force_ocr and self.check_has_text_layer(pdf_path):
            # Use PyMuPDF text extraction with page number detection
            return self._extract_with_text_layer(pdf_path, progress_callback), "text_layer"
        else:
            # Use full Vision OCR
            return self.process_pdf(pdf_path, progress_callback), "vision_ocr"

    def _extract_with_text_layer(
        self,
        pdf_path: str,
        progress_callback: Optional[callable] = None,
    ) -> List[OCRPage]:
        """Extract text from PDF with text layer, still detecting page numbers.

        For PDFs with text layers, we extract text directly but still need
        to detect printed page numbers (which may differ from PDF page numbers).

        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional progress callback

        Returns:
            List of OCRPage objects
        """
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        ocr_pages: List[OCRPage] = []

        for pdf_page_num in range(total_pages):
            if progress_callback:
                progress_callback(pdf_page_num + 1, total_pages)

            page = doc[pdf_page_num]
            text = page.get_text()

            # Try to detect page number from text
            book_page = self._detect_page_number_from_text(text, pdf_page_num + 1)

            ocr_pages.append(OCRPage(
                pdf_page=pdf_page_num + 1,
                book_page=book_page,
                text=text.strip(),
                confidence=0.7 if book_page != pdf_page_num + 1 else 0.5,
                is_landscape_half=False,
                layout_type="single",
            ))

        doc.close()
        return ocr_pages

    def _detect_page_number_from_text(self, text: str, pdf_page: int) -> int:
        """Detect printed page number from extracted text.

        Looks for page numbers in common locations (first/last lines).

        Args:
            text: Extracted page text
            pdf_page: PDF page number as fallback

        Returns:
            Detected or fallback page number
        """
        lines = text.strip().split("\n")
        if not lines:
            return pdf_page

        # Check first and last few lines for page numbers
        candidates = []

        # Check first 3 lines (header area)
        for line in lines[:3]:
            candidates.extend(self._extract_page_candidates(line))

        # Check last 3 lines (footer area)
        for line in lines[-3:]:
            candidates.extend(self._extract_page_candidates(line))

        if candidates:
            # Prefer numbers that are reasonable page numbers (1-9999)
            valid = [n for n in candidates if 1 <= n <= 9999]
            if valid:
                return valid[0]

        return pdf_page

    def _extract_page_candidates(self, line: str) -> List[int]:
        """Extract potential page numbers from a line.

        Args:
            line: Single line of text

        Returns:
            List of candidate page numbers
        """
        candidates = []
        line = line.strip()

        # Pattern 1: Standalone number (common in footers)
        if re.match(r'^\d{1,4}$', line):
            candidates.append(int(line))

        # Pattern 2: "Page X" or "p. X"
        match = re.search(r'(?:page|p\.?)\s*(\d{1,4})', line, re.IGNORECASE)
        if match:
            candidates.append(int(match.group(1)))

        # Pattern 3: Number at end of line (running header)
        match = re.search(r'\s(\d{1,4})\s*$', line)
        if match:
            candidates.append(int(match.group(1)))

        # Pattern 4: Number at start of line
        match = re.search(r'^\s*(\d{1,4})\s', line)
        if match:
            candidates.append(int(match.group(1)))

        return candidates


def process_pdf_vision(
    pdf_path: str,
    gemini_api_key: str,
    force_ocr: bool = True,
    progress_callback: Optional[callable] = None,
) -> List[OCRPage]:
    """Convenience function to process PDF with Vision OCR.

    Args:
        pdf_path: Path to PDF file
        gemini_api_key: Gemini API key
        force_ocr: If True, always use OCR
        progress_callback: Optional progress callback

    Returns:
        List of OCRPage objects with verified page numbers
    """
    processor = VisionOCRProcessor(gemini_api_key)

    if force_ocr:
        return processor.process_pdf(pdf_path, progress_callback)
    else:
        pages, _ = processor.process_pdf_with_fallback(pdf_path, False, progress_callback)
        return pages
