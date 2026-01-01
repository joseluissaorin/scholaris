"""AI-powered metadata extraction from PDFs.

Uses Gemini Vision to extract bibliographic information from PDF first pages
when pdf2bib or other automated methods fail.
"""

import base64
import json
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import fitz  # PyMuPDF
import google.generativeai as genai


class MetadataExtractor:
    """Extract bibliographic metadata from PDFs using Gemini Vision."""

    def __init__(
        self,
        gemini_api_key: str,
        model_name: str = "gemini-flash-lite-latest",
        max_retries: int = 3,
    ):
        """Initialize metadata extractor.

        Args:
            gemini_api_key: Gemini API key
            model_name: Gemini model to use
            max_retries: Maximum retries on API errors
        """
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)
        self.max_retries = max_retries

    def extract_metadata(
        self,
        pdf_path: str,
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract or complete bibliographic metadata from PDF.

        Args:
            pdf_path: Path to PDF file
            existing_metadata: Partial metadata to fill gaps in

        Returns:
            Dictionary with bibliographic fields:
            - title: str
            - authors: List[str]
            - year: int
            - source: str (journal/publisher/conference)
            - entry_type: str (article/book/inproceedings)
            - volume: Optional[str]
            - issue: Optional[str]
            - pages: Optional[str]
            - doi: Optional[str]
            - citation_key: str (generated if not provided)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        existing = existing_metadata or {}

        # Check what fields are missing
        required_fields = ["title", "authors", "year", "source"]
        missing_fields = [f for f in required_fields if not existing.get(f)]

        if not missing_fields:
            # All required fields present, just generate citation key if missing
            if not existing.get("citation_key"):
                existing["citation_key"] = self._generate_citation_key(existing)
            return existing

        # Extract metadata using Vision
        extracted = self._extract_with_vision(pdf_path, missing_fields)

        # Merge with existing (existing takes precedence)
        result = {**extracted, **{k: v for k, v in existing.items() if v}}

        # Generate citation key if not present
        if not result.get("citation_key"):
            result["citation_key"] = self._generate_citation_key(result)

        return result

    def _extract_with_vision(
        self,
        pdf_path: Path,
        missing_fields: List[str],
    ) -> Dict[str, Any]:
        """Extract metadata from PDF using Gemini Vision.

        Args:
            pdf_path: Path to PDF
            missing_fields: List of fields to extract

        Returns:
            Extracted metadata dictionary
        """
        # Render first 2-3 pages to images
        doc = fitz.open(str(pdf_path))
        pages_to_check = min(3, len(doc))

        image_parts = []
        for i in range(pages_to_check):
            page = doc[i]
            mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
            pix = page.get_pixmap(matrix=mat)
            image_bytes = pix.tobytes("png")
            image_parts.append({
                "mime_type": "image/png",
                "data": base64.b64encode(image_bytes).decode("utf-8"),
            })

        doc.close()

        # Build extraction prompt
        prompt = self._build_extraction_prompt(missing_fields)

        # Call Gemini Vision
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    [prompt] + image_parts,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=2048,
                    ),
                )
                return self._parse_extraction_response(response.text)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    # Return empty dict on failure
                    return {}

        return {}

    def _build_extraction_prompt(self, missing_fields: List[str]) -> str:
        """Build prompt for metadata extraction."""
        fields_str = ", ".join(missing_fields)

        prompt = f"""Analyze these pages from an academic document and extract bibliographic metadata.

I need to extract these fields: {fields_str}

Look for:
1. TITLE: Usually the largest text on the first page, or in the header
2. AUTHORS: Names listed below the title, or in the header/footer
3. YEAR: Publication year (look in copyright, header, or first page footer)
4. SOURCE: Journal name, publisher, or conference name
5. VOLUME/ISSUE: For journal articles
6. PAGES: Page range (e.g., "243-281")
7. DOI: Digital Object Identifier if present

For ENTRY_TYPE, determine if this is:
- "article" - Journal article
- "book" - Book or book chapter
- "inproceedings" - Conference paper

OUTPUT FORMAT (JSON only, no markdown):
{{
    "title": "Full Title of the Document",
    "authors": ["First Author Name", "Second Author Name"],
    "year": 2023,
    "source": "Journal Name or Publisher",
    "entry_type": "article",
    "volume": "8",
    "issue": "3",
    "pages": "243-281",
    "doi": "10.1234/example"
}}

Notes:
- For authors, use full names in "First Last" format
- Year should be an integer
- Set null for any field you cannot determine with confidence
- If this appears to be a book, use the publisher as source

IMPORTANT: Return ONLY valid JSON, no explanation or markdown."""

        return prompt

    def _parse_extraction_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response into metadata dictionary."""
        # Clean response
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return {}
            else:
                return {}

        # Normalize the data
        result = {}

        if data.get("title"):
            result["title"] = str(data["title"]).strip()

        if data.get("authors"):
            authors = data["authors"]
            if isinstance(authors, list):
                result["authors"] = [str(a).strip() for a in authors if a]
            elif isinstance(authors, str):
                # Split by "and" or comma
                result["authors"] = [a.strip() for a in re.split(r',|and', authors) if a.strip()]

        if data.get("year"):
            try:
                result["year"] = int(data["year"])
            except (ValueError, TypeError):
                # Try to extract year from string
                year_match = re.search(r'(19|20)\d{2}', str(data["year"]))
                if year_match:
                    result["year"] = int(year_match.group())

        if data.get("source"):
            result["source"] = str(data["source"]).strip()

        if data.get("entry_type"):
            entry_type = str(data["entry_type"]).lower()
            if entry_type in ["article", "book", "inproceedings", "incollection", "phdthesis"]:
                result["entry_type"] = entry_type
            else:
                result["entry_type"] = "article"

        # Optional fields
        for field in ["volume", "issue", "pages", "doi"]:
            if data.get(field):
                result[field] = str(data[field]).strip()

        return result

    def _generate_citation_key(self, metadata: Dict[str, Any]) -> str:
        """Generate a citation key from metadata.

        Format: firstauthorlastname + year (e.g., "halliday1976")
        """
        authors = metadata.get("authors", [])
        year = metadata.get("year", 0)

        if authors:
            # Get first author's last name
            first_author = authors[0]
            # Handle "Last, First" or "First Last" format
            if "," in first_author:
                last_name = first_author.split(",")[0].strip()
            else:
                parts = first_author.split()
                last_name = parts[-1] if parts else "unknown"

            # Clean and lowercase
            last_name = re.sub(r'[^a-zA-Z]', '', last_name).lower()
        else:
            last_name = "unknown"

        return f"{last_name}{year}"

    def extract_from_filename(self, filename: str) -> Dict[str, Any]:
        """Try to extract metadata from PDF filename.

        Common patterns:
        - Author_Year_Title.pdf
        - Author1_Author2_Year_Title.pdf

        Args:
            filename: PDF filename (not full path)

        Returns:
            Partial metadata dictionary
        """
        name = Path(filename).stem  # Remove .pdf extension

        # Replace underscores with spaces for parsing
        parts = name.replace("_", " ").split()

        result = {}

        # Look for year (4 digit number starting with 19 or 20)
        for i, part in enumerate(parts):
            year_match = re.search(r'^(19|20)\d{2}$', part)
            if year_match:
                result["year"] = int(year_match.group())

                # Everything before year is likely authors
                if i > 0:
                    author_parts = parts[:i]
                    # Group by potential author names
                    result["authors"] = self._parse_author_parts(author_parts)

                # Everything after year is likely title
                if i < len(parts) - 1:
                    result["title"] = " ".join(parts[i+1:])

                break

        return result

    def _parse_author_parts(self, parts: List[str]) -> List[str]:
        """Parse author name parts into author list."""
        authors = []
        current_author = []

        for part in parts:
            # Check if this starts a new author (capitalized, not a common word)
            if part[0].isupper() and part.lower() not in ["and", "the", "of", "in"]:
                if current_author and len(current_author) >= 1:
                    # Save previous author
                    authors.append(" ".join(current_author))
                    current_author = []
                current_author.append(part)
            else:
                current_author.append(part)

        if current_author:
            authors.append(" ".join(current_author))

        return authors


def extract_pdf_metadata(
    pdf_path: str,
    gemini_api_key: str,
    existing_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function to extract metadata from a PDF.

    Args:
        pdf_path: Path to PDF file
        gemini_api_key: Gemini API key
        existing_metadata: Partial metadata to complete

    Returns:
        Complete metadata dictionary
    """
    extractor = MetadataExtractor(gemini_api_key)

    # First try to get info from filename
    filename_info = extractor.extract_from_filename(Path(pdf_path).name)

    # Merge with existing
    base_metadata = {**filename_info, **(existing_metadata or {})}

    # Use Vision to fill gaps
    return extractor.extract_metadata(pdf_path, base_metadata)


def batch_extract_metadata(
    pdf_paths: List[str],
    gemini_api_key: str,
    existing_metadata: Optional[List[Dict[str, Any]]] = None,
    progress_callback: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """Extract metadata from multiple PDFs.

    Args:
        pdf_paths: List of PDF file paths
        gemini_api_key: Gemini API key
        existing_metadata: List of partial metadata dicts (same order as pdf_paths)
        progress_callback: Optional callback(current, total, citation_key)

    Returns:
        List of complete metadata dictionaries
    """
    extractor = MetadataExtractor(gemini_api_key)
    existing = existing_metadata or [{}] * len(pdf_paths)

    results = []
    for i, (pdf_path, existing_meta) in enumerate(zip(pdf_paths, existing)):
        if progress_callback:
            key = existing_meta.get("citation_key", Path(pdf_path).stem)
            progress_callback(i + 1, len(pdf_paths), key)

        try:
            # Get from filename first
            filename_info = extractor.extract_from_filename(Path(pdf_path).name)
            base = {**filename_info, **existing_meta}

            # Fill gaps with Vision
            result = extractor.extract_metadata(pdf_path, base)
            results.append(result)
        except Exception as e:
            # On error, return existing with generated key
            fallback = existing_meta.copy()
            if not fallback.get("citation_key"):
                fallback["citation_key"] = Path(pdf_path).stem.lower().replace(" ", "_")
            results.append(fallback)

    return results
