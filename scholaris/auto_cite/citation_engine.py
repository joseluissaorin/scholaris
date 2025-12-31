"""Citation engine powered by Gemini 2.0 Flash.

This module implements the Full Context Mode where the entire bibliography
is loaded into Gemini's 1M token context window for intelligent citation matching.
"""

import logging
import json
from typing import List, Dict, Any, Tuple
import google.generativeai as genai

from .models import PageAwarePDF, Citation, CitationStyle

logger = logging.getLogger(__name__)


class GeminiCitationEngine:
    """Citation engine using Gemini 2.0 Flash with full context loading.

    This engine leverages Gemini's 1M token context to load entire bibliographies
    (up to ~50 PDFs with full text) and intelligently match claims to sources.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3-flash-preview",
        max_context_tokens: int = 500000,
    ):
        """Initialize Gemini citation engine.

        Args:
            api_key: Google Gemini API key
            model: Gemini model to use (default: gemini-3-flash-preview)
            max_context_tokens: Maximum tokens to use (~500k for safety)
        """
        self.api_key = api_key
        self.model = model
        self.max_context_tokens = max_context_tokens

        # Configure Gemini
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

        logger.info(f"Gemini citation engine initialized: model={model}")

    def analyze_and_cite(
        self,
        document_text: str,
        bibliography: List[PageAwarePDF],
        style: CitationStyle,
        max_citations_per_claim: int = 3,
    ) -> List[Citation]:
        """Analyze document and generate citations using Full Context Mode.

        This is the core method that:
        1. Builds a mega-prompt with full bibliography context
        2. Sends to Gemini 2.0 Flash
        3. Receives structured citation suggestions
        4. Creates Citation objects with page numbers

        Args:
            document_text: User's document text
            bibliography: List of PageAwarePDF objects
            style: Citation style (APA7 or Chicago17)
            max_citations_per_claim: Max citations per claim

        Returns:
            List of Citation objects to insert
        """
        logger.info(
            f"Analyzing document ({len(document_text)} chars) "
            f"against {len(bibliography)} sources"
        )

        # Build Full Context prompt
        prompt = self._build_full_context_prompt(
            document_text=document_text,
            bibliography=bibliography,
            style=style,
            max_citations_per_claim=max_citations_per_claim,
        )

        # Send to Gemini
        logger.info("Sending request to Gemini 2.0 Flash...")
        response = self.client.generate_content(prompt)

        # Parse response into Citation objects
        citations = self._parse_citations_from_response(
            response=response,
            bibliography=bibliography,
            style=style,
        )

        logger.info(f"✓ Generated {len(citations)} citations")
        return citations

    def _build_full_context_prompt(
        self,
        document_text: str,
        bibliography: List[PageAwarePDF],
        style: CitationStyle,
        max_citations_per_claim: int,
    ) -> str:
        """Build comprehensive prompt with full bibliography context.

        This prompt includes:
        - System instructions for citation matching
        - Complete user document
        - Full text of all PDFs with page tracking
        - Citation style requirements
        - Output format specification

        Args:
            document_text: User's document
            bibliography: List of PageAwarePDF objects
            style: Citation style
            max_citations_per_claim: Max citations per claim

        Returns:
            Complete prompt string
        """
        # Determine citation style requirements
        if style == CitationStyle.APA7:
            style_guide = """
APA 7th Edition Requirements:
- Format: (Author, Year, p. PageNumber)
- Direct quotes: MUST include page number
- Paraphrases: Page number recommended for specific claims
- Multiple authors: (Smith & Jones, 2023, p. 42) for 2 authors
- 3+ authors: (Smith et al., 2023, p. 42)
"""
        else:  # Chicago 17th
            style_guide = """
Chicago 17th Edition (Notes & Bibliography) Requirements:
- Format: Superscript footnote number in text
- Footnote: Author, "Title," Source Volume, no. Issue (Year): Page.
- First citation: Full format
- Subsequent citations: Shortened format
"""

        # Build bibliography context (full text with page tracking)
        bibliography_context = self._build_bibliography_context(bibliography)

        # Construct mega-prompt
        prompt = f"""You are an expert academic citation assistant. Your task is to analyze a user's document and suggest accurate in-text citations with PRECISE page numbers.

{style_guide}

CRITICAL REQUIREMENTS:
1. Only suggest citations where there is clear evidence in the source
2. Include EXACT page numbers where the evidence appears
3. Maximum {max_citations_per_claim} citations per claim
4. Prioritize citations with high relevance and confidence
5. Page numbers MUST be accurate - this is for publication

---
USER DOCUMENT TO ANALYZE:
---

{document_text}

---
BIBLIOGRAPHY (Full Text with Page Tracking):
---

{bibliography_context}

---
TASK:
---

Analyze the user's document and identify claims that need citations. For each claim:

1. Extract the exact sentence/phrase needing citation
2. Find supporting evidence in the bibliography
3. Identify the EXACT page number where evidence appears
4. Suggest a citation with high confidence

OUTPUT FORMAT (JSON):
```json
{{
  "citations": [
    {{
      "claim_text": "Exact text from user document",
      "citation_key": "author2023key",
      "pdf_page_number": 15,
      "evidence_text": "Exact quote or paraphrase from source",
      "confidence": 0.95,
      "reason": "Why this source supports this claim"
    }}
  ]
}}
```

IMPORTANT:
- Be conservative - only suggest citations you're confident about
- Page numbers MUST be accurate (this is critical!)
- If unsure about page number, set confidence < 0.7
- Include evidence_text to show what supports the claim
"""

        return prompt

    def _build_bibliography_context(
        self,
        bibliography: List[PageAwarePDF],
    ) -> str:
        """Build bibliography context with full PDF text and page tracking.

        Args:
            bibliography: List of PageAwarePDF objects

        Returns:
            Formatted bibliography context string
        """
        context_parts = []

        for i, pdf in enumerate(bibliography, 1):
            # Build header with metadata
            authors_str = ", ".join(pdf.reference.authors[:3]) if pdf.reference.authors else "Unknown"
            if len(pdf.reference.authors) > 3:
                authors_str += " et al."

            header = f"""
========================================
SOURCE {i}: {pdf.citation_key}
========================================
Authors: {authors_str}
Title: {pdf.reference.title}
Year: {pdf.reference.year}
Journal: {pdf.reference.source}
Page Offset: {pdf.page_offset_result.offset}
PDF Pages → Journal Pages: PDF page 1 = Journal page {pdf.get_journal_page(1)}

"""
            context_parts.append(header)

            # Add page-by-page text (with page tracking)
            for page in pdf.pages[:20]:  # Limit to first 20 pages for token budget
                page_header = f"\n--- PDF Page {page.pdf_page_number} (Journal Page {page.journal_page_number}) ---\n"
                context_parts.append(page_header)
                context_parts.append(page.text_content[:3000])  # Limit text per page

        return "".join(context_parts)

    def _parse_citations_from_response(
        self,
        response: Any,
        bibliography: List[PageAwarePDF],
        style: CitationStyle,
    ) -> List[Citation]:
        """Parse Gemini response into Citation objects.

        Args:
            response: Gemini API response
            bibliography: List of PageAwarePDF objects
            style: Citation style

        Returns:
            List of Citation objects
        """
        citations = []

        try:
            # Extract JSON from response
            response_text = response.text

            # Find JSON block
            if "```json" in response_text:
                json_start = response_text.index("```json") + 7
                json_end = response_text.index("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.index("{")
                json_end = response_text.rindex("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                logger.warning("No JSON found in Gemini response")
                return []

            data = json.loads(json_text)

            # Parse each citation
            for cite_data in data.get("citations", []):
                # Find the source PDF
                citation_key = cite_data.get("citation_key")
                source_pdf = None
                for pdf in bibliography:
                    if pdf.citation_key == citation_key:
                        source_pdf = pdf
                        break

                if not source_pdf:
                    logger.warning(f"Citation key not found: {citation_key}")
                    continue

                # Create Citation object
                citation = Citation(
                    source_pdf=source_pdf,
                    page_number=cite_data.get("pdf_page_number", 1),
                    claim_text=cite_data.get("claim_text", ""),
                    evidence_text=cite_data.get("evidence_text", ""),
                    confidence=cite_data.get("confidence", 0.5),
                )

                # Format citation string based on style
                if style == CitationStyle.APA7:
                    citation.citation_string = citation.format_apa7()
                else:  # Chicago17
                    citation.citation_string = citation.format_chicago17()

                citations.append(citation)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini response: {e}")
        except Exception as e:
            logger.error(f"Error parsing citations: {e}")

        return citations
