"""Citation engine powered by Gemini.

This module implements citation matching with three modes:
1. Full document mode: Send entire document at once
2. Paragraph-by-paragraph mode: Process each paragraph for thorough coverage
3. Grounded RAG mode: Use verified page numbers from retrieval (NEW)

MODIFIED: Grounded RAG for accurate page citations.
"""

import logging
import json
import re
from typing import List, Dict, Any, Tuple, Optional, Union, TYPE_CHECKING
import google.generativeai as genai

from .models import PageAwarePDF, Citation, CitationStyle, RetrievedChunk

if TYPE_CHECKING:
    from .page_aware_rag import PageAwareRAG
    from .citation_export import CitationExporter

logger = logging.getLogger(__name__)


class GeminiCitationEngine:
    """Citation engine using Gemini with aggressive citation matching.

    Supports two processing modes:
    - Full document: Faster, sends entire document at once
    - Paragraph-by-paragraph: More thorough, processes each paragraph
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3-flash-preview",
        max_context_tokens: int = 900000,
        aggressive_mode: bool = True,
        paragraph_mode: bool = True,  # NEW: Process paragraph by paragraph
    ):
        """Initialize Gemini citation engine.

        Args:
            api_key: Google Gemini API key
            model: Gemini model to use
            max_context_tokens: Maximum tokens to use
            aggressive_mode: If True, find more citations (default: True)
            paragraph_mode: If True, process paragraph by paragraph (default: True)
        """
        self.api_key = api_key
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.aggressive_mode = aggressive_mode
        self.paragraph_mode = paragraph_mode

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

        logger.info(
            f"Gemini citation engine initialized: model={model}, "
            f"aggressive={aggressive_mode}, paragraph_mode={paragraph_mode}"
        )

    def analyze_and_cite(
        self,
        document_text: str,
        bibliography: List[PageAwarePDF],
        style: CitationStyle,
        max_citations_per_claim: int = 3,
    ) -> List[Citation]:
        """Analyze document and generate citations.

        Uses paragraph-by-paragraph mode if enabled for thorough coverage.
        """
        logger.info(
            f"Analyzing document ({len(document_text)} chars) "
            f"against {len(bibliography)} sources"
        )

        if self.paragraph_mode and self.aggressive_mode:
            return self._analyze_by_paragraphs(
                document_text=document_text,
                bibliography=bibliography,
                style=style,
                max_citations_per_claim=max_citations_per_claim,
            )
        else:
            return self._analyze_full_document(
                document_text=document_text,
                bibliography=bibliography,
                style=style,
                max_citations_per_claim=max_citations_per_claim,
            )

    def _analyze_by_paragraphs(
        self,
        document_text: str,
        bibliography: List[PageAwarePDF],
        style: CitationStyle,
        max_citations_per_claim: int,
        batch_size: int = 4,
    ) -> List[Citation]:
        """Process document in paragraph batches for thorough citation coverage."""

        # Split into paragraphs (by double newline or markdown headers)
        paragraphs = self._split_into_paragraphs(document_text)
        # Filter out short paragraphs
        paragraphs = [p for p in paragraphs if len(p.strip()) >= 50]
        logger.info(f"Split document into {len(paragraphs)} paragraphs (batch size: {batch_size})")

        # Build bibliography context once (reused for all batches)
        bibliography_context = self._build_compact_bibliography_context(bibliography)

        all_citations = []
        seen_claims = set()  # Avoid duplicate citations

        # Process in batches of batch_size paragraphs
        total_batches = (len(paragraphs) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(paragraphs))
            batch_paragraphs = paragraphs[start_idx:end_idx]

            logger.info(f"Processing batch {batch_idx+1}/{total_batches} (paragraphs {start_idx+1}-{end_idx})...")

            try:
                # Combine paragraphs with markers
                combined_text = "\n\n".join([
                    f"[PARAGRAPH {start_idx + i + 1}]\n{p}"
                    for i, p in enumerate(batch_paragraphs)
                ])

                prompt = self._build_paragraph_prompt(
                    paragraph=combined_text,
                    paragraph_number=batch_idx + 1,
                    total_paragraphs=total_batches,
                    bibliography_context=bibliography_context,
                    bibliography=bibliography,
                    style=style,
                )

                response = self.client.generate_content(prompt)

                citations = self._parse_citations_from_response(
                    response=response,
                    bibliography=bibliography,
                    style=style,
                )

                # Deduplicate
                for citation in citations:
                    claim_key = citation.claim_text[:100]
                    if claim_key not in seen_claims:
                        seen_claims.add(claim_key)
                        all_citations.append(citation)

            except Exception as e:
                logger.warning(f"Failed to process batch {batch_idx+1}: {e}")
                continue

        logger.info(f"✓ Generated {len(all_citations)} citations from {len(paragraphs)} paragraphs")
        return all_citations

    def _split_into_paragraphs(self, document_text: str) -> List[str]:
        """Split document into processable paragraphs."""
        # Split by double newlines or markdown section breaks
        raw_paragraphs = re.split(r'\n\s*\n|\n-{3,}\n|\n#{1,6}\s+', document_text)

        # Merge very short paragraphs with next one
        paragraphs = []
        buffer = ""

        for p in raw_paragraphs:
            p = p.strip()
            if not p:
                continue

            buffer += "\n\n" + p if buffer else p

            # If buffer is substantial enough, add it
            if len(buffer) > 200:
                paragraphs.append(buffer)
                buffer = ""

        if buffer:
            paragraphs.append(buffer)

        return paragraphs

    def _build_paragraph_prompt(
        self,
        paragraph: str,
        paragraph_number: int,
        total_paragraphs: int,
        bibliography_context: str,
        bibliography: List[PageAwarePDF],
        style: CitationStyle,
    ) -> str:
        """Build prompt for a single paragraph."""

        # Citation key mapping
        key_author_map = []
        for pdf in bibliography:
            author_name = pdf.reference.authors[0].split()[-1] if pdf.reference.authors else "Unknown"
            key_author_map.append(f"- {pdf.citation_key}: {author_name} ({pdf.reference.year})")

        key_mapping_str = "\n".join(key_author_map)

        prompt = f"""You are an expert academic citation assistant. Analyze this paragraph and identify ALL claims that need citations.

AVAILABLE SOURCES:
{key_mapping_str}

CITATION FORMAT (APA 7th):
- (Author, Year, p. PageNumber)
- 2 authors: (Smith & Jones, 2023, p. 42)
- 3+ authors: (Smith et al., 2023, p. 42)

=== PARAGRAPH TO ANALYZE ({paragraph_number}/{total_paragraphs}) ===

{paragraph}

=== BIBLIOGRAPHY EXCERPTS ===

{bibliography_context}

=== INSTRUCTIONS ===

Find EVERY citation opportunity in this paragraph:
1. Author mentions (Halliday, Beaugrande, Van Dijk, Langacker, etc.)
2. Theoretical concepts (cohesion, coherence, construal, textuality, etc.)
3. Specific claims about linguistics or NLP
4. References to specific theories (RST, Centering, BPE, etc.)

For EACH opportunity, provide:
- claim_text: The EXACT text from the paragraph (copy precisely)
- citation_key: Which source to cite
- pdf_page_number: Page number in the PDF (use 1 if unsure)
- confidence: 0.5-1.0

OUTPUT (JSON only, no explanation):
```json
{{
  "citations": [
    {{
      "claim_text": "exact text needing citation",
      "citation_key": "halliday1976",
      "pdf_page_number": 1,
      "evidence_text": "brief evidence from source",
      "confidence": 0.8,
      "reason": "why this needs citation"
    }}
  ]
}}
```

If this paragraph has no citable claims, return: {{"citations": []}}
"""
        return prompt

    def _build_compact_bibliography_context(
        self,
        bibliography: List[PageAwarePDF],
    ) -> str:
        """Build compact bibliography for paragraph mode (all pages, full text)."""
        context_parts = []

        for pdf in bibliography:
            authors_str = ", ".join(pdf.reference.authors[:2]) if pdf.reference.authors else "Unknown"

            header = f"""
=== {pdf.citation_key} ===
{authors_str} ({pdf.reference.year}). {pdf.reference.title}

Full content:
"""
            context_parts.append(header)

            # NO PAGE LIMIT - include ALL pages with FULL text
            for page in pdf.pages:
                context_parts.append(f"[p.{page.pdf_page_number}] {page.text_content}\n")

        return "".join(context_parts)

    def _analyze_full_document(
        self,
        document_text: str,
        bibliography: List[PageAwarePDF],
        style: CitationStyle,
        max_citations_per_claim: int,
    ) -> List[Citation]:
        """Original full document analysis."""
        if self.aggressive_mode:
            prompt = self._build_aggressive_prompt(
                document_text=document_text,
                bibliography=bibliography,
                style=style,
                max_citations_per_claim=max_citations_per_claim,
            )
        else:
            prompt = self._build_conservative_prompt(
                document_text=document_text,
                bibliography=bibliography,
                style=style,
                max_citations_per_claim=max_citations_per_claim,
            )

        response = self.client.generate_content(prompt)

        citations = self._parse_citations_from_response(
            response=response,
            bibliography=bibliography,
            style=style,
        )

        logger.info(f"✓ Generated {len(citations)} citations")
        return citations

    def _build_aggressive_prompt(
        self,
        document_text: str,
        bibliography: List[PageAwarePDF],
        style: CitationStyle,
        max_citations_per_claim: int,
    ) -> str:
        """Build aggressive prompt for thorough citation matching."""
        author_list = []
        for pdf in bibliography:
            for author in pdf.reference.authors:
                parts = author.replace(",", "").split()
                if parts:
                    last_name = parts[-1] if len(parts) == 1 else parts[0]
                    author_list.append(last_name)

        authors_str = ", ".join(set(author_list))

        key_author_map = []
        for pdf in bibliography:
            author_name = pdf.reference.authors[0].split()[-1] if pdf.reference.authors else "Unknown"
            key_author_map.append(f"- {pdf.citation_key}: {author_name} ({pdf.reference.year})")

        key_mapping_str = "\n".join(key_author_map)

        if style == CitationStyle.APA7:
            style_guide = """
APA 7th Edition:
- Format: (Author, Year, p. PageNumber)
- 2 authors: (Smith & Jones, 2023, p. 42)
- 3+ authors: (Smith et al., 2023, p. 42)
"""
        else:
            style_guide = """
Chicago 17th Edition (Notes & Bibliography):
- Format: Superscript footnote number
- Footnote: Author, "Title," Source (Year): Page.
"""

        bibliography_context = self._build_rich_bibliography_context(bibliography)

        spanish_indicators = ["el", "la", "los", "las", "de", "que", "en", "es", "un", "una"]
        doc_lower = document_text.lower()[:1000]
        is_spanish = sum(1 for w in spanish_indicators if f" {w} " in doc_lower) > 3

        language_instruction = """
IDIOMA: El documento está en ESPAÑOL. Busca menciones de autores y conceptos en español.
""" if is_spanish else ""

        prompt = f"""You are an expert academic citation assistant. Your task is to analyze an academic document and insert bibliographic citations EXHAUSTIVELY.

AUTHORS AVAILABLE IN BIBLIOGRAPHY:
{authors_str}

CITATION KEY MAPPING:
{key_mapping_str}

{style_guide}
{language_instruction}

=== CRITICAL INSTRUCTIONS ===

1. AGGRESSIVELY SEARCH for citation opportunities:
   - Every author mention (Halliday, Beaugrande, Langacker, Van Dijk, etc.)
   - Every theoretical concept attributable to a source
   - Every paraphrase of ideas from sources
   - Every empirical or theoretical claim needing support

2. BE GENEROUS WITH CITATIONS:
   - More citations are better than fewer
   - A dense theoretical paragraph may need 3-5 citations
   - Author mentions should ALWAYS have citations

---
DOCUMENT TO ANALYZE:
---

{document_text}

---
FULL BIBLIOGRAPHY:
---

{bibliography_context}

---
OUTPUT (JSON):
```json
{{
  "citations": [
    {{
      "claim_text": "exact text from document",
      "citation_key": "halliday1976",
      "pdf_page_number": 15,
      "evidence_text": "quote from source",
      "confidence": 0.85,
      "reason": "why"
    }}
  ]
}}
```

Generate AT LEAST 20-40 citations for a long academic document.
"""

        return prompt

    def _build_conservative_prompt(
        self,
        document_text: str,
        bibliography: List[PageAwarePDF],
        style: CitationStyle,
        max_citations_per_claim: int,
    ) -> str:
        """Build conservative prompt (original behavior)."""
        if style == CitationStyle.APA7:
            style_guide = """
APA 7th Edition Requirements:
- Format: (Author, Year, p. PageNumber)
- Direct quotes: MUST include page number
- Paraphrases: Page number recommended for specific claims
"""
        else:
            style_guide = """
Chicago 17th Edition (Notes & Bibliography) Requirements:
- Format: Superscript footnote number in text
- Footnote: Author, "Title," Source Volume, no. Issue (Year): Page.
"""

        bibliography_context = self._build_bibliography_context(bibliography)

        prompt = f"""You are an expert academic citation assistant. Analyze the document and suggest accurate in-text citations.

{style_guide}

---
USER DOCUMENT:
---

{document_text}

---
BIBLIOGRAPHY:
---

{bibliography_context}

---
OUTPUT FORMAT (JSON):
```json
{{
  "citations": [
    {{
      "claim_text": "Exact text from document",
      "citation_key": "author2023key",
      "pdf_page_number": 15,
      "evidence_text": "Quote from source",
      "confidence": 0.95,
      "reason": "Why this source supports this claim"
    }}
  ]
}}
```

Be conservative - only suggest citations you're confident about.
"""

        return prompt

    def _build_rich_bibliography_context(
        self,
        bibliography: List[PageAwarePDF],
    ) -> str:
        """Build rich bibliography context - NO LIMITS, full PDF text."""
        context_parts = []

        for i, pdf in enumerate(bibliography, 1):
            authors_str = ", ".join(pdf.reference.authors[:3]) if pdf.reference.authors else "Unknown"
            if len(pdf.reference.authors) > 3:
                authors_str += " et al."

            header = f"""
================================================================================
SOURCE {i}: {pdf.citation_key}
================================================================================
Authors: {authors_str}
Title: {pdf.reference.title}
Year: {pdf.reference.year}
Journal/Publisher: {pdf.reference.source}

FULL CONTENT ({len(pdf.pages)} pages):
"""
            context_parts.append(header)

            # NO LIMITS - include ALL pages with FULL text
            for page in pdf.pages:
                page_header = f"\n--- PDF Page {page.pdf_page_number} ---\n"
                context_parts.append(page_header)
                context_parts.append(page.text_content)

        return "".join(context_parts)

    def _build_bibliography_context(
        self,
        bibliography: List[PageAwarePDF],
    ) -> str:
        """Build bibliography context - NO LIMITS, full PDF text."""
        context_parts = []

        for i, pdf in enumerate(bibliography, 1):
            authors_str = ", ".join(pdf.reference.authors[:3]) if pdf.reference.authors else "Unknown"

            header = f"""
========================================
SOURCE {i}: {pdf.citation_key}
========================================
Authors: {authors_str}
Title: {pdf.reference.title}
Year: {pdf.reference.year}
Total pages: {len(pdf.pages)}
"""
            context_parts.append(header)

            # NO LIMITS - include ALL pages with FULL text
            for page in pdf.pages:
                page_header = f"\n--- Page {page.pdf_page_number} ---\n"
                context_parts.append(page_header)
                context_parts.append(page.text_content)

        return "".join(context_parts)

    def _parse_citations_from_response(
        self,
        response: Any,
        bibliography: List[PageAwarePDF],
        style: CitationStyle,
    ) -> List[Citation]:
        """Parse Gemini response into Citation objects."""
        citations = []

        try:
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

            pdf_map = {pdf.citation_key: pdf for pdf in bibliography}

            for cite_data in data.get("citations", []):
                citation_key = cite_data.get("citation_key")
                source_pdf = pdf_map.get(citation_key)

                # Fuzzy matching
                if not source_pdf and citation_key:
                    for key, pdf in pdf_map.items():
                        if (citation_key.lower() in key.lower() or
                            key.lower() in citation_key.lower()):
                            source_pdf = pdf
                            break

                if not source_pdf:
                    logger.warning(f"Citation key not found: {citation_key}")
                    continue

                citation = Citation(
                    source_pdf=source_pdf,
                    page_number=cite_data.get("pdf_page_number", 1),
                    claim_text=cite_data.get("claim_text", ""),
                    evidence_text=cite_data.get("evidence_text", ""),
                    confidence=cite_data.get("confidence", 0.7),
                )

                if style == CitationStyle.APA7:
                    citation.citation_string = citation.format_apa7()
                else:
                    citation.citation_string = citation.format_chicago17()

                citations.append(citation)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
        except Exception as e:
            logger.error(f"Error parsing citations: {e}")

        return citations

    # ==================== GROUNDED RAG MODE ====================

    def analyze_with_grounded_rag(
        self,
        document_text: str,
        rag: "PageAwareRAG",
        style: CitationStyle,
        max_citations_per_claim: int = 3,
        batch_size: int = 4,
        exporter: Optional["CitationExporter"] = None,
    ) -> List["GroundedCitation"]:
        """Analyze document using grounded RAG for verified page numbers.

        This is the preferred mode for accurate citations:
        - Page numbers come from retrieval metadata, not guessing
        - Gemini can only cite pages that exist in retrieved chunks
        - Every citation is grounded in verified evidence
        - Processes paragraphs in batches to reduce API calls

        Args:
            document_text: Document to analyze
            rag: PageAwareRAG instance with indexed bibliography
            style: Citation style (APA7 or Chicago17)
            max_citations_per_claim: Maximum citations per claim
            batch_size: Number of paragraphs to process per batch
            exporter: Optional CitationExporter for CSV/JSON export

        Returns:
            List of GroundedCitation objects with verified page numbers
        """
        logger.info(f"Starting grounded RAG citation analysis ({len(document_text)} chars)")

        # Split into paragraphs and filter short ones
        paragraphs = self._split_into_paragraphs(document_text)
        paragraphs = [p for p in paragraphs if len(p.strip()) >= 50]
        logger.info(f"Split document into {len(paragraphs)} paragraphs (batch size: {batch_size})")

        # Register paragraphs with exporter if provided
        if exporter:
            for i, para in enumerate(paragraphs):
                exporter.add_sentence(i, para)

        all_citations = []
        seen_claims = set()

        # Process in batches
        total_batches = (len(paragraphs) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(paragraphs))
            batch_paragraphs = paragraphs[start_idx:end_idx]

            logger.info(f"Processing batch {batch_idx+1}/{total_batches} (paragraphs {start_idx+1}-{end_idx}) with RAG...")

            try:
                # Combine paragraphs with markers
                combined_text = "\n\n".join([
                    f"[PARAGRAPH {start_idx + i + 1}]\n{p}"
                    for i, p in enumerate(batch_paragraphs)
                ])

                citations, retrieved_chunks = self._cite_paragraph_with_rag(
                    paragraph=combined_text,
                    paragraph_number=batch_idx + 1,
                    total_paragraphs=total_batches,
                    rag=rag,
                    style=style,
                    return_chunks=True,
                )

                # Add to exporter if provided
                if exporter and retrieved_chunks:
                    # Add chunks for each paragraph in batch
                    for i, para in enumerate(batch_paragraphs):
                        para_idx = start_idx + i
                        exporter.add_retrieved_chunks(para_idx, retrieved_chunks, max_chunks=5)

                # Deduplicate and track source paragraph
                for citation in citations:
                    claim_key = citation.claim_text[:100]
                    if claim_key not in seen_claims:
                        seen_claims.add(claim_key)

                        # Find which paragraph this citation belongs to
                        if exporter:
                            best_para_idx = start_idx
                            best_overlap = 0
                            for i, para in enumerate(batch_paragraphs):
                                if citation.claim_text and citation.claim_text in para:
                                    best_para_idx = start_idx + i
                                    break
                                overlap = len(set(citation.claim_text.lower().split()) &
                                            set(para.lower().split()))
                                if overlap > best_overlap:
                                    best_overlap = overlap
                                    best_para_idx = start_idx + i

                            citation.source_paragraph_id = best_para_idx
                            exporter.add_citation(best_para_idx, citation)

                        all_citations.append(citation)

            except Exception as e:
                logger.warning(f"Failed to process batch {batch_idx+1}: {e}")
                continue

        logger.info(f"✓ Generated {len(all_citations)} grounded citations from {len(paragraphs)} paragraphs")
        return all_citations

    def _cite_paragraph_with_rag(
        self,
        paragraph: str,
        paragraph_number: int,
        total_paragraphs: int,
        rag: "PageAwareRAG",
        style: CitationStyle,
        return_chunks: bool = False,
    ) -> Union[List["GroundedCitation"], Tuple[List["GroundedCitation"], List[RetrievedChunk]]]:
        """Generate citations for a paragraph using RAG retrieval.

        Page numbers come from retrieval metadata, NOT from Gemini guessing.

        Args:
            paragraph: Paragraph text to analyze
            paragraph_number: Current paragraph number
            total_paragraphs: Total paragraph count
            rag: PageAwareRAG instance
            style: Citation style
            return_chunks: If True, also return retrieved chunks

        Returns:
            If return_chunks=False: List of GroundedCitation
            If return_chunks=True: Tuple of (citations, retrieved_chunks)
        """
        # Step 1: Retrieve relevant chunks from RAG
        chunks = rag.query_for_paragraph(paragraph, n_results=30)

        if not chunks:
            return ([], []) if return_chunks else []

        # Step 2: Format evidence with VERIFIED page numbers
        evidence_lines = []
        for c in chunks:
            text_preview = c.text[:600] + "..." if len(c.text) > 600 else c.text
            evidence_lines.append(
                f"[{c.citation_key}, p.{c.book_page}] (similarity: {c.similarity:.2f})\n{text_preview}"
            )

        evidence_str = "\n\n---\n\n".join(evidence_lines)

        # Step 3: Build prompt that ONLY allows retrieved page numbers
        prompt = self._build_grounded_rag_prompt(
            paragraph=paragraph,
            paragraph_number=paragraph_number,
            total_paragraphs=total_paragraphs,
            evidence_str=evidence_str,
            chunks=chunks,
        )

        # Step 4: Call Gemini
        response = self.client.generate_content(prompt)

        # Step 5: Parse and validate citations
        citations = self._parse_grounded_citations(response, chunks, style)

        if return_chunks:
            return citations, chunks
        return citations

    def _build_grounded_rag_prompt(
        self,
        paragraph: str,
        paragraph_number: int,
        total_paragraphs: int,
        evidence_str: str,
        chunks: List[RetrievedChunk],
    ) -> str:
        """Build prompt for grounded RAG citation matching."""
        # Build list of valid citation targets with page ranges
        valid_targets = []
        pages_by_source = {}  # citation_key -> sorted list of pages
        seen = set()

        for c in chunks:
            key = f"{c.citation_key}_p{c.book_page}"
            if key not in seen:
                seen.add(key)
                valid_targets.append(f"- {c.citation_key}, p.{c.book_page}")
                # Track pages per source for range detection
                if c.citation_key not in pages_by_source:
                    pages_by_source[c.citation_key] = []
                pages_by_source[c.citation_key].append(c.book_page)

        # Sort pages for each source
        for key in pages_by_source:
            pages_by_source[key] = sorted(set(pages_by_source[key]))

        valid_targets_str = "\n".join(valid_targets[:50])

        # Build page range info for sources with consecutive pages
        range_info = []
        for citation_key, pages in pages_by_source.items():
            if len(pages) > 1:
                range_info.append(f"  {citation_key}: pages {pages}")
        range_info_str = "\n".join(range_info) if range_info else "  (no multi-page sources detected)"

        prompt = f"""You are an expert academic citation assistant using GROUNDED evidence matching.

=== PARAGRAPH TO ANALYZE ({paragraph_number}/{total_paragraphs}) ===

{paragraph}

=== RETRIEVED EVIDENCE (with VERIFIED page numbers) ===

{evidence_str}

=== VALID CITATION TARGETS ===

{valid_targets_str}

=== AVAILABLE PAGE RANGES ===

{range_info_str}

=== CRITICAL RULES ===

1. ONLY cite using (citation_key, page_number) pairs from RETRIEVED EVIDENCE above
2. Do NOT guess page numbers - they are VERIFIED from OCR
3. If no evidence matches a claim, do NOT cite it
4. Copy the EXACT page number from evidence header [citation_key, p.XX]
5. Match claims to evidence based on semantic similarity

=== PAGE RANGES ===

If a claim is supported by evidence from MULTIPLE CONSECUTIVE pages (e.g., p.3, p.4, p.5),
you may specify a page range using "book_page_end". Both start and end pages MUST exist
in the retrieved evidence. Example: book_page=3, book_page_end=5 for pp. 3-5.

Only use page ranges when:
- The claim is a broad topic/concept spanning multiple pages
- You have evidence from at least 2-3 consecutive pages
- The evidence from different pages supports the SAME claim

=== OUTPUT FORMAT (JSON only) ===

```json
{{
  "citations": [
    {{
      "claim_text": "exact text from paragraph needing citation",
      "citation_key": "halliday1976",
      "book_page": 322,
      "book_page_end": null,
      "evidence_match": "brief quote from matching evidence",
      "confidence": 0.85
    }},
    {{
      "claim_text": "topic spanning multiple pages",
      "citation_key": "beaugrande1981",
      "book_page": 3,
      "book_page_end": 11,
      "evidence_match": "concept introduced on p.3, elaborated through p.11",
      "confidence": 0.90
    }}
  ]
}}
```

If no claims match the evidence, return: {{"citations": []}}
"""
        return prompt

    def _parse_grounded_citations(
        self,
        response: Any,
        chunks: List[RetrievedChunk],
        style: CitationStyle,
    ) -> List["GroundedCitation"]:
        """Parse grounded citations and validate page numbers.

        Supports page ranges (e.g., pp. 3-11) when Gemini provides book_page_end.
        """
        citations = []

        try:
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
                logger.warning("No JSON found in grounded RAG response")
                return []

            data = json.loads(json_text)

            # Build lookup for validation: (citation_key, page) -> chunk
            valid_pages = {}
            for chunk in chunks:
                key = (chunk.citation_key, chunk.book_page)
                if key not in valid_pages:
                    valid_pages[key] = chunk

            for cite_data in data.get("citations", []):
                citation_key = cite_data.get("citation_key")
                book_page = cite_data.get("book_page", 1)
                book_page_end = cite_data.get("book_page_end")  # NEW: page range end

                # Validate start page exists in retrieval
                lookup_key = (citation_key, book_page)
                if lookup_key not in valid_pages:
                    # Try fuzzy match
                    matched = False
                    for (ck, pg), chunk in valid_pages.items():
                        if citation_key and citation_key.lower() in ck.lower():
                            lookup_key = (ck, pg)
                            matched = True
                            break
                    if not matched:
                        logger.warning(f"Skipping unverified: {citation_key} p.{book_page}")
                        continue

                chunk = valid_pages[lookup_key]

                # Validate page range if specified
                page_end = None
                pdf_page_end = None
                if book_page_end is not None and book_page_end != book_page:
                    # Validate end page exists
                    end_key = (citation_key, book_page_end)
                    if end_key in valid_pages:
                        end_chunk = valid_pages[end_key]
                        page_end = book_page_end
                        pdf_page_end = end_chunk.pdf_page
                        logger.info(f"Page range validated: {citation_key} pp. {book_page}-{book_page_end}")
                    else:
                        # Try to find closest valid end page
                        valid_source_pages = sorted([
                            pg for (ck, pg) in valid_pages.keys()
                            if ck == citation_key and pg > book_page
                        ])
                        if valid_source_pages:
                            # Use highest available page up to requested end
                            for pg in reversed(valid_source_pages):
                                if pg <= book_page_end:
                                    end_chunk = valid_pages[(citation_key, pg)]
                                    page_end = pg
                                    pdf_page_end = end_chunk.pdf_page
                                    logger.info(f"Page range adjusted: {citation_key} pp. {book_page}-{pg} (requested {book_page_end})")
                                    break

                # Format citation string with page range support
                author_str = chunk.authors if chunk.authors else "Unknown"
                year = chunk.year if chunk.year else 0

                if page_end is not None and page_end != book_page:
                    # Page range: pp. X-Y
                    if style == CitationStyle.APA7:
                        citation_string = f"({author_str}, {year}, pp. {book_page}-{page_end})"
                    else:
                        title = chunk.title if chunk.title else "Unknown"
                        citation_string = f'{author_str}, "{title}" ({year}): {book_page}-{page_end}.'
                else:
                    # Single page: p. X
                    if style == CitationStyle.APA7:
                        citation_string = f"({author_str}, {year}, p. {book_page})"
                    else:
                        title = chunk.title if chunk.title else "Unknown"
                        citation_string = f'{author_str}, "{title}" ({year}): {book_page}.'

                citations.append(GroundedCitation(
                    citation_key=chunk.citation_key,
                    page_number=chunk.book_page,
                    pdf_page_number=chunk.pdf_page,
                    claim_text=cite_data.get("claim_text", ""),
                    evidence_text=cite_data.get("evidence_match", ""),
                    confidence=cite_data.get("confidence", 0.7),
                    citation_string=citation_string,
                    authors=chunk.authors,
                    year=year,
                    title=chunk.title,
                    page_end=page_end,
                    pdf_page_end=pdf_page_end,
                ))

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse grounded JSON: {e}")
        except Exception as e:
            logger.error(f"Error parsing grounded citations: {e}")

        return citations


class GroundedCitation:
    """Citation with verified page from RAG retrieval.

    Supports page ranges for concepts spanning multiple pages (e.g., pp. 123-126).
    """

    def __init__(
        self,
        citation_key: str,
        page_number: int,
        pdf_page_number: int,
        claim_text: str,
        evidence_text: str,
        confidence: float,
        citation_string: str,
        authors: str = "",
        year: int = 0,
        title: str = "",
        source_paragraph_id: int = -1,
        page_end: Optional[int] = None,
        pdf_page_end: Optional[int] = None,
    ):
        self.citation_key = citation_key
        self.page_number = page_number  # Start page (or single page)
        self.page_end = page_end  # End page for ranges (None if single page)
        self.pdf_page_number = pdf_page_number
        self.pdf_page_end = pdf_page_end  # End PDF page for ranges
        self.claim_text = claim_text
        self.evidence_text = evidence_text
        self.confidence = confidence
        self.citation_string = citation_string
        self.authors = authors
        self.year = year
        self.title = title
        self.source_pdf = None
        self.source_paragraph_id = source_paragraph_id  # Which paragraph this citation is for

    @property
    def is_page_range(self) -> bool:
        """Return True if this citation spans multiple pages."""
        return self.page_end is not None and self.page_end != self.page_number

    @property
    def page_string(self) -> str:
        """Format page(s) as 'p. X' or 'pp. X-Y'."""
        if self.is_page_range:
            return f"pp. {self.page_number}-{self.page_end}"
        return f"p. {self.page_number}"

    @property
    def journal_page(self) -> int:
        return self.page_number

    def format_apa7(self) -> str:
        return f"({self.authors}, {self.year}, p. {self.page_number})"

    def format_chicago17(self) -> str:
        return f'{self.authors}, "{self.title}" ({self.year}): {self.page_number}.'
