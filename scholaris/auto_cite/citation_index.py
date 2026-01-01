"""CitationIndex: Runtime index for citation matching across multiple sources.

Loads multiple .spdf files and provides unified search and citation generation.
Supports auto-processing of bibliography folders.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

import numpy as np

from .processed_pdf import ProcessedPDF, SUPPORTED_EXTENSIONS
from .models import CitationStyle, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class IndexedSource:
    """A source in the citation index."""
    citation_key: str
    processed_pdf: ProcessedPDF
    chunk_offset: int  # Offset in global embedding matrix
    chunk_count: int
    source_type: str  # "spdf" or "auto-processed"


class CitationIndex:
    """Runtime index for citation matching across multiple sources.

    Loads multiple .spdf files and provides:
    - Unified vector search across all sources
    - Citation generation with verified page numbers
    - Bibliography folder auto-processing

    Example usage:
        # From bibliography folder
        index = CitationIndex.from_bibliography(
            folder="./bibliography/",
            gemini_api_key="...",
            auto_process=True,
        )

        # Generate citations
        citations = index.cite(
            document_text="Your research...",
            style=CitationStyle.APA7,
        )

        # Or build manually
        index = CitationIndex()
        index.add(ProcessedPDF.load("source1.spdf"))
        index.add(ProcessedPDF.load("source2.spdf"))
    """

    def __init__(self, gemini_api_key: Optional[str] = None):
        """Initialize empty index.

        Args:
            gemini_api_key: API key for querying (embeddings) and citations
        """
        self._sources: Dict[str, IndexedSource] = {}
        self._embedding_matrix: Optional[np.ndarray] = None
        self._chunk_lookup: List[Tuple[str, int]] = []  # (citation_key, chunk_id) for each row
        self._gemini_api_key = gemini_api_key
        self._embedder = None
        self._bibliography_folder: Optional[Path] = None
        self._needs_rebuild = False

    def add(self, processed_pdf: ProcessedPDF) -> None:
        """Add a processed PDF to the index.

        Args:
            processed_pdf: ProcessedPDF instance to add
        """
        if not processed_pdf.metadata:
            raise ValueError("ProcessedPDF has no metadata")

        citation_key = processed_pdf.metadata.citation_key

        if citation_key in self._sources:
            logger.warning(f"Replacing existing source: {citation_key}")

        chunk_offset = len(self._chunk_lookup)

        self._sources[citation_key] = IndexedSource(
            citation_key=citation_key,
            processed_pdf=processed_pdf,
            chunk_offset=chunk_offset,
            chunk_count=len(processed_pdf.chunks),
            source_type="spdf",
        )

        # Add to chunk lookup
        for i, chunk in enumerate(processed_pdf.chunks):
            self._chunk_lookup.append((citation_key, chunk.id))

        self._needs_rebuild = True
        logger.info(f"Added source: {citation_key} ({len(processed_pdf.chunks)} chunks)")

    def add_many(self, processed_pdfs: List[ProcessedPDF]) -> None:
        """Add multiple processed PDFs.

        Args:
            processed_pdfs: List of ProcessedPDF instances
        """
        for pdf in processed_pdfs:
            self.add(pdf)

    def add_directory(self, folder: Union[str, Path]) -> int:
        """Load all .spdf files from a directory.

        Args:
            folder: Path to directory

        Returns:
            Number of files loaded
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")

        count = 0
        for ext in SUPPORTED_EXTENSIONS:
            for path in folder.glob(f"*{ext}"):
                try:
                    processed = ProcessedPDF.load(path)
                    self.add(processed)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")

        logger.info(f"Loaded {count} .spdf files from {folder}")
        return count

    @classmethod
    def from_bibliography(
        cls,
        folder: Union[str, Path],
        gemini_api_key: Optional[str] = None,
        auto_process: bool = True,
        save_processed: bool = True,
        include_previews: bool = True,
        metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> "CitationIndex":
        """Create index from a bibliography folder.

        Automatically:
        1. Loads all .spdf/.scholaris/.scpdf files
        2. Processes .pdf files that don't have corresponding .spdf
        3. Optionally saves processed files for future reuse

        Args:
            folder: Path to bibliography folder
            gemini_api_key: API key for processing and querying
            auto_process: Process PDFs without existing .spdf
            save_processed: Save .spdf files after processing
            include_previews: Include previews in processed files
            metadata_map: Dict mapping PDF filename to metadata:
                {"paper.pdf": {"citation_key": "...", "authors": [...], ...}}

        Returns:
            CitationIndex with all sources loaded
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")

        index = cls(gemini_api_key=gemini_api_key)
        index._bibliography_folder = folder

        # Find all spdf files
        spdf_files = []
        for ext in SUPPORTED_EXTENSIONS:
            spdf_files.extend(folder.glob(f"*{ext}"))

        spdf_keys = set()
        for path in spdf_files:
            # Get citation key from the file
            stem = path.stem
            spdf_keys.add(stem)

        # Load existing spdf files
        logger.info(f"Found {len(spdf_files)} .spdf files in {folder}")
        for path in spdf_files:
            try:
                processed = ProcessedPDF.load(path)
                index.add(processed)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")

        # Find PDFs without corresponding spdf
        if auto_process and gemini_api_key:
            pdf_files = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))

            pdfs_to_process = []
            for pdf_path in pdf_files:
                stem = pdf_path.stem
                # Check if any spdf exists for this PDF
                has_spdf = any(
                    (folder / f"{stem}{ext}").exists()
                    for ext in SUPPORTED_EXTENSIONS
                )
                if not has_spdf:
                    pdfs_to_process.append(pdf_path)

            if pdfs_to_process:
                logger.info(f"Auto-processing {len(pdfs_to_process)} PDF files...")

                for pdf_path in pdfs_to_process:
                    try:
                        # Get metadata from map or generate defaults
                        meta = (metadata_map or {}).get(pdf_path.name, {})
                        citation_key = meta.get("citation_key", pdf_path.stem)
                        authors = meta.get("authors", [])
                        year = meta.get("year", 0)
                        title = meta.get("title", pdf_path.stem)

                        logger.info(f"Processing: {pdf_path.name}")

                        processed = ProcessedPDF.from_pdf(
                            pdf_path=pdf_path,
                            citation_key=citation_key,
                            authors=authors,
                            year=year,
                            title=title,
                            gemini_api_key=gemini_api_key,
                            include_previews=include_previews,
                        )

                        # Save if requested
                        if save_processed:
                            spdf_path = pdf_path.with_suffix(".spdf")
                            processed.save(spdf_path)
                            logger.info(f"Saved: {spdf_path.name}")

                        # Mark as auto-processed
                        index.add(processed)
                        if citation_key in index._sources:
                            index._sources[citation_key].source_type = "auto-processed"

                    except Exception as e:
                        logger.error(f"Failed to process {pdf_path}: {e}")

        # Build the index
        index._build_index()

        return index

    def _build_index(self) -> None:
        """Build the embedding matrix for fast search."""
        if not self._sources:
            self._embedding_matrix = None
            return

        # Collect all embeddings
        embeddings = []
        for source in self._sources.values():
            embeddings.extend(source.processed_pdf.embeddings)

        if embeddings:
            self._embedding_matrix = np.vstack(embeddings)
            logger.info(f"Built index: {self._embedding_matrix.shape[0]} chunks, {self._embedding_matrix.shape[1]} dims")
        else:
            self._embedding_matrix = None

        self._needs_rebuild = False

    def _ensure_index(self) -> None:
        """Ensure index is built."""
        if self._needs_rebuild or self._embedding_matrix is None:
            self._build_index()

    def _get_embedder(self):
        """Get or create embedder."""
        if self._embedder is None:
            if not self._gemini_api_key:
                raise ValueError("No API key provided for embedding queries")
            from .page_aware_rag import GeminiEmbedder
            self._embedder = GeminiEmbedder(self._gemini_api_key)
        return self._embedder

    def query(
        self,
        text: str,
        n_results: int = 20,
        min_similarity: float = 0.3,
        filter_sources: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """Query the index for relevant chunks.

        Args:
            text: Query text
            n_results: Maximum results
            min_similarity: Minimum cosine similarity threshold
            filter_sources: Optional list of citation_keys to filter by

        Returns:
            List of RetrievedChunk objects
        """
        self._ensure_index()

        if self._embedding_matrix is None or len(self._embedding_matrix) == 0:
            return []

        # Embed query
        embedder = self._get_embedder()
        query_embedding = np.array(embedder.embed(text), dtype=np.float32)

        # Compute cosine similarities
        # Normalize query
        query_norm_val = np.linalg.norm(query_embedding)
        if query_norm_val < 1e-10:
            return []
        query_norm = query_embedding / query_norm_val

        # Normalize matrix (row-wise)
        norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms < 1e-10, 1.0, norms)
        matrix_norm = self._embedding_matrix / norms

        # Dot product for cosine similarity
        similarities = np.dot(matrix_norm, query_norm)
        # Handle any NaN values
        similarities = np.nan_to_num(similarities, nan=0.0)

        # Get top indices
        top_indices = np.argsort(similarities)[::-1]

        # Build results
        chunks = []
        for idx in top_indices:
            if len(chunks) >= n_results:
                break

            similarity = float(similarities[idx])
            if similarity < min_similarity:
                continue

            citation_key, chunk_id = self._chunk_lookup[idx]

            # Apply source filter
            if filter_sources and citation_key not in filter_sources:
                continue

            source = self._sources[citation_key]
            chunk_data = source.processed_pdf.chunks[chunk_id]

            chunks.append(RetrievedChunk(
                chunk_id=f"{citation_key}_c{chunk_id}",
                citation_key=citation_key,
                text=chunk_data.text,
                book_page=chunk_data.book_page,
                pdf_page=chunk_data.pdf_page,
                similarity=similarity,
                authors=", ".join(source.processed_pdf.metadata.authors) if source.processed_pdf.metadata.authors else "",
                year=source.processed_pdf.metadata.year,
                title=source.processed_pdf.metadata.title,
            ))

        return chunks

    def cite(
        self,
        document_text: str,
        style: CitationStyle = CitationStyle.APA7,
        max_citations_per_claim: int = 3,
        batch_size: int = 4,
        exporter: Optional[Any] = None,
    ) -> List[Any]:
        """Generate citations for a document.

        Args:
            document_text: Text to add citations to
            style: Citation style (APA7 or CHICAGO17)
            max_citations_per_claim: Maximum citations per claim
            batch_size: Paragraphs to process in parallel
            exporter: Optional CitationExporter for tracking

        Returns:
            List of GroundedCitation objects
        """
        if not self._gemini_api_key:
            raise ValueError("No API key provided for citation generation")

        self._ensure_index()

        from .citation_engine import GeminiCitationEngine

        engine = GeminiCitationEngine(api_key=self._gemini_api_key)

        # Create a RAG-like interface for the engine
        index_rag = _CitationIndexRAGAdapter(self)

        citations = engine.analyze_with_grounded_rag(
            document_text=document_text,
            rag=index_rag,
            style=style,
            max_citations_per_claim=max_citations_per_claim,
            batch_size=batch_size,
            exporter=exporter,
        )

        return citations

    def sources(self) -> List[Dict[str, Any]]:
        """List all sources in the index.

        Returns:
            List of source info dicts
        """
        result = []
        for key, source in self._sources.items():
            meta = source.processed_pdf.metadata
            result.append({
                "citation_key": key,
                "title": meta.title if meta else "",
                "authors": meta.authors if meta else [],
                "year": meta.year if meta else 0,
                "chunks": source.chunk_count,
                "pages": meta.total_pages if meta else 0,
                "source_type": source.source_type,
            })
        return result

    def get_source(self, citation_key: str) -> Optional[ProcessedPDF]:
        """Get a specific source by citation key.

        Args:
            citation_key: Citation key to look up

        Returns:
            ProcessedPDF or None
        """
        source = self._sources.get(citation_key)
        return source.processed_pdf if source else None

    def remove(self, citation_key: str) -> bool:
        """Remove a source from the index.

        Args:
            citation_key: Citation key to remove

        Returns:
            True if removed, False if not found
        """
        if citation_key not in self._sources:
            return False

        del self._sources[citation_key]

        # Rebuild chunk lookup
        self._chunk_lookup = []
        for source in self._sources.values():
            source.chunk_offset = len(self._chunk_lookup)
            for chunk in source.processed_pdf.chunks:
                self._chunk_lookup.append((source.citation_key, chunk.id))

        self._needs_rebuild = True
        logger.info(f"Removed source: {citation_key}")
        return True

    def refresh(self) -> int:
        """Re-scan bibliography folder for new files.

        Returns:
            Number of new files processed
        """
        if not self._bibliography_folder:
            raise ValueError("No bibliography folder set. Use from_bibliography() first.")

        if not self._gemini_api_key:
            raise ValueError("No API key available for processing new files")

        # Find new PDFs
        existing_keys = set(self._sources.keys())
        folder = self._bibliography_folder

        new_count = 0

        # Check for new spdf files
        for ext in SUPPORTED_EXTENSIONS:
            for path in folder.glob(f"*{ext}"):
                stem = path.stem
                if stem not in existing_keys:
                    try:
                        processed = ProcessedPDF.load(path)
                        self.add(processed)
                        new_count += 1
                    except Exception as e:
                        logger.error(f"Failed to load {path}: {e}")

        # Check for new PDFs without spdf
        for pdf_path in folder.glob("*.pdf"):
            stem = pdf_path.stem
            if stem not in existing_keys and stem not in self._sources:
                has_spdf = any(
                    (folder / f"{stem}{ext}").exists()
                    for ext in SUPPORTED_EXTENSIONS
                )
                if not has_spdf:
                    try:
                        processed = ProcessedPDF.from_pdf(
                            pdf_path=pdf_path,
                            citation_key=stem,
                            authors=[],
                            year=0,
                            title=stem,
                            gemini_api_key=self._gemini_api_key,
                        )
                        processed.save(pdf_path.with_suffix(".spdf"))
                        self.add(processed)
                        self._sources[stem].source_type = "auto-processed"
                        new_count += 1
                    except Exception as e:
                        logger.error(f"Failed to process {pdf_path}: {e}")

        if new_count > 0:
            self._build_index()

        logger.info(f"Refresh complete: {new_count} new sources")
        return new_count

    def save(self, path: Union[str, Path]) -> str:
        """Save index state for faster reload.

        Note: This saves the index metadata, not the full data.
        The .spdf files must still exist for full reload.

        Args:
            path: Output path (will add .idx extension)

        Returns:
            Path to saved file
        """
        import json

        path = Path(path)
        if path.suffix != ".idx":
            path = path.with_suffix(".idx")

        data = {
            "version": 1,
            "bibliography_folder": str(self._bibliography_folder) if self._bibliography_folder else None,
            "sources": [
                {
                    "citation_key": s.citation_key,
                    "source_type": s.source_type,
                    "chunk_count": s.chunk_count,
                    "source_path": s.processed_pdf._source_path,
                }
                for s in self._sources.values()
            ],
        }

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"Saved index: {path}")
        return str(path)

    @classmethod
    def load_index(cls, path: Union[str, Path], gemini_api_key: Optional[str] = None) -> "CitationIndex":
        """Load index state and reload sources.

        Args:
            path: Path to .idx file
            gemini_api_key: API key for querying

        Returns:
            CitationIndex with sources reloaded
        """
        import json

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))

        index = cls(gemini_api_key=gemini_api_key)

        if data.get("bibliography_folder"):
            index._bibliography_folder = Path(data["bibliography_folder"])

        # Reload sources
        for source_info in data.get("sources", []):
            source_path = source_info.get("source_path")
            if source_path and Path(source_path).exists():
                try:
                    processed = ProcessedPDF.load(source_path)
                    index.add(processed)
                    index._sources[processed.metadata.citation_key].source_type = source_info.get("source_type", "spdf")
                except Exception as e:
                    logger.error(f"Failed to reload {source_path}: {e}")

        index._build_index()
        return index

    @property
    def total_chunks(self) -> int:
        """Total number of chunks across all sources."""
        return len(self._chunk_lookup)

    @property
    def total_sources(self) -> int:
        """Number of sources in the index."""
        return len(self._sources)

    def __len__(self) -> int:
        return len(self._sources)

    def __contains__(self, citation_key: str) -> bool:
        return citation_key in self._sources

    def __repr__(self) -> str:
        return f"CitationIndex({len(self._sources)} sources, {len(self._chunk_lookup)} chunks)"


class _CitationIndexRAGAdapter:
    """Adapter to make CitationIndex compatible with GeminiCitationEngine.

    Provides the same interface as PageAwareRAG.query_for_paragraph().
    """

    def __init__(self, index: CitationIndex):
        self._index = index

    def query_for_paragraph(self, paragraph: str, n_results: int = 30) -> List[RetrievedChunk]:
        """Query method matching PageAwareRAG interface."""
        return self._index.query(paragraph, n_results=n_results, min_similarity=0.25)

    def query(self, text: str, n_results: int = 20) -> List[RetrievedChunk]:
        """Generic query method."""
        return self._index.query(text, n_results=n_results)
