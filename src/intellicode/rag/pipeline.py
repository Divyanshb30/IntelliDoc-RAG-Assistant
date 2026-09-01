"""High-level RAG pipeline orchestrating retrieval, reranking, and indexing.

This is the primary public interface for the retrieval subsystem.  Typical
usage::

    from intellicode.rag import RAGPipeline

    pipeline = RAGPipeline()
    pipeline.build_index_from_directory("data/")
    results = pipeline.query("What products does TechCorp offer?")
"""

from __future__ import annotations

import logging
from pathlib import Path

from intellicode.config import Settings
from intellicode.rag.reranker import Reranker
from intellicode.rag.retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end retrieval pipeline: ingest → index → query.

    Args:
        settings: Optional configuration override.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._retriever = HybridRetriever(self._settings)
        self._reranker: Reranker | None = (
            Reranker(self._settings) if self._settings.use_reranker else None
        )

    # ── Index management ─────────────────────────────────────────────────

    def build_index_from_directory(self, data_dir: str | Path) -> int:
        """Read all ``.txt`` files in *data_dir*, chunk, embed, and index them.

        Args:
            data_dir: Directory containing text files.

        Returns:
            Number of chunks indexed.

        Raises:
            FileNotFoundError: If *data_dir* does not exist.
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        text_files = sorted(data_path.glob("*.txt"))
        if not text_files:
            logger.warning("No .txt files found in %s", data_path)
            return 0

        documents: list[str] = []
        filenames: list[str] = []
        for fp in text_files:
            try:
                content = fp.read_text(encoding="utf-8")
                documents.append(content)
                filenames.append(fp.name)
                logger.info("Read %s (%d chars)", fp.name, len(content))
            except (OSError, UnicodeDecodeError) as exc:
                logger.error("Skipping %s: %s", fp.name, exc)

        count = self._retriever.build_index(documents, filenames)
        logger.info("Indexed %d chunks from %d files", count, len(documents))
        return count

    def build_index(self, documents: list[str], source_files: list[str] | None = None) -> int:
        """Build index from raw document strings.

        Args:
            documents: List of document texts.
            source_files: Optional parallel list of filenames.

        Returns:
            Number of chunks indexed.
        """
        return self._retriever.build_index(documents, source_files)

    def save_index(self, directory: str | Path) -> None:
        """Persist the current index to disk."""
        self._retriever.save_index(directory)

    def load_index(self, directory: str | Path) -> int:
        """Load a previously saved index.

        Returns:
            Number of chunks loaded.
        """
        return self._retriever.load_index(directory)

    # ── Query ────────────────────────────────────────────────────────────

    def query(
        self,
        query: str,
        top_k: int | None = None,
        *,
        use_reranker: bool | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve and optionally rerank chunks for *query*.

        Args:
            query: Natural-language search query.
            top_k: Final number of results (after reranking).
            use_reranker: Override the config's ``use_reranker`` setting.

        Returns:
            Ranked list of :class:`RetrievalResult`.

        Raises:
            IndexNotBuiltError: If no index is available.
        """
        rerank = use_reranker if use_reranker is not None else self._settings.use_reranker
        final_k = top_k or self._settings.rerank_top_k

        # Retrieve more candidates when reranking
        retrieve_k = self._settings.retrieval_top_k if rerank else final_k
        candidates = self._retriever.retrieve(query, top_k=retrieve_k)

        if rerank and self._reranker is not None and candidates:
            return self._reranker.rerank(query, candidates, top_k=final_k)

        return candidates[:final_k]

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def is_built(self) -> bool:
        """Whether the index has been built or loaded."""
        return self._retriever.is_built

    @property
    def retriever(self) -> HybridRetriever:
        """Direct access to the underlying retriever (for advanced use)."""
        return self._retriever
