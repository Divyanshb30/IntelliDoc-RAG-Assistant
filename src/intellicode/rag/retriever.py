"""Hybrid dense + sparse retriever with Reciprocal Rank Fusion.

Dense:  FAISS IndexFlatIP on L2-normalised embeddings (≡ cosine similarity).
Sparse: BM25 via rank_bm25 for lexical keyword matching.
Fusion: RRF — ``score = Σ 1 / (k + rank_i)`` — merges both rankings without
        score normalisation.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from intellicode.config import Settings
from intellicode.rag.chunking import Chunk, chunk_document

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """A single retrieval hit."""

    text: str
    score: float
    source_file: str = ""
    chunk_index: int = 0


# ── Exceptions ───────────────────────────────────────────────────────────────


class IndexNotBuiltError(Exception):
    """Raised when querying an index that has not been built yet."""


# ── Retriever ────────────────────────────────────────────────────────────────


class HybridRetriever:
    """Hybrid dense + BM25 retriever with RRF fusion.

    Args:
        settings: Application configuration.  Defaults are used when *None*.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        logger.info("Loading embedding model '%s' …", self._settings.embedding_model)
        self._encoder = SentenceTransformer(self._settings.embedding_model)
        self._dimension: int = self._encoder.get_sentence_embedding_dimension()

        # State populated by build_index / load_index
        self._faiss_index: faiss.IndexFlatIP | None = None
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []

    # ── Index building ───────────────────────────────────────────────────

    def build_index(self, documents: Sequence[str], source_files: Sequence[str] | None = None) -> int:
        """Chunk, embed, and index a list of raw document strings.

        Args:
            documents: Raw text contents, one per document.
            source_files: Optional parallel list of filenames for metadata.

        Returns:
            Number of chunks indexed.
        """
        all_chunks: list[Chunk] = []
        for i, doc in enumerate(documents):
            src = source_files[i] if source_files else f"doc_{i}"
            chunks = chunk_document(
                doc,
                strategy=self._settings.chunking_strategy,
                chunk_size=self._settings.chunk_size,
                chunk_overlap=self._settings.chunk_overlap,
                source_file=src,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks produced — nothing to index.")
            return 0

        # Renumber chunk_index globally so it is unique across the whole index.
        # (chunk_text numbers per-document, which would collide during fusion.)
        self._chunks = [replace(c, chunk_index=i) for i, c in enumerate(all_chunks)]
        texts = [c.text for c in self._chunks]

        # Dense index (FAISS cosine via normalised IP)
        logger.info("Encoding %d chunks …", len(texts))
        embeddings = self._encoder.encode(texts, show_progress_bar=False)
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)
        self._faiss_index = faiss.IndexFlatIP(self._dimension)
        self._faiss_index.add(embeddings)

        # Sparse index (BM25)
        self._tokenized_corpus = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info("Index built — %d chunks, dim=%d", len(all_chunks), self._dimension)
        return len(all_chunks)

    # ── Retrieval ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        use_hybrid: bool | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for *query*.

        Args:
            query: Natural-language search query.
            top_k: Number of results to return (defaults to ``settings.retrieval_top_k``).
            use_hybrid: Override ``settings.use_hybrid_search`` for this call.

        Returns:
            Ranked list of :class:`RetrievalResult`.

        Raises:
            IndexNotBuiltError: If :meth:`build_index` has not been called.
        """
        if self._faiss_index is None or not self._chunks:
            raise IndexNotBuiltError(
                "No documents indexed. Call build_index() or load_index() first."
            )

        top_k = top_k or self._settings.retrieval_top_k
        hybrid = use_hybrid if use_hybrid is not None else self._settings.use_hybrid_search
        top_k = min(top_k, len(self._chunks))

        dense_results = self._dense_search(query, top_k)

        if hybrid and self._bm25 is not None:
            sparse_results = self._bm25_search(query, top_k)
            fused = self._rrf_fuse(dense_results, sparse_results, top_k)
            return fused

        return dense_results

    # ── Dense search ─────────────────────────────────────────────────────

    def _dense_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        q_emb = self._encode_query(query)
        scores, indices = self._faiss_index.search(q_emb, top_k)  # type: ignore[union-attr]

        results: list[RetrievalResult] = []
        for idx, score in zip(indices[0], scores[0], strict=True):
            if 0 <= idx < len(self._chunks):
                c = self._chunks[idx]
                results.append(
                    RetrievalResult(
                        text=c.text,
                        score=float(score),
                        source_file=c.source_file,
                        chunk_index=c.chunk_index,
                    )
                )
        return results

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode and L2-normalise a single query."""
        emb = self._encoder.encode([query]).astype("float32")
        faiss.normalize_L2(emb)
        return emb

    # ── Sparse search (BM25) ─────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        assert self._bm25 is not None
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            if scores[idx] > 0:
                c = self._chunks[idx]
                results.append(
                    RetrievalResult(
                        text=c.text,
                        score=float(scores[idx]),
                        source_file=c.source_file,
                        chunk_index=c.chunk_index,
                    )
                )
        return results

    # ── RRF fusion ───────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        dense: list[RetrievalResult],
        sparse: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Reciprocal Rank Fusion: ``score = Σ 1 / (k + rank)``."""
        k = self._settings.rrf_k
        scores: dict[int, float] = {}
        result_map: dict[int, RetrievalResult] = {}

        for rank, r in enumerate(dense):
            key = r.chunk_index
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            result_map[key] = r

        for rank, r in enumerate(sparse):
            key = r.chunk_index
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in result_map:
                result_map[key] = r

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        fused: list[RetrievalResult] = []
        for chunk_idx, fused_score in ranked:
            r = result_map[chunk_idx]
            fused.append(
                RetrievalResult(
                    text=r.text,
                    score=fused_score,
                    source_file=r.source_file,
                    chunk_index=r.chunk_index,
                )
            )
        return fused

    # ── Persistence ──────────────────────────────────────────────────────

    def save_index(self, directory: str | Path) -> None:
        """Persist FAISS index + chunk metadata to *directory*.

        Args:
            directory: Target directory (created if absent).
        """
        if self._faiss_index is None:
            raise IndexNotBuiltError("No index to save.")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._faiss_index, str(path / "index.faiss"))

        meta = {
            "chunks": [asdict(c) for c in self._chunks],
            "tokenized_corpus": self._tokenized_corpus,
            "dimension": self._dimension,
        }
        (path / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        logger.info("Index saved to %s", path)

    def load_index(self, directory: str | Path) -> int:
        """Load a previously saved index from *directory*.

        Args:
            directory: Directory containing ``index.faiss`` and ``metadata.json``.

        Returns:
            Number of chunks in the loaded index.
        """
        path = Path(directory)
        self._faiss_index = faiss.read_index(str(path / "index.faiss"))

        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        self._chunks = [Chunk(**c) for c in meta["chunks"]]
        self._tokenized_corpus = meta.get("tokenized_corpus", [])
        self._dimension = meta.get("dimension", self._dimension)

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info("Index loaded from %s — %d chunks", path, len(self._chunks))
        return len(self._chunks)

    # ── Utilities ────────────────────────────────────────────────────────

    @property
    def chunks(self) -> list[Chunk]:
        """All indexed chunks (read-only copy)."""
        return list(self._chunks)

    @property
    def is_built(self) -> bool:
        """Whether an index is available for querying."""
        return self._faiss_index is not None and len(self._chunks) > 0
