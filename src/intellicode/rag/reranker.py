"""Cross-encoder reranker for second-stage relevance scoring.

Takes top-N candidates from the retriever and re-scores them with a
cross-encoder that sees (query, passage) jointly — much more accurate than
bi-encoder similarity but too slow to run over the full corpus.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from intellicode.config import Settings

if TYPE_CHECKING:
    from intellicode.rag.retriever import RetrievalResult

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker using ``sentence-transformers`` CrossEncoder.

    The model is lazily loaded on first call to :meth:`rerank` so that import
    time stays fast when reranking is disabled.

    Args:
        settings: Application configuration.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._model = None  # lazy load

    def _load_model(self) -> None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker model '%s' …", self._settings.reranker_model)
        self._model = CrossEncoder(self._settings.reranker_model)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Re-score *candidates* against *query* and return top-k by relevance.

        Args:
            query: The user query.
            candidates: Results from the retriever (first stage).
            top_k: How many to keep (defaults to ``settings.rerank_top_k``).

        Returns:
            Reranked list, highest-relevance first.
        """
        if not candidates:
            return []

        top_k = top_k or self._settings.rerank_top_k

        if self._model is None:
            self._load_model()
        assert self._model is not None

        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)

        scored = sorted(
            zip(candidates, scores, strict=True),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        results: list[RetrievalResult] = []
        for candidate, score in scored[:top_k]:
            from intellicode.rag.retriever import RetrievalResult as RR

            results.append(
                RR(
                    text=candidate.text,
                    score=float(score),
                    source_file=candidate.source_file,
                    chunk_index=candidate.chunk_index,
                )
            )

        logger.debug("Reranked %d → %d candidates", len(candidates), len(results))
        return results
