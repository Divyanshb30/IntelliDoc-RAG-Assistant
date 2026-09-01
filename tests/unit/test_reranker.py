"""Unit tests for the cross-encoder reranker.

Marked ``slow`` because the first run downloads a small cross-encoder model.
"""

from __future__ import annotations

import pytest

from intellicode.config import Settings
from intellicode.rag.reranker import Reranker
from intellicode.rag.retriever import RetrievalResult

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def reranker() -> Reranker:
    return Reranker(Settings())


def _candidates() -> list[RetrievalResult]:
    return [
        RetrievalResult(text="The capital of France is Paris.", score=0.5, chunk_index=0),
        RetrievalResult(text="Bananas are a good source of potassium.", score=0.6, chunk_index=1),
        RetrievalResult(text="Paris is the largest city in France.", score=0.4, chunk_index=2),
    ]


def test_rerank_promotes_relevant_result(reranker):
    ranked = reranker.rerank("What is the capital of France?", _candidates(), top_k=3)
    # The France/Paris chunks should outrank the banana chunk.
    assert ranked[0].chunk_index in {0, 2}
    assert ranked[-1].chunk_index == 1


def test_rerank_respects_top_k(reranker):
    ranked = reranker.rerank("France", _candidates(), top_k=1)
    assert len(ranked) == 1


def test_rerank_empty_candidates(reranker):
    assert reranker.rerank("anything", [], top_k=5) == []


def test_rerank_single_candidate(reranker):
    single = [RetrievalResult(text="Only option.", score=0.1, chunk_index=7)]
    ranked = reranker.rerank("query", single, top_k=5)
    assert len(ranked) == 1
    assert ranked[0].chunk_index == 7
