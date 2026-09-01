"""Unit tests for the hybrid retriever.

Uses the real embedding model (loaded once per session) on a tiny corpus, so
these run without a GPU in a few seconds.
"""

from __future__ import annotations

import pytest

from intellicode.config import Settings
from intellicode.rag.retriever import HybridRetriever, IndexNotBuiltError

_DOCS = [
    "TechCorp offers CloudSync Pro for encrypted cloud storage and file sync.",
    "The engineering team uses Python and Go, with PostgreSQL as the database.",
    "All passwords are hashed with bcrypt and multi-factor authentication is required.",
]
_NAMES = ["products.txt", "engineering.txt", "security.txt"]


@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    r = HybridRetriever(Settings())
    r.build_index(_DOCS, _NAMES)
    return r


def test_index_reports_built(retriever):
    assert retriever.is_built
    assert len(retriever.chunks) == len(_DOCS)


def test_dense_retrieval_finds_relevant(retriever):
    results = retriever.retrieve("cloud storage", top_k=1, use_hybrid=False)
    assert results
    assert "CloudSync" in results[0].text


def test_bm25_keyword_match(retriever):
    results = retriever.retrieve("PostgreSQL", top_k=1, use_hybrid=True)
    assert "PostgreSQL" in results[0].text


def test_hybrid_returns_unique_chunks(retriever):
    results = retriever.retrieve("Python database bcrypt", top_k=3, use_hybrid=True)
    indices = [r.chunk_index for r in results]
    assert len(indices) == len(set(indices)), "fusion must not return duplicate chunks"


def test_top_k_is_respected(retriever):
    assert len(retriever.retrieve("anything", top_k=2)) <= 2


def test_query_on_empty_index_raises():
    empty = HybridRetriever(Settings())
    with pytest.raises(IndexNotBuiltError):
        empty.retrieve("query")


def test_save_and_load_roundtrip(tmp_path):
    original = HybridRetriever(Settings())
    original.build_index(_DOCS, _NAMES)
    original.save_index(tmp_path)

    loaded = HybridRetriever(Settings())
    n = loaded.load_index(tmp_path)
    assert n == len(_DOCS)

    r1 = original.retrieve("bcrypt passwords", top_k=1)
    r2 = loaded.retrieve("bcrypt passwords", top_k=1)
    assert r1[0].chunk_index == r2[0].chunk_index


def test_empty_documents_index_is_empty():
    r = HybridRetriever(Settings())
    assert r.build_index([], []) == 0
    assert not r.is_built
