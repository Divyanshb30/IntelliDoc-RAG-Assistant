"""End-to-end integration test for the RAG pipeline.

Ingests the real corpus, builds a hybrid index, and verifies that queries
return the answer-bearing chunks. Runs on CPU with real embeddings.
"""

from __future__ import annotations

import pytest

from intellicode.config import Settings
from intellicode.rag import RAGPipeline
from intellicode.rag.retriever import IndexNotBuiltError


@pytest.fixture(scope="module")
def pipeline(documents, document_names) -> RAGPipeline:
    p = RAGPipeline(Settings(use_reranker=False, use_hybrid_search=True))
    p.build_index(documents, document_names)
    return p


def test_index_builds_from_corpus(pipeline):
    assert pipeline.is_built


@pytest.mark.parametrize(
    ("query", "expected_span"),
    [
        ("What products does TechCorp offer?", "CloudSync Pro"),
        ("How are files encrypted at rest?", "AES-256"),
        ("What database is used?", "PostgreSQL"),
        ("What is the API rate limit?", "1000 requests per minute"),
        ("How are passwords hashed?", "bcrypt"),
    ],
)
def test_query_retrieves_answer_bearing_chunk(pipeline, query, expected_span):
    results = pipeline.query(query, top_k=3)
    assert results, "expected at least one result"
    joined = " ".join(r.text for r in results).lower()
    assert expected_span.lower() in joined


def test_directory_ingestion(tmp_path):
    (tmp_path / "a.txt").write_text("The sky is blue during the day.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Grass is green in the summer.", encoding="utf-8")

    p = RAGPipeline(Settings(use_reranker=False))
    count = p.build_index_from_directory(tmp_path)
    assert count >= 2

    results = p.query("What color is the sky?", top_k=1)
    assert "sky" in results[0].text.lower()


def test_missing_directory_raises():
    p = RAGPipeline(Settings(use_reranker=False))
    with pytest.raises(FileNotFoundError):
        p.build_index_from_directory("nonexistent_dir_98765")


def test_query_before_index_raises():
    p = RAGPipeline(Settings(use_reranker=False))
    with pytest.raises(IndexNotBuiltError):
        p.query("anything")
