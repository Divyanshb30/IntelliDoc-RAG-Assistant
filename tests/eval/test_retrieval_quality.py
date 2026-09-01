"""Retrieval-quality gates.

These tests fail if retrieval metrics regress below the pinned baselines in
``eval/baselines.json``.  They run on CPU with embeddings + FAISS only — no GPU
and no API keys required.
"""

from __future__ import annotations

import pytest

from intellicode.config import Settings
from intellicode.evaluation import evaluate_retrieval
from intellicode.rag import RAGPipeline

pytestmark = pytest.mark.eval


def test_hybrid_meets_baseline(hybrid_pipeline, eval_queries, baselines):
    """Hybrid retrieval (no rerank) must meet the pinned floors."""
    metrics = evaluate_retrieval(hybrid_pipeline, eval_queries, use_reranker=False)
    floor = baselines["retrieval_hybrid"]

    assert metrics.mrr_at_5 >= floor["mrr_at_5"], (
        f"MRR@5 {metrics.mrr_at_5:.3f} < baseline {floor['mrr_at_5']}"
    )
    assert metrics.recall_at_3 >= floor["recall_at_3"], (
        f"Recall@3 {metrics.recall_at_3:.3f} < baseline {floor['recall_at_3']}"
    )
    assert metrics.ndcg_at_5 >= floor["ndcg_at_5"], (
        f"NDCG@5 {metrics.ndcg_at_5:.3f} < baseline {floor['ndcg_at_5']}"
    )


def test_hybrid_beats_dense(eval_corpus, eval_queries):
    """Hybrid search should not underperform dense-only on MRR@5."""
    docs, names = eval_corpus

    dense = RAGPipeline(Settings(use_reranker=False, use_hybrid_search=False))
    dense.build_index(docs, names)
    dense_metrics = evaluate_retrieval(dense, eval_queries, use_reranker=False)

    hybrid = RAGPipeline(Settings(use_reranker=False, use_hybrid_search=True))
    hybrid.build_index(docs, names)
    hybrid_metrics = evaluate_retrieval(hybrid, eval_queries, use_reranker=False)

    assert hybrid_metrics.mrr_at_5 >= dense_metrics.mrr_at_5


@pytest.mark.slow
def test_reranking_meets_baseline(eval_corpus, eval_queries, baselines):
    """Cross-encoder reranking must reach the higher reranked baseline.

    Downloads a small cross-encoder on first run.
    """
    docs, names = eval_corpus
    pipeline = RAGPipeline(Settings(use_reranker=True, use_hybrid_search=True))
    pipeline.build_index(docs, names)
    metrics = evaluate_retrieval(pipeline, eval_queries, use_reranker=True)
    floor = baselines["retrieval_reranked"]

    assert metrics.mrr_at_5 >= floor["mrr_at_5"], (
        f"Reranked MRR@5 {metrics.mrr_at_5:.3f} < baseline {floor['mrr_at_5']}"
    )
    assert metrics.recall_at_1 >= floor["recall_at_1"], (
        f"Reranked Recall@1 {metrics.recall_at_1:.3f} < baseline {floor['recall_at_1']}"
    )


def test_negative_queries_are_rejected(hybrid_pipeline, eval_queries):
    """Out-of-corpus queries should mostly fall below the confidence threshold."""
    metrics = evaluate_retrieval(hybrid_pipeline, eval_queries, use_reranker=False)
    assert metrics.n_negative > 0
    assert metrics.negative_rejection_rate >= 0.5
