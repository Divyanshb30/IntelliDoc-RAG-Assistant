"""Shared fixtures for the evaluation suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "tests" / "fixtures" / "documents"
DATASET_PATH = ROOT / "tests" / "eval" / "datasets" / "retrieval_eval.json"
BASELINES_PATH = ROOT / "eval" / "baselines.json"
BUGGY_FIXTURE = ROOT / "tests" / "fixtures" / "sample_buggy_code.py"
CLEAN_FIXTURE = ROOT / "tests" / "fixtures" / "sample_clean_code.py"


@pytest.fixture(scope="session")
def baselines() -> dict:
    """Load the pinned quality-gate thresholds."""
    return json.loads(BASELINES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def eval_queries() -> list:
    """Load the labeled retrieval evaluation queries."""
    from intellicode.evaluation import load_dataset

    return load_dataset(DATASET_PATH)


@pytest.fixture(scope="session")
def eval_corpus() -> tuple[list[str], list[str]]:
    """Load the evaluation document corpus."""
    from intellicode.evaluation import load_corpus

    return load_corpus(CORPUS_DIR)


@pytest.fixture(scope="session")
def hybrid_pipeline(eval_corpus):
    """A pipeline with hybrid search, reranker off (fast, deterministic)."""
    from intellicode.config import Settings
    from intellicode.rag import RAGPipeline

    docs, names = eval_corpus
    pipeline = RAGPipeline(Settings(use_reranker=False, use_hybrid_search=True))
    pipeline.build_index(docs, names)
    return pipeline
