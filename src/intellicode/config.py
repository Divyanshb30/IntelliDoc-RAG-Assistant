"""Centralized configuration via Pydantic Settings with env-var overrides."""

from __future__ import annotations

import logging

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, overridable with INTELLICODE_* env vars.

    Example:
        ``INTELLICODE_CHUNK_SIZE=512 python app.py`` overrides *chunk_size*.
    """

    model_config = SettingsConfigDict(env_prefix="INTELLICODE_")

    # ── Embedding & retrieval ────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = 256
    chunk_overlap: int = 64
    chunking_strategy: str = "sentence"  # "sentence" | "word"

    retrieval_top_k: int = 20
    rerank_top_k: int = 5

    use_reranker: bool = True
    use_hybrid_search: bool = True
    rrf_k: int = 60

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    llm_max_new_tokens: int = 150
    llm_temperature: float = 0.3

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"


def configure_logging(settings: Settings | None = None) -> None:
    """Set up stdlib logging from *settings.log_level*."""
    level = (settings or Settings()).log_level.upper()
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        level=getattr(logging, level, logging.INFO),
    )
