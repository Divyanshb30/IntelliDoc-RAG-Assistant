"""Unit tests for text chunking (pure text — no models)."""

from __future__ import annotations

import pytest

from intellicode.rag.chunking import (
    Chunk,
    chunk_document,
    chunk_text,
    chunk_text_word_split,
)

# ── Sentence-aware chunking ──────────────────────────────────────────────────


def test_returns_chunk_objects():
    chunks = chunk_text("Hello world. This is a test.", source_file="t.txt")
    assert chunks
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source_file == "t.txt" for c in chunks)


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_is_single_chunk():
    chunks = chunk_text("A short sentence.", chunk_size=256)
    assert len(chunks) == 1
    assert chunks[0].text == "A short sentence."


def test_long_text_splits_into_multiple_chunks():
    # ~1000 words → well over a 64-token chunk
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_size=64, chunk_overlap=16)
    assert len(chunks) > 1


@pytest.mark.parametrize("size", [32, 64, 128, 256])
def test_chunks_respect_approximate_size(size):
    text = ". ".join(f"This is sentence number {i} with some filler words" for i in range(200))
    chunks = chunk_text(text, chunk_size=size, chunk_overlap=size // 4)
    # Estimated tokens ≈ chars / 4; allow generous slack for overlap + merges.
    for c in chunks:
        assert len(c.text) / 4 <= size * 2.5


def test_chunk_indices_are_sequential():
    text = ". ".join(f"Sentence {i} here with words" for i in range(100))
    chunks = chunk_text(text, chunk_size=32)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_paragraph_boundaries_respected():
    text = "First paragraph content here.\n\nSecond paragraph content here."
    chunks = chunk_text(text, chunk_size=8, chunk_overlap=0)
    # Small chunk size forces the two paragraphs apart.
    assert len(chunks) >= 2


# ── Legacy word-split chunking ───────────────────────────────────────────────


def test_word_split_basic():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text_word_split(text, chunk_size=30, chunk_overlap=5)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_word_split_overlap():
    text = " ".join(str(i) for i in range(50))
    chunks = chunk_text_word_split(text, chunk_size=20, chunk_overlap=10)
    # With step 10 over 50 words we expect several overlapping windows.
    assert len(chunks) >= 4


# ── Strategy dispatcher ──────────────────────────────────────────────────────


def test_dispatch_sentence_strategy():
    chunks = chunk_document("Hello. World.", strategy="sentence", source_file="f")
    assert all(isinstance(c, Chunk) for c in chunks)


def test_dispatch_word_strategy_wraps_in_chunks():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_document(text, strategy="word", chunk_size=30, chunk_overlap=5, source_file="f")
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source_file == "f" for c in chunks)
