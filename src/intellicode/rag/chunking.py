"""Sentence-aware recursive text chunking.

Splits documents on natural boundaries (paragraphs → sentences → words) so
each chunk is semantically coherent and fits within the embedding model's
optimal token window.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Lightweight token estimator — ~1 token per 4 chars for English text.
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Chunk:
    """A document chunk with provenance metadata."""

    text: str
    source_file: str = ""
    chunk_index: int = 0
    char_start: int = 0
    char_end: int = 0


def _estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


# ── Splitters (ordered coarse → fine) ────────────────────────────────────────

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _split_words(text: str) -> list[str]:
    return text.split()


# ── Public API ───────────────────────────────────────────────────────────────


def chunk_text(
    text: str,
    *,
    chunk_size: int = 256,
    chunk_overlap: int = 64,
    source_file: str = "",
) -> list[Chunk]:
    """Split *text* into overlapping chunks of ≤ *chunk_size* estimated tokens.

    Strategy (recursive, coarse-to-fine):
      1. Split on paragraph boundaries (``\\n\\n``).
      2. If a paragraph exceeds *chunk_size*, split on sentence boundaries.
      3. If a sentence still exceeds, fall back to word-level splitting.
      4. Merge consecutive small segments until they approach *chunk_size*.
      5. Create overlap by repeating the last *chunk_overlap* tokens of each
         chunk at the start of the next.

    Args:
        text: Raw document text.
        chunk_size: Target chunk size in estimated tokens.
        chunk_overlap: Overlap between consecutive chunks in estimated tokens.
        source_file: Filename to record in each :class:`Chunk`.

    Returns:
        Ordered list of :class:`Chunk` objects.
    """
    if not text or not text.strip():
        return []

    segments = _recursive_split(text, chunk_size)
    merged = _merge_segments(segments, chunk_size)
    chunks = _add_overlap(merged, chunk_overlap, chunk_size)

    result: list[Chunk] = []
    char_cursor = 0
    for idx, chunk_text_str in enumerate(chunks):
        start = text.find(chunk_text_str[:80], max(0, char_cursor - 50))
        if start == -1:
            start = char_cursor
        end = start + len(chunk_text_str)
        result.append(
            Chunk(
                text=chunk_text_str,
                source_file=source_file,
                chunk_index=idx,
                char_start=start,
                char_end=end,
            )
        )
        char_cursor = start + 1  # allow overlapping finds

    logger.debug("Chunked %s into %d chunks (target %d tokens)", source_file, len(result), chunk_size)
    return result


# ── Internals ────────────────────────────────────────────────────────────────


def _recursive_split(text: str, max_tokens: int) -> list[str]:
    """Recursively split text until every segment fits in *max_tokens*."""
    if _estimate_tokens(text) <= max_tokens:
        return [text]

    # Try paragraph split first
    parts = _split_paragraphs(text)
    if len(parts) > 1:
        result: list[str] = []
        for part in parts:
            result.extend(_recursive_split(part, max_tokens))
        return result

    # Try sentence split
    parts = _split_sentences(text)
    if len(parts) > 1:
        result = []
        for part in parts:
            result.extend(_recursive_split(part, max_tokens))
        return result

    # Fall back to word split
    words = _split_words(text)
    step = max(1, max_tokens * _CHARS_PER_TOKEN // (sum(len(w) for w in words) // len(words) + 1))
    result = []
    for i in range(0, len(words), step):
        segment = " ".join(words[i : i + step])
        if segment.strip():
            result.append(segment)
    return result


def _merge_segments(segments: list[str], max_tokens: int) -> list[str]:
    """Merge consecutive small segments until they approach *max_tokens*."""
    if not segments:
        return []

    merged: list[str] = []
    current = segments[0]

    for segment in segments[1:]:
        combined = current + " " + segment
        if _estimate_tokens(combined) <= max_tokens:
            current = combined
        else:
            merged.append(current.strip())
            current = segment

    if current.strip():
        merged.append(current.strip())

    return merged


def _add_overlap(chunks: list[str], overlap_tokens: int, max_tokens: int) -> list[str]:
    """Prepend the tail of each chunk to the start of the next one."""
    if len(chunks) <= 1 or overlap_tokens <= 0:
        return chunks

    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN
    result = [chunks[0]]

    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        tail = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
        # Find a clean word boundary in the tail
        space_idx = tail.find(" ")
        if space_idx > 0:
            tail = tail[space_idx + 1 :]
        combined = tail + " " + chunks[i]
        # Trim if overlap pushed it over limit
        if _estimate_tokens(combined) > max_tokens * 1.2:
            combined = chunks[i]
        result.append(combined.strip())

    return result


# ── Legacy adapter ───────────────────────────────────────────────────────────


def chunk_text_word_split(
    text: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Original word-count sliding-window chunker (kept for benchmarking).

    Args:
        text: Raw text.
        chunk_size: Window size in words.
        chunk_overlap: Overlap in words.

    Returns:
        List of plain text chunks.
    """
    words = text.split()
    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ── Strategy dispatcher ──────────────────────────────────────────────────────


def chunk_document(
    text: str,
    *,
    strategy: str = "sentence",
    chunk_size: int = 256,
    chunk_overlap: int = 64,
    source_file: str = "",
) -> list[Chunk]:
    """Chunk *text* using the named *strategy*.

    Args:
        text: Raw document text.
        strategy: ``"sentence"`` (recursive sentence-aware) or ``"word"``
            (legacy word-count sliding window).
        chunk_size: Chunk size (tokens for "sentence", words for "word").
        chunk_overlap: Overlap in the same units as *chunk_size*.
        source_file: Filename recorded on each chunk.

    Returns:
        List of :class:`Chunk` objects.
    """
    if strategy == "word":
        raw = chunk_text_word_split(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return [
            Chunk(text=t, source_file=source_file, chunk_index=i)
            for i, t in enumerate(raw)
        ]
    return chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_file=source_file,
    )
