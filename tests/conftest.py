"""Shared fixtures for the whole test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
DOCUMENTS = FIXTURES / "documents"


@pytest.fixture(scope="session")
def documents() -> list[str]:
    """The evaluation corpus as a list of raw document strings."""
    return [fp.read_text(encoding="utf-8") for fp in sorted(DOCUMENTS.glob("*.txt"))]


@pytest.fixture(scope="session")
def document_names() -> list[str]:
    """Filenames parallel to the ``documents`` fixture."""
    return [fp.name for fp in sorted(DOCUMENTS.glob("*.txt"))]


@pytest.fixture
def buggy_code_path() -> Path:
    """Path to the annotated buggy-code fixture."""
    return FIXTURES / "sample_buggy_code.py"


@pytest.fixture
def clean_code_path() -> Path:
    """Path to the clean-code fixture."""
    return FIXTURES / "sample_clean_code.py"
