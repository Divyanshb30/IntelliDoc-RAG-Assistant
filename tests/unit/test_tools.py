"""Unit tests for the tool registry adapters."""

from __future__ import annotations

import pytest

from intellicode.config import Settings
from intellicode.rag import RAGPipeline
from intellicode.tools import (
    CodeAnalyzerTool,
    CodeExecutorTool,
    CodeFixerTool,
    DocumentSearchTool,
    SecurityScannerTool,
    SummarizerTool,
    TestGeneratorTool,
    get_tools,
)


@pytest.fixture(scope="module")
def pipeline(documents, document_names) -> RAGPipeline:
    p = RAGPipeline(Settings(use_reranker=False))
    p.build_index(documents, document_names)
    return p


def test_get_tools_returns_all_seven(pipeline):
    tools = get_tools(pipeline)
    assert set(tools) == {
        "document_search",
        "summarizer",
        "code_analyzer",
        "security_scanner",
        "test_generator",
        "code_executor",
        "code_fixer",
    }


def test_document_search_tool(pipeline):
    out = DocumentSearchTool(pipeline).execute("What products does TechCorp offer?", top_k=3)
    assert out["success"]
    assert out["chunks"]
    assert "context" in out


def test_summarizer_tool(pipeline):
    out = SummarizerTool(pipeline).execute("company overview")
    assert out["success"]
    assert out["highlights"]


def test_document_search_on_unbuilt_pipeline():
    empty = RAGPipeline(Settings(use_reranker=False))
    out = DocumentSearchTool(empty).execute("query")
    assert not out["success"]
    assert "error" in out


def test_code_analyzer_tool(buggy_code_path):
    out = CodeAnalyzerTool().execute(str(buggy_code_path))
    assert out["success"]
    assert out["issues"]


def test_security_scanner_tool():
    out = SecurityScannerTool().execute("password = 'secret123'")
    assert out["success"]
    assert out["total_issues"] >= 1


def test_test_generator_tool(clean_code_path):
    out = TestGeneratorTool().execute(str(clean_code_path))
    assert out["success"]
    assert out["functions_found"] >= 1


def test_code_executor_tool():
    out = CodeExecutorTool().execute("print('hi')")
    assert out["success"]
    assert "hi" in out["output"]


def test_code_fixer_tool(buggy_code_path):
    out = CodeFixerTool().execute(str(buggy_code_path))
    assert out["success"]
    assert out["fixes_suggested"] >= 1
