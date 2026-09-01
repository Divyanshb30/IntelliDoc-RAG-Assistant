"""Tool registry wiring analysis + retrieval components behind a common API.

Each tool exposes an ``execute`` method returning a plain ``dict`` so the
:class:`~intellicode.agent.RAGAgent` and the Gradio UI can consume results
uniformly.
"""

from __future__ import annotations

import logging
from typing import Any

from intellicode.analysis import (
    CodeAnalyzer,
    CodeExecutor,
    CodeFixer,
    SecurityScanner,
    TestGenerator,
)
from intellicode.rag.pipeline import RAGPipeline
from intellicode.rag.retriever import IndexNotBuiltError

logger = logging.getLogger(__name__)


class DocumentSearchTool:
    """RAG document search backed by a :class:`RAGPipeline`."""

    name = "document_search"

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline

    def execute(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Retrieve relevant chunks and assemble a context string.

        Args:
            query: The search query.
            top_k: Number of chunks to return.

        Returns:
            Dict with ``success``, ``chunks``, ``context``, and ``query``.
        """
        try:
            results = self._pipeline.query(query, top_k=top_k)
        except IndexNotBuiltError as exc:
            return {"success": False, "error": str(exc)}

        chunks = [
            {"chunk_id": i, "text": r.text, "score": r.score}
            for i, r in enumerate(results)
        ]
        context = "\n".join(f"[Chunk {i + 1}]: {r.text}" for i, r in enumerate(results))
        return {"success": True, "chunks": chunks, "context": context, "query": query}


class SummarizerTool:
    """Summarise retrieved content for a query."""

    name = "summarizer"

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline

    def execute(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Retrieve the top chunks and expose highlights for summarisation.

        Args:
            query: The topic to summarise.
            top_k: Number of chunks to pull.

        Returns:
            Dict with ``success``, ``highlights``, ``full_text``, ``num_chunks``.
        """
        try:
            results = self._pipeline.query(query, top_k=top_k)
        except IndexNotBuiltError as exc:
            return {"success": False, "error": str(exc)}

        if not results:
            return {"success": False, "error": "No documents to summarise"}

        highlights = [r.text[:100] for r in results[:3]]
        full_text = " ".join(r.text for r in results)[:2000]
        return {
            "success": True,
            "highlights": highlights,
            "full_text": full_text,
            "num_chunks": len(results),
        }


class CodeAnalyzerTool:
    """Adapter exposing :class:`CodeAnalyzer` with a dict interface."""

    name = "code_analyzer"

    def __init__(self, analyzer: CodeAnalyzer | None = None) -> None:
        self._analyzer = analyzer or CodeAnalyzer()

    def execute(self, code_file_path: str) -> dict[str, Any]:
        """Analyze a file and return the result as a dict."""
        return self._analyzer.analyze_file(code_file_path).to_dict()


class SecurityScannerTool:
    """Adapter exposing :class:`SecurityScanner` with a dict interface."""

    name = "security_scanner"

    def __init__(self, scanner: SecurityScanner | None = None) -> None:
        self._scanner = scanner or SecurityScanner()

    def execute(self, code: str) -> dict[str, Any]:
        """Scan a code string and return the result as a dict."""
        return self._scanner.scan(code).to_dict()


class TestGeneratorTool:
    """Adapter exposing :class:`TestGenerator` with a dict interface."""

    name = "test_generator"

    def __init__(self, generator: TestGenerator | None = None) -> None:
        self._generator = generator or TestGenerator()

    def execute(self, code_file_path: str) -> dict[str, Any]:
        """Generate tests for a file and return the result as a dict."""
        return self._generator.generate_from_file(code_file_path).to_dict()


class CodeExecutorTool:
    """Adapter exposing :class:`CodeExecutor` with a dict interface."""

    name = "code_executor"

    def __init__(self, executor: CodeExecutor | None = None) -> None:
        self._executor = executor or CodeExecutor()

    def execute(self, code: str, timeout: int | None = None) -> dict[str, Any]:
        """Run code and return the result as a dict."""
        return self._executor.execute(code, timeout=timeout).to_dict()


class CodeFixerTool:
    """Adapter exposing :class:`CodeFixer` with a dict interface."""

    name = "code_fixer"

    def __init__(self, fixer: CodeFixer | None = None) -> None:
        self._fixer = fixer or CodeFixer()

    def execute(self, code_file_path: str) -> dict[str, Any]:
        """Suggest fixes for a file and return the result as a dict."""
        return self._fixer.fix_file(code_file_path).to_dict()


def get_tools(pipeline: RAGPipeline) -> dict[str, Any]:
    """Instantiate and return the full tool registry.

    Args:
        pipeline: A :class:`RAGPipeline` for the retrieval-backed tools.

    Returns:
        Mapping of tool name → tool instance.
    """
    return {
        DocumentSearchTool.name: DocumentSearchTool(pipeline),
        SummarizerTool.name: SummarizerTool(pipeline),
        CodeAnalyzerTool.name: CodeAnalyzerTool(),
        SecurityScannerTool.name: SecurityScannerTool(),
        TestGeneratorTool.name: TestGeneratorTool(),
        CodeExecutorTool.name: CodeExecutorTool(),
        CodeFixerTool.name: CodeFixerTool(),
    }
