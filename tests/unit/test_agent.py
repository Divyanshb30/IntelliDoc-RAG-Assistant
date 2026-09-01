"""Unit tests for the intent-routing agent.

These use stub tools and no LLM, so they run instantly and deterministically.
"""

from __future__ import annotations

import pytest

from intellicode.agent import RAGAgent


class StubTool:
    """Records the last call and returns a preset output."""

    def __init__(self, output: dict) -> None:
        self.output = output
        self.calls: list = []

    def execute(self, *args, **kwargs) -> dict:
        self.calls.append((args, kwargs))
        return self.output


@pytest.fixture
def stub_tools() -> dict:
    return {
        name: StubTool({"success": True})
        for name in [
            "document_search",
            "summarizer",
            "code_analyzer",
            "security_scanner",
            "test_generator",
            "code_executor",
            "code_fixer",
        ]
    }


@pytest.fixture
def agent(stub_tools) -> RAGAgent:
    return RAGAgent(stub_tools, llm=None)


# ── Intent detection ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is the API endpoint?", "document_search"),
        ("Summarize this document", "summarizer"),
        ("Give me an overview", "summarizer"),
    ],
)
def test_intent_without_file_context(agent, query, expected):
    assert agent.detect_intent(query) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("run this code", "code_executor"),
        ("check for security vulnerabilities", "security_scanner"),
        ("fix the bugs", "code_fixer"),
        ("generate pytest tests", "test_generator"),
        ("analyze the code quality", "code_analyzer"),
        ("just look at this", "code_analyzer"),  # default for code files
    ],
)
def test_intent_with_code_context(agent, query, expected):
    ctx = {"type": "code", "path": "x.py"}
    assert agent.detect_intent(query, ctx) == expected


# ── Execution & routing ──────────────────────────────────────────────────────


def test_execute_routes_to_document_search(agent, stub_tools):
    stub_tools["document_search"].output = {
        "success": True,
        "chunks": [{"text": "TechCorp was founded in 2010. It is based in SF.", "score": 0.9}],
    }
    response = agent.execute("When was TechCorp founded?")
    assert response.success
    assert response.tool_used == "document_search"
    assert "2010" in response.answer


def test_execute_handles_missing_tool():
    agent = RAGAgent({}, llm=None)
    response = agent.execute("anything")
    assert not response.success
    assert response.tool_used == "error"


def test_execute_propagates_tool_failure(agent, stub_tools):
    stub_tools["document_search"].output = {"success": False, "error": "index empty"}
    response = agent.execute("query")
    assert not response.success
    assert "index empty" in response.answer


def test_analyzer_answer_summarizes_issues(agent, stub_tools):
    stub_tools["code_analyzer"].output = {
        "success": True,
        "severity": "HIGH",
        "issues": [{"type": "Mutable default argument", "severity": "HIGH", "line": 3}],
        "metrics": {"comment_ratio": 12.0},
    }
    response = agent.execute("analyze this", {"type": "code", "path": "x.py"})
    assert response.success
    assert "HIGH" in response.answer


# ── LLM integration ──────────────────────────────────────────────────────────


class FakeLLM:
    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        return "A generated factual answer about the topic."


def test_llm_used_for_document_answer(stub_tools):
    stub_tools["document_search"].output = {
        "success": True,
        "chunks": [{"text": "Some context here.", "score": 0.8}],
    }
    agent = RAGAgent(stub_tools, llm=FakeLLM())
    response = agent.execute("tell me about it")
    assert "generated factual answer" in response.answer


def test_llm_failure_falls_back_to_template(stub_tools):
    class BrokenLLM:
        def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
            raise RuntimeError("model crashed")

    stub_tools["document_search"].output = {
        "success": True,
        "chunks": [{"text": "TechCorp offers CloudSync Pro. It is encrypted.", "score": 0.8}],
    }
    agent = RAGAgent(stub_tools, llm=BrokenLLM())
    response = agent.execute("what is offered")
    # Falls back to first-sentences template rather than raising.
    assert response.success
    assert "CloudSync" in response.answer
