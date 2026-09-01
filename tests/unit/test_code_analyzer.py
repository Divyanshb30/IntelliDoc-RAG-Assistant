"""Unit tests for the AST code analyzer."""

from __future__ import annotations

import pytest

from intellicode.analysis import CodeAnalyzer


@pytest.fixture
def analyzer() -> CodeAnalyzer:
    return CodeAnalyzer()


# ── Individual anti-pattern detection ────────────────────────────────────────


@pytest.mark.parametrize(
    ("code", "expected_type"),
    [
        ("def f(x=[]):\n    return x", "Mutable default argument"),
        ("def f(x={}):\n    return x", "Mutable default argument"),
        ("def f(x=set()):\n    return x", "Mutable default argument"),
        ("try:\n    pass\nexcept:\n    pass", "Bare except clause"),
        ("try:\n    pass\nexcept Exception:\n    pass", "Broad exception handler"),
        ("from os import *", "Star import"),
        ("def f():\n    global x\n    x = 1", "Global statement"),
        ("def f(x):\n    assert x > 0\n    return x", "Assert in production code"),
    ],
)
def test_detects_pattern(analyzer, code, expected_type):
    """Each anti-pattern is flagged with the correct issue type."""
    result = analyzer.analyze_code(code)
    assert result.success
    assert expected_type in {issue.type for issue in result.issues}


def test_detects_unused_variable(analyzer):
    result = analyzer.analyze_code("def f():\n    unused = 5\n    return 10")
    assert "Unused variable" in {i.type for i in result.issues}


def test_underscore_variable_not_flagged_as_unused(analyzer):
    """Names prefixed with underscore are intentionally-unused and ignored."""
    result = analyzer.analyze_code("def f():\n    _ignored = 5\n    return 10")
    assert "Unused variable" not in {i.type for i in result.issues}


# ── Async support (the original bug) ─────────────────────────────────────────


def test_async_function_mutable_default(analyzer):
    """AsyncFunctionDef must be analyzed, not skipped (regression guard)."""
    result = analyzer.analyze_code("async def fetch(cache=[]):\n    return cache")
    assert "Mutable default argument" in {i.type for i in result.issues}


def test_async_function_bare_except(analyzer):
    code = "async def run():\n    try:\n        await x()\n    except:\n        pass"
    result = analyzer.analyze_code(code)
    assert "Bare except clause" in {i.type for i in result.issues}


# ── Complexity & nesting ─────────────────────────────────────────────────────


def test_deep_nesting_detected(analyzer):
    code = (
        "def f(data):\n"
        "    for a in data:\n"
        "        if a:\n"
        "            while a:\n"
        "                if a > 1:\n"
        "                    a -= 1"
    )
    assert "Deep nesting" in {i.type for i in analyzer.analyze_code(code).issues}


def test_high_complexity_detected():
    analyzer = CodeAnalyzer(complexity_threshold=3)
    code = "def f(a, b, c):\n    if a:\n        pass\n    if b:\n        pass\n    if c:\n        pass"
    assert "High cyclomatic complexity" in {i.type for i in analyzer.analyze_code(code).issues}


# ── Metrics ──────────────────────────────────────────────────────────────────


def test_metrics_counts(analyzer):
    code = "# a comment\nclass A:\n    def m(self):\n        return 1\n\ndef g():\n    return 2"
    result = analyzer.analyze_code(code)
    assert result.metrics["functions"] == 2
    assert result.metrics["classes"] == 1
    assert result.metrics["comment_lines"] == 1


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_syntax_error_returns_failure(analyzer):
    result = analyzer.analyze_code("def f(:\n    pass")
    assert not result.success
    assert "Syntax Error" in result.error


def test_empty_code_succeeds_with_no_issues(analyzer):
    result = analyzer.analyze_code("")
    assert result.success
    assert result.issues == []


def test_clean_code_has_no_high_severity(analyzer, clean_code_path):
    result = analyzer.analyze_file(str(clean_code_path))
    assert result.success
    assert all(i.severity != "HIGH" for i in result.issues)


def test_missing_file_returns_failure(analyzer):
    result = analyzer.analyze_file("does_not_exist_12345.py")
    assert not result.success
    assert "not found" in result.error.lower()
