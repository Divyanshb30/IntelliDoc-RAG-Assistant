"""Unit tests for the sandboxed code executor."""

from __future__ import annotations

import pytest

from intellicode.analysis import CodeExecutor


@pytest.fixture
def executor() -> CodeExecutor:
    return CodeExecutor(timeout=5)


def test_successful_execution_captures_stdout(executor):
    result = executor.execute("print('hello world')")
    assert result.success
    assert "hello world" in result.output
    assert result.exit_code == 0


def test_failed_execution_captures_error(executor):
    result = executor.execute("raise ValueError('boom')")
    assert not result.success
    assert result.exit_code != 0
    assert "ValueError" in result.error


def test_timeout_is_enforced():
    executor = CodeExecutor(timeout=1)
    result = executor.execute("while True:\n    pass")
    assert result.timed_out
    assert not result.success


def test_output_is_truncated():
    executor = CodeExecutor(timeout=5, max_output_size=50)
    result = executor.execute("print('x' * 1000)")
    assert "truncated" in result.output
    assert len(result.output) < 200


def test_execution_time_is_recorded(executor):
    result = executor.execute("print(1)")
    assert result.execution_time >= 0
    assert "s" in result.to_dict()["execution_time"]


def test_syntax_error_in_code_is_a_failure(executor):
    result = executor.execute("def bad(:\n    pass")
    assert not result.success
    assert "SyntaxError" in result.error
