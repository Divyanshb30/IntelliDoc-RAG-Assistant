"""Unit tests for the pytest scaffold generator."""

from __future__ import annotations

import pytest

from intellicode.analysis import TestGenerator


@pytest.fixture
def generator() -> TestGenerator:
    return TestGenerator()


def test_generates_tests_for_public_functions(generator):
    result = generator.generate_from_code("def add(a, b):\n    return a + b")
    assert result.success
    assert result.functions_found == 1
    assert "add" in result.functions
    assert "def test_add_normal" in result.test_code


def test_private_functions_are_skipped(generator):
    result = generator.generate_from_code("def _helper():\n    return 1")
    assert not result.success
    assert "No testable" in result.error


def test_generates_edge_cases_for_parametrized_functions(generator):
    result = generator.generate_from_code("def scale(values):\n    return values")
    assert result.test_cases_generated >= 2  # normal + edge cases
    assert "test_scale_empty" in result.test_code


def test_type_inference_from_names(generator):
    result = generator.generate_from_code("def greet(name):\n    return name")
    # 'name' → str, so the generated normal input should be a string literal
    assert "name='test'" in result.test_code or 'name="test"' in result.test_code


def test_syntax_error_returns_failure(generator):
    result = generator.generate_from_code("def broken(:\n    pass")
    assert not result.success
    assert "Syntax Error" in result.error


def test_async_functions_are_extracted(generator):
    result = generator.generate_from_code("async def fetch(url):\n    return url")
    assert result.success
    assert "fetch" in result.functions


def test_generated_code_is_valid_python(generator):
    import ast

    result = generator.generate_from_code("def add(a, b):\n    return a + b")
    ast.parse(result.test_code)  # raises SyntaxError if invalid
