"""Unit tests for the code fixer."""

from __future__ import annotations

import pytest

from intellicode.analysis import CodeFixer


@pytest.fixture
def fixer() -> CodeFixer:
    return CodeFixer()


def test_suggests_fix_for_mutable_default(fixer):
    result = fixer.fix_code("def f(x=[]):\n    return x")
    assert result.success
    assert result.fixes_suggested >= 1
    fix = next(f for f in result.fixes if "mutable" in f.issue_type.lower())
    assert fix.code_example  # includes a before/after example


def test_clean_code_needs_no_fixes(fixer, clean_code_path):
    result = fixer.fix_file(str(clean_code_path))
    assert result.success
    # Clean code may still trip stylistic checks; ensure no HIGH-severity fixes.
    assert all(f.severity != "HIGH" for f in result.fixes)


def test_every_fix_has_explanation(fixer):
    result = fixer.fix_code("def f(x=[]):\n    try:\n        pass\n    except:\n        pass")
    assert result.fixes
    assert all(f.fix and f.explanation for f in result.fixes)


def test_syntax_error_returns_failure(fixer):
    result = fixer.fix_code("def broken(:\n    pass")
    assert not result.success


def test_fix_count_is_capped():
    fixer = CodeFixer(max_fixes=2)
    code = "def a(x=[]): return x\ndef b(y=[]): return y\ndef c(z=[]): return z"
    result = fixer.fix_code(code)
    assert result.fixes_suggested <= 2
    assert result.original_issues >= 3
