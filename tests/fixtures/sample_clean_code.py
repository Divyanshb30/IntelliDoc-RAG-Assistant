"""Clean-code fixture for the analyzer false-positive-rate benchmark.

This module is intentionally well-written: fully type-hinted public functions,
no mutable defaults, specific exception handling, no globals, no bare asserts,
and shallow nesting.  The analyzer should report zero target-class issues here.
"""

from __future__ import annotations


def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


def safe_divide(numerator: float, denominator: float) -> float | None:
    """Divide two numbers, returning None on division by zero."""
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return None


def accumulate(items: list[int] | None = None) -> list[int]:
    """Append to a fresh list, avoiding the mutable-default pitfall."""
    if items is None:
        items = []
    items.append(0)
    return items


def categorize(value: int) -> str:
    """Classify a value using flat guard clauses instead of deep nesting."""
    if value < 0:
        return "negative"
    if value == 0:
        return "zero"
    return "positive"


def validate_positive(value: int) -> int:
    """Validate input by raising an explicit exception (not assert)."""
    if value <= 0:
        raise ValueError("value must be positive")
    return value


class Counter:
    """A small stateful helper that avoids global state."""

    def __init__(self) -> None:
        self._count = 0

    def increment(self) -> int:
        """Increment the counter and return the new value."""
        self._count += 1
        return self._count
