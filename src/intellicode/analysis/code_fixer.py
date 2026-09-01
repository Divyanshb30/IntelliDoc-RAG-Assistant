"""Fix-suggestion generator built on top of :class:`CodeAnalyzer`.

Runs the analyzer, then maps each detected issue to a concrete, copy-pasteable
before/after remediation snippet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from intellicode.analysis.code_analyzer import CodeAnalyzer, Issue

logger = logging.getLogger(__name__)


@dataclass
class FixSuggestion:
    """A remediation suggestion for one issue."""

    issue_type: str
    line: int
    severity: str
    fix: str
    explanation: str
    code_example: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_type": self.issue_type,
            "line": self.line,
            "severity": self.severity,
            "fix": self.fix,
            "explanation": self.explanation,
            "code_example": self.code_example,
        }


@dataclass
class FixResult:
    """Result of generating fix suggestions."""

    success: bool
    original_issues: int = 0
    fixes: list[FixSuggestion] = field(default_factory=list)
    severity: str = "LOW"
    error: str = ""

    @property
    def fixes_suggested(self) -> int:
        return len(self.fixes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "original_issues": self.original_issues,
            "fixes_suggested": self.fixes_suggested,
            "fixes": [f.to_dict() for f in self.fixes],
            "severity": self.severity,
            "error": self.error,
        }


# Issue-type substring → (fix, explanation, example)
_FIX_TEMPLATES: dict[str, tuple[str, str, str]] = {
    "mutable default": (
        "Replace the mutable default with None and initialise inside the function",
        "Mutable defaults are created once and shared across all calls.",
        "# Before\ndef f(items=[]):\n    items.append(x)\n\n# After\ndef f(items=None):\n    if items is None:\n        items = []\n    items.append(x)",
    ),
    "bare except": (
        "Catch a specific exception instead of bare 'except:'",
        "Bare except swallows KeyboardInterrupt and SystemExit, hiding real failures.",
        "# Before\ntry:\n    run()\nexcept:\n    pass\n\n# After\ntry:\n    run()\nexcept ValueError as e:\n    logger.error(e)",
    ),
    "broad exception": (
        "Narrow 'except Exception' to the specific error you expect",
        "Broad handlers mask bugs that should surface during development.",
        "# Before\nexcept Exception:\n    ...\n\n# After\nexcept (KeyError, IndexError):\n    ...",
    ),
    "unused variable": (
        "Remove the unused variable or prefix it with an underscore",
        "Dead assignments clutter code and mislead readers.",
        "# Before\nunused = compute()\n\n# After\n_ = compute()  # or delete the line",
    ),
    "deep nesting": (
        "Flatten with early returns / guard clauses",
        "Deep nesting hurts readability and raises cyclomatic complexity.",
        "# Before\nif a:\n    if b:\n        do()\n\n# After\nif not a or not b:\n    return\ndo()",
    ),
    "complexity": (
        "Extract helper functions to reduce branching",
        "High cyclomatic complexity correlates with defect density.",
        "# Split one large branching function into small, named helpers.",
    ),
    "star import": (
        "Import only the names you use",
        "Star imports pollute the namespace and defeat static analysis.",
        "# Before\nfrom os import *\n\n# After\nfrom os import path, getcwd",
    ),
    "return type": (
        "Add a return type annotation",
        "Return hints document intent and enable type checking.",
        "# Before\ndef total(xs):\n    ...\n\n# After\ndef total(xs: list[int]) -> int:\n    ...",
    ),
    "global": (
        "Avoid 'global'; pass state explicitly or encapsulate it",
        "Global mutable state makes code hard to test and reason about.",
        "# Prefer returning values or wrapping state in a class.",
    ),
    "assert": (
        "Replace 'assert' with an explicit raised exception",
        "Assertions are stripped when Python runs with -O, silently disabling checks.",
        "# Before\nassert x > 0\n\n# After\nif x <= 0:\n    raise ValueError('x must be positive')",
    ),
}


class CodeFixer:
    """Generate fix suggestions for issues found by :class:`CodeAnalyzer`."""

    def __init__(self, analyzer: CodeAnalyzer | None = None, *, max_fixes: int = 10) -> None:
        self._analyzer = analyzer or CodeAnalyzer()
        self._max_fixes = max_fixes

    def fix_file(self, path: str) -> FixResult:
        """Analyze *path* and return remediation suggestions.

        Args:
            path: Path to a ``.py`` file.

        Returns:
            A :class:`FixResult`.
        """
        analysis = self._analyzer.analyze_file(path)
        if not analysis.success:
            return FixResult(success=False, error=analysis.error)

        fixes = [self._suggest(issue) for issue in analysis.issues[: self._max_fixes]]
        return FixResult(
            success=True,
            original_issues=len(analysis.issues),
            fixes=fixes,
            severity=analysis.severity,
        )

    def fix_code(self, code: str) -> FixResult:
        """Analyze a code string and return remediation suggestions."""
        analysis = self._analyzer.analyze_code(code)
        if not analysis.success:
            return FixResult(success=False, error=analysis.error)

        fixes = [self._suggest(issue) for issue in analysis.issues[: self._max_fixes]]
        return FixResult(
            success=True,
            original_issues=len(analysis.issues),
            fixes=fixes,
            severity=analysis.severity,
        )

    # ── Internals ────────────────────────────────────────────────────────

    def _suggest(self, issue: Issue) -> FixSuggestion:
        key = issue.type.lower()
        for token, (fix, explanation, example) in _FIX_TEMPLATES.items():
            if token in key:
                return FixSuggestion(
                    issue_type=issue.type,
                    line=issue.line,
                    severity=issue.severity,
                    fix=fix,
                    explanation=explanation,
                    code_example=example,
                )

        return FixSuggestion(
            issue_type=issue.type,
            line=issue.line,
            severity=issue.severity,
            fix=f"Review and address: {issue.message}",
            explanation=issue.suggestion or "Review the flagged code.",
        )
