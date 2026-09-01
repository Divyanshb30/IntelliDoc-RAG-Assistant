"""AST-based Python code analyzer.

Detects common anti-patterns and code smells by walking the abstract syntax
tree.  Every function-level check covers both synchronous (``FunctionDef``) and
asynchronous (``AsyncFunctionDef``) definitions.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Function-like AST nodes (sync + async) — the original analyzer missed async.
FunctionNode = (ast.FunctionDef, ast.AsyncFunctionDef)


class AnalysisError(Exception):
    """Raised when analysis fails for reasons other than a syntax error."""


@dataclass
class Issue:
    """A single detected code issue."""

    type: str
    severity: str  # "HIGH" | "MEDIUM" | "LOW"
    line: int
    message: str
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "line": self.line,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass
class AnalysisResult:
    """Result of analyzing a single file or code string."""

    success: bool
    issues: list[Issue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    severity: str = "LOW"
    total_lines: int = 0
    file: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
            "severity": self.severity,
            "total_lines": self.total_lines,
            "file": self.file,
            "error": self.error,
        }


class CodeAnalyzer:
    """Analyze Python source for anti-patterns and compute quality metrics.

    Args:
        complexity_threshold: Cyclomatic complexity above which a function is
            flagged.
        long_function_lines: Line count above which a function is flagged.
        max_nesting_depth: Control-flow nesting depth that triggers a warning.
    """

    def __init__(
        self,
        *,
        complexity_threshold: int = 10,
        long_function_lines: int = 50,
        max_nesting_depth: int = 4,
    ) -> None:
        self.complexity_threshold = complexity_threshold
        self.long_function_lines = long_function_lines
        self.max_nesting_depth = max_nesting_depth

    # ── Public API ───────────────────────────────────────────────────────

    def analyze_file(self, code_file_path: str) -> AnalysisResult:
        """Analyze a Python file on disk.

        Args:
            code_file_path: Path to a ``.py`` file.

        Returns:
            An :class:`AnalysisResult`.
        """
        try:
            with open(code_file_path, encoding="utf-8") as f:
                code = f.read()
        except FileNotFoundError:
            return AnalysisResult(success=False, error=f"File not found: {code_file_path}")
        except OSError as exc:
            return AnalysisResult(success=False, error=f"Could not read file: {exc}")

        import os

        return self.analyze_code(code, file_name=os.path.basename(code_file_path))

    def analyze_code(self, code: str, *, file_name: str = "<string>") -> AnalysisResult:
        """Analyze Python source provided as a string.

        Args:
            code: Python source code.
            file_name: Name to record in the result.

        Returns:
            An :class:`AnalysisResult`.  ``success`` is ``False`` on syntax error.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return AnalysisResult(
                success=False,
                file=file_name,
                error=f"Syntax Error: {exc.msg} (line {exc.lineno})",
            )

        issues: list[Issue] = []
        issues.extend(self._check_mutable_defaults(tree))
        issues.extend(self._check_bare_except(tree))
        issues.extend(self._check_broad_except(tree))
        issues.extend(self._check_unused_variables(tree))
        issues.extend(self._check_deep_nesting(tree))
        issues.extend(self._check_long_functions(tree))
        issues.extend(self._check_cyclomatic_complexity(tree))
        issues.extend(self._check_star_imports(tree))
        issues.extend(self._check_missing_return_hints(tree))
        issues.extend(self._check_global_statements(tree))
        issues.extend(self._check_asserts(tree))
        issues.extend(self._check_nested_function_depth(tree))

        metrics = self._calculate_metrics(code, tree)
        severity = self._overall_severity(issues)

        return AnalysisResult(
            success=True,
            issues=issues,
            metrics=metrics,
            severity=severity,
            total_lines=len(code.splitlines()),
            file=file_name,
        )

    # ── Checks ───────────────────────────────────────────────────────────

    # Builtins whose calls construct a fresh mutable object (e.g. ``list()``).
    _MUTABLE_CONSTRUCTORS = frozenset({"list", "dict", "set", "bytearray"})

    def _check_mutable_defaults(self, tree: ast.AST) -> list[Issue]:
        """Flag mutable default arguments (list/dict/set literals or constructors)."""
        issues: list[Issue] = []
        for node in ast.walk(tree):
            if isinstance(node, FunctionNode):
                for default in node.args.defaults + node.args.kw_defaults:
                    if self._is_mutable_default(default):
                        issues.append(
                            Issue(
                                type="Mutable default argument",
                                severity="HIGH",
                                line=node.lineno,
                                message=f"Function '{node.name}' uses a mutable default argument",
                                suggestion="Use None as default and initialise inside the function",
                            )
                        )
        return issues

    def _is_mutable_default(self, default: ast.expr | None) -> bool:
        """Return True if *default* is a mutable literal or constructor call."""
        if isinstance(default, (ast.List, ast.Dict, ast.Set)):
            return True
        return (
            isinstance(default, ast.Call)
            and isinstance(default.func, ast.Name)
            and default.func.id in self._MUTABLE_CONSTRUCTORS
        )

    def _check_bare_except(self, tree: ast.AST) -> list[Issue]:
        """Flag bare ``except:`` clauses."""
        issues: list[Issue] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append(
                    Issue(
                        type="Bare except clause",
                        severity="MEDIUM",
                        line=node.lineno,
                        message="Bare 'except:' catches all exceptions, including KeyboardInterrupt",
                        suggestion="Catch specific exceptions, e.g. 'except ValueError as e:'",
                    )
                )
        return issues

    def _check_broad_except(self, tree: ast.AST) -> list[Issue]:
        """Flag overly broad ``except Exception`` handlers."""
        issues: list[Issue] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ExceptHandler)
                and isinstance(node.type, ast.Name)
                and node.type.id in {"Exception", "BaseException"}
            ):
                issues.append(
                    Issue(
                        type="Broad exception handler",
                        severity="LOW",
                        line=node.lineno,
                        message=f"Catching '{node.type.id}' is very broad and can hide bugs",
                        suggestion="Catch the most specific exception type that applies",
                    )
                )
        return issues

    def _check_unused_variables(self, tree: ast.AST) -> list[Issue]:
        """Flag variables assigned but never read within a function."""
        issues: list[Issue] = []
        for node in ast.walk(tree):
            if not isinstance(node, FunctionNode):
                continue
            assigned: dict[str, int] = {}
            used: set[str] = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            assigned[target.id] = child.lineno
                elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    used.add(child.id)
            for var, line in assigned.items():
                if var not in used and not var.startswith("_"):
                    issues.append(
                        Issue(
                            type="Unused variable",
                            severity="LOW",
                            line=line,
                            message=f"Variable '{var}' is assigned but never used",
                            suggestion="Remove it or prefix with underscore",
                        )
                    )
        return issues

    def _check_deep_nesting(self, tree: ast.AST) -> list[Issue]:
        """Flag functions whose control-flow nesting is too deep."""
        issues: list[Issue] = []

        def depth(node: ast.AST, current: int = 0) -> int:
            deepest = current
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith)):
                    deepest = max(deepest, depth(child, current + 1))
                else:
                    deepest = max(deepest, depth(child, current))
            return deepest

        for node in ast.walk(tree):
            if isinstance(node, FunctionNode):
                d = depth(node)
                if d >= self.max_nesting_depth:
                    issues.append(
                        Issue(
                            type="Deep nesting",
                            severity="MEDIUM",
                            line=node.lineno,
                            message=f"Function '{node.name}' has nesting depth {d}",
                            suggestion="Reduce nesting with early returns or guard clauses",
                        )
                    )
        return issues

    def _check_long_functions(self, tree: ast.AST) -> list[Issue]:
        """Flag functions exceeding the configured line count."""
        issues: list[Issue] = []
        for node in ast.walk(tree):
            if isinstance(node, FunctionNode) and hasattr(node, "end_lineno") and node.end_lineno:
                length = node.end_lineno - node.lineno
                if length > self.long_function_lines:
                    issues.append(
                        Issue(
                            type="Long function",
                            severity="LOW",
                            line=node.lineno,
                            message=f"Function '{node.name}' is {length} lines long",
                            suggestion="Break it into smaller, single-purpose functions",
                        )
                    )
        return issues

    def _check_cyclomatic_complexity(self, tree: ast.AST) -> list[Issue]:
        """Flag functions with high cyclomatic complexity."""
        issues: list[Issue] = []
        decision_nodes = (
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.ExceptHandler,
            ast.With,
            ast.AsyncWith,
            ast.BoolOp,
            ast.IfExp,
        )
        for node in ast.walk(tree):
            if isinstance(node, FunctionNode):
                complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (*decision_nodes, ast.comprehension)):
                        complexity += 1
                if complexity > self.complexity_threshold:
                    issues.append(
                        Issue(
                            type="High cyclomatic complexity",
                            severity="MEDIUM",
                            line=node.lineno,
                            message=f"Function '{node.name}' has cyclomatic complexity {complexity}",
                            suggestion="Simplify branching logic or extract helper functions",
                        )
                    )
        return issues

    def _check_star_imports(self, tree: ast.AST) -> list[Issue]:
        """Flag ``from module import *``."""
        issues: list[Issue] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                issues.append(
                    Issue(
                        type="Star import",
                        severity="MEDIUM",
                        line=node.lineno,
                        message=f"'from {node.module or '?'} import *' pollutes the namespace",
                        suggestion="Import only the names you use",
                    )
                )
        return issues

    def _check_missing_return_hints(self, tree: ast.AST) -> list[Issue]:
        """Flag public functions lacking a return type annotation."""
        issues: list[Issue] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, FunctionNode)
                and not node.name.startswith("_")
                and node.returns is None
            ):
                issues.append(
                    Issue(
                        type="Missing return type hint",
                        severity="LOW",
                        line=node.lineno,
                        message=f"Public function '{node.name}' has no return type annotation",
                        suggestion="Add a '-> ReturnType' annotation",
                    )
                )
        return issues

    def _check_global_statements(self, tree: ast.AST) -> list[Issue]:
        """Flag use of the ``global`` statement."""
        issues: list[Issue] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                issues.append(
                    Issue(
                        type="Global statement",
                        severity="LOW",
                        line=node.lineno,
                        message=f"Use of 'global {', '.join(node.names)}' introduces shared mutable state",
                        suggestion="Pass values as arguments or encapsulate in a class",
                    )
                )
        return issues

    def _check_asserts(self, tree: ast.AST) -> list[Issue]:
        """Flag ``assert`` statements (stripped when Python runs with -O)."""
        issues: list[Issue] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                issues.append(
                    Issue(
                        type="Assert in production code",
                        severity="LOW",
                        line=node.lineno,
                        message="'assert' is removed when Python runs with -O; do not use it for validation",
                        suggestion="Raise an explicit exception instead",
                    )
                )
        return issues

    def _check_nested_function_depth(self, tree: ast.AST) -> list[Issue]:
        """Flag functions nested more than two levels deep."""
        issues: list[Issue] = []

        def visit(node: ast.AST, func_depth: int) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, FunctionNode):
                    if func_depth + 1 > 2:
                        issues.append(
                            Issue(
                                type="Deeply nested function",
                                severity="MEDIUM",
                                line=child.lineno,
                                message=f"Function '{child.name}' is nested {func_depth + 1} levels deep",
                                suggestion="Extract nested functions to module level",
                            )
                        )
                    visit(child, func_depth + 1)
                else:
                    visit(child, func_depth)

        visit(tree, 0)
        return issues

    # ── Metrics ──────────────────────────────────────────────────────────

    def _calculate_metrics(self, code: str, tree: ast.AST) -> dict[str, Any]:
        """Compute basic size and structure metrics."""
        lines = code.splitlines()
        total = len(lines)
        code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
        functions = sum(1 for n in ast.walk(tree) if isinstance(n, FunctionNode))
        classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))

        return {
            "total_lines": total,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "comment_ratio": (comment_lines / total * 100) if total else 0.0,
            "functions": functions,
            "classes": classes,
        }

    @staticmethod
    def _overall_severity(issues: list[Issue]) -> str:
        """Roll individual issue severities up into a single label."""
        if any(i.severity == "HIGH" for i in issues):
            return "HIGH"
        if len(issues) > 3:
            return "MEDIUM"
        return "LOW"
