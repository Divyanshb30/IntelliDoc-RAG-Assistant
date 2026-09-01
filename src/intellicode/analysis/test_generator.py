"""Automatic pytest test scaffold generation from Python source.

Extracts public function signatures via AST, infers parameter types from
naming conventions, and emits a pytest module exercising each function with
normal, empty, and large inputs.
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

FunctionNode = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass
class FunctionSignature:
    """A public function's name, parameters, and generated test cases."""

    name: str
    params: list[dict[str, str]]
    test_cases: list[dict[str, Any]]
    line: int


@dataclass
class GenerationResult:
    """Result of generating a test scaffold."""

    success: bool
    functions_found: int = 0
    test_cases_generated: int = 0
    test_code: str = ""
    functions: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "functions_found": self.functions_found,
            "test_cases_generated": self.test_cases_generated,
            "test_code": self.test_code,
            "functions": self.functions,
            "error": self.error,
        }


# Naming-convention → type heuristics, most specific first.
_TYPE_HEURISTICS: list[tuple[frozenset[str], str]] = [
    (frozenset({"list", "items", "array", "values", "numbers"}), "list"),
    (frozenset({"price", "rate", "ratio", "percent", "score"}), "float"),
    (frozenset({"name", "text", "message", "title", "word"}), "str"),
    (frozenset({"dict", "map", "data", "config"}), "dict"),
    (frozenset({"flag", "is_", "has_", "can_", "should_"}), "bool"),
    (frozenset({"num", "count", "size", "length", "age", "id"}), "int"),
]

_SAMPLE_VALUES: dict[str, dict[str, Any]] = {
    "normal": {"int": 5, "float": 10.5, "str": "test", "list": [1, 2, 3], "dict": {"k": "v"}, "bool": True, "Any": "test"},
    "empty": {"int": 0, "float": 0.0, "str": "", "list": [], "dict": {}, "bool": False, "Any": None},
    "large": {"int": 1_000_000, "float": 999999.99, "str": "x" * 100, "list": list(range(100)), "dict": {f"k{i}": i for i in range(10)}, "bool": True, "Any": "large"},
}


class TestGenerator:
    """Generate pytest scaffolds for the public functions in a source file."""

    def generate_from_code(self, code: str, *, module_name: str = "module") -> GenerationResult:
        """Generate a pytest module for the functions defined in *code*.

        Args:
            code: Python source code.
            module_name: Import name used in the generated ``from … import …``.

        Returns:
            A :class:`GenerationResult`; ``success`` is ``False`` on syntax
            error or if no testable functions are found.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return GenerationResult(success=False, error=f"Syntax Error: {exc.msg}")

        functions = self._extract_functions(tree)
        if not functions:
            return GenerationResult(success=False, error="No testable public functions found")

        test_code = self._render(functions, module_name)
        return GenerationResult(
            success=True,
            functions_found=len(functions),
            test_cases_generated=sum(len(f.test_cases) for f in functions),
            test_code=test_code,
            functions=[f.name for f in functions],
        )

    def generate_from_file(self, path: str) -> GenerationResult:
        """Generate tests for a file on disk.

        Args:
            path: Path to a ``.py`` file.

        Returns:
            A :class:`GenerationResult`.
        """
        try:
            with open(path, encoding="utf-8") as f:
                code = f.read()
        except OSError as exc:
            return GenerationResult(success=False, error=f"Could not read file: {exc}")

        module = os.path.basename(path).removesuffix(".py")
        return self.generate_from_code(code, module_name=module)

    # ── Extraction ───────────────────────────────────────────────────────

    def _extract_functions(self, tree: ast.AST) -> list[FunctionSignature]:
        functions: list[FunctionSignature] = []
        for node in ast.walk(tree):
            if isinstance(node, FunctionNode) and not node.name.startswith("_"):
                params = [
                    {"name": arg.arg, "type": self._infer_type(arg.arg)}
                    for arg in node.args.args
                    if arg.arg != "self"
                ]
                functions.append(
                    FunctionSignature(
                        name=node.name,
                        params=params,
                        test_cases=self._make_cases(params),
                        line=node.lineno,
                    )
                )
        return functions

    @staticmethod
    def _infer_type(param_name: str) -> str:
        lowered = param_name.lower()
        for keywords, type_name in _TYPE_HEURISTICS:
            if any(k in lowered for k in keywords):
                return type_name
        return "Any"

    def _make_cases(self, params: list[dict[str, str]]) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = [{"name": "normal", "inputs": self._inputs(params, "normal")}]
        if params:
            cases.append({"name": "empty", "inputs": self._inputs(params, "empty")})
            cases.append({"name": "large", "inputs": self._inputs(params, "large")})
        return cases

    @staticmethod
    def _inputs(params: list[dict[str, str]], kind: str) -> dict[str, Any]:
        table = _SAMPLE_VALUES[kind]
        return {p["name"]: table.get(p["type"], table["Any"]) for p in params}

    # ── Rendering ────────────────────────────────────────────────────────

    def _render(self, functions: list[FunctionSignature], module_name: str) -> str:
        names = ", ".join(f.name for f in functions)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        header = (
            f'"""Auto-generated tests for {module_name}.\n\n'
            f"Generated by IntelliCode on {timestamp}.\n"
            '"""\n\n'
            "import pytest\n\n"
            f"from {module_name} import {names}\n\n\n"
        )

        body = ""
        for func in functions:
            for case in func.test_cases:
                inputs = case["inputs"]
                call_args = ", ".join(f"{name}={value!r}" for name, value in inputs.items())
                body += (
                    f"def test_{func.name}_{case['name']}():\n"
                    f'    """Exercise {func.name} with {case["name"]} inputs."""\n'
                    f"    result = {func.name}({call_args})\n"
                    f"    assert result is not None\n\n\n"
                )

        return header + body
