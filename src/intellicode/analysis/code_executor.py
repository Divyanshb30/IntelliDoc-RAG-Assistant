"""Sandboxed Python code execution via subprocess with a hard timeout.

Runs untrusted code in a separate interpreter process so a crash, infinite
loop, or resource exhaustion cannot take down the host application.  Output is
size-capped to avoid unbounded memory use.

.. warning::
   Subprocess isolation is *not* a security sandbox.  It bounds runtime and
   captures output, but does not restrict filesystem or network access.  Do
   not run genuinely hostile code without OS-level sandboxing (containers,
   seccomp, etc.).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a snippet of code."""

    success: bool
    output: str = ""
    error: str = ""
    exit_code: int = 0
    execution_time: float = 0.0
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output or "(no output)",
            "error": self.error,
            "exit_code": self.exit_code,
            "execution_time": f"{self.execution_time:.2f}s",
            "timeout": self.timed_out,
        }


class CodeExecutor:
    """Execute Python code in an isolated subprocess.

    Args:
        timeout: Default wall-clock timeout in seconds.
        max_output_size: Maximum captured characters per stream (stdout/stderr).
    """

    def __init__(self, timeout: int = 5, max_output_size: int = 5000) -> None:
        self.timeout = timeout
        self.max_output_size = max_output_size

    def execute(self, code: str, timeout: int | None = None) -> ExecutionResult:
        """Run *code* and capture its output.

        Args:
            code: Python source to execute.
            timeout: Override the default timeout for this call.

        Returns:
            An :class:`ExecutionResult` describing the run.
        """
        timeout = timeout or self.timeout
        temp_file = self._write_temp(code)

        try:
            start = time.perf_counter()
            try:
                completed = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tempfile.gettempdir(),
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    check=False,
                )
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - start
                logger.info("Execution timed out after %ss", timeout)
                return ExecutionResult(
                    success=False,
                    error=f"Execution timed out after {timeout}s",
                    exit_code=-1,
                    execution_time=elapsed,
                    timed_out=True,
                )

            elapsed = time.perf_counter() - start
            stdout = self._truncate(completed.stdout)
            stderr = self._truncate(completed.stderr)

            return ExecutionResult(
                success=completed.returncode == 0,
                output=stdout,
                error=stderr,
                exit_code=completed.returncode,
                execution_time=elapsed,
            )
        finally:
            self._cleanup(temp_file)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _write_temp(code: str) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            return f.name

    def _truncate(self, text: str) -> str:
        if len(text) > self.max_output_size:
            return text[: self.max_output_size] + "\n… (output truncated)"
        return text

    @staticmethod
    def _cleanup(path: str) -> None:
        try:
            os.unlink(path)
        except OSError as exc:
            logger.debug("Could not delete temp file %s: %s", path, exc)
