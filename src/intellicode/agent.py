"""Intent-routing agent that dispatches queries to the right tool.

The agent performs keyword-based intent detection, executes the selected tool,
and renders a natural-language answer.  When an LLM is supplied it is used for
free-form answers (document Q&A, summaries); otherwise the agent degrades
gracefully to deterministic template answers — which also keeps the whole
system testable without a GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class LLMBackend(Protocol):
    """Minimal protocol for a text-generation backend."""

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Return generated text for *prompt*."""
        ...


@dataclass
class AgentResponse:
    """Structured response from :meth:`RAGAgent.execute`."""

    answer: str
    tool_used: str
    raw_output: dict[str, Any]
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "tool_used": self.tool_used,
            "raw_output": self.raw_output,
            "success": self.success,
        }


# Intent → keyword triggers (checked in priority order for code files).
_CODE_INTENTS: list[tuple[str, tuple[str, ...]]] = [
    ("code_executor", ("run this", "execute this", "run the code", "execute the code", "run code")),
    ("security_scanner", ("security", "vulnerability", "vulnerabilities", "secure", "exploit", "injection", "scan")),
    ("code_fixer", ("fix", "repair", "suggest fix")),
    ("test_generator", ("generate test", "create test", "write test", "pytest", "unit test", "test case")),
    ("code_analyzer", ("analyze", "analyse", "check", "review", "find bugs", "issues", "quality", "inspect", "metric")),
]

_DOC_SUMMARY_KEYWORDS = ("summary", "summarize", "summarise", "overview", "brief")


class RAGAgent:
    """Route a query to a tool and render an answer.

    Args:
        tools: Registry from :func:`intellicode.tools.get_tools`.
        llm: Optional text-generation backend.  When ``None``, template
            answers are used everywhere.
    """

    def __init__(self, tools: dict[str, Any], llm: LLMBackend | None = None) -> None:
        self._tools = tools
        self._llm = llm

    # ── Intent detection ─────────────────────────────────────────────────

    def detect_intent(self, query: str, file_context: dict[str, Any] | None = None) -> str:
        """Determine which tool should handle *query*.

        Args:
            query: The user's query.
            file_context: Optional dict describing an uploaded file; a
                ``type == "code"`` entry routes to the code tools.

        Returns:
            The name of the selected tool.
        """
        q = query.lower()

        if file_context and file_context.get("type") == "code":
            for tool_name, keywords in _CODE_INTENTS:
                if any(kw in q for kw in keywords):
                    return tool_name
            return "code_analyzer"  # default for code files

        if any(kw in q for kw in _DOC_SUMMARY_KEYWORDS):
            return "summarizer"

        return "document_search"

    # ── Execution ────────────────────────────────────────────────────────

    def execute(self, query: str, file_context: dict[str, Any] | None = None) -> AgentResponse:
        """Detect intent, run the tool, and render an answer.

        Args:
            query: The user's query.
            file_context: Optional uploaded-file context.

        Returns:
            An :class:`AgentResponse`.
        """
        tool_name = self.detect_intent(query, file_context)
        tool = self._tools.get(tool_name)
        if tool is None:
            return AgentResponse(
                answer=f"No tool available for intent '{tool_name}'.",
                tool_used="error",
                raw_output={"error": "tool_not_found"},
                success=False,
            )

        try:
            output = self._dispatch(tool, tool_name, query, file_context)
        except (OSError, ValueError) as exc:
            logger.error("Tool '%s' failed: %s", tool_name, exc)
            return AgentResponse(
                answer=f"Error running {tool_name}: {exc}",
                tool_used=tool_name,
                raw_output={"error": str(exc)},
                success=False,
            )

        if not output.get("success"):
            return AgentResponse(
                answer=f"Error: {output.get('error', 'unknown error')}",
                tool_used=tool_name,
                raw_output=output,
                success=False,
            )

        answer = self._render_answer(query, output, tool_name)
        return AgentResponse(answer=answer, tool_used=tool_name, raw_output=output, success=True)

    def _dispatch(
        self,
        tool: Any,
        tool_name: str,
        query: str,
        file_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Call the tool with the arguments its execute() expects."""
        needs_code_string = tool_name in {"security_scanner", "code_executor"}
        needs_file_path = tool_name in {"code_analyzer", "test_generator", "code_fixer"}

        if needs_code_string and file_context:
            with open(file_context["path"], encoding="utf-8") as f:
                return tool.execute(f.read())
        if needs_file_path and file_context:
            return tool.execute(file_context["path"])
        if tool_name in {"document_search", "summarizer"}:
            return tool.execute(query)

        return {"success": False, "error": "invalid tool invocation"}

    # ── Answer rendering ─────────────────────────────────────────────────

    def _render_answer(self, query: str, output: dict[str, Any], tool_name: str) -> str:
        """Produce a human-readable answer from raw tool output."""
        renderers = {
            "test_generator": self._render_test_generator,
            "code_executor": self._render_executor,
            "code_fixer": self._render_fixer,
            "code_analyzer": self._render_analyzer,
            "security_scanner": self._render_security,
            "document_search": self._render_document_search,
            "summarizer": self._render_summary,
        }
        renderer = renderers.get(tool_name)
        return renderer(query, output) if renderer else "Done."

    # Deterministic renderers ------------------------------------------------

    @staticmethod
    def _render_test_generator(_query: str, output: dict[str, Any]) -> str:
        funcs = output.get("functions", [])
        preview = ", ".join(funcs[:3]) + (f" and {len(funcs) - 3} more" if len(funcs) > 3 else "")
        return (
            f"Generated {output.get('test_cases_generated', 0)} test cases across "
            f"{output.get('functions_found', 0)} functions ({preview})."
        )

    @staticmethod
    def _render_executor(_query: str, output: dict[str, Any]) -> str:
        if output.get("timeout"):
            return f"Execution timed out ({output.get('execution_time', '?')})."
        if output.get("exit_code") == 0:
            return f"Executed successfully ({output.get('execution_time', '?')}).\n\n{output.get('output', '')}"
        return f"Execution failed (exit {output.get('exit_code')}).\n\n{output.get('error', '')}"

    @staticmethod
    def _render_fixer(_query: str, output: dict[str, Any]) -> str:
        n = output.get("fixes_suggested", 0)
        if n == 0:
            return "No issues found — nothing to fix."
        lines = [f"Generated {n} fix suggestion(s) for {output.get('original_issues', 0)} issue(s):"]
        for i, fix in enumerate(output.get("fixes", [])[:3], 1):
            lines.append(f"{i}. Line {fix.get('line')}: {fix.get('issue_type')} — {fix.get('fix')}")
        return "\n".join(lines)

    def _render_analyzer(self, _query: str, output: dict[str, Any]) -> str:
        issues = output.get("issues", [])
        metrics = output.get("metrics", {})
        counts = _severity_counts(issues, ("HIGH", "MEDIUM", "LOW"))
        base = (
            f"Found {len(issues)} issue(s) — severity {output.get('severity', 'LOW')} "
            f"({counts['HIGH']} high, {counts['MEDIUM']} medium, {counts['LOW']} low). "
            f"Comment coverage {metrics.get('comment_ratio', 0):.0f}%."
        )
        return self._maybe_llm_summary(base, output)

    def _render_security(self, _query: str, output: dict[str, Any]) -> str:
        vulns = output.get("vulnerabilities", [])
        counts = _severity_counts(vulns, ("CRITICAL", "HIGH", "MEDIUM"))
        base = (
            f"Risk level {output.get('risk_level', 'LOW')} — {len(vulns)} vulnerability(ies) "
            f"({counts['CRITICAL']} critical, {counts['HIGH']} high, {counts['MEDIUM']} medium)."
        )
        return self._maybe_llm_summary(base, output)

    def _render_document_search(self, query: str, output: dict[str, Any]) -> str:
        chunks = output.get("chunks", [])
        if not chunks:
            return "No relevant information found in the indexed documents."

        context = " ".join(c["text"] for c in chunks[:3])[:1800]
        if self._llm is None:
            return _first_sentences(chunks[0]["text"], 3)

        prompt = (
            "Answer the question using only the information below. State facts "
            "directly; do not ask questions.\n\n"
            f"QUESTION: {query}\n\nINFORMATION:\n{context}\n\nANSWER:"
        )
        answer = self._safe_generate(prompt)
        return answer or _first_sentences(chunks[0]["text"], 3)

    def _render_summary(self, _query: str, output: dict[str, Any]) -> str:
        highlights = output.get("highlights", [])
        if self._llm is None:
            if highlights:
                return "Key points:\n" + "\n".join(f"- {h}" for h in highlights)
            return "Unable to generate a summary."

        prompt = f"Summarise the following in 3 concise bullet points.\n\nText: {output.get('full_text', '')[:1000]}\n\nSummary:"
        answer = self._safe_generate(prompt)
        if answer:
            return answer
        return "Key points:\n" + "\n".join(f"- {h}" for h in highlights)

    # LLM helpers -----------------------------------------------------------

    def _maybe_llm_summary(self, fallback: str, output: dict[str, Any]) -> str:
        """Use the LLM for a one-line summary if available, else the fallback."""
        if self._llm is None:
            return fallback
        prompt = f"Write a one-sentence assessment.\n\nData: {fallback}\n\nAssessment:"
        answer = self._safe_generate(prompt, max_new_tokens=60)
        return answer or fallback

    def _safe_generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Call the LLM, returning an empty string on any failure."""
        if self._llm is None:
            return ""
        try:
            return self._llm.generate(prompt, max_new_tokens=max_new_tokens).strip()
        except Exception as exc:  # noqa: BLE001 — LLM backends raise varied errors
            logger.error("LLM generation failed: %s", exc)
            return ""


# ── Module helpers ───────────────────────────────────────────────────────────


def _severity_counts(items: list[dict[str, Any]], levels: tuple[str, ...]) -> dict[str, int]:
    """Count items per severity level."""
    return {level: sum(1 for it in items if it.get("severity") == level) for level in levels}


def _first_sentences(text: str, n: int) -> str:
    """Return the first *n* sentences of *text*."""
    sentences = text.split(".")[:n]
    return ". ".join(s.strip() for s in sentences if s.strip()).rstrip(".") + "."
