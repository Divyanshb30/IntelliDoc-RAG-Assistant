"""Gradio front-end for IntelliCode, deployed on HuggingFace Spaces (ZeroGPU).

Two modes:
  * Documents     — RAG Q&A over uploaded files (Qwen2.5-3B + hybrid retrieval).
  * Code Analysis — AST-powered analysis, security scan, tests, execution.

The LLM is optional: if it cannot be loaded (e.g. no GPU), the agent falls
back to deterministic template answers so the app still works.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys

# Make the src/ layout importable on HuggingFace Spaces without a pip install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr

# Work around a gradio 4.44.0 bug where boolean JSON schemas crash type parsing.
import gradio_client.utils as _gcu  # noqa: E402

_orig_json_schema = _gcu._json_schema_to_python_type


def _patched_json_schema(schema, defs=None):
    if isinstance(schema, bool):
        return "any"
    return _orig_json_schema(schema, defs)


_gcu._json_schema_to_python_type = _patched_json_schema

from intellicode.agent import RAGAgent  # noqa: E402
from intellicode.config import Settings, configure_logging  # noqa: E402
from intellicode.rag import RAGPipeline  # noqa: E402
from intellicode.tools import get_tools  # noqa: E402

# ── ZeroGPU shim ──────────────────────────────────────────────────────────────
try:
    import spaces  # type: ignore

    gpu_decorator = spaces.GPU
except ImportError:  # running locally without the `spaces` package

    def gpu_decorator(fn):  # type: ignore
        return fn


configure_logging()
logger = logging.getLogger(__name__)

# ── Core components ───────────────────────────────────────────────────────────
settings = Settings()
pipeline = RAGPipeline(settings)
tools = get_tools(pipeline)


def _load_llm():
    """Load the LLM, returning None on failure so the app degrades gracefully."""
    try:
        from intellicode.llm import QwenLLM

        return QwenLLM.from_pretrained(settings)
    except Exception as exc:  # noqa: BLE001 — model load can fail many ways
        logger.warning("LLM unavailable, using template answers: %s", exc)
        return None


llm = _load_llm()
agent = RAGAgent(tools, llm=llm)

current_code_file: dict[str, str | None] = {"path": None, "name": None}


# ── File handlers ─────────────────────────────────────────────────────────────


def handle_doc_upload(files) -> str:
    """Copy uploaded documents into the data directory (PDF → text)."""
    if not files:
        return "No files uploaded."
    os.makedirs("data", exist_ok=True)
    names: list[str] = []
    for f in files:
        fname = os.path.basename(f.name)
        dest = os.path.join("data", fname)
        try:
            if fname.endswith(".pdf"):
                import pypdf

                reader = pypdf.PdfReader(f.name)
                text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
                with open(dest.replace(".pdf", ".txt"), "w", encoding="utf-8") as out:
                    out.write(text)
            else:
                shutil.copy(f.name, dest)
            names.append(fname)
        except (OSError, ValueError) as exc:
            logger.error("Failed to ingest %s: %s", fname, exc)
            return f"Error ingesting {fname}: {exc}"
    return f"Uploaded: {', '.join(names)}"


def build_index() -> str:
    """Build the retrieval index from the data directory."""
    try:
        count = pipeline.build_index_from_directory("data")
        if count == 0:
            return "No documents found. Upload files first."
        return f"Knowledge base ready — {count} chunks indexed."
    except FileNotFoundError:
        return "No data directory. Upload documents first."
    except Exception as exc:  # noqa: BLE001
        logger.error("Index build failed: %s", exc)
        return f"Error building index: {exc}"


def handle_code_upload(file) -> str:
    """Stash an uploaded Python file for the code-analysis tools."""
    if not file:
        return "No file uploaded."
    os.makedirs("uploads", exist_ok=True)
    fname = os.path.basename(file.name)
    dest = os.path.join("uploads", fname)
    try:
        shutil.copy(file.name, dest)
    except OSError as exc:
        return f"Error loading {fname}: {exc}"
    current_code_file["path"] = dest
    current_code_file["name"] = fname
    return f"Loaded: {fname}"


# ── Chat handler ──────────────────────────────────────────────────────────────


@gpu_decorator
def chat(message: str, history: list, mode: str):
    """Route a message through the agent and append the answer to history."""
    if not message.strip():
        return history, ""

    file_context = None
    if mode == "Code Analysis" and current_code_file["path"]:
        file_context = {
            "path": current_code_file["path"],
            "name": current_code_file["name"],
            "type": "code",
        }

    result = agent.execute(message, file_context=file_context).to_dict()
    answer = result.get("answer", "No response generated.")
    answer = _augment_answer(answer, result.get("tool_used", ""), result.get("raw_output", {}))

    history.append((message, answer))
    return history, ""


def _augment_answer(answer: str, tool: str, raw: dict) -> str:
    """Append structured tool detail to the natural-language answer."""
    if tool == "code_analyzer" and raw.get("issues"):
        issues = raw["issues"]
        answer += f"\n\n**Issues:** {len(issues)} | **Severity:** {raw.get('severity', 'N/A')}"
        for i in issues[:3]:
            answer += f"\n- Line {i.get('line', '?')}: `{i.get('type', '')}` ({i.get('severity', '')})"
    elif tool == "security_scanner" and raw.get("vulnerabilities"):
        vulns = raw["vulnerabilities"]
        answer += f"\n\n**Risk:** {raw.get('risk_level', 'N/A')} | **Findings:** {len(vulns)}"
        for v in vulns[:3]:
            answer += f"\n- Line {v.get('line', '?')}: `{v.get('type', '')}` ({v.get('severity', '')})"
    elif tool == "code_executor":
        if raw.get("output"):
            answer += f"\n\n```\n{raw['output'][:500]}\n```"
        if raw.get("error"):
            answer += f"\n\n**Error:**\n```\n{raw['error'][:300]}\n```"
    elif tool == "test_generator" and raw.get("test_code"):
        answer += f"\n\n```python\n{raw['test_code'][:600]}\n```"
    elif tool == "code_fixer" and raw.get("fixes"):
        for i, fix in enumerate(raw["fixes"][:3], 1):
            answer += f"\n\n**Fix #{i} — line {fix.get('line', '?')}:** {fix.get('fix', '')}"
            if fix.get("code_example"):
                answer += f"\n```python\n{fix['code_example']}\n```"
    return answer


# ── UI ────────────────────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""
    with gr.Blocks(title="IntelliCode RAG Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# IntelliCode RAG Assistant\n"
            "*Hybrid RAG (dense + BM25 + reranking) & AST code analysis — "
            "Qwen2.5-3B · FAISS · Python AST*"
        )
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(["Documents", "Code Analysis"], value="Documents", label="Mode")

                with gr.Group(visible=True) as doc_panel:
                    gr.Markdown("### Documents")
                    doc_upload = gr.File(
                        label="Upload files",
                        file_types=[".txt", ".pdf", ".csv", ".md"],
                        file_count="multiple",
                    )
                    doc_status = gr.Textbox(label="", interactive=False, lines=1)
                    build_btn = gr.Button("Build Knowledge Base", variant="primary")
                    build_status = gr.Textbox(label="", interactive=False, lines=1)

                with gr.Group(visible=False) as code_panel:
                    gr.Markdown("### Code Analysis")
                    code_upload = gr.File(label="Upload Python file", file_types=[".py"])
                    code_status = gr.Textbox(label="", interactive=False, lines=1)
                    gr.Markdown("### Quick Actions")
                    with gr.Row():
                        btn_analyze = gr.Button("Analyze", size="sm")
                        btn_security = gr.Button("Security", size="sm")
                    with gr.Row():
                        btn_tests = gr.Button("Gen Tests", size="sm")
                        btn_fix = gr.Button("Fix Issues", size="sm")
                    with gr.Row():
                        btn_run = gr.Button("Run Code", size="sm")

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="IntelliCode", height=550)
                with gr.Row():
                    msg = gr.Textbox(placeholder="Ask about your code or documents…", show_label=False, scale=5)
                    send_btn = gr.Button("Send", variant="primary", scale=1)

        def toggle_mode(m):
            return gr.update(visible=m == "Documents"), gr.update(visible=m == "Code Analysis")

        mode.change(toggle_mode, inputs=mode, outputs=[doc_panel, code_panel])
        doc_upload.change(handle_doc_upload, inputs=doc_upload, outputs=doc_status)
        build_btn.click(build_index, outputs=build_status)
        code_upload.change(handle_code_upload, inputs=code_upload, outputs=code_status)
        send_btn.click(chat, inputs=[msg, chatbot, mode], outputs=[chatbot, msg])
        msg.submit(chat, inputs=[msg, chatbot, mode], outputs=[chatbot, msg])

        for button, prompt in [
            (btn_analyze, "Analyze this code"),
            (btn_security, "Check security vulnerabilities"),
            (btn_tests, "Generate pytest tests"),
            (btn_fix, "Fix the issues"),
            (btn_run, "Run this code"),
        ]:
            button.click(
                lambda h, m, p=prompt: chat(p, h, m),
                inputs=[chatbot, mode],
                outputs=[chatbot, msg],
            )

    return demo


if __name__ == "__main__":
    build_ui().launch()
