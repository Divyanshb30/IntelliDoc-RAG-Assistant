import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rag import RAGPipeline
from tools import get_tools
from agent import RAGAgent
import os
import pypdf
import spaces  # required for ZeroGPU

# Patch gradio 4.44.0 bool schema bug
import gradio_client.utils as _gcu
_orig = _gcu._json_schema_to_python_type
def _patched(schema, defs=None):
    if isinstance(schema, bool):
        return "any"
    return _orig(schema, defs)
_gcu._json_schema_to_python_type = _patched

# ── Global state ──────────────────────────────────────────────────────────────
rag = RAGPipeline()
tools = get_tools(rag)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
print("Loading Qwen2.5-3B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
llm = {"model": model, "tokenizer": tokenizer}
agent = RAGAgent(llm, tools)
print("Model loaded ✓")

current_code_file = {"path": None, "name": None}

# ── File handlers ─────────────────────────────────────────────────────────────
def handle_doc_upload(files):
    if not files:
        return "No files uploaded."
    os.makedirs("data", exist_ok=True)
    names = []
    for f in files:
        fname = os.path.basename(f.name)
        dest = os.path.join("data", fname)
        if fname.endswith(".pdf"):
            reader = pypdf.PdfReader(f.name)
            text = "\n\n".join([p.extract_text() for p in reader.pages])
            with open(dest.replace(".pdf", ".txt"), "w", encoding="utf-8") as out:
                out.write(text)
        else:
            import shutil
            shutil.copy(f.name, dest)
        names.append(fname)
    return f"✅ Uploaded: {', '.join(names)}"

def build_index():
    try:
        rag.build_index("data")
        return "✅ Knowledge base ready!"
    except Exception as e:
        return f"❌ Error: {e}"

def handle_code_upload(file):
    if not file:
        return "No file uploaded."
    os.makedirs("temp", exist_ok=True)
    fname = os.path.basename(file.name)
    dest = os.path.join("temp", fname)
    import shutil
    shutil.copy(file.name, dest)
    current_code_file["path"] = dest
    current_code_file["name"] = fname
    return f"✅ Loaded: {fname}"

# ── Chat handler ──────────────────────────────────────────────────────────────
@spaces.GPU
def chat(message, history, mode):
    if not message.strip():
        return history, ""

    file_context = None
    if mode == "Code Analysis" and current_code_file["path"]:
        file_context = {
            "path": current_code_file["path"],
            "name": current_code_file["name"],
            "type": "code"
        }

    result = agent.execute(message, file_context=file_context)
    answer = result.get("answer", "No response generated.")
    tool = result.get("tool_used", "")
    raw = result.get("raw_output", {})

    # Append tool detail to answer
    if tool == "code_analyzer" and raw.get("issues"):
        issues = raw["issues"]
        answer += f"\n\n**Issues Found:** {len(issues)} | **Severity:** {raw.get('severity','N/A')}"
        for i in issues[:3]:
            answer += f"\n- Line {i.get('line','?')}: `{i.get('type','')}` ({i.get('severity','')}) — {i.get('message','')}"

    elif tool == "security_scanner" and raw.get("vulnerabilities"):
        vulns = raw["vulnerabilities"]
        answer += f"\n\n**Risk Level:** {raw.get('risk_level','N/A')} | **Vulnerabilities:** {len(vulns)}"
        for v in vulns[:3]:
            answer += f"\n- Line {v.get('line','?')}: `{v.get('type','')}` ({v.get('severity','')}) — {v.get('description','')}"

    elif tool == "code_executor":
        output = raw.get("output", "")
        error = raw.get("error", "")
        if output:
            answer += f"\n\n```\n{output[:500]}\n```"
        if error:
            answer += f"\n\n**Error:**\n```\n{error[:300]}\n```"

    elif tool == "test_generator":
        answer += f"\n\n**Test file:** `{raw.get('test_file','N/A')}`"
        funcs = raw.get("functions", [])
        if funcs:
            answer += "\n**Functions tested:** " + ", ".join([f"`{f}()`" for f in funcs])

    elif tool == "code_fixer" and raw.get("fixes"):
        fixes = raw["fixes"][:3]
        for i, fix in enumerate(fixes, 1):
            answer += f"\n\n**Fix #{i} — Line {fix.get('line','?')} ({fix.get('severity','')}):**"
            answer += f"\n> {fix.get('fix','')}"
            if fix.get("code_example"):
                answer += f"\n```python\n{fix['code_example']}\n```"

    history.append((message, answer))
    return history, ""

def quick_action(action, history, mode):
    return chat(action, history, mode)

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="IntelliCode RAG Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 IntelliCode RAG Assistant\n* AI for code analysis & document Q&A — Qwen2.5-3B | FAISS | Python AST*")

    with gr.Row():
        # ── Left sidebar ──
        with gr.Column(scale=1):
            mode = gr.Radio(
                ["Documents", "Code Analysis"],
                value="Documents",
                label="Mode"
            )

            with gr.Group(visible=True) as doc_panel:
                gr.Markdown("### 📄 Documents")
                doc_upload = gr.File(
                    label="Upload files",
                    file_types=[".txt", ".pdf", ".csv", ".md"],
                    file_count="multiple"
                )
                doc_status = gr.Textbox(label="", interactive=False, lines=1)
                build_btn = gr.Button("Build Knowledge Base", variant="primary")
                build_status = gr.Textbox(label="", interactive=False, lines=1)

            with gr.Group(visible=False) as code_panel:
                gr.Markdown("### 💻 Code Analysis")
                code_upload = gr.File(
                    label="Upload Python file",
                    file_types=[".py"]
                )
                code_status = gr.Textbox(label="", interactive=False, lines=1)

                gr.Markdown("### ⚡ Quick Actions")
                with gr.Row():
                    btn_analyze = gr.Button("Analyze Code", size="sm")
                    btn_security = gr.Button("Security Scan", size="sm")
                with gr.Row():
                    btn_tests = gr.Button("Generate Tests", size="sm")
                    btn_fix = gr.Button("Fix Issues", size="sm")
                with gr.Row():
                    btn_run = gr.Button("Run Code", size="sm")
                    btn_explain = gr.Button("Explain Code", size="sm")

        # ── Chat area ──
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="IntelliCode RAG",
                height=550,
                bubble_full_width=False
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about your code or documents...",
                    show_label=False,
                    scale=5
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Markdown("*Powered by Qwen2.5-3B | FAISS | Python AST | v2.0*")

    # ── Mode toggle ──
    def toggle_mode(m):
        return gr.update(visible=m == "Documents"), gr.update(visible=m == "Code Analysis")

    mode.change(toggle_mode, inputs=mode, outputs=[doc_panel, code_panel])

    # ── File events ──
    doc_upload.change(handle_doc_upload, inputs=doc_upload, outputs=doc_status)
    build_btn.click(build_index, outputs=build_status)
    code_upload.change(handle_code_upload, inputs=code_upload, outputs=code_status)

    # ── Chat events ──
    send_btn.click(chat, inputs=[msg, chatbot, mode], outputs=[chatbot, msg])
    msg.submit(chat, inputs=[msg, chatbot, mode], outputs=[chatbot, msg])

    # ── Quick action buttons ──
    btn_analyze.click(lambda h, m: quick_action("Analyze this code", h, m), inputs=[chatbot, mode], outputs=[chatbot, msg])
    btn_security.click(lambda h, m: quick_action("Check security vulnerabilities", h, m), inputs=[chatbot, mode], outputs=[chatbot, msg])
    btn_tests.click(lambda h, m: quick_action("Generate pytest tests", h, m), inputs=[chatbot, mode], outputs=[chatbot, msg])
    btn_fix.click(lambda h, m: quick_action("Fix the issues", h, m), inputs=[chatbot, mode], outputs=[chatbot, msg])
    btn_run.click(lambda h, m: quick_action("Run this code", h, m), inputs=[chatbot, mode], outputs=[chatbot, msg])
    btn_explain.click(lambda h, m: quick_action("Explain this code", h, m), inputs=[chatbot, mode], outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch()


