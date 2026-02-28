---
title: IntelliCode RAG Assistant
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
hardware: zero-gpu
---


# 🧠 IntelliCode RAG Assistant

> Dual-mode AI assistant for code analysis & document Q&A — Qwen2.5-3B · FAISS · Python AST · HuggingFace ZeroGPU

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/Divb30/intellicode-rag)
[![GitHub](https://img.shields.io/badge/GitHub-IntelliCode-black?logo=github)](https://github.com/Divyanshb30/IntelliDoc-RAG-Assistant)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--3B-orange)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

##  Live Demo

**[→ Try it live on HuggingFace Spaces](https://huggingface.co/spaces/Divb30/intellicode-rag)**

GPU-accelerated via ZeroGPU (A100). No login required.

---

## Two Modes, Two Engines

IntelliCode runs two completely different engines depending on the task:

| | Mode 1: Document Q&A | Mode 2: Code Analysis |
|---|---|---|
| **Engine** | Qwen2.5-3B + FAISS | Python AST (local, no LLM) |
| **Speed** | GPU inference | Instant (no model call) |
| **Input** | PDF, MD, CSV, TXT | `.py` files |
| **Output** | RAG-grounded answers | Structured issue reports |

---

## Mode 1 — Document Q&A (RAG)

Upload any document and ask questions in natural language.

**How it works:**
1. Document is chunked and embedded via `sentence-transformers`
2. Embeddings stored in a local **FAISS** index
3. On query → top-5 most relevant chunks retrieved
4. **Qwen2.5-3B** answers strictly from retrieved context — no hallucination beyond the document

**Tools:**
- `document_search` — retrieves top-5 chunks, generates a factual answer
- `summarizer` — identifies prominent sections, returns key bullet points

---

## Mode 2 — Code Analysis (AST-Powered)

Python code is parsed into an **Abstract Syntax Tree locally** — no LLM involved. Faster, safer, deterministic.

###  Code Analyzer
Detects Python anti-patterns with severity classification:

| Severity | Pattern |
|---|---|
| HIGH | Mutable default arguments (`def foo(x=[])`) |
| MEDIUM | Bare `except:` clauses, overly broad exception handling |
| LOW | Unused variables, deeply nested functions, long functions, low comment coverage |

###  Security Scanner
Regex-based detection of critical vulnerabilities:
- Hardcoded API keys and passwords
- SQL injection via string formatting inside `.execute()`
- Weak hashing: `MD5`, `SHA1`
- Dangerous builtins: `eval()`, `exec()`, `os.system()`
- Insecure deserialization: `pickle.loads()`

### ▶ Code Executor
- Runs uploaded code in an **isolated shell**
- **5-second timeout** prevents infinite loops
- Captures stdout + stderr and returns terminal output

###  Test Generator
- Reads all functions in the uploaded file
- **Infers parameter types** from naming conventions (`count` → int, `name` → str)
- Auto-generates a full `pytest` suite with edge cases (`0`, `""`, `None`)
- Writes tests to a `test/` directory

### Code Fixer
- Takes issues found by the Analyzer
- Generates structured fix suggestions with:
  - Why it's a bug
  - Before/after code snippet


## Tech Stack

| Component | Technology |
|---|---|
| LLM | Qwen2.5-3B-Instruct (HuggingFace ZeroGPU) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| Code Analysis | Python `ast` module |
| Security Scanning | Python `re` (regex) |
| UI | Gradio |
| Hosting | HuggingFace Spaces + ZeroGPU |

---

