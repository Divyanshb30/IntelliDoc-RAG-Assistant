# IntelliCode RAG Assistant

A production-ready AI-powered code analysis and document intelligence system built with local LLMs. This assistant provides comprehensive code quality analysis, security scanning, automated test generation, and intelligent document search capabilities without requiring external API keys or internet connectivity.

## Overview

IntelliCode RAG Assistant is an enterprise-grade tool that combines Retrieval-Augmented Generation (RAG) with advanced code analysis to help developers maintain high-quality, secure codebases while providing instant answers from technical documentation. The system runs entirely on local hardware, ensuring complete privacy and eliminating API costs.

## Features

### Code Analysis Tools

**Static Code Analyzer**
- AST-based Python code inspection
- Detects mutable default arguments, bare except clauses, and unused variables
- Identifies deep nesting and overly complex functions
- Calculates code metrics including comment coverage and cyclomatic complexity
- Provides severity-based issue classification (HIGH, MEDIUM, LOW)

**Security Scanner**
- Detects hardcoded credentials and API keys
- Identifies SQL injection vulnerabilities
- Flags weak cryptographic algorithms (MD5, SHA1)
- Warns about dangerous function usage (eval, exec, os.system)
- Checks for insecure deserialization patterns
- Assigns risk levels from LOW to CRITICAL

**Automated Test Generator**
- Generates pytest-compatible test suites automatically
- Creates normal, edge, and parametrized test cases
- Infers parameter types from function signatures
- Supports multiple test scenarios per function
- Outputs ready-to-run test files

**Code Executor**
- Safe execution in isolated subprocess environment
- Configurable timeout protection
- Captures stdout and stderr separately
- Provides execution time metrics
- Prevents resource leaks with automatic cleanup

**Code Fixer**
- Analyzes code issues and suggests specific fixes
- Provides before/after code examples
- Explains why each fix is necessary
- Covers common Python antipatterns
- Generates actionable improvement recommendations

### Document Intelligence Tools

**RAG-Powered Document Search**
- Semantic search across uploaded documents
- Supports TXT, PDF, CSV, and Markdown formats
- Uses FAISS for efficient vector similarity search
- Provides context-aware answers to queries
- Processes documents with configurable chunk sizes

**Document Summarizer**
- Extracts key highlights from large documents
- Generates concise summaries
- Supports multiple document formats
- Maintains context across document sections

## Technical Architecture

### Core Components

**Language Model**
- Model: Qwen 2.5 3B Instruct (quantized Q4_K_M)
- Context window: 4096 tokens
- Runs on CPU with optimized threading
- No GPU required for operation

**Vector Database**
- FAISS (Facebook AI Similarity Search)
- IndexFlatL2 for exact similarity search
- Optimized for CPU-based retrieval
- Supports dynamic index updates

**Embedding Model**
- Model: all-MiniLM-L6-v2
- Dimension: 384
- Fast inference on CPU
- Balanced accuracy and speed

**Web Interface**
- Built with Streamlit
- Responsive design for desktop use
- Real-time chat interface
- Professional UI with custom CSS styling
- Expandable detail panels for analysis results

