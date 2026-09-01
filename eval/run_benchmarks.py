"""Comparative retrieval benchmarks for IntelliCode.

Runs four controlled experiments over the labeled evaluation set and prints
Markdown tables (also written to ``eval/results/``).  Each experiment changes
exactly one variable so the effect is attributable.

Runs on CPU with embeddings + FAISS only.  The reranking experiment downloads
a small cross-encoder (~90 MB) on first use.

Usage:
    python eval/run_benchmarks.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from intellicode.config import Settings
from intellicode.evaluation import evaluate_retrieval, load_corpus, load_dataset
from intellicode.rag import RAGPipeline

logging.disable(logging.CRITICAL)  # keep benchmark output clean

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "tests" / "eval" / "datasets" / "retrieval_eval.json"
CORPUS = ROOT / "tests" / "fixtures" / "documents"
RESULTS_DIR = ROOT / "eval" / "results"


def _run(settings: Settings, queries, docs, names, *, use_reranker: bool) -> dict[str, Any]:
    """Build an index with *settings* and return its metrics dict."""
    pipeline = RAGPipeline(settings)
    pipeline.build_index(docs, names)
    metrics = evaluate_retrieval(pipeline, queries, use_reranker=use_reranker)
    return metrics.to_dict()


def _table(title: str, rows: list[tuple[str, dict[str, Any]]], columns: list[str]) -> str:
    """Render a Markdown metrics table."""
    header = "| Configuration | " + " | ".join(columns) + " |"
    sep = "|" + "---|" * (len(columns) + 1)
    lines = [f"### {title}", "", header, sep]
    for label, metrics in rows:
        cells = " | ".join(f"{metrics[c]:.3f}" if isinstance(metrics[c], float) else str(metrics[c]) for c in columns)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run all benchmark experiments and write results."""
    queries = load_dataset(DATASET)
    docs, names = load_corpus(CORPUS)
    print(f"Loaded {len(queries)} queries over {len(docs)} documents "
          f"({sum(1 for q in queries if not q.expect_no_answer)} answerable).\n")

    metric_cols = ["mrr_at_5", "recall_at_1", "recall_at_3", "ndcg_at_5"]
    all_results: dict[str, Any] = {}

    # ── Experiment 1: search mode ────────────────────────────────────────
    dense = _run(Settings(use_reranker=False, use_hybrid_search=False), queries, docs, names, use_reranker=False)
    hybrid = _run(Settings(use_reranker=False, use_hybrid_search=True), queries, docs, names, use_reranker=False)
    exp1 = _table("Experiment 1 - Search mode (chunk=256, no rerank)",
                  [("Dense only", dense), ("Hybrid (dense + BM25)", hybrid)], metric_cols)
    all_results["search_mode"] = {"dense_only": dense, "hybrid": hybrid}
    print(exp1)

    # ── Experiment 2: reranking ──────────────────────────────────────────
    no_rerank = hybrid  # reuse
    with_rerank = _run(Settings(use_reranker=True, use_hybrid_search=True), queries, docs, names, use_reranker=True)
    exp2 = _table("Experiment 2 - Cross-encoder reranking (hybrid, chunk=256)",
                  [("Hybrid, no rerank", no_rerank), ("Hybrid + rerank", with_rerank)], metric_cols)
    all_results["reranking"] = {"no_rerank": no_rerank, "with_rerank": with_rerank}
    print(exp2)

    # ── Experiment 3: chunk size ─────────────────────────────────────────
    chunk_rows = []
    all_results["chunk_size"] = {}
    for size in (128, 256, 512):
        overlap = size // 4
        m = _run(
            Settings(use_reranker=False, use_hybrid_search=True, chunk_size=size, chunk_overlap=overlap),
            queries, docs, names, use_reranker=False,
        )
        chunk_rows.append((f"{size} tokens", m))
        all_results["chunk_size"][str(size)] = m
    exp3 = _table("Experiment 3 - Chunk size (hybrid, no rerank)", chunk_rows, metric_cols)
    print(exp3)

    # ── Experiment 4: chunking method (matched ~256-token size) ──────────
    # Hold chunk size roughly constant so the comparison isolates *method*,
    # not size: ~190 words ≈ 256 tokens.
    word = _run(
        Settings(use_reranker=False, use_hybrid_search=True, chunking_strategy="word", chunk_size=190, chunk_overlap=48),
        queries, docs, names, use_reranker=False,
    )
    sentence = hybrid  # sentence-aware @256 from experiment 1
    exp4 = _table("Experiment 4 - Chunking method at matched size (hybrid, no rerank)",
                  [("Word-split (~256 tok)", word), ("Sentence-aware (256 tok)", sentence)], metric_cols)
    all_results["chunking_method"] = {"word_split": word, "sentence_aware": sentence}
    print(exp4)

    # ── Persist ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "benchmark_results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "benchmark_tables.md").write_text(
        "\n".join([exp1, exp2, exp3, exp4]), encoding="utf-8"
    )
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
