"""Retrieval evaluation metrics and harness.

Relevance is defined by *answer spans*: a retrieved chunk is relevant if its
text contains at least one of a query's gold answer spans (case-insensitive
substring match).  This keeps the labels robust to changes in chunking — no
brittle chunk-ID bookkeeping — while still measuring whether the answer-bearing
passage was retrieved and ranked highly.

All metrics run on embeddings + FAISS only: no GPU, no API keys.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from intellicode.analysis.code_analyzer import CodeAnalyzer
    from intellicode.rag.pipeline import RAGPipeline
    from intellicode.rag.retriever import RetrievalResult


# ── Relevance predicate ──────────────────────────────────────────────────────


def is_relevant(chunk_text: str, answer_spans: list[str]) -> bool:
    """Return True if *chunk_text* contains any of *answer_spans* (case-insensitive)."""
    haystack = chunk_text.lower()
    return any(span.lower() in haystack for span in answer_spans)


# ── Ranking metrics ──────────────────────────────────────────────────────────


def reciprocal_rank(results: list[RetrievalResult], answer_spans: list[str]) -> float:
    """Reciprocal rank of the first relevant result (0 if none)."""
    for rank, r in enumerate(results, start=1):
        if is_relevant(r.text, answer_spans):
            return 1.0 / rank
    return 0.0


def recall_at_k(results: list[RetrievalResult], answer_spans: list[str], k: int) -> float:
    """1.0 if a relevant chunk appears in the top *k*, else 0.0."""
    return 1.0 if any(is_relevant(r.text, answer_spans) for r in results[:k]) else 0.0


def ndcg_at_k(results: list[RetrievalResult], answer_spans: list[str], k: int) -> float:
    """Binary NDCG@k with a single ideal relevant document.

    With binary relevance and one ideal hit, the ideal DCG is 1, so NDCG@k
    reduces to ``1 / log2(rank + 1)`` for the first relevant result within *k*.
    """
    for rank, r in enumerate(results[:k], start=1):
        if is_relevant(r.text, answer_spans):
            return 1.0 / math.log2(rank + 1)
    return 0.0


# ── Dataset loading ──────────────────────────────────────────────────────────


@dataclass
class EvalQuery:
    """A single labeled evaluation query."""

    id: str
    category: str
    query: str
    answer_spans: list[str]
    expect_no_answer: bool = False


def load_dataset(path: str | Path) -> list[EvalQuery]:
    """Load the retrieval evaluation dataset from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalQuery(
            id=q["id"],
            category=q["category"],
            query=q["query"],
            answer_spans=q.get("answer_spans", []),
            expect_no_answer=q.get("expect_no_answer", False),
        )
        for q in data["queries"]
    ]


def load_corpus(corpus_dir: str | Path) -> tuple[list[str], list[str]]:
    """Load all ``.txt`` documents in *corpus_dir*.

    Returns:
        A ``(documents, filenames)`` tuple.
    """
    docs: list[str] = []
    names: list[str] = []
    for fp in sorted(Path(corpus_dir).glob("*.txt")):
        docs.append(fp.read_text(encoding="utf-8"))
        names.append(fp.name)
    return docs, names


# ── Evaluation harness ───────────────────────────────────────────────────────


@dataclass
class RetrievalMetrics:
    """Aggregate retrieval metrics over an evaluation set."""

    mrr_at_5: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    ndcg_at_5: float
    n_answerable: int
    negative_rejection_rate: float = 0.0
    n_negative: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mrr_at_5": round(self.mrr_at_5, 4),
            "recall_at_1": round(self.recall_at_1, 4),
            "recall_at_3": round(self.recall_at_3, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "n_answerable": self.n_answerable,
            "negative_rejection_rate": round(self.negative_rejection_rate, 4),
            "n_negative": self.n_negative,
        }


def evaluate_retrieval(
    pipeline: RAGPipeline,
    queries: list[EvalQuery],
    *,
    top_k: int = 5,
    use_reranker: bool | None = None,
    negative_threshold: float = 0.35,
) -> RetrievalMetrics:
    """Evaluate a built *pipeline* against labeled *queries*.

    Answerable queries contribute to MRR / Recall / NDCG.  Negative queries
    contribute to the *negative rejection rate*: the fraction whose top raw
    dense similarity falls below *negative_threshold* (i.e. correctly treated
    as "no confident answer").

    Args:
        pipeline: A :class:`RAGPipeline` with an index already built.
        queries: Labeled evaluation queries.
        top_k: Cutoff for the @k metrics.
        use_reranker: Force reranking on/off (defaults to pipeline config).
        negative_threshold: Dense-similarity cutoff for negative rejection.

    Returns:
        Aggregated :class:`RetrievalMetrics`.
    """
    answerable = [q for q in queries if not q.expect_no_answer]
    negatives = [q for q in queries if q.expect_no_answer]

    rr = r1 = r3 = r5 = ndcg = 0.0
    for q in answerable:
        results = pipeline.query(q.query, top_k=top_k, use_reranker=use_reranker)
        rr += reciprocal_rank(results, q.answer_spans)
        r1 += recall_at_k(results, q.answer_spans, 1)
        r3 += recall_at_k(results, q.answer_spans, 3)
        r5 += recall_at_k(results, q.answer_spans, 5)
        ndcg += ndcg_at_k(results, q.answer_spans, 5)

    n = max(1, len(answerable))

    # Negative rejection: use raw dense similarity (comparable across queries).
    rejected = 0
    for q in negatives:
        dense = pipeline.retriever.retrieve(q.query, top_k=1, use_hybrid=False)
        top_score = dense[0].score if dense else 0.0
        if top_score < negative_threshold:
            rejected += 1
    neg_rate = rejected / len(negatives) if negatives else 0.0

    return RetrievalMetrics(
        mrr_at_5=rr / n,
        recall_at_1=r1 / n,
        recall_at_3=r3 / n,
        recall_at_5=r5 / n,
        ndcg_at_5=ndcg / n,
        n_answerable=len(answerable),
        negative_rejection_rate=neg_rate,
        n_negative=len(negatives),
    )


# ── Analyzer accuracy evaluation ─────────────────────────────────────────────

# Anti-pattern classes scored by the analyzer benchmark. The stylistic
# "Missing return type hint" check is excluded — it fires on every un-hinted
# function and would dominate the counts without signalling detection quality.
ANALYZER_TARGET_CLASSES: frozenset[str] = frozenset(
    {
        "Mutable default argument",
        "Bare except clause",
        "Broad exception handler",
        "Unused variable",
        "Deep nesting",
        "High cyclomatic complexity",
        "Global statement",
        "Assert in production code",
        "Long function",
    }
)

_EXPECT_RE = re.compile(r"#\s*EXPECT:\s*(.+?)\s*$")


def parse_expect_markers(path: str | Path) -> set[tuple[int, str]]:
    """Extract ``(line, issue_type)`` ground-truth pairs from ``# EXPECT:`` markers."""
    ground_truth: set[tuple[int, str]] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        match = _EXPECT_RE.search(line)
        if match:
            ground_truth.add((lineno, match.group(1).strip()))
    return ground_truth


def analyzer_detections(
    analyzer: CodeAnalyzer,
    path: str | Path,
    target_classes: frozenset[str],
) -> set[tuple[int, str]]:
    """Run *analyzer* on *path* and return ``(line, type)`` pairs for target classes."""
    result = analyzer.analyze_file(str(path))
    return {
        (issue.line, issue.type)
        for issue in result.issues
        if issue.type in target_classes
    }


@dataclass
class AnalyzerMetrics:
    """Precision/recall/F1 for the analyzer, with per-class breakdown."""

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    false_positives_on_clean: int
    per_class: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "false_positives_on_clean": self.false_positives_on_clean,
            "per_class": self.per_class,
        }


def evaluate_analyzer(
    analyzer: CodeAnalyzer,
    buggy_path: str | Path,
    clean_path: str | Path | None = None,
    *,
    target_classes: frozenset[str] = ANALYZER_TARGET_CLASSES,
) -> AnalyzerMetrics:
    """Measure analyzer precision/recall against ``# EXPECT`` markers.

    Recall and precision are computed on *buggy_path* by matching detected
    ``(line, type)`` pairs against the file's markers.  If *clean_path* is
    given, any target-class detection there is counted as a false positive
    (a well-written file should produce none).

    Args:
        analyzer: The analyzer under test.
        buggy_path: Fixture annotated with ``# EXPECT:`` markers.
        clean_path: Optional clean fixture expected to yield zero findings.
        target_classes: Issue types included in scoring.

    Returns:
        Aggregated :class:`AnalyzerMetrics`.
    """
    ground_truth = {(ln, t) for ln, t in parse_expect_markers(buggy_path) if t in target_classes}
    detected = analyzer_detections(analyzer, buggy_path, target_classes)

    tp = len(ground_truth & detected)
    fn = len(ground_truth - detected)
    fp = len(detected - ground_truth)

    fp_clean = 0
    if clean_path is not None:
        fp_clean = len(analyzer_detections(analyzer, clean_path, target_classes))

    total_fp = fp + fp_clean
    precision = tp / (tp + total_fp) if (tp + total_fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Per-class recall breakdown
    per_class: dict[str, dict[str, int]] = {}
    for cls in sorted(target_classes):
        gt_cls = {p for p in ground_truth if p[1] == cls}
        det_cls = {p for p in detected if p[1] == cls}
        if gt_cls or det_cls:
            per_class[cls] = {
                "expected": len(gt_cls),
                "detected": len(gt_cls & det_cls),
                "false_positive": len(det_cls - gt_cls),
            }

    return AnalyzerMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        false_positives_on_clean=fp_clean,
        per_class=per_class,
    )
