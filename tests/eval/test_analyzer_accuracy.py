"""Analyzer-accuracy gates.

Measures the AST analyzer's precision/recall against the annotated
``sample_buggy_code.py`` ground truth and the ``sample_clean_code.py``
false-positive control.  Pure Python — no models, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from intellicode.analysis import CodeAnalyzer
from intellicode.evaluation import (
    ANALYZER_TARGET_CLASSES,
    evaluate_analyzer,
    parse_expect_markers,
)

pytestmark = pytest.mark.eval

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
BUGGY_FIXTURE = _FIXTURES / "sample_buggy_code.py"
CLEAN_FIXTURE = _FIXTURES / "sample_clean_code.py"


def test_analyzer_f1_meets_baseline(baselines):
    """Analyzer F1 over the ground-truth fixture must meet the pinned floor."""
    metrics = evaluate_analyzer(CodeAnalyzer(), BUGGY_FIXTURE, CLEAN_FIXTURE)
    floor = baselines["analyzer"]

    assert metrics.f1 >= floor["f1"], f"F1 {metrics.f1:.3f} < baseline {floor['f1']}"
    assert metrics.false_positives_on_clean <= floor["max_false_positives_on_clean"], (
        f"{metrics.false_positives_on_clean} false positive(s) on clean code"
    )


def test_no_false_positives_on_clean_code():
    """Well-written code should produce zero target-class findings."""
    detections = CodeAnalyzer().analyze_file(str(CLEAN_FIXTURE)).issues
    target_hits = [i for i in detections if i.type in ANALYZER_TARGET_CLASSES]
    assert target_hits == [], f"Unexpected findings on clean code: {target_hits}"


def test_every_marked_pattern_is_detected():
    """Recall must be perfect on the annotated fixture (no missed patterns)."""
    metrics = evaluate_analyzer(CodeAnalyzer(), BUGGY_FIXTURE, CLEAN_FIXTURE)
    assert metrics.false_negatives == 0, f"Missed {metrics.false_negatives} pattern(s)"


def test_ground_truth_covers_target_classes():
    """The fixture should exercise a broad set of target anti-patterns."""
    marked_types = {t for _, t in parse_expect_markers(BUGGY_FIXTURE)}
    covered = marked_types & ANALYZER_TARGET_CLASSES
    assert len(covered) >= 8, f"Only {len(covered)} target classes exercised: {covered}"
