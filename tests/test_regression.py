"""Unit tests for eval/regression.py — the logic behind the CI regression gate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.regression import check_regression


BASELINE = {"retrieval_recall_at_k": 0.90, "answer_quality_avg": 0.85}


def test_identical_metrics_pass():
    result = check_regression(dict(BASELINE), dict(BASELINE))
    assert result.passed is True


def test_improved_metrics_pass():
    current = {"retrieval_recall_at_k": 0.95, "answer_quality_avg": 0.90}
    result = check_regression(current, dict(BASELINE))
    assert result.passed is True


def test_small_drop_within_tolerance_passes():
    # default tolerance is 0.02 (2 percentage points)
    current = {"retrieval_recall_at_k": 0.885, "answer_quality_avg": 0.85}
    result = check_regression(current, dict(BASELINE))
    assert result.passed is True


def test_drop_beyond_tolerance_fails():
    current = {"retrieval_recall_at_k": 0.80, "answer_quality_avg": 0.85}
    result = check_regression(current, dict(BASELINE))
    assert result.passed is False
    assert any("retrieval_recall_at_k" in m and "FAIL" in m for m in result.messages)


def test_both_metrics_regressing_reports_both():
    current = {"retrieval_recall_at_k": 0.5, "answer_quality_avg": 0.5}
    result = check_regression(current, dict(BASELINE))
    assert result.passed is False
    fail_messages = [m for m in result.messages if m.startswith("FAIL")]
    assert len(fail_messages) == 2


def test_missing_metric_in_current_fails_closed():
    current = {"retrieval_recall_at_k": 0.95}  # answer_quality_avg missing
    result = check_regression(current, dict(BASELINE))
    assert result.passed is False
    assert any("answer_quality_avg" in m and "missing from current run" in m for m in result.messages)


def test_missing_metric_in_baseline_fails_closed():
    baseline = {"retrieval_recall_at_k": 0.90}  # answer_quality_avg missing
    current = {"retrieval_recall_at_k": 0.95, "answer_quality_avg": 0.95}
    result = check_regression(current, baseline)
    assert result.passed is False
    assert any("missing from baseline.json" in m for m in result.messages)


def test_custom_tolerance_override():
    current = {"retrieval_recall_at_k": 0.80, "answer_quality_avg": 0.85}
    # widen tolerance enough that the 0.10 drop in recall is now allowed
    result = check_regression(current, dict(BASELINE), tolerance_overrides={"retrieval_recall_at_k": 0.15})
    assert result.passed is True


def test_exactly_at_floor_passes():
    # current == baseline - tolerance should NOT count as a regression (>=)
    current = {"retrieval_recall_at_k": 0.88, "answer_quality_avg": 0.85}
    result = check_regression(current, dict(BASELINE))
    assert result.passed is True
