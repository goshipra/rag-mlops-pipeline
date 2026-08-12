"""Unit tests for eval/scoring.py — pure functions, no live services needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.scoring import QueryResult, aggregate_metrics, answer_quality, retrieval_hit


def test_retrieval_hit_true_when_expected_source_present():
    assert retrieval_hit(["a", "b", "c"], "b") is True


def test_retrieval_hit_false_when_expected_source_missing():
    assert retrieval_hit(["a", "b", "c"], "z") is False


def test_retrieval_hit_empty_list():
    assert retrieval_hit([], "a") is False


def test_answer_quality_all_phrases_present():
    assert answer_quality("PEP 518 introduced pyproject.toml", ["PEP 518"]) == 1.0


def test_answer_quality_case_insensitive():
    assert answer_quality("pep 518 introduced it", ["PEP 518"]) == 1.0


def test_answer_quality_partial_match():
    score = answer_quality("PEP 518 introduced it", ["PEP 518", "PEP 621"])
    assert score == 0.5


def test_answer_quality_no_match():
    assert answer_quality("totally unrelated text", ["PEP 518"]) == 0.0


def test_answer_quality_empty_phrase_list_returns_zero():
    # No phrases to check for is "no signal", not a free perfect score.
    assert answer_quality("anything", []) == 0.0


def test_aggregate_metrics_empty_results():
    metrics = aggregate_metrics([])
    assert metrics == {"retrieval_recall_at_k": 0.0, "answer_quality_avg": 0.0, "n_questions": 0}


def test_aggregate_metrics_mixed_results():
    results = [
        QueryResult(question_id="q1", retrieval_hit=True, answer_quality=1.0),
        QueryResult(question_id="q2", retrieval_hit=False, answer_quality=0.5),
        QueryResult(question_id="q3", retrieval_hit=True, answer_quality=0.0),
        QueryResult(question_id="q4", retrieval_hit=True, answer_quality=0.5),
    ]
    metrics = aggregate_metrics(results)
    assert metrics["n_questions"] == 4
    assert metrics["retrieval_recall_at_k"] == 0.75
    assert metrics["answer_quality_avg"] == 0.5


def test_aggregate_metrics_all_perfect():
    results = [QueryResult(question_id=f"q{i}", retrieval_hit=True, answer_quality=1.0) for i in range(5)]
    metrics = aggregate_metrics(results)
    assert metrics["retrieval_recall_at_k"] == 1.0
    assert metrics["answer_quality_avg"] == 1.0
