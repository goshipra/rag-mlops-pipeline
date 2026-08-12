"""
eval/scoring.py — pure, dependency-free scoring functions for the eval harness.

These functions have no dependency on Qdrant, Ollama, BentoML, or MLflow on
purpose: they are the part of the eval pipeline that CI's regression gate
actually trusts, so they need to be simple, deterministic, and unit-testable
in isolation (see tests/test_scoring.py). Everything that talks to a live
service lives in run_eval.py instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def retrieval_hit(retrieved_source_ids: list[str], expected_source_id: str) -> bool:
    """Return True if ``expected_source_id`` appears anywhere in the top-k
    retrieved source ids (recall@k for a single query, k = len(retrieved))."""
    return expected_source_id in retrieved_source_ids


def answer_quality(answer: str, expected_key_phrases: list[str]) -> float:
    """Deterministic, dependency-free answer-quality score.

    Scores the fraction of ``expected_key_phrases`` that appear in ``answer``
    as a case-insensitive substring match. This intentionally avoids an
    LLM-as-judge call (slow, flaky, non-deterministic, costs an API key) so
    it is safe to run as a hard CI gate. See README "What CI validates" for
    where an LLM-judge would plug in as a *local* extension instead.

    Returns 0.0 for an empty phrase list (nothing to check counts as no
    signal, not perfect score) and otherwise the fraction in [0.0, 1.0].
    """
    if not expected_key_phrases:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for phrase in expected_key_phrases if phrase.lower() in answer_lower)
    return hits / len(expected_key_phrases)


@dataclass
class QueryResult:
    question_id: str
    retrieval_hit: bool
    answer_quality: float
    retrieval_score: float = 0.0
    latency_ms: float = 0.0
    extra: dict = field(default_factory=dict)


def aggregate_metrics(results: list[QueryResult]) -> dict:
    """Aggregate a list of per-question QueryResult into run-level metrics.

    - ``retrieval_recall_at_k``: fraction of questions whose expected source
      appeared in the top-k retrieved chunks.
    - ``answer_quality_avg``: mean per-question answer-quality score.
    - ``n_questions``: how many questions were scored (denominator sanity
      check — a change in testset size should be visible in the run).
    """
    n = len(results)
    if n == 0:
        return {"retrieval_recall_at_k": 0.0, "answer_quality_avg": 0.0, "n_questions": 0}

    recall = sum(1 for r in results if r.retrieval_hit) / n
    quality = sum(r.answer_quality for r in results) / n
    return {
        "retrieval_recall_at_k": round(recall, 4),
        "answer_quality_avg": round(quality, 4),
        "n_questions": n,
    }
