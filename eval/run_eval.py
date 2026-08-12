#!/usr/bin/env python3
"""
eval/run_eval.py — run every question in eval/testset.jsonl through the RAG
pipeline, score retrieval recall@k and answer quality, log the run to
MLflow, and write a results JSON that eval/check_regression.py compares
against eval/baseline.json.

Usage:
    python eval/run_eval.py [--top-k 5] [--testset eval/testset.jsonl] \\
        [--out eval/results/latest.json]

Generation mode:
    Set GENERATOR=stub (or CI=true) to use the deterministic, dependency-free
    stub generator instead of calling Ollama — this is what CI does, since
    CI runners cannot realistically run Ollama. See README for details.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# MLflow >=3.15 refuses to use the plain filesystem tracking store unless
# this is set, nudging users toward a database backend. This project
# intentionally keeps MLflow filesystem-based (./mlruns) by default so eval
# runs work with zero extra infrastructure — no MLflow server, no database.
# See README "MLflow tracking" for the tradeoffs and how to switch to a
# database backend via MLFLOW_TRACKING_URI if you outgrow the file store.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402

import rag_core  # noqa: E402
from eval.scoring import QueryResult, aggregate_metrics, answer_quality, retrieval_hit  # noqa: E402


def load_testset(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run(testset_path: Path, top_k: int) -> tuple[dict, list[QueryResult]]:
    questions = load_testset(testset_path)
    results: list[QueryResult] = []

    for row in questions:
        pipeline_result, chunks, _tokens = rag_core.answer_question(row["question"], top_k=top_k)
        retrieved_source_ids = [c.source_id for c in chunks]

        hit = retrieval_hit(retrieved_source_ids, row["expected_source_id"])
        quality = answer_quality(pipeline_result.answer, row["expected_key_phrases"])

        results.append(
            QueryResult(
                question_id=row["id"],
                retrieval_hit=hit,
                answer_quality=quality,
                retrieval_score=pipeline_result.retrieval_score,
                latency_ms=pipeline_result.latency_ms,
                extra={
                    "question": row["question"],
                    "expected_source_id": row["expected_source_id"],
                    "retrieved_source_ids": retrieved_source_ids,
                    "answer": pipeline_result.answer,
                },
            )
        )

    metrics = aggregate_metrics(results)
    return metrics, results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", default=str(REPO_ROOT / "eval" / "testset.jsonl"))
    parser.add_argument("--out", default=str(REPO_ROOT / "eval" / "results" / "latest.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--experiment", default="rag-eval", help="MLflow experiment name to log the run under."
    )
    args = parser.parse_args()

    testset_path = Path(args.testset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generator = "stub" if rag_core.use_stub_generator() else "ollama"
    print(f"running eval: testset={testset_path} top_k={args.top_k} generator={generator}")

    mlflow.set_tracking_uri(f"file://{REPO_ROOT / 'mlruns'}")
    mlflow.set_experiment(args.experiment)

    start = time.time()
    metrics, results = run(testset_path, args.top_k)
    wall_seconds = time.time() - start

    with mlflow.start_run(run_name=f"eval-{generator}"):
        mlflow.log_param("top_k", args.top_k)
        mlflow.log_param("generator", generator)
        mlflow.log_param("embedding_model", rag_core.EMBEDDING_MODEL_NAME)
        mlflow.log_param("qdrant_collection", rag_core.QDRANT_COLLECTION)
        mlflow.log_param("n_questions", metrics["n_questions"])
        mlflow.log_metric("retrieval_recall_at_k", metrics["retrieval_recall_at_k"])
        mlflow.log_metric("answer_quality_avg", metrics["answer_quality_avg"])
        mlflow.log_metric("wall_seconds", wall_seconds)

        per_question_path = out_path.parent / "per_question.json"
        per_question_path.write_text(
            json.dumps([r.__dict__ for r in results], indent=2), encoding="utf-8"
        )
        mlflow.log_artifact(str(per_question_path))

    output = {
        "generator": generator,
        "top_k": args.top_k,
        **metrics,
    }
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))
    print(f"wrote results to {out_path}")

    n_failed_retrieval = sum(1 for r in results if not r.retrieval_hit)
    if n_failed_retrieval:
        print(f"note: {n_failed_retrieval}/{len(results)} questions missed retrieval@{args.top_k}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
