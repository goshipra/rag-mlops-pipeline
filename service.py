"""
service.py — BentoML service exposing the RAG pipeline.

Endpoints (see README "Interface contract"):
    GET  /healthz  -> 200 {"status": "ok", ...}
    GET  /metrics  -> Prometheus text exposition format
    POST /query    -> {"question": str, "top_k": int = 5}
                       -> {"answer": str, "sources": [...],
                           "retrieval_score": float, "latency_ms": float}

Implementation note: BentoML's `@bentoml.api` decorator generates a
POST-only JSON-RPC-style endpoint per method. To get exact control over
HTTP verbs and response formats for /healthz (GET, JSON) and /metrics (GET,
Prometheus text) we mount a small FastAPI app onto the BentoML service via
`bentoml.asgi_app` instead of using three separate `@bentoml.api` methods.
This keeps the service a single BentoML-managed process (so `bentoml serve`,
containerization, and scaling all work normally) while giving us a plain,
predictable REST surface that matches the interface contract byte-for-byte.

Gotcha discovered while building this (see `ContractOverrideMiddleware`
below): BentoML's own HTTP app *unconditionally* registers its own
`/healthz` route (a plain "200 + blank body" k8s liveness probe, not JSON)
and, when its built-in Prometheus integration is enabled, its own
`/metrics` route too — both are added to the Starlette route table before
a `bentoml.asgi_app`-mounted app, so under plain routing those built-ins
always win and our FastAPI `/healthz`/`/metrics` handlers become
unreachable dead code. Verified this by hand: `bentoml serve service:svc`
+ `curl /healthz` returned BentoML's blank `text/plain` body, not our JSON,
until the middleware below was added. The fix is a small ASGI middleware
registered via `svc.add_asgi_middleware(...)` that short-circuits exactly
those two paths *before* BentoML's router ever sees the request — Starlette
middleware wraps the whole app, including routing, so this reliably wins
regardless of route registration order. `/query` has no such collision
(it isn't a BentoML system route) and is served normally by the mounted
FastAPI app.

Serving:
    bentoml serve service:svc          # binds 0.0.0.0:3000 by default
"""
from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

import bentoml
import rag_core

logger = logging.getLogger("rag_service")

# --------------------------------------------------------------------------
# Prometheus metrics (see README "Prometheus metrics" table for the full
# contract). Registered against the default global CollectorRegistry, which
# is correct for the single-process `bentoml serve` default used in this
# project; see README's "Known limitations" for the multi-worker note.
# --------------------------------------------------------------------------
RAG_REQUESTS_TOTAL = Counter(
    "rag_requests_total", "Total /query requests, labeled by outcome status.", ["status"]
)
RAG_REQUEST_LATENCY_SECONDS = Histogram(
    "rag_request_latency_seconds", "End-to-end /query latency in seconds."
)
RAG_RETRIEVAL_SCORE = Histogram(
    "rag_retrieval_score",
    "Top-1 retrieval similarity score per query (cosine similarity, 0-1).",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
RAG_TOKENS_GENERATED_TOTAL = Counter(
    "rag_tokens_generated_total", "Total tokens generated across all /query responses."
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class SourceChunk(BaseModel):
    source_id: str
    source_file: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    retrieval_score: float
    latency_ms: float


def healthz_payload() -> dict:
    """Liveness/readiness payload. Does not touch Qdrant/Ollama on the hot
    path — a healthy process response here means the service is up, not
    that its dependencies are reachable (dependency health shows up as
    5xx / status="error" on /query, tracked via rag_requests_total)."""
    return {
        "status": "ok",
        "qdrant_url": rag_core.QDRANT_URL,
        "qdrant_collection": rag_core.QDRANT_COLLECTION,
        "generator": "stub" if rag_core.use_stub_generator() else "ollama",
        "ollama_model": rag_core.OLLAMA_MODEL,
    }


def metrics_response_body() -> bytes:
    return generate_latest()


fastapi_app = FastAPI(title="rag-mlops-pipeline", version="0.1.0")


@fastapi_app.get("/healthz")
def healthz() -> dict:
    """Handler used by direct unit tests against `fastapi_app` (see
    tests/test_service.py). In real `bentoml serve` traffic this exact
    path is actually served by `ContractOverrideMiddleware` below, since
    BentoML's own built-in /healthz route would otherwise shadow it — see
    the module docstring "Gotcha" note. Kept here (sharing
    `healthz_payload()`) so the handler logic itself stays testable without
    booting a full BentoML server."""
    return healthz_payload()


@fastapi_app.get("/metrics")
def metrics() -> Response:
    """See `healthz` docstring above — same shadowing situation, same fix
    (`ContractOverrideMiddleware`)."""
    return Response(content=metrics_response_body(), media_type=CONTENT_TYPE_LATEST)


@fastapi_app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    status = "ok"
    try:
        result, _chunks, tokens_generated = rag_core.answer_question(
            request.question, top_k=request.top_k
        )
    except Exception:
        status = "error"
        logger.exception("query failed: question=%r top_k=%s", request.question, request.top_k)
        raise
    finally:
        RAG_REQUEST_LATENCY_SECONDS.observe(time.perf_counter() - start)
        RAG_REQUESTS_TOTAL.labels(status=status).inc()

    RAG_RETRIEVAL_SCORE.observe(result.retrieval_score)
    RAG_TOKENS_GENERATED_TOTAL.inc(tokens_generated)

    return QueryResponse(
        answer=result.answer,
        sources=[SourceChunk(**s) for s in result.sources],
        retrieval_score=result.retrieval_score,
        latency_ms=result.latency_ms,
    )


class ContractOverrideMiddleware:
    """ASGI middleware that serves our exact /healthz and /metrics contract,
    short-circuiting *before* BentoML's router runs.

    Why this exists: BentoML's HTTP app always registers its own
    `/healthz` route (plain 200, blank body — a generic k8s liveness probe)
    and, when its built-in Prometheus integration is on, its own
    `/metrics` route too. Those are added to the Starlette route table
    ahead of any `bentoml.asgi_app`-mounted app, so under normal routing
    they'd always win and our FastAPI handlers for those two paths would
    never actually run in `bentoml serve`. Starlette middleware wraps the
    *entire* app, including routing, so intercepting here reliably serves
    our contract responses regardless of what BentoML registers beneath.

    /query has no such collision and is left to the mounted FastAPI app.
    """

    def __init__(self, app) -> None:  # noqa: ANN001 - ASGI app, untyped by convention
        self.app = app

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] == "http" and scope["path"] in ("/healthz", "/metrics"):
            request = Request(scope, receive=receive)
            if request.method == "GET":
                if scope["path"] == "/healthz":
                    response = JSONResponse(healthz_payload())
                else:
                    response = Response(
                        content=metrics_response_body(), media_type=CONTENT_TYPE_LATEST
                    )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


@bentoml.service(name="rag-mlops-pipeline")
@bentoml.asgi_app(fastapi_app, path="/")
class RagService:
    """Thin BentoML wrapper around the FastAPI app above. All request
    handling lives in the plain FastAPI routes so the HTTP surface is exact
    and framework-independent; BentoML supplies process management,
    containerization (`bentoml build` / `bentoml containerize`), and
    deployment tooling on top."""

    def __init__(self) -> None:
        logger.info(
            "rag-mlops-pipeline starting: QDRANT_URL=%s QDRANT_COLLECTION=%s "
            "OLLAMA_HOST=%s OLLAMA_MODEL=%s generator=%s",
            rag_core.QDRANT_URL,
            rag_core.QDRANT_COLLECTION,
            rag_core.OLLAMA_HOST,
            rag_core.OLLAMA_MODEL,
            "stub" if rag_core.use_stub_generator() else "ollama",
        )


svc = RagService
svc.add_asgi_middleware(ContractOverrideMiddleware)
