"""
rag_core.py — shared retrieval + generation pipeline used by both
service.py (the BentoML service) and eval/run_eval.py (the eval harness).

Centralizing this here means the eval suite exercises the exact same
retrieval/prompt/generation code path the live service uses (minus the
Ollama call, which CI replaces with a deterministic stub — see
``generate_stub`` and the ``GENERATOR`` env var).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import requests
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "rag_docs")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context below.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

_embedder = None
_qdrant_client = None


def get_embedder():
    """Lazily load the sentence-transformers embedding model (kept lazy so
    importing this module — e.g. for the CI stub generation path — never
    pulls in torch/sentence-transformers unless embeddings are actually
    needed)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client


@dataclass
class RetrievedChunk:
    source_id: str
    source_file: str
    chunk_index: int
    text: str
    score: float


@dataclass
class PipelineResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    retrieval_score: float = 0.0
    latency_ms: float = 0.0


def embed_query(question: str) -> list[float]:
    vector = get_embedder().encode(question, normalize_embeddings=True)
    return vector.tolist()


def retrieve(question: str, top_k: int = 5, collection: str = QDRANT_COLLECTION) -> list[RetrievedChunk]:
    vector = embed_query(question)
    client = get_qdrant_client()
    # query_points is the current qdrant-client API (search() is deprecated
    # as of qdrant-client 1.10+).
    response = client.query_points(collection_name=collection, query=vector, limit=top_k)
    hits = response.points
    return [
        RetrievedChunk(
            source_id=hit.payload.get("source_id", ""),
            source_file=hit.payload.get("source_file", ""),
            chunk_index=hit.payload.get("chunk_index", -1),
            text=hit.payload.get("text", ""),
            score=float(hit.score),
        )
        for hit in hits
    ]


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(f"[{c.source_id}] {c.text}" for c in chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_stub(question: str, chunks: list[RetrievedChunk]) -> tuple[str, int]:
    """Deterministic, dependency-free "generation" used when CI=true (or
    GENERATOR=stub). It does NOT call any LLM — it template-joins the
    retrieved chunk text so the eval harness can score retrieval quality and
    exercise the full request/response/scoring pipeline without needing
    Ollama in CI. See README: "What CI validates vs. what requires local
    Ollama."
    """
    if not chunks:
        answer = "I don't know — no relevant context was retrieved."
    else:
        snippets = " ".join(c.text for c in chunks)
        answer = f"Based on the retrieved context: {snippets}"
    return answer, len(answer.split())


def generate_ollama(prompt: str, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST) -> tuple[str, int]:
    """Call Ollama's HTTP generation API. Requires ``ollama serve`` running
    locally (or reachable at OLLAMA_HOST) with ``model`` pulled."""
    resp = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    payload = resp.json()
    answer = payload.get("response", "").strip()
    tokens_generated = payload.get("eval_count", len(answer.split()))
    return answer, tokens_generated


def use_stub_generator() -> bool:
    """CI (and any environment without Ollama available) sets CI=true or
    GENERATOR=stub to use the deterministic stub instead of calling Ollama.
    """
    if os.environ.get("GENERATOR", "").lower() == "stub":
        return True
    if os.environ.get("GENERATOR", "").lower() == "ollama":
        return False
    return os.environ.get("CI", "").lower() == "true"


def answer_question(question: str, top_k: int = 5) -> tuple[PipelineResult, list[RetrievedChunk], int]:
    """Run the full retrieve -> prompt -> generate pipeline for one question.

    Returns (PipelineResult matching the /query response contract,
    the raw retrieved chunks (used by the eval harness for recall@k),
    tokens_generated (for the rag_tokens_generated_total metric)).
    """
    start = time.perf_counter()
    chunks = retrieve(question, top_k=top_k)

    if use_stub_generator():
        answer, tokens_generated = generate_stub(question, chunks)
    else:
        prompt = build_prompt(question, chunks)
        answer, tokens_generated = generate_ollama(prompt)

    latency_ms = (time.perf_counter() - start) * 1000
    retrieval_score = chunks[0].score if chunks else 0.0

    result = PipelineResult(
        answer=answer,
        sources=[
            {
                "source_id": c.source_id,
                "source_file": c.source_file,
                "chunk_index": c.chunk_index,
                "score": c.score,
            }
            for c in chunks
        ],
        retrieval_score=retrieval_score,
        latency_ms=latency_ms,
    )
    return result, chunks, tokens_generated
