#!/usr/bin/env python3
"""
ingest.py — chunk the sample document corpus, embed it, and upsert into Qdrant.

Usage:
    python ingest.py [--docs-dir data/sample_docs] [--collection rag_docs]

Environment variables (see README for the full contract):
    QDRANT_URL         default: http://localhost:6333
    QDRANT_COLLECTION  default: rag_docs

This script is intentionally dependency-light and side-effect-obvious: it
reads every ``*.md`` file under ``--docs-dir``, splits each into overlapping
word-based chunks, embeds the chunks with a local sentence-transformers
model, and upserts them into a Qdrant collection (recreating the collection
so re-running ingest.py is idempotent).
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # native output size of all-MiniLM-L6-v2

DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "rag_docs")


def chunk_text(text: str, chunk_words: int = 120, overlap_words: int = 30) -> list[str]:
    """Split ``text`` into overlapping chunks of ~``chunk_words`` words.

    Word-based chunking keeps this dependency-free (no tokenizer needed) and
    is plenty precise for the short markdown docs in this corpus. Overlap
    keeps a sentence that straddles a chunk boundary retrievable from either
    side of the split.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text.strip()]

    chunks = []
    start = 0
    step = max(chunk_words - overlap_words, 1)
    while start < len(words):
        window = words[start : start + chunk_words]
        chunks.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks


def iter_source_files(docs_dir: Path) -> Iterable[Path]:
    for path in sorted(docs_dir.glob("*.md")):
        yield path
    for path in sorted(docs_dir.glob("*.txt")):
        yield path


def build_points(docs_dir: Path, model: SentenceTransformer) -> list[qmodels.PointStruct]:
    points: list[qmodels.PointStruct] = []
    for source_path in iter_source_files(docs_dir):
        source_id = source_path.stem  # e.g. "01_pyproject_basics"
        text = source_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        if not chunks:
            continue
        embeddings = model.encode(chunks, normalize_embeddings=True)
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}::{idx}"))
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={
                        "source_id": source_id,
                        "source_file": source_path.name,
                        "chunk_index": idx,
                        "text": chunk,
                    },
                )
            )
    return points


def ensure_collection(client: QdrantClient, collection: str, dim: int) -> None:
    """(Re)create the collection so re-running ingest.py is idempotent."""
    if client.collection_exists(collection_name=collection):
        client.delete_collection(collection_name=collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-dir",
        default=str(Path(__file__).parent / "data" / "sample_docs"),
        help="Directory of source markdown/text docs to ingest.",
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_dir():
        print(f"error: docs dir not found: {docs_dir}", file=sys.stderr)
        return 1

    print(f"loading embedding model {EMBEDDING_MODEL_NAME!r} ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"chunking + embedding docs from {docs_dir} ...")
    points = build_points(docs_dir, model)
    if not points:
        print(f"error: no chunks produced from {docs_dir}", file=sys.stderr)
        return 1
    print(f"produced {len(points)} chunks from "
          f"{len(list(iter_source_files(docs_dir)))} source docs")

    print(f"connecting to Qdrant at {args.qdrant_url} ...")
    client = QdrantClient(url=args.qdrant_url)

    print(f"(re)creating collection {args.collection!r} (dim={EMBEDDING_DIM}) ...")
    ensure_collection(client, args.collection, EMBEDDING_DIM)

    print(f"upserting {len(points)} points ...")
    client.upsert(collection_name=args.collection, points=points)

    count = client.count(collection_name=args.collection, exact=True).count
    print(f"done. collection {args.collection!r} now has {count} points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
