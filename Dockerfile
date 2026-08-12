# Dockerfile — builds the BentoML RAG service image.
#
# This container runs the service only. It expects Qdrant and Ollama to be
# reachable over the network (see docker-compose.yml, which wires Qdrant as
# a sibling container and expects Ollama on the host).
FROM python:3.11-slim

WORKDIR /app

# System deps: none beyond build-essentials-lite are required at runtime;
# sentence-transformers/torch wheels are pre-built (CPU) for this image's
# platform via pip.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_core.py service.py ingest.py ./
COPY data/ ./data/

# Env var defaults match the interface contract (README "Environment
# variables"); override at `docker run`/compose time as needed.
ENV QDRANT_URL=http://qdrant:6333 \
    QDRANT_COLLECTION=rag_docs \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    OLLAMA_MODEL=llama3.2 \
    MLFLOW_TRACKING_URI=file:///app/mlruns

EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -sf http://localhost:3000/healthz || exit 1

CMD ["bentoml", "serve", "service:svc", "--host", "0.0.0.0", "--port", "3000"]
