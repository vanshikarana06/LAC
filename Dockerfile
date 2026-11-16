FROM python:3.12-slim

WORKDIR /app

# Install production dependencies only (no torch, no transformers)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Pre-download FastEmbed model into cache at build time so runtime is fully offline
ENV FASTEMBED_CACHE_PATH=/app/models
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/app/models')"

# Ingest and build ChromaDB inside the image (data_vector_db is gitignored)
COPY data_raw/ ./data_raw/
COPY src/embeddings.py ./src/embeddings.py
COPY src/ingestion/ ./src/ingestion/
RUN python src/ingestion/embedder.py

# Copy application source code and static assets
COPY src/api/ ./src/api/
COPY static/ ./static/

# Run as non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

# Health check using standard-library urllib without curl/wget dependencies
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, os; port = os.environ.get('PORT', '8001'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health')"

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
