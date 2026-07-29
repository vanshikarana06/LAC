# syntax=docker/dockerfile:1.7
# The line above enables BuildKit's newer secret-mounting syntax used below.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

ENV RENDER=true

COPY data_raw/ ./data_raw/
COPY src/ingestion/ ./src/ingestion/

# Render's Secret Files (/etc/secrets/hf_token) or local BuildKit's
# --secret mount (/run/secrets/hf_token). Neither path is baked into
# any image layer either way — same security property as before, now
# working on both my machine and Render.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /etc/secrets/hf_token 2>/dev/null || cat /run/secrets/hf_token 2>/dev/null) \
    python src/ingestion/embedder.py

COPY src/api/ ./src/api/
COPY static/ ./static/

EXPOSE 8001

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]