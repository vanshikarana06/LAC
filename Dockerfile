# Base image: an official, minimal Linux distro with Python 3.12 already installed.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Set RENDER=true at BUILD time too — main.py's embeddings branch checks this,
# and we want the lightweight API-based path even during the build step below.
ENV RENDER=true

# Copy source PDFs and the ingestion script, then BUILD the vector database
# right here, during the image build. 
# NOTE: this requires GROQ_API_KEY / HF_TOKEN to be available at BUILD time,
# into the Docker build step. 
COPY data_raw/ ./data_raw/
COPY src/ingestion/ ./src/ingestion/
RUN python src/ingestion/embedder.py

# Now copy the rest of the application code.
COPY src/api/ ./src/api/
COPY static/ ./static/

EXPOSE 8001

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]