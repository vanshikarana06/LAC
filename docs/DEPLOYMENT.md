# Production Deployment Guide (Render & Docker)

This guide covers production deployment on Render's free tier, environment configurations, container memory budgets, and maintenance procedures.

---

## 1. Architecture & Constraints on Render

Render's free tier operates under strict resource limits:
- **RAM**: 512 MB memory ceiling. Exceeding this limit causes an immediate out-of-memory (`OOMKilled`) termination.
- **Persistent Disks**: Not available on the free tier. Any files written outside the container image at runtime are lost upon service restart or deployment.
- **Cold Start**: Free-tier web services spin down after 15 minutes of inactivity. The first incoming request initiates a container boot, which typically takes 50–70 seconds.

### Why the Vector Database is Baked into the Image
Because free-tier instances have ephemeral local filesystems and no persistent disk mounts, `data_vector_db` is generated during `docker build` using `src/ingestion/embedder.py` and baked into the image layer alongside the ONNX FastEmbed model cache (`/app/models`). This guarantees instant, offline vector database readiness upon boot without runtime downloading or indexing.

### Elimination of PyTorch & Hugging Face Hosted Inference
Earlier versions encountered two production blockers:
1. **PyTorch OOM**: Container images bundling `torch` and `transformers` consumed >600 MB of resident set size (RSS), failing on Render's 512 MB cap.
2. **Hosted API Quota Exhaustion (402 Payment Required)**: Using `router.huggingface.co` hosted embeddings failed once monthly free quota credits depleted.

Both issues are resolved by using `fastembed` (`sentence-transformers/all-MiniLM-L6-v2` via ONNX Runtime). The entire runtime container footprint runs at **176.9 MiB RSS (34.56% of Render's 512 MB cap)**, leaving over 335 MiB of memory headroom.

---

## 2. Environment Variables

Configure these environment variables in the Render Dashboard (**Environment** tab):

| Variable | Required | Default | Purpose / Notes |
| :--- | :---: | :--- | :--- |
| `GROQ_API_KEY` | **Yes** | — | Authentication key for Groq Cloud API. Service will fail startup if missing. |
| `TAVILY_API_KEY` | **Yes** | — | API key for web search fallback when legal queries are out of corpus scope. |
| `GROQ_MODEL` | No | `openai/gpt-oss-20b` | Chat model ID on Groq. **Important**: If updating model, set explicitly in dashboard. |
| `ALLOWED_ORIGINS` | No | `http://localhost:8001,http://127.0.0.1:8001` | Comma-separated list of allowed CORS origins. Add your production Render URL (e.g. `https://your-service.onrender.com`). |
| `EMBEDDING_BACKEND` | No | `fastembed` | Model backend. Leave as `fastembed` for production. (`local-torch` is dev-only). |
| `LOG_LEVEL` | No | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `PORT` | Auto | `8001` | Injected dynamically by Render runtime; Dockerfile CMD respects `${PORT:-8001}`. |
| `LANGCHAIN_TRACING_V2` | No | `false` | Set to `true` to enable LangSmith traces. |
| `LANGCHAIN_API_KEY` | No | — | LangSmith API key for tracing. |
| `LANGCHAIN_PROJECT` | No | `legal-advisor-ai` | LangSmith project name. |

> [!WARNING]
> **Remove Old Secrets**: If migrating from previous deployments, delete any `hf_token` Secret File or `HF_TOKEN` environment variable from your Render dashboard. The build no longer relies on Hugging Face tokens.

---

## 3. Deployment Steps on Render

1. **Create Web Service**:
   - Go to [dashboard.render.com](https://dashboard.render.com).
   - Click **New +** $\rightarrow$ **Web Service**.
   - Connect your GitHub repository (`vanshikarana06/LAC`).
2. **Select Runtime**:
   - Environment: **Docker**
   - Region: Choose closest to target users (e.g., Oregon or Frankfurt).
   - Branch: `main`
   - Plan: **Free** (0.1 CPU, 512 MB RAM).
3. **Configure Environment Variables**:
   - Add `GROQ_API_KEY`, `TAVILY_API_KEY`, `GROQ_MODEL`, and `ALLOWED_ORIGINS` as described above.
4. **Deploy**:
   - Click **Create Web Service**.
   - Render will clone the repository, run `docker build` (installing `requirements-docker.txt`, pre-caching the FastEmbed model to `/app/models`, and building `data_vector_db`), and start the container as non-root `appuser`.

---

## 4. Updating the Knowledge Base

To add or update legal acts in the vector store:
1. Place new PDF files in `data_raw/`.
2. Commit and push to GitHub.
3. Render automatically triggers a new Docker build, re-running `embedder.py` to rebuild the baked-in vector store inside the new container image.

---

## 5. Local Docker Verification & Memory Profiling

Before deploying to Render, verify the container builds cleanly and runs within the 512 MB memory budget:

```powershell
# 1. Build image
docker build -t lac:test .

# 2. Run container with Render's 512m memory limit
docker run -d --name lac_verify --memory=512m -p 8001:8001 --env-file .env lac:test

# 3. Inspect resident memory consumption
docker stats --no-stream lac_verify

# 4. Clean up
docker stop lac_verify
docker rm lac_verify
```

