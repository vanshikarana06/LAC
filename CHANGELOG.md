# Changelog

All notable changes to the Legal Advisor AI project, summarized from the git history and grouped by functional theme.

---

## [Unreleased] — Production Hardening & FastEmbed Migration

### Embeddings & Vector Search
- **Replaced Hosted API Embeddings with Self-Hosted FastEmbed**: Migrated from Hugging Face hosted endpoint (`router.huggingface.co`, which returned 402 Payment Required) to self-hosted ONNX Runtime embeddings using `fastembed.TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")`.
- **Zero PyTorch Footprint**: Completely eliminated `torch` and `transformers` dependencies from production dependencies, keeping container resident memory consumption well under 150 MB RAM (fitting Render's 512 MB free tier ceiling).
- **Backend Selection**: Added `EMBEDDING_BACKEND` environment variable (`fastembed` default; `local-torch` retained for legacy offline development).
- **Baked-in Model Cache**: Configured `FASTEMBED_CACHE_PATH=/app/models` in Docker build, downloading the model into the image so runtime never makes outbound network calls.
- **Threshold Verification**: Added `scripts/eval/threshold_check.py` to evaluate similarity score distribution and separation across answerable and out-of-scope queries.

### Security & Hardening
- **Prompt Injection Defense**: Implemented multi-tier defense:
  - 1,000-character input cap returning `400 Bad Request`.
  - Deterministic regex screening (`looks_like_injection()`) returning refusal with `query_source = "blocked"`.
  - XML tag isolation (`<user_question>` and `<context>`) in all system prompts with untrusted content instructions.
- **XSS Sanitization**: Sanitized all dynamic user inputs, error messages, and session history previews in `static/index.html` using `escapeHtml()`; migrated all inline event listeners (`onclick`, `onkeydown`, `oninput`) to standard `addEventListener`.
- **CORS Configuration**: Replaced permissive `allow_origins=["*"]` with configurable `ALLOWED_ORIGINS` environment variable (default: `localhost:8001`, `127.0.0.1:8001`).
- **SQLite Concurrency**: Standardized SQLite database access across all endpoints via `get_conn()` configuring `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000`.
- **Logging Sanitization**: Removed raw questions and context chunk payloads from `INFO` logs; introduced `LOG_LEVEL` environment variable.

### Agentic Architecture & Graph Pipeline
- **9-Node Multi-Turn Pipeline**: Implemented LangGraph workflow: history loading, query rephrasing, classification, MMR retrieval, similarity threshold check, agentic planner, Tavily web search fallback, answer generation, and critic risk gate.
- **Self-Correction Retry Loop**: Integrated `node_critic` with `is_critic_risky()` gate, reviewing ungrounded claims and triggering revisions up to 2 retries.
- **Model Validation on Startup**: Added startup verification against Groq Cloud API; raises actionable `RuntimeError` listing available chat models if configured `GROQ_MODEL` is unavailable.
- **Tavily Error Handling**: Wrapped web search fallback in try/except blocks handling timeouts and string-type responses safely.

### Containerization & CI
- **Hardened Dockerfile**: Multi-stage build on `python:3.12-slim` running as non-root `appuser`; pre-bakes vector database and FastEmbed model cache into image layers.
- **Port Dynamic Binding**: CMD uses `sh -c "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8001}"` to support Render's dynamic port assignment.
- **Native Healthcheck**: Implemented container healthcheck using Python's standard library `urllib` (no external `curl` needed).
- **GitHub Actions CI**: Added `.github/workflows/ci.yml` running syntax/runtime error checks with Ruff (`--select E9,F63,F7,F82`) and the 46-test unit/integration test suite.

---

## [Historical Commits]

### Evaluation & Chunking Ablation
- `853f016`, `b8c7435`, `aa5842f`: Implemented structure-aware hierarchical chunking ablation against fixed-size chunking; generated RAGAS benchmark datasets and verified ground truth alignments.

### Model Upgrades & Maintenance
- `7ae20ac`: Updated default Groq model to `openai/gpt-oss-20b` after deprecation of legacy Llama models.
- `e6f6382`: Removed unused Blueprint configuration in favor of dashboard-managed deployment.
- `be60b2b`: Cleaned up git tracking for generated CSV files.

