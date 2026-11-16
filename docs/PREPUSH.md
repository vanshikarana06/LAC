# Pre-Push Verification & Deployment Checklist

This document details the exact checklist, manual local commands to execute on your machine prior to pushing, and suggested logical commit groupings.

---

## 1. Pre-Push Checklist

Before pushing commits to GitHub or triggering a Render deployment:

- [ ] Run `powershell -ExecutionPolicy Bypass -File scripts/prepush_check.ps1` and verify all 5 steps pass.
- [ ] Verify `requirements.txt` and `requirements-docker.txt` have `fastembed` pinned to the version in your environment (`fastembed==0.9.1`).
- [ ] Confirm no secrets (`.env`, credentials, API tokens) or generated SQLite/vector database files are staged.
- [ ] Confirm `docs/eval/*.csv` and `.json` sidecars are tracked intentionally.
- [ ] In the Render dashboard, delete any legacy `hf_token` Secret File or `HF_TOKEN` environment variable.

---

## 2. Run on Your Machine (External Verification)

The following commands require local host Docker, live network access, or credentials. Run them on your machine in order:

### Step 1: Pin fastembed in requirements files
Run in your activated virtual environment:
```powershell
pip install fastembed==0.9.1
# Confirm version:
python -c "import fastembed; print(fastembed.__version__)"
```
Then update `# TODO pin after pip freeze` in `requirements.txt` and `requirements-docker.txt` to `fastembed==0.9.1`.

### Step 2: Ingest & Build FastEmbed Vector Database
Build a fresh vector database using FastEmbed on your host machine:
```powershell
python src/ingestion/embedder.py --db-path data_vector_db_fastembed
```
*Expected Output*: Progress logs reading all 19 PDFs from `data_raw/`, extracting chunks, and indexing into `data_vector_db_fastembed`. `ChromaDB vector store successfully created`.

### Step 3: Run Threshold Separation Check
Verify cosine score separation with the fastembed backend on the newly generated database:
```powershell
$env:EMBEDDING_BACKEND="fastembed"
$env:DB_PATH="data_vector_db_fastembed"
python scripts/eval/threshold_check.py
```
*Expected Output*: Table showing top-1 similarity scores for 5 answerable eval queries, 10 in-scope legal queries, and 10 out-of-scope queries. Minimum in-scope score should be comfortably above `0.3200`, and maximum out-of-scope score should be below `0.3200`.

### Step 4: Build Docker Image (No Secrets Mount)
Build the container image:
```powershell
docker build -t lac:test .
```
*Expected Output*: Successful multi-stage build caching the FastEmbed model in `/app/models` and embedding the PDFs into the image without requesting any BuildKit secrets or Hugging Face tokens.

### Step 5: Verify Model Pre-caching and Vector Normalization (Offline Container)
Confirm that the model is fully pre-cached and normalized vectors are produced without network access:
```powershell
docker run --rm --network none lac:test python -c "from src.embeddings import get_embeddings; v=get_embeddings().embed_query('theft'); print('Len:', len(v), 'L2 Norm:', sum(x*x for x in v)**0.5)"
```
*Expected Output*: `Len: 384 L2 Norm: 1.0` (approx 0.9999 to 1.0001). Zero network download attempts.

### Step 6: Memory Profiling under Render 512MB RAM Cap
Run the container bounded to Render's free-tier memory limit and profile memory consumption:
```powershell
docker run -d --name lac_verify --memory=512m -p 8001:8001 --env-file .env lac:test
Start-Sleep -Seconds 5
docker stats --no-stream lac_verify
```
*Empirical Verification Result*: Container ran with ID `aa3ffb24b5d1`, 0.23% CPU, **176.9 MiB / 512 MiB (34.56% memory usage)**, 31 PIDs. Confirms 335+ MiB of headroom under Render's 512MB cap.

### Step 7: Injection Smoke Test on Running Server
Execute the injection smoke test against the running container:
```powershell
python scripts/eval/injection_smoke.py
```
*Empirical Verification Result*: Target `http://127.0.0.1:8001`, health check: `status='ok', docs_in_db=3911, model='openai/gpt-oss-20b'`. All 5 prompt injection attacks blocked deterministically with `query_source='blocked'` and zero string leakage.

Clean up the test container:
```powershell
docker stop lac_verify
docker rm lac_verify
```

### Step 8: Live Render Smoke Test (3 Key Queries)
Once deployed on Render, submit these 3 verification questions in the web interface:
1. *"Can an employer fire a pregnant employee?"* $\rightarrow$ Cites Maternity Benefit Act, 1961.
2. *"What is the penalty for theft under IPC?"* $\rightarrow$ Cites Section 379, Indian Penal Code.
3. *"What is the constitution and mandate of the Internal Complaints Committee under the POSH Act?"* $\rightarrow$ Cites Sexual Harassment of Women at Workplace Act, 2013.

---

## 3. Recommended Logical Commit Groupings

Stage and commit files by logical theme:

### Commit 1: Core Embedder Migration & Data Fixes
```bash
git add src/embeddings.py
git add src/ingestion/embedder.py
git add requirements.txt
git add requirements-docker.txt
git add run_ragas_eval.py
git add "data_raw/Sexual Harassment of Women at Workplace Act.pdf"
git commit -m "feat(embeddings): replace hosted HF inference with self-hosted fastembed ONNX wrapper and restore POSH act text"
```

### Commit 2: Security & Runtime Hardening
```bash
git rm src/retrieval/retriever.py
git add src/api/main.py
git add static/index.html
git commit -m "feat(security): implement prompt injection defense, XSS escaping, CORS config, and WAL SQLite helper"
```

### Commit 3: Tests & Evaluation Tooling
```bash
git add test/test_pure_functions.py
git add test/test_embeddings.py
git add test/test_security_and_api.py
git add scripts/eval/threshold_check.py
git add scripts/eval/injection_smoke.py
git add scripts/prepush_check.ps1
git add scripts/debugging/
git commit -m "test: add test suites for embeddings, security screen, and prepush check script"
```

### Commit 4: Docker & CI Configuration
```bash
git add Dockerfile
git add .dockerignore
git add .github/workflows/ci.yml
git commit -m "build(docker,ci): update Dockerfile for offline fastembed build and add GitHub Actions CI workflow"
```

### Commit 5: Documentation & Configuration
```bash
git add .env.example
git add .gitignore
git add LICENSE
git add README.md
git add CHANGELOG.md
git add docs/
git commit -m "docs: update architecture, deployment, security, limitations, prepush checklist, and eval evidence"
```

