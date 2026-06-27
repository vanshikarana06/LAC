# ⚖️ Legal Advisor AI
 
A Retrieval-Augmented Generation (RAG) system for querying Indian legal documents via a clean web interface.
 
---
 
## 📁 Project Structure
 
```
LEGAL-ADVISOR-AI/
├── data_raw/               ← Your PDF files go here (already have these)
├── data_vector_db/         ← ChromaDB persisted here (already built)
├── static/
│   └── index.html          ← ✅ Frontend (HTML/CSS/JS)
├── src/
│   ├── api/
│   │   └── main.py         ← ✅ FastAPI backend
│   ├── ingestion/
│   │   └── embedder.py     ← ✅ Already done
│   └── retrieval/
│       └── retriever.py    ← ✅ ChromaDB wrapper
├── .env                    ← Create from .env.example
├── requirements.txt        ← ✅ Full dependencies
└── README.md
```
 
---
 
## 🚀 Setup
 
### 1. Install dependencies
```bash
pip install -r requirements.txt
```
 
### 2. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your HuggingFace token
```
Get a free HuggingFace token at: https://huggingface.co/settings/tokens
 
### 3. Make sure your vector DB is built
If you haven't run the embedder yet:
```bash
python src/ingestion/embedder.py
```
 
### 4. Run the server
```bash
# From the project root:
uvicorn src.api.main:app --reload --port 8000
```
 
### 5. Open the app
Visit: **https://legal-advisor-ai-go7g.onrender.com/**
 
---
 
## 🔌 API Endpoints
 
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Serves the web UI |
| `POST` | `/ask`   | Ask a legal question |
| `GET`  | `/health`| DB status & chunk count |
| `GET`  | `/acts`  | List all loaded legal acts |
| `GET`  | `/docs`  | Auto-generated Swagger UI |
 
### Example `/ask` request
```json
POST /ask
{
  "question": "What are the maternity leave entitlements?"
}
```
 
### Example response
```json
{
  "answer": "Under the Maternity Benefit Act...",
  "sources": [
    { "source": "THE MATERNITY BENEFIT ACT.pdf", "page": 3 }
  ]
}
```
 
---
 
## 🤖 Switching the LLM
 
The default uses **Mistral-7B via HuggingFace Hub** (free). To switch:
 
### Option A – Different HuggingFace model
In `src/api/main.py`, change:
```python
LLM_REPO_ID = "google/flan-t5-xxl"   # smaller, faster
```
 
### Option B – OpenAI GPT
```bash
pip install langchain-openai openai
```
In `src/api/main.py`, replace the `llm =` block:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
```
And add `OPENAI_API_KEY=sk-...` to your `.env`.
 
---
 
## 💡 Tips
 
- **Slow first response?** HuggingFace Hub cold-starts models. The second query is faster.
- **Better accuracy?** Increase `k` in `retriever.py` (more context chunks).
- **More PDFs?** Drop them in `data_raw/` and re-run `embedder.py`.
