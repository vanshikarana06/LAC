# ⚖️ Legal Advisor AI
 
> A production-grade RAG (Retrieval-Augmented Generation) system that makes Indian law accessible to ordinary citizens — factory workers, women denied maternity benefits, consumers cheated by companies — who need legal guidance but cannot afford a lawyer.
 
**🌐 Live Demo:** [https://legal-advisor-ai-go7g.onrender.com](https://legal-advisor-ai-go7g.onrender.com)
 
---
 
## 🏗️ Architecture
 
This project implements a **LangGraph stateful workflow** with 7 discrete nodes, replacing a monolithic RAG function with a declarative, testable graph pipeline.
 
```
User Question
      ↓
[node_load_history]        — Load conversation from SQLite
      ↓
[node_classify_question]   — Route: simple / scenario / multi_hop / compound
      ↓
[node_rephrase_query]      — Contextualize follow-up questions
      ↓
[node_retrieve_chunks]     — Semantic search via ChromaDB (MMR, k=6)
      ↓
[node_check_threshold]     — Score-based three-way routing
      ↓
   ┌──────────────────────────────────────┐
   │  score ≥ 0.45     score < 0.45      │
   │  📚 Legal DB      + legal topic      │
   │       ↓           🌐 Web Search      │
   │       ↓           + not legal        │
   │       ↓           ⚠️ Out of scope    │
   └──────────────────────────────────────┘
      ↓
[node_generate_answer]     — LLM generation with specialized prompt
      ↓
[node_save_history]        — Persist to SQLite
      ↓
   Response
```
 
### Three-Way Routing
 
| Route | Condition | Source Badge |
|---|---|---|
| ChromaDB | similarity score ≥ 0.35 | Legal DB |
| Tavily web search | score < 0.35 + legal question | Web Search |
| Polite refusal | score < 0.35 + non-legal | Out of scope |

 *Threshold determined empirically by analyzing score distribution across 
representative queries — legal questions scored 0.45–0.55, out-of-scope 
scored below 0.38. Production threshold set to 0.35 to account for slight 
score differences between local and API-based embeddings.* 
---
 
## ✨ Features
 
### Phase 1 — Advanced Question Understanding
- **Scenario detection** — classifies every question into `simple`, `scenario`, `multi_hop`, or `compound`
- **Query decomposition** — breaks compound questions into sub-questions, answers each independently
- **Specialized prompts** — 3 different prompt templates routed by question type
### Phase 2 — Hybrid Retrieval
- **Semantic similarity search** against 19 indexed Indian legal Acts
- **Web search fallback** via Tavily, restricted to trusted Indian legal domains (`indiankanoon.org`, `sci.gov.in`, `livelaw.in`, `barandbench.com`)
- **Topic classification** — prevents non-legal questions from triggering web search
- **Retry logic** with exponential backoff for all LLM API calls
### Phase 3 — LangGraph Stateful Pipeline
- 7-node graph with `TypedDict` state for type-safe inter-node communication
- Conditional edges for declarative routing
- Replaces monolithic function with single-responsibility nodes
### Session Persistence
- **Two-layer durability:** SQLite on server + localStorage on browser
- Conversations survive server restarts, page refreshes, browser closes
- Normalized schema with indexed `session_id` lookups
- TTL-based cleanup (30-day retention policy)
### Production Safeguards
- Template injection vulnerability prevention (curly brace escaping in PDF content)
- Anti-hallucination prompts — LLM explicitly instructed not to fabricate case citations
- Token overflow prevention via history trimming before LLM calls
- Negative similarity score clamping
---
 
## 📚 Knowledge Base
 
**19 Indian Acts** across 5 legal domains — 3,739 semantic chunks indexed:
 
| Category | Acts |
|---|---|
| Constitutional | Constitution of India |
| Labour & Employment | Minimum Wages Act, Industrial Disputes Act, Factories Act, Contract Labour Act, EPF Act, Equal Remuneration Act, Workmen's Compensation Act |
| Women's Rights | Maternity Benefit Act, Sexual Harassment at Workplace Act (POSH), Domestic Violence Act |
| Criminal | Indian Penal Code 1860, Code of Criminal Procedure 1973 |
| Consumer & Civil | Consumer Protection Act 2019, RTI Act 2005, Transfer of Property Act, Indian Succession Act |
| Family | Hindu Marriage Act |
 
---
 
## 🛠️ Tech Stack
 
| Component | Technology |
|---|---|
| Backend | FastAPI (Python) |
| AI Workflow | LangGraph |
| LLM | Groq — LLaMA 3.1 8B Instant |
| Embeddings | HuggingFace Inference API (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (MMR search) |
| Web Search | Tavily API |
| Session Storage | SQLite + localStorage |
| Frontend | HTML / CSS / JavaScript |
| Deployment | Render (free tier) |
 
---
 
## 🚀 Local Setup
 
### Prerequisites
- Python 3.10+
- Free API keys: [Groq](https://console.groq.com) · [Tavily](https://tavily.com) · [HuggingFace](https://huggingface.co/settings/tokens)
### Installation
 
```bash
# 1. Clone the repo
git clone https://github.com/vanshikarana06/LAC.git
cd LAC
 
# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux
 
# 3. Install dependencies
pip install -r requirements.txt
 
# 4. Set up environment variables
cp .env.example .env
# Add your API keys to .env
```
 
### Environment Variables
 
```env
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly_...
HF_TOKEN=hf_...
DB_PATH=data_vector_db
```
 
### Run
 
```bash
python -m uvicorn src.api.main:app --reload --port 8001
```
 
Visit: **http://localhost:8001**
 
### Rebuild Vector Database (if adding new PDFs)
 
```bash
# Drop new PDFs into data_raw/
# Delete old database
Remove-Item -Recurse -Force data_vector_db   # Windows
# rm -rf data_vector_db                       # Mac/Linux
 
# Rebuild
python src/ingestion/embedder.py
```
 
---
 
## 🔌 API Reference
 
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `POST` | `/ask` | Submit a legal question |
| `GET` | `/health` | Server status + chunk count + active model |
| `GET` | `/acts` | List all indexed legal Acts |
| `DELETE` | `/session/{id}` | Clear conversation history |
| `GET` | `/docs` | Auto-generated Swagger UI |
 
### Request
 
```json
POST /ask
{
  "question": "What if my employer refuses to pay gratuity after 5 years?",
  "session_id": "optional-existing-session-uuid"
}
```
 
### Response
 
```json
{
  "answer": "Under the Payment of Gratuity Act, 1972...",
  "sources": [{ "source": "THE PAYMENT OF GRATUITY ACT.pdf", "page": 5 }],
  "session_id": "a3f2-bc12-...",
  "question_type": "scenario",
  "query_source": "vector_search"
}
```
 
---
 
## 🧠 Key Engineering Decisions
 
**Why LangGraph over a monolithic function?**
Each pipeline step is an isolated, testable node. Conditional edges make routing explicit and declarative. The graph can be extended with new nodes (e.g., clarification questions, user feedback) without modifying existing nodes.
 
**Why SQLite over Redis for sessions?**
At current scale (single server, resume project), SQLite provides full persistence with zero infrastructure overhead. Redis would be the correct migration when horizontal scaling becomes necessary.
 
**Why empirical threshold tuning?**
The 0.45 similarity threshold was determined by testing representative queries and identifying the natural score gap between in-domain legal questions (0.45–0.55) and out-of-scope questions (below 0.38) — not by guessing.
 
**Why restrict web search to Indian legal domains?**
Unrestricted web search returned US Supreme Court cases for Indian law queries. Domain restriction to `indiankanoon.org`, `sci.gov.in`, and `livelaw.in` ensures results are jurisdictionally relevant.
 
---
 
## 📊 Chunking Strategy
 
```python
RecursiveCharacterTextSplitter(
    chunk_size=1500,      # captures full legal clauses
    chunk_overlap=200,    # prevents mid-sentence splits
    separators=[
        "\nSection", "\nSECTION",
        "\nArticle", "\nARTICLE",
        "\nClause",  "\n\n", "\n"
    ]
)
```
 
Legal-aware separators ensure chunks split at section boundaries rather than mid-clause.
 
---
 
## ⚠️ Disclaimer
 
This application provides general legal information for educational purposes only. It is not a substitute for professional legal advice. Always consult a qualified lawyer for legal matters.

