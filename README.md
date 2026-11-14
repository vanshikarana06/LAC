# ⚖️ Legal Advisor AI

> An agentic RAG (Retrieval-Augmented Generation) system that makes Indian law accessible to ordinary citizens — factory workers, women denied maternity benefits, consumers cheated by companies — who need legal guidance but cannot afford a lawyer.

**🌐 Live Demo:** [https://lac-oebg.onrender.com/](https://lac-oebg.onrender.com/)

---

## 🏗️ Architecture

A **LangGraph stateful workflow**, now with **9 nodes and a self-correction cycle** — an LLM-driven planning layer replaces pure keyword routing for ambiguous questions, and a critic step reviews generated answers and can trigger a retry.

```
User Question
      ↓
[node_load_history]        — Load conversation from SQLite
      ↓
[node_classify_question]   — Route: simple / scenario / multi_hop / compound
      ↓
[node_rephrase_query]      — Contextualize follow-up questions
      ↓
[node_retrieve_chunks]     — Semantic similarity search via ChromaDB (k=6, k=10 on retry)
      ↓
[node_check_threshold]     — Score-based routing
      ↓
   ┌───────────────────────────────────────────────────────┐
   │  score ≥ 0.32            score < 0.32, ambiguous       │
   │  📚 Legal DB             or fails keyword gate         │
   │       ↓                  🧭 [node_plan_action]         │
   │       ↓                  LLM decides: web_search /     │
   │       ↓                  clarify / refuse              │
   └───────────────────────────────────────────────────────┘
      ↓
[node_generate_answer]     — LLM generation with specialized prompt
      ↓
[node_critic_review]       — Groundedness check on web_search / low-confidence
      ↓                       answers only; retries retrieval once if flagged
      ↓ (retry loops back to node_retrieve_chunks)
[node_save_history]        — Persist to SQLite
      ↓
   Response
```

### Routing

| Route | Condition | Source Badge |
|---|---|---|
| ChromaDB | similarity score ≥ 0.32, unambiguous | Legal DB |
| Agentic planner → web search | ambiguous score, or vague/short question | Web Search |
| Agentic planner → clarification | question is legal-adjacent but too vague to answer usefully | Clarification |
| Agentic planner → refusal | question has nothing to do with law | Out of scope |

*Threshold (0.32) determined empirically by analyzing score distribution across representative queries — legal questions scored 0.40–0.55, out-of-scope scored below 0.30.*

---

## 🤖 Agentic Capabilities

Originally a fixed retrieve→generate pipeline with keyword-based routing. Two agentic upgrades replace parts of that fixed logic with LLM judgment:

### Query Planning (`node_plan_action`)
Deterministic keyword/threshold routing handles clear-cut cases cheaply (no LLM call). Genuinely ambiguous cases — vague questions ("what is the law"), short questions, or questions that fail the keyword gate despite being legitimate (e.g. "can I sue my neighbor for noise" — no exact keyword match, but clearly a legal question) — are handed to an LLM planner that decides between three actions: search the web, ask a clarifying question, or refuse. This closed a real gap where legitimate questions were being wrongly refused by rigid keyword matching, and added a capability the fixed pipeline never had: asking the user a clarifying question instead of forcing a weak answer.

### Self-Correction Loop (`node_critic_review`)
After generation, a critic step reviews answers from the two paths measured (via RAGAS evaluation, below) to be less reliable — web search results and low-confidence vector matches. High-confidence vector-search answers skip the critic entirely, so the added cost only applies where evaluation data showed real risk. If the critic flags low confidence, retrieval re-runs once with a wider `k` before generating again. Testing against a known-difficult case (a fabricated court case name) showed the loop reliably improves answer precision/confidence-calibration but — an honest, documented limitation — cannot fix answers limited by weak underlying search results unless the retry query itself is reformulated, which it currently is not.

---

## 📊 Evaluation

A [RAGAS](https://github.com/explodinggradients/ragas) harness scores the pipeline on faithfulness, answer relevancy, context precision, and context recall against hand-authored ground truth, split into answerable and should-refuse test categories (context precision/recall are not meaningful for refusal cases).

| Metric | Before (baseline) | After chunking fix + prompt tightening |
|---|---|---|
| Faithfulness | 0.831 | **0.910** |
| Context Precision | 0.716 | **0.900** |

A low-faithfulness result during evaluation was traced — not assumed — to a document-chunking defect: the splitter's separators didn't match the source PDFs' actual section-numbering format, fragmenting key clauses across chunk boundaries. Confirmed via direct database inspection (bypassing similarity search entirely), fixed with a structure-aware splitter (see Chunking Strategy below), and re-measured to confirm the improvement.

RAGAS also confirmed a hypothesis formed from manual testing: faithfulness on the web-search fallback path (0.500) is measurably lower than the vector-DB path (0.83+) — the basis for which path the self-correction critic reviews.

---

## 🐳 Deployment

Containerized with Docker; the vector database (3,901 chunks) is built directly into the image during `docker build` rather than relying on a persistent disk (not available on Render's free tier). API secrets required at build time (for embedding the knowledge base) are passed via Docker BuildKit secret mounts / Render Secret Files — never baked into image layers, unlike environment variables passed via `ARG`/`ENV`.

```bash
docker build --secret id=hf_token,env=HF_TOKEN -t legal-advisor-ai .
docker run -p 8001:8001 --env-file .env legal-advisor-ai
```

---

## ✨ Features

### Chat History Sidebar
Lists past conversations (via `/sessions`) with a preview of the first question; clicking one loads its full history (`/session/{id}/history`) back into the chat window without losing your current session.

### Advanced Question Understanding
- **Scenario detection** — classifies every question into `simple`, `scenario`, `multi_hop`, or `compound`
- **Specialized prompts** — 3 different prompt templates routed by question type

### Hybrid Retrieval
- Semantic similarity search against 19 indexed Indian legal Acts
- Web search fallback via Tavily, restricted to trusted Indian legal domains (`indiankanoon.org`, `sci.gov.in`, `livelaw.in`, `barandbench.com`)
- Retry logic with exponential backoff for all LLM API calls

### Session Persistence
- Two-layer durability: SQLite on server + localStorage on browser
- Normalized schema with indexed `session_id` lookups
- TTL-based cleanup (30-day retention policy)

### Production Safeguards
- Template injection vulnerability prevention (curly brace escaping in PDF content)
- Anti-hallucination prompts — LLM explicitly instructed not to fabricate case citations or add claims beyond retrieved context
- Token overflow prevention via history trimming and context truncation
- Negative similarity score clamping

---

## 📚 Knowledge Base

**19 Indian Acts** across 5 legal domains — 3,901 semantic chunks indexed:

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
| AI Workflow | LangGraph (9-node stateful workflow with self-correction cycle) |
| LLM | Groq — openai/gpt-oss-20b (configurable via GROQ_MODEL) |
| Embeddings | HuggingFace Inference API (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (similarity search) |
| Web Search | Tavily API |
| Evaluation | RAGAS (faithfulness, relevancy, precision, recall) |
| Observability | LangSmith tracing |
| Session Storage | SQLite + localStorage |
| Frontend | HTML / CSS / JavaScript |
| Containerization | Docker (BuildKit secrets) |
| Deployment | Render (Docker runtime) |

---

## 🚀 Local Setup

### Prerequisites
- Python 3.12+ (or Docker — see Deployment above)
- Free API keys: [Groq](https://console.groq.com) · [Tavily](https://tavily.com) · [HuggingFace](https://huggingface.co/settings/tokens)

### Installation

```bash
git clone https://github.com/vanshikarana06/LAC.git
cd LAC

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt

cp .env.example .env
# Add your API keys to .env
```

### Environment Variables

```env
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly_...
HF_TOKEN=hf_...
GROQ_MODEL=openai/gpt-oss-20b    # Note: On Render, configure this in Environment Variables dashboard
DB_PATH=data_vector_db
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=legal-advisor-ai
```

### Run

```bash
python -m uvicorn src.api.main:app --reload --port 8001
```

Visit: **http://localhost:8001**

### Rebuild Vector Database (if adding new PDFs)

```bash
# Drop new PDFs into data_raw/
Remove-Item -Recurse -Force data_vector_db   # Windows
# rm -rf data_vector_db                       # Mac/Linux

python src/ingestion/embedder.py
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `POST` | `/ask` | Submit a legal question |
| `GET` | `/ask/stream` | Submit a question via SSE streaming |
| `GET` | `/sessions` | List past conversations with previews |
| `GET` | `/session/{id}/history` | Load a specific conversation's full history |
| `DELETE` | `/session/{id}` | Clear conversation history |
| `GET` | `/health` | Server status + chunk count + active model |
| `GET` | `/acts` | List all indexed legal Acts |
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
Each pipeline step is an isolated, testable node. Conditional edges make routing explicit and declarative — including a real cycle (critic → retry → retrieval), not just a linear chain.

**Why a hybrid planner instead of an LLM call on every question?**
Clear-cut cases (obvious keyword matches, high/low similarity scores) are handled deterministically at zero extra LLM cost. Only genuinely ambiguous cases invoke the planner — cost and latency scale with actual uncertainty, not with every request.

**Why only critique web_search and low-confidence answers?**
RAGAS evaluation measured faithfulness specifically lower on the web-search path (0.500) than vector-search (0.83+). The critic targets the path with evidence of real risk, rather than adding latency to every response uniformly.

**Why SQLite over Redis for sessions?**
At current scale (single server, resume project), SQLite provides full persistence with zero infrastructure overhead. Redis would be the correct migration when horizontal scaling becomes necessary.

**Why restrict web search to Indian legal domains?**
Unrestricted web search returned US Supreme Court cases for Indian law queries. Domain restriction to `indiankanoon.org`, `sci.gov.in`, and `livelaw.in` ensures results are jurisdictionally relevant.

---

## 📊 Chunking Strategy

```python
RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=[
        r"\n\d{1,3}[A-Z]?\.\s",   # matches bare section numbering, e.g. "379. " or "124A. "
        "\nSection", "\nSECTION",
        "\nArticle", "\nARTICLE",
        "\nClause", "\nCLAUSE",
        "\n\n", "\n", ". ", " "
    ],
    is_separator_regex=True,
    keep_separator="start",
)
```

The primary separator matches each source document's actual numbering format directly (most Acts number provisions as bare digits, e.g. `379.`, rather than the word "Section"), so chunks split cleanly at the start of each legal provision instead of at arbitrary character-count boundaries.

---

## 🗺️ Roadmap

Guardrails (prompt-injection/jailbreak testing), tool use (citation verification), and a larger-scale evaluation set (30-50 queries) are planned next.

---

## ⚠️ Disclaimer

This application provides general legal information for educational purposes only. It is not a substitute for professional legal advice. Always consult a qualified lawyer for legal matters.