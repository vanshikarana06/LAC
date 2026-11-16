# RAGAS Evaluation Documentation — Legal Advisor AI

## Overview & Methodology Disclosure

This directory contains evaluation benchmarks for the Legal Advisor AI pipeline using the [RAGAS](https://github.com/explodinggradients/ragas) framework.

### ⚠️ Honest Benchmark Caveats (Screener / Recruiter Notice)
- **Dataset Size:** The test set currently comprises only **7 hand-authored questions** (5 answerable legal inquiries and 2 should-refuse control cases). It represents an initial smoke test / sanity check, not a statistically power-tested validation benchmark.
- **Ground Truth Status:** The references in `ragas_eval_dataset.py` are hand-drafted approximations and are **unverified by accredited legal counsel**. They serve as directional references for development.
- **Self-Evaluation Bias:** The evaluation judge LLM (`run_ragas_eval.py`) uses the same model as the generation pipeline (`llm.model_name`), which introduces known self-preference bias. For production validation, cross-evaluation against frontier models (e.g. Claude 3.5 Sonnet or GPT-4o) is recommended.

---

## Evaluation Dataset Breakdown

| # | Question Type | Category | Target Evaluation Behavior |
|---|---|---|---|
| 1 | Simple Factual (IPC Theft) | Answerable | Retrieve Section 379 IPC from knowledge base; cite 3-year term & fine. |
| 2 | Simple Factual (Maternity Benefit) | Answerable | Retrieve Maternity Benefit Act 1961; cite 26 weeks (12 for 2+ children). |
| 3 | Simple Factual (Gratuity) | Answerable | Retrieve Payment of Gratuity Act 1972; cite 5-year eligibility criteria. |
| 4 | Scenario (Arrest Without Warrant) | Answerable | Retrieve CrPC Sections 41/41A; step-by-step rights on cognizable arrest. |
| 5 | Multihop (Cognizable vs Non-Cognizable) | Answerable | Synthesize CrPC definitions and police powers. |
| 6 | Hallucination Trap (Fake case name) | Should-Refuse | State that no verifiable Supreme Court case exists under this name. |
| 7 | Out-of-Scope (Commodity price) | Should-Refuse | Politely decline as outside Indian legal domain. |

---

## How to Run Evaluation

### Prerequisites
1. Ensure your `.env` contains:
   ```env
   GROQ_API_KEY=gsk_...
   TAVILY_API_KEY=tvly-...
   HF_TOKEN=hf_...
   GROQ_MODEL=openai/gpt-oss-20b
   DB_PATH=data_vector_db
   ```
2. Activate your virtual environment and install eval dependencies:
   ```bash
   pip install ragas datasets pandas
   ```

### Execution
Run the runner script from project root:
```bash
python run_ragas_eval.py
```

### Rate Limit Awareness
Groq free tier limits requests to 6,000 Tokens Per Minute (TPM). `run_ragas_eval.py` is configured with:
- `asyncio.sleep(15)` between pipeline questions.
- `RunConfig(max_workers=1, timeout=300)` for sequential evaluation judging.
- `strictness=1` on `AnswerRelevancy` to prevent rejected $n > 1$ variant calls.

---

## Ablation Benchmark: Structure-Aware vs. Standard-Fixed Chunking

Evaluated on 16-Nov-2026 with `openai/gpt-oss-20b` (Generation and Judge, 4096 judge max tokens).

### 1. Summary Comparison (Answerable Questions, N=5)

| Metric | Structure-Aware (`newsplit`) | Standard-Fixed (`oldsplit`) | Delta |
|---|:---:|:---:|:---:|
| **Context Precision** | **0.7500** (1 NaN) | **0.0000** (4 NaNs) | **+0.7500** |
| **Context Recall** | **0.6250** (1 NaN) | **0.0000** (4 NaNs) | **+0.6250** |
| **Answer Relevancy** | **0.6852** (0 NaNs) | **0.4157** (2 NaNs) | **+0.2695** |
| **Faithfulness** | **0.8556** (2 NaNs) | **0.9286** (3 NaNs) | -0.0730 |

### 2. NaN Analysis

| Metric | `ragas_newsplit.csv` (Total) | `ragas_oldsplit.csv` (Total) |
|---|:---:|:---:|
| Faithfulness | 2 / 7 | 5 / 7 |
| Answer Relevancy | 1 / 7 | 3 / 7 |
| Context Precision | 1 / 7 | 6 / 7 |
| Context Recall | 1 / 7 | 6 / 7 |
| **Total NaNs** | **5 / 28** | **20 / 28** |

*Root Cause of NaNs:* Due to Groq TPM limits (6,000 TPM), long context inputs in the RAGAS batch executor intermittently triggered `RateLimitError (429)` and timeouts. Standard-fixed chunking suffered significantly more timeouts (20 NaNs) because arbitrary character splits produced larger, noisier chunk sequences that repeatedly exhausted the token budget.

### 3. Key Observations & Takeaways
1. **Precision & Recall Gains:** Structure-aware chunking boundaries preserve statutory sections whole (e.g. IPC §379, Maternity Benefit Act §4/§5, Gratuity Act §4), resulting in a massive boost in Context Precision (+0.7500) and Context Recall (+0.6250).
2. **Q1 Retrieval Verification:** In `newsplit`, IPC §379 theft penalty was cleanly retrieved with Context Precision = 0.8333, whereas `oldsplit` retrieved unrelated CrPC text (Precision = 0.0000).


