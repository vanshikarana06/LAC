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

