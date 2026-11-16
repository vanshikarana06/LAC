# Known Limitations & Design Trade-offs

This document details explicit technical and domain limitations, known edge cases, and architectural constraints in Legal Advisor AI.

---

## 1. Statutory Corpus Scope & Criminal Law Transition

1. **Pre-2024 Criminal Statutes (IPC / CrPC / IEA)**:
   - The knowledge base indexes the historic criminal codes: Indian Penal Code, 1860 (IPC), Code of Criminal Procedure, 1973 (CrPC), and Indian Evidence Act, 1872.
   - It does **not** currently index the newly enacted criminal statutes effective July 1, 2024:
     - Bharatiya Nyaya Sanhita, 2023 (BNS)
     - Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)
     - Bharatiya Sakshya Adhiniyam, 2023 (BSA)
   - Inquiries on BNS/BNSS/BSA fall back to web search if similarity scoring identifies low corpus overlap.

2. **Statutes Only (No Case Law Corpus)**:
   - The primary vector database indexes statutory enactments only. It contains no judicial precedents, High Court rulings, or Supreme Court case law databases. Judicial interpretations rely on live web search fallback.

3. **External Gazette Ceilings (e.g., Gratuity Limit)**:
   - Certain statutory numbers (such as the current ₹20 lakh tax-free gratuity ceiling) were enacted via central government gazette notifications rather than amendments to the base text of the Payment of Gratuity Act, 1972 (which still prints ₹10 lakh in its historic PDF print). The model may cite the statutory ₹10 lakh limit unless augmented by web search.

4. **Typographical Filenames Kept for Backward Compatibility**:
   - Several PDF files in `data_raw/` contain typos (e.g. `rigjt_to_information_act_2005pdf.pdf`, `protection_of_women_from_domestic_voilance2005.pdf`, `consumer_protection_sct_2019.pdf`).
   - These filenames are intentionally retained because ChromaDB metadata and existing client citations directly key off `os.path.basename(source)`. Renaming them would break citation consistency without a full database re-indexing.

---

## 2. Evaluation Suite Limitations

1. **Small Pilot Sample ($N=5$)**:
   - The RAGAS benchmark dataset currently contains 7 questions (5 answerable, 2 unanswerable/out-of-scope). While this pilot provides initial comparative feedback on chunking strategies, it is too small to compute statistically significant confidence intervals.
2. **Judge = Generator (Self-Evaluation Bias)**:
   - The RAGAS judge model and the generation model are both `openai/gpt-oss-20b` (via Groq). Using the same LLM for both generation and judgment introduces self-preference bias in faithfulness and relevancy scoring.
3. **High NaN Rates in Standard Fixed Splitter**:
   - On the old fixed-size splitter, 4 out of 5 answerable questions generated NaN for context precision and context recall because the retrieved context was missing ground-truth statutory clauses.

---

## 3. Runtime & Infrastructure Trade-offs

1. **LLM Non-Determinism in Planning & Critic Cycles**:
   - While `temperature=0.2` reduces randomness, `node_plan_action` and `node_critic` are generative nodes whose reasoning paths can vary across identical queries.
2. **Static Retry Context**:
   - When the critic flags an ungrounded or incomplete answer, the graph retries answer generation with critic feedback, but does not reformulate or re-run the ChromaDB vector query.
3. **Groq Model Availability & Deprecation**:
   - Cloud inference relies on hosted models on Groq. Model deprecations (such as earlier deprecation of `llama-3.1-8b-instant`) require updating `GROQ_MODEL`. The server lifespan fails fast on startup with an actionable error if the configured model is unavailable.
4. **Single-Process SQLite Session Storage**:
   - Conversation history is stored in local SQLite (`session_store.db`) using WAL mode. This is optimized for single-instance container deployments (e.g. Render free tier). It does not support horizontal multi-replica deployments without moving to a managed PostgreSQL instance.
5. **Embedding Model Threshold Re-validation**:
   - The `SIMILARITY_THRESHOLD = 0.32` was empirically tuned using cosine relevance scores. Switching between embedding implementations (e.g., PyTorch vs ONNX FastEmbed) produces slightly different score distributions and requires verifying score separation via `scripts/eval/threshold_check.py`.

