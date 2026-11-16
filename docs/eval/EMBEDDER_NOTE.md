# Embedder Configuration & RAGAS Benchmark Note

## Model Architecture & Weight Lineage
- **Model Name:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense embeddings).
- **Benchmark Embedder (Local Dev):** The RAGAS evaluation runs recorded in `docs/eval/ragas_newsplit.csv` and `docs/eval/ragas_oldsplit.csv` were executed using the PyTorch-based `langchain_huggingface.HuggingFaceEmbeddings` implementation (`EMBEDDING_BACKEND="local-torch"`).
- **Production Embedder (FastEmbed ONNX):** Production on Render and default ingestion use the ONNX runtime build via `fastembed.TextEmbedding` (`FastEmbedEmbeddings` in `src/embeddings.py`, default `EMBEDDING_BACKEND="fastembed"`).

## Evaluation Status Disclosure
- The benchmark evaluations in this directory have **not been re-run** on the FastEmbed ONNX build.
- Because both implementations share the identical underlying model weights (`sentence-transformers/all-MiniLM-L6-v2`) and output L2-normalized 384-dimensional embeddings, retrieval behavior is semantically equivalent.
- Future benchmark re-runs with `run_ragas_eval.py` will record `"embedding_backend": "fastembed"` in their `.json` metadata sidecars.

