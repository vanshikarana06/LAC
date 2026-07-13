"""
RAGAS Evaluation Runner — Legal Advisor AI

WHAT THIS DOES:
1. Manually runs the same startup logic FastAPI's lifespan() would run
   (this script never starts uvicorn, so vectordb/embeddings would
   otherwise never be initialized).
2. Runs each question in ragas_eval_dataset.py through your actual
   LangGraph pipeline, with a pause between questions to stay under
   Groq's 6000 TPM rate limit.
3. Scores faithfulness, answer_relevancy, context_precision, context_recall
   using RAGAS, with your existing Groq LLM and HuggingFace embeddings
   as the judge — using max_workers=1 and strictness=1 so RAGAS doesn't
   fire concurrent or multi-sample (n>1) calls Groq's API will reject.
4. Saves a CSV and prints a results table + averages.

BEFORE RUNNING — confirm this import path matches your project structure.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.main import (
    run_legal_advisor,
    llm,
    embeddings,
    app,
    lifespan,
)
from ragas import evaluate, RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from ragas_eval_dataset import EVAL_QUESTIONS


async def run_pipeline_for_question(question: str) -> dict:
    """Runs one question through the real graph and extracts
    answer + retrieved context, exactly as a live /ask request would."""
    result = await run_legal_advisor({
        "original_question": question,
        "session_id": "ragas-eval-session",
        "chat_history": [],
        "question_type": "",
        "standalone_question": "",
        "retrieved_docs": [],
        "similarity_score": 0.0,
        "context": "",
        "query_source": "",
        "answer": "",
    })

    contexts = [doc.page_content for doc in result.get("retrieved_docs", [])]
    if not contexts and result.get("context"):
        contexts = [result["context"]] if result["context"] else [""]
    if not contexts:
        contexts = [""]

    return {
        "answer": result["answer"],
        "contexts": contexts,
        "query_source": result.get("query_source", ""),
    }


async def build_eval_dataset():
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    query_sources = []

    for i, item in enumerate(EVAL_QUESTIONS):
        print(f"Running: {item['question']}")
        result = await run_pipeline_for_question(item["question"])

        questions.append(item["question"])
        answers.append(result["answer"])
        contexts_list.append(result["contexts"])
        ground_truths.append(item["ground_truth"])
        query_sources.append(result["query_source"])

        print(f"  -> query_source={result['query_source']}, "
              f"answer_preview={result['answer'][:80]!r}")

        if i < len(EVAL_QUESTIONS) - 1:
            print("  (pausing 15s to stay under Groq's rate limit...)")
            await asyncio.sleep(15)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })
    return dataset, query_sources


async def main():
    # vectordb/retriever are normally created by FastAPI's lifespan() when
    # uvicorn starts the app. This script never starts uvicorn, so we run
    # the same startup logic manually before touching the graph.
    async with lifespan(app):
        print("Building evaluation dataset by running questions through the live pipeline...\n")
        dataset, query_sources = await build_eval_dataset()

        print("\nRunning RAGAS evaluation (this calls the LLM several times per question — may take several minutes)...\n")

        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

        # strictness=1: default is 3, which asks the LLM for 3 variant
        # completions in ONE call (n=3). Groq's API rejects any n > 1 —
        # every answer_relevancy score was silently failing without this.
        metrics = [
            Faithfulness(),
            AnswerRelevancy(strictness=1),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ]

        # max_workers=1: default concurrency fires many scoring calls at
        # once, colliding with Groq's 6000 TPM limit and causing the
        # widespread TimeoutErrors from the last run. Scoring one at a
        # time is slower but reliable against a rate-limited API.
        run_config = RunConfig(max_workers=1, timeout=300)

        result = evaluate(
            dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
        )

        df = result.to_pandas()
        df["query_source"] = query_sources

        output_path = "ragas_results.csv"
        df.to_csv(output_path, index=False)

        # RAGAS 0.3.x may use different column names than older versions
        # (e.g. user_input instead of question). Print actual columns so
        # nothing below silently breaks on a naming mismatch again.
        print("\nActual result columns:", df.columns.tolist())

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        id_cols = [c for c in ["user_input", "question", "query_source"] if c in df.columns]
        metric_cols = [c for c in
                        ["faithfulness", "answer_relevancy",
                         "llm_context_precision_with_reference", "context_precision",
                         "llm_context_recall", "context_recall"]
                        if c in df.columns]
        print(df[id_cols + metric_cols].to_string(index=False))

        print("\n" + "-" * 60)
        print("AVERAGES (overall)")
        print("-" * 60)
        for metric in metric_cols:
            print(f"{metric}: {df[metric].mean():.3f}")

        print("\n" + "-" * 60)
        print("AVERAGES BY ROUTING PATH")
        print("-" * 60)
        print(df.groupby("query_source")[metric_cols].mean())

        print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())