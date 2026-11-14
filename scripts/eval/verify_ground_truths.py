"""
Verification script for RAGAS ground-truth checklist.
For each of the 5 answerable questions, prints the ground truth and the top 3
retrieved chunks from ChromaDB (source, page, score, content preview).
"""

import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ragas_eval_dataset import ANSWERABLE_QUESTIONS
from src.api.main import app, lifespan


async def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs", "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "ground_truth_check.txt")

    lines = []
    lines.append("=" * 80)
    lines.append("GROUND TRUTH VERIFICATION CHECKLIST (Legal Advisor AI)")
    lines.append("=" * 80)
    lines.append("Review each ground truth against the retrieved knowledge base chunks below.\n")

    async with lifespan(app):
        from src.api.main import vectordb

        if vectordb is None:
            raise RuntimeError("VectorDB could not be initialized.")

        for i, item in enumerate(ANSWERABLE_QUESTIONS, 1):
            q = item["question"]
            gt = item["ground_truth"]

            lines.append("-" * 80)
            lines.append(f"QUESTION #{i}: {q}")
            lines.append(f"GROUND TRUTH:\n{gt}\n")
            lines.append("TOP 3 RETRIEVED CHUNKS (by embedding ground truth):")

            docs_with_scores = vectordb.similarity_search_with_relevance_scores(gt, k=3)
            for rank, (doc, score) in enumerate(docs_with_scores, 1):
                src = os.path.basename(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page", "?")
                preview = doc.page_content.strip().replace("\n", " ")
                if len(preview) > 300:
                    preview = preview[:300] + "..."

                lines.append(f"  [{rank}] Score: {score:.4f} | Source: {src} | Page: {page}")
                lines.append(f"      {preview}")
            lines.append("")

    content = "\n".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(content)
    print(f"\nSaved checklist to: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())

