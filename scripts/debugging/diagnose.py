"""
Retrieval diagnostic — bypasses the whole graph, queries ChromaDB directly
for one question, and prints the raw top-6 chunks with their scores.

Run this 2-3 times in a row. If you get DIFFERENT chunks each time for the
exact same question, that proves retrieval itself is non-deterministic —
a serious finding. If you get the SAME chunks every time, retrieval is
stable, and the RAGAS eval run's odd result was a one-off (e.g. maybe a
different question's context leaked in — worth checking the eval script
for a bug instead).
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.main import app, lifespan


async def main():
    async with lifespan(app):
        # Import vectordb AFTER lifespan has run and set the global
        from src.api.main import vectordb

        question = "What is the punishment for theft under IPC?"
        print(f"Querying: {question!r}\n")

        docs_with_scores = vectordb.similarity_search_with_relevance_scores(question, k=10)

        for i, (doc, score) in enumerate(docs_with_scores):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"[{i+1}] score={score:.4f}  source={source}  page={page}")
            print(f"    {preview}...")
            print()


if __name__ == "__main__":
    asyncio.run(main())