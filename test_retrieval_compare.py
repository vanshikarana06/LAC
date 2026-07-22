
"""
Compares retrieval for "What is the punishment for theft under IPC?"
between your REAL database (old chunking) and the TEST database (new,
structure-aware chunking). Run this after test_ingest_rechunk.py finishes.
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import dotenv

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

QUESTION = "What is the punishment for theft under IPC?"


def show_results(label, db_path):
    print(f"\n{'=' * 70}")
    print(f"{label}  (db: {db_path})")
    print('=' * 70)

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN
    )
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)

    docs_with_scores = db.similarity_search_with_relevance_scores(QUESTION, k=6)

    for i, (doc, score) in enumerate(docs_with_scores):
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:600].replace("\n", " ")
        print(f"[{i+1}] score={score:.4f}  page={page}")
        print(f"    {preview}...")
        print()


def main():
    show_results("OLD CHUNKING (your real production DB)", "data_vector_db")
    show_results("NEW CHUNKING (structure-aware, test DB, IPC only)", "data_vector_db_test")

    print("\n" + "=" * 70)
    print("What to check: does page 174 (Section 379, the actual theft")
    print("punishment text) appear in the NEW results but not the OLD ones?")
    print("Also check whether NEW chunks look cleaner — each one starting")
    print("cleanly at a section number, instead of mid-sentence fragments.")
    print("=" * 70)


if __name__ == "__main__":
    main()