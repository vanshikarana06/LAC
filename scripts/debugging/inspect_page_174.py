"""
Bypasses similarity search entirely. Pulls every chunk whose metadata page
is 173, 174, or 175 directly from both databases, so we can see the raw
text on the page that should contain Section 379 — regardless of whether
similarity search ranks it highly or not.
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import dotenv

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def show_pages(label, db_path, target_pages):
    print(f"\n{'=' * 70}")
    print(f"{label}  (db: {db_path})")
    print('=' * 70)

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN
    )
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)

    # Pull everything, then filter by metadata page in Python — simplest
    # way to see raw stored content regardless of similarity ranking.
    all_data = db.get(include=["metadatas", "documents"])

    found_any = False
    for doc_text, meta in zip(all_data["documents"], all_data["metadatas"]):
        page = meta.get("page")
        if page in target_pages:
            found_any = True
            print(f"\n--- page={page} ---")
            print(doc_text[:600])
            print("...")

    if not found_any:
        print(f"No chunks found with page in {target_pages}")


def main():
    target_pages = [172, 173, 174, 175]

    show_pages("OLD CHUNKING (real production DB)", "data_vector_db", target_pages)
    show_pages("NEW CHUNKING (test DB, IPC only)", "data_vector_db_test", target_pages)

    print("\n" + "=" * 70)
    print("Looking for: '379. Punishment for theft' or similar text.")
    print("If it's not on any of these pages, Section 379 may be on a")
    print("different physical page than expected (PDF page numbering can")
    print("differ from the printed page number in the book).")
    print("=" * 70)


if __name__ == "__main__":
    main()