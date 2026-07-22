"""
Searches every chunk in the IPC-only test database for the literal text
"379" to find exactly which chunk(s) contain Section 379, what page
they're on, and how clean the chunk boundary is — without relying on
similarity search ranking at all.
"""

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
import dotenv

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def main():
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN
    )
    db = Chroma(persist_directory="data_vector_db_test", embedding_function=embeddings)

    all_data = db.get(include=["metadatas", "documents"])

    print(f"Searching {len(all_data['documents'])} total chunks for '379'...\n")

    found = 0
    for doc_text, meta in zip(all_data["documents"], all_data["metadatas"]):
        if "379" in doc_text:
            found += 1
            page = meta.get("page")
            print(f"--- MATCH #{found}  page={page} ---")
            print(doc_text)
            print()

    if found == 0:
        print("No chunk contains '379' at all. Section 379 text may have been")
        print("dropped during PDF extraction or text cleaning — worth checking")
        print("the raw PDF page directly.")
    else:
        print(f"\nTotal chunks containing '379': {found}")



if __name__ == "__main__":
    main()