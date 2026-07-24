"""
Test ingestion — structure-aware chunking, IPC ONLY, separate test database.

Does NOT touch your real data_vector_db. Builds data_vector_db_test instead,
containing only THE-INDIAN-PENAL-CODE-1860.pdf, re-chunked with a splitter
that actually matches the bare-number section format ("379." not "Section 379").

Run this, then run test_retrieval_compare.py to check whether the Section 379
theft-punishment text now retrieves correctly.
"""

import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
import dotenv

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

DATA_PATH = "data_raw"
TEST_DB_PATH = "data_vector_db_test"
TARGET_FILE = "THE-INDIAN-PENAL-CODE-1860.pdf"  # only re-chunk this one file for the test


def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def main():
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN
    )

    target_path = os.path.join(DATA_PATH, TARGET_FILE)
    if not os.path.exists(target_path):
        print(f"ERROR: {target_path} not found. Check the filename matches exactly.")
        return

    print(f"Loading: {TARGET_FILE}")
    loader = PyPDFLoader(target_path)
    raw_docs = loader.load()

    cleaned_documents = []
    for doc in raw_docs:
        cleaned_content = clean_text(doc.page_content)
        if len(cleaned_content) > 50:
            cleaned_documents.append(Document(page_content=cleaned_content, metadata=doc.metadata))

    print(f"Loaded {len(cleaned_documents)} pages")

    # The actual fix: highest-priority separator now matches the real
    # bare-number section format ("379. ", "124A. ") instead of only
    # "\nSection"/"\nArticle", which never appear in IPC's numbering style.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=[
            r"\n\d{1,3}[A-Z]?\.\s",   # "\n379. " or "\n124A. " — matches this document's real format
            "\nSection", "\nSECTION",
            "\nArticle", "\nARTICLE",
            "\nClause", "\nCLAUSE",
            "\n\n", "\n", ". ", " "
        ],
        is_separator_regex=True,
        keep_separator="start",
    )

    chunks = splitter.split_documents(cleaned_documents)
    print(f"Split into {len(chunks)} chunks (structure-aware)")

    print("\nBuilding TEST vector database (separate from your real one)...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=TEST_DB_PATH
    )
    print(f"Done. Test DB at: {TEST_DB_PATH}  ({len(chunks)} chunks, IPC only)")


if __name__ == "__main__":
    main()