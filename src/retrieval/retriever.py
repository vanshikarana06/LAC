"""
retriever.py – thin wrapper around ChromaDB for similarity search.
Used by the API but also importable from notebooks.
"""
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
 
 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
 
 
class LegalRetriever:
    def __init__(self, db_path: str = "data_vector_db", k: int = 5):
        self.k = k
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
        )
 
    def get_retriever(self):
        return self.vectordb.as_retriever(search_kwargs={"k": self.k})
 
    def similarity_search(self, query: str):
        return self.vectordb.similarity_search(query, k=self.k)
 
    def doc_count(self) -> int:
        return self.vectordb._collection.count()
