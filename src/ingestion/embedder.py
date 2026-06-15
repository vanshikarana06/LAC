import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import dotenv
dotenv.load_dotenv()
 
HF_TOKEN = os.getenv("HF_TOKEN")
 
class LegalIngestor:
    def __init__(self, data_path, db_path):
        self.data_path = data_path
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
    def clean_text(self, text):
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove page numbers like "Page 1 of 50" or just "1"
        text = re.sub(r'Page \d+ of \d+', '', text)
        # Remove standalone numbers (PDF page number artifacts)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        # Remove excessive whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
 
    def process_pdfs(self):
        cleaned_documents = []
 
        for file in os.listdir(self.data_path):
            if file.endswith(".pdf"):
                print(f" Ingesting: {file}")
                loader = PyPDFLoader(os.path.join(self.data_path, file))
                raw_docs = loader.load()
 
                for doc in raw_docs:
                    cleaned_content = self.clean_text(doc.page_content)
                    if len(cleaned_content) > 50:   # skip near-empty pages
                        cleaned_documents.append(
                            Document(page_content=cleaned_content, metadata=doc.metadata)
                        )
 
        print(f"\n Loaded {len(cleaned_documents)} pages from PDFs")
 
        #  Chunking
        # Larger chunks = more context per retrieval = better answers
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,        
            chunk_overlap=200,      
            separators=[
                "\nSection", "\nSECTION",
                "\nArticle", "\nARTICLE",
                "\nClause", "\nCLAUSE",
                "\n\n", "\n", ". ", " "
            ]
        )
 
        chunks = splitter.split_documents(cleaned_documents)
        print(f" Split into {len(chunks)} chunks")
 
        # Build Vector DB 
        print("\n Building vector database (this may take a few minutes)...")
        Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print(" Success! Knowledge base is built and persistent.")
        print(f"   Chunks stored: {len(chunks)}")
 
 
if __name__ == "__main__":
    ingestor = LegalIngestor(data_path="data_raw", db_path="data_vector_db")
    ingestor.process_pdfs()