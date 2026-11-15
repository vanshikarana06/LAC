"""Ground-truth checklist: for each answerable question, show the top 3 chunks
retrieved by embedding the ground-truth text. Read-only; does not import the app."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ragas_eval_dataset import ANSWERABLE_QUESTIONS

load_dotenv()
DB_PATH = os.getenv("DB_PATH", "data_vector_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

lines = [f"DB: {DB_PATH} | chunks: {db._collection.count()}", "=" * 80]
for i, item in enumerate(ANSWERABLE_QUESTIONS, 1):
    lines += ["-" * 80, f"Q{i}: {item['question']}", f"GROUND TRUTH:\n{item['ground_truth']}\n",
              "TOP 3 CHUNKS:"]
    for rank, (doc, score) in enumerate(db.similarity_search_with_relevance_scores(item["ground_truth"], k=3), 1):
        src = os.path.basename(doc.metadata.get("source", "?"))
        text = doc.page_content.strip().replace("\n", " ")[:300]
        lines.append(f"  [{rank}] {score:.4f} | {src} | p.{doc.metadata.get('page', '?')}\n      {text}...")

out = os.path.join("docs", "eval", "ground_truth_check.txt")
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))

