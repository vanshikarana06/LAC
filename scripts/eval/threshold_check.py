import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_chroma import Chroma
from src.embeddings import get_embeddings
from src.api.main import SIMILARITY_THRESHOLD

EVAL_ANSWERABLE_QUESTIONS = [
    "What is the punishment for theft under IPC?",
    "What is maternity leave duration in India?",
    "What is gratuity and who is eligible for it?",
    "Suppose I am arrested without a warrant — is that legal?",
    "What are the key differences between a cognizable and non-cognizable offense in India?",
]

IN_SCOPE_LEGAL_QUESTIONS = [
    "What is the procedure to file an FIR under Indian criminal law?",
    "What are the fundamental rights guaranteed under Article 21 of the Indian Constitution?",
    "What constitutes sexual harassment in the workplace under the POSH Act?",
    "How many days does an employer have to pay gratuity under the Payment of Gratuity Act?",
    "Can an employer terminate a female employee during maternity leave?",
    "What are consumer rights regarding defective products under the Consumer Protection Act?",
    "What is the limitation period for filing an appeal in a civil suit in India?",
    "What are the legal grounds for seeking anticipatory bail under CrPC?",
    "What is the punishment for criminal breach of trust under IPC Section 405?",
    "What are the duties of the Internal Complaints Committee under the POSH Act?",
]

OUT_OF_SCOPE_QUESTIONS = [
    "What is the best recipe for chocolate chip cookies?",
    "How far is the Moon from the Earth in kilometers?",
    "What is the current stock price of Apple on NASDAQ?",
    "Who won the FIFA World Cup final in 2022?",
    "How do I fix a leaking bathroom faucet at home?",
    "What is the population of Tokyo in 2024?",
    "How to train for a half marathon in 12 weeks?",
    "What are the system requirements for playing Cyberpunk 2077?",
    "How does photosynthesis convert sunlight into glucose?",
    "What is the capital city of Australia?",
]


def check_thresholds():
    db_path = os.getenv("DB_PATH", "data_vector_db")
    embeddings = get_embeddings()

    print(f"=== SIMILARITY THRESHOLD CHECK ===")
    print(f"DB Path:            {db_path}")
    print(f"Active Embedder:    {embeddings.__class__.__name__}")
    print(f"Configured Threshold: {SIMILARITY_THRESHOLD}\n")

    if not os.path.exists(db_path):
        print(f"ERROR: Vector DB directory '{db_path}' does not exist.")
        sys.exit(1)

    vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)

    def get_top1_score(query: str) -> float:
        results = vectordb.similarity_search_with_relevance_scores(query, k=1)
        if results and len(results) > 0:
            return float(results[0][1])
        return 0.0

    print("--- 1. Answerable Eval Questions (Top-1 Relevance Scores) ---")
    eval_scores = []
    for q in EVAL_ANSWERABLE_QUESTIONS:
        score = get_top1_score(q)
        eval_scores.append(score)
        passed = "[PASS]" if score >= SIMILARITY_THRESHOLD else "[FAIL]"
        print(f"{passed} {score:.4f} | {q}")

    print("\n--- 2. In-Scope Legal Questions (Top-1 Relevance Scores) ---")
    in_scope_scores = []
    for q in IN_SCOPE_LEGAL_QUESTIONS:
        score = get_top1_score(q)
        in_scope_scores.append(score)
        passed = "[PASS]" if score >= SIMILARITY_THRESHOLD else "[FAIL]"
        print(f"{passed} {score:.4f} | {q}")

    print("\n--- 3. Clearly Out-of-Scope Questions (Top-1 Relevance Scores) ---")
    out_of_scope_scores = []
    for q in OUT_OF_SCOPE_QUESTIONS:
        score = get_top1_score(q)
        out_of_scope_scores.append(score)
        filtered = "[FILTERED]" if score < SIMILARITY_THRESHOLD else "[LEAKED]"
        print(f"{filtered} {score:.4f} | {q}")

    all_in_scope = eval_scores + in_scope_scores
    min_in_scope = min(all_in_scope)
    max_out_of_scope = max(out_of_scope_scores)

    print("\n" + "=" * 60)
    print(f"SUMMARY & SEPARATION VERDICT:")
    print(f"  Total In-Scope Evaluated:     {len(all_in_scope)} (5 eval + 10 additional)")
    print(f"  Min In-Scope Score:           {min_in_scope:.4f}")
    print(f"  Max Out-of-Scope Score:       {max_out_of_scope:.4f}")
    print(f"  Current SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD:.4f}")

    cleanly_separates = min_in_scope >= SIMILARITY_THRESHOLD > max_out_of_scope
    if cleanly_separates:
        print(f"  VERDICT: CLEAN SEPARATION (min in-scope >= threshold > max out-of-scope)")
    else:
        print(f"  VERDICT: MARGINAL OR OVERLAPPING (min in-scope: {min_in_scope:.4f}, max out-of-scope: {max_out_of_scope:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    check_thresholds()

