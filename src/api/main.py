from contextlib import asynccontextmanager
import os
import re
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import uuid

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

import groq as groq_sdk
import sqlite3
from langchain_tavily import TavilySearch 

# DB_SESSION_PATH = "session_store.db"
DB_SESSION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "session_store.db")

DB_PATH  = os.getenv("DB_PATH", "data_vector_db")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


# Auto-pick working Groq model
def get_working_model():
    client    = groq_sdk.Groq(api_key=GROQ_KEY)
    available = [m.id for m in client.models.list().data]
    preferred = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
    ]
    for m in preferred:
        if m in available:
            print(f" Using model: {m}")
            return m
    fallback = available[0]
    print(f" Falling back to: {fallback}")
    return fallback

MODEL = get_working_model()

#Vector DB 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb   = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever  = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 15}   
)

# LLM 
llm = ChatGroq(
    model=MODEL,
    api_key=GROQ_KEY,
    temperature=0.2,
    max_tokens=768,   
)

# SCENARIO DETECTOR
# Classifies every incoming question into one of 4 types
# so we can route it to the correct prompt and strategy

SCENARIO_KEYWORDS = [
    "what if", "suppose", "assuming", "hypothetically", "in a situation",
    "if my employer", "if i am", "if someone", "can my employer", "is it legal",
    "what happens if", "what should i do if", "what would happen", "if a company",
    "scenario", "case where", "situation where"
]

DOMAINS = {
    "labour":    ["worker", "employee", "employer", "salary", "wage", "job", "termination", "fired"],
    "maternity": ["pregnant", "maternity", "pregnancy", "baby", "child", "mother"],
    "criminal":  ["arrest", "police", "fir", "crime", "punishment", "jail", "prison"],
    "consumer":  ["product", "defect", "refund", "consumer", "purchase", "bought"],
    "property":  ["property", "land", "rent", "lease", "mortgage", "house"],
    "family":    ["divorce", "marriage", "wife", "husband", "alimony", "custody"],
}
# ── Topic classifier — prevents off-topic web searches ───────────────────────
LEGAL_KEYWORDS = [
    "law", "act", "legal", "court", "judge", "rights", "section",
    "punishment", "crime", "contract", "property", "marriage", "divorce",
    "employment", "worker", "salary", "wage", "compensation", "gratuity",
    "maternity", "consumer", "rti", "constitution", "fundamental", "article",
    "ipc", "judgement", "verdict", "case", "petition", "bail", "warrant",
    "harassment", "discrimination", "termination", "dispute", "police",
    "arrest", "fir", "complaint", "tribunal", "relief", "damages", "penalty",
    "offence", "accused", "defendant", "plaintiff", "advocate", "solicitor"
]

def is_legal_question(question: str) -> bool:
    """
    Returns True if question is legal in nature.
    Prevents web search from being triggered for 
    completely off-topic questions like iPhone prices.
    """
    q = question.lower()
    return any(keyword in q for keyword in LEGAL_KEYWORDS)

def detect_question_type(question: str) -> str:
    """
    Returns one of: 'scenario', 'multi_hop', 'compound', 'simple'
    Used to route the question to the correct prompt template.
    """
    q = question.lower()

    # Check 1: hypothetical / what-if question
    if any(kw in q for kw in SCENARIO_KEYWORDS):
        return "scenario"

    # Check 2: spans 2+ legal domains → needs multi-hop retrieval
    matched = sum(1 for kws in DOMAINS.values() if any(kw in q for kw in kws))
    if matched >= 2:
        return "multi_hop"

    # Check 3: multiple questions in one sentence
    compound_signals = ["and how", "and what", "also", "additionally", "as well as"]
    if any(s in q for s in compound_signals) and len(question) > 80:
        return "compound"

    # Default: simple factual question
    return "simple"

# QUERY DECOMPOSITION
# Breaks compound questions into sub-questions using the LLM,
# answers each separately, then joins the results

decompose_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a legal query analyzer. Break the following compound legal question
into 2-3 clear simple sub-questions that can each be answered independently.
Return ONLY the sub-questions as a numbered list, nothing else.
Example:
Input: What are my maternity rights and how do I file a complaint if denied?
Output:
1. What are the maternity rights of an employee under Indian law?
2. What is the process to file a complaint if maternity benefits are denied?"""),
    ("human", "{question}")
])

decompose_chain = decompose_prompt | llm | StrOutputParser()

def decompose_question(question: str) -> list[str]:
    """Uses LLM to break a compound question into sub-questions."""
    
    result = call_with_retry(decompose_chain, {"question": question})
    lines  = [l.strip() for l in result.strip().split("\n") if l.strip()]
    sub_questions = []
    for line in lines:
        # Remove "1. " or "2) " prefix using regex
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if cleaned and len(cleaned) > 10:
            sub_questions.append(cleaned)
    # Fallback: if decomposition fails, return original question
    return sub_questions if sub_questions else [question]

#SPECIALIZED PROMPTS
# Different prompt templates for different question types.
# Same LLM + different instructions = significantly better answers.

# For hypothetical / what-if questions
SCENARIO_SYSTEM = """You are an expert Indian legal advisor. The user is asking a hypothetical or scenario-based legal question.

Use the context below to:
1. Identify which Indian law(s) apply to this scenario
2. State what the law says about this exact situation
3. Give a practical step-by-step answer of what the person should do
4. Quote specific section numbers where possible
5. Mention any exceptions or conditions that apply

If the context does not contain a real, verifiable answer, explicitly say so.
NEVER invent case names, party names, or judgement years that are not in the context.

Context:
{context}"""

# For questions spanning multiple Acts
MULTIHOP_SYSTEM = """You are an expert Indian legal advisor. This question involves multiple areas of Indian law.

Use the context below to:
1. Identify ALL relevant Acts that apply
2. Answer each legal aspect separately with its Act name
3. Explain how these laws interact or overlap in this situation
4. Give a unified practical answer
5. Quote section numbers from each relevant Act

If the context does not contain a real, verifiable answer, explicitly say so.
NEVER invent case names, party names, or judgement years that are not in the context.

Context:
{context}"""

# For simple and compound questions
DEFAULT_SYSTEM = """You are an expert Indian legal advisor with deep knowledge of Indian law.

Use ONLY the context provided below to answer the question.
- Be specific and detailed.
- Quote the relevant section or article number when possible.
- Structure your answer with numbered points if there are multiple conditions or rights.
- If the answer is not in the context, say what related information you found instead.

If the context does not contain a real, verifiable answer, explicitly say so.
NEVER invent case names, party names, or judgement years that are not in the context.

Context:
{context}"""

# Dict to look up correct prompt by question type
SYSTEM_PROMPTS = {
    "scenario":  SCENARIO_SYSTEM,
    "multi_hop": MULTIHOP_SYSTEM,
    "compound":  DEFAULT_SYSTEM,
    "simple":    DEFAULT_SYSTEM,
}

# Prompt to rephrase follow-up questions into standalone questions
# e.g. "Who is eligible for it?" → "Who is eligible for gratuity?"
REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Your ONLY job is to rephrase the follow-up question as a standalone question.
DO NOT answer the question.
DO NOT add any information.
DO NOT say you you lack  access to information.
ONLY output the rephrased question and nothing else.
If the question is already standalone, return it exactly as-is."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

rephrase_chain = REPHRASE_PROMPT | llm | StrOutputParser()


def call_with_retry(chain , input_data , max_retries=3):
    """" Wraps any LangChain .invoke() call with retry logic.
    Handles transient network failures with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        
        except Exception as e:
            if attempt == max_retries - 1:
                raise  
            else:
                wait_time = 2 ** attempt

                print(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)


# SESSION STORE + HISTORY MANAGEMENT
# Stores conversation history per session_id in memory.
# trim_history_to_fit prevents token overflow on long conversations.

def init_db():
    conn= sqlite3.connect(DB_SESSION_PATH)
    cursor= conn.cursor()
    # Use triple quotes for multi-line SQL strings
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id          INTEGER  PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT     NOT NULL,
        msg_type    TEXT     NOT NULL,
        msg_content TEXT     NOT NULL,
        msg_time    REAL     NOT NULL
    )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS IDX_SESSION_ID ON messages(SESSION_ID);
    """)
    conn.commit()  
    conn.close()  

def get_history(session_id: str) -> list:
    conn = sqlite3.connect(DB_SESSION_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM messages 
            WHERE session_id = ? 
            ORDER BY msg_time ASC
        """, (session_id,))
        
        # 2. fetch all rows
        rows = cursor.fetchall()
        
        # 3. convert each row to HumanMessage or AIMessage
        history = []
        for _, _, msg_type, msg_content, _ in rows:
            if msg_type == "human":
                history.append(HumanMessage(content=msg_content))
            elif msg_type == "ai":
                history.append(AIMessage(content=msg_content))  

        return history

    finally:
        # 4. close connection (finally ensures this runs even if an error happens)
        conn.close()
    
def save_history(session_id: str, human: str, ai: str):
    conn = sqlite3.connect(DB_SESSION_PATH)
    try:
        cursor = conn.cursor()
        # YOUR TASK:
        cursor.execute("""INSERT INTO messages (session_id, msg_type, msg_content, msg_time)
        VALUES (?, ?, ?, ?)""", (session_id, "human", human, time.time())) # VALUES ();
        cursor.execute("""INSERT INTO messages (session_id, msg_type, msg_content, msg_time)
        VALUES (?, ?, ?, ?)""", (session_id, "ai", ai, time.time())) # VALUES ();
        # commit
        conn.commit()  #Without commit()  data sits in buffer in  ram 

    finally:
          
        conn.close()

def cleanup_old_sessions(days: int = 30):
    """
    Delete messages older than X days.
    Called periodically — not on every request.
    This is called a TTL — Time To Live policy.
    """
    conn = sqlite3.connect(DB_SESSION_PATH)
    try:
        cursor = conn.cursor()
        cutoff = time.time() - (days * 24 * 60 * 60)
        cursor.execute("""
            DELETE FROM messages 
            WHERE msg_time < ?
        """, (cutoff,))
        conn.commit()
    finally:
        conn.close()

def trim_history_to_fit(chat_history: list, max_history_tokens: int = 1500) -> list:
    """
    Trim oldest messages until history fits within token budget.
    Prevents 413 token limit errors on long conversations.
    Estimate: 1 token ≈ 4 characters (rough but effective)
    """
    while chat_history:
        total_chars = sum(len(m.content) for m in chat_history)
        estimated_tokens = total_chars // 4
        if estimated_tokens <= max_history_tokens:
            break
        # Remove oldest human+AI pair
        chat_history = chat_history[2:]
    return chat_history


# Flow: trim history → rephrase → retrieve → build prompt → call LLM
 
def get_active_session_count() -> int:
    conn = sqlite3.connect(DB_SESSION_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM messages")
        return cursor.fetchone()[0]   # fetchone() returns one row, [0] gets first column
    finally:
        conn.close()
def web_search_fallback(question: str) -> str:
    query = question[:200]

    INDIAN_LEGAL_DOMAINS = [
        "indiankanoon.org",
        "sci.gov.in",
        "livelaw.in",
        "barandbench.com",
        "indiacode.nic.in",
    ]

    tool = TavilySearch(
        max_results=5,
        api_key=TAVILY_KEY,
        include_domains=INDIAN_LEGAL_DOMAINS,
        country="india",
    )
    print("Performing web search for:", query)
    results = tool.invoke({"query": query})

    if "results" not in results or not results["results"]:
        return "No relevant results found from Indian legal sources for this query."

    results_text = "Web Search Results (Indian Legal Sources):\n\n" + "\n\n---\n\n".join(
        f"{res['content']}\nSource: {res['url']}"
        for res in results['results']
    )
    return results_text
    


def answer_single_question(
    question: str,
    chat_history: list,
    question_type: str,
    
) -> tuple[str, list, str]:

    #trim history to prevent token overflow
    chat_history = trim_history_to_fit(chat_history)

    # rephrase follow-up using history (only if history exists)
    # Converts "Who is eligible?" → "Who is eligible for gratuity?"
    if chat_history:
        standalone = call_with_retry(rephrase_chain, {
           "input": question,
           "chat_history": chat_history,
})
        print(f"Rephrased: {standalone}")
    else:
        standalone = question

    # 2 retrieve relevant chunks from ChromaDB using rephrased question
    # NEW — returns (doc, score) tuples
    docs_with_scores = vectordb.similarity_search_with_relevance_scores(
    standalone, k=6
          )

    # separate docs and scores
    docs          = [doc for doc, score in docs_with_scores]
    # After getting docs_with_scores:
    highest_score = max(
    [max(score, 0) for doc, score in docs_with_scores],
    default=0)
    print(f"Highest similarity score: {highest_score:.3f}")
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    SIMILARITY_THRESHOLD = 0.4

    if highest_score < SIMILARITY_THRESHOLD:
        if not is_legal_question(standalone):
            # completely off-topic — refuse politely
            return (
                "I am a Legal Advisor AI specializing in Indian law. "
                "I can only answer questions related to Indian legal acts, "
                "rights, and court judgements. Please ask a legal question.",
                [],
                "out_of_scope"
            )
        # legal but not in DB → web search
        print(f" Below threshold → web search")
        context      = web_search_fallback(standalone)
        docs         = []
        query_source = "web_search"
    
    else:
        print(f"✅ Score {highest_score:.3f} above threshold → using ChromaDB")
        context      = "\n\n---\n\n".join(doc.page_content for doc in docs)
        query_source = "vector_search"

    

    # 3 pick correct system prompt based on question type
    # escape curly braces in PDF content to prevent template injection
    safe_context = context.replace("{", "{{").replace("}", "}}")
    system_msg = SYSTEM_PROMPTS[question_type].format(context=safe_context)
 

    # 4 build full message list: system + history + current question
    messages = [("system", system_msg)]
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            messages.append(("human", msg.content))
        elif isinstance(msg, AIMessage):
            messages.append(("ai", msg.content))
    messages.append(("human", question))

    # Step 5 — build chain and call LLM
    prompt = ChatPromptTemplate.from_messages(messages)
    chain  = prompt | llm | StrOutputParser()
    answer = call_with_retry(chain, {})

    return answer, docs, query_source



@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    cleanup_old_sessions(days=30)    # ← called here at startup
    yield
# FastAPI App
app = FastAPI(title="Legal Advisor AI", lifespan=lifespan)   

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class Source(BaseModel):
    source: str
    page: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    session_id: str
    question_type: str 
    query_source: str          


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id   = req.session_id or str(uuid.uuid4())
    chat_history = get_history(session_id)

    # Step 1: Detect question type 
    question_type = detect_question_type(question)
    print(f"\nType: {question_type} | Q: {question}")

    try:
        # Step 2: Route to correct handler
        if question_type == "compound":
            # Decompose → answer each sub-question → join with divider
            sub_questions = decompose_question(question)
            print(f"🔀 Sub-questions: {sub_questions}")

            all_answers = []
            all_docs    = []
            for sub_q in sub_questions:
                ans, docs, source = answer_single_question(sub_q, chat_history, "simple")
                all_answers.append(f"**{sub_q}**\n{ans}")
                all_docs.extend(docs)

            answer = "\n\n---\n\n".join(all_answers)
            docs   = all_docs

        else:
            # scenario, multi_hop, simple → answer directly with correct prompt
            answer, docs , query_source = answer_single_question(question, chat_history, question_type)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    #  Step 3: Save to history 
    save_history(session_id, question, answer)

    #  Step 4: De-duplicate sources 
    seen    = set()
    sources = []
    for doc in docs:
        meta = doc.metadata
        src  = os.path.basename(meta.get("source", "Unknown"))
        page = meta.get("page")
        if (src, page) not in seen:
            seen.add((src, page))
            sources.append(Source(source=src, page=page))

    return QueryResponse(
        answer=answer,
        sources=sources,
        session_id=session_id,
        question_type=question_type,
        query_source=query_source, 
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear chat history — called when user clicks New Chat."""
    conn = sqlite3.connect(DB_SESSION_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()
    return {"cleared": session_id}


@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "docs_in_db":      vectordb._collection.count(),
        "model":           MODEL,
        "active_sessions": get_active_session_count(),
    }

@app.get("/acts")
async def list_acts():
    all_meta = vectordb._collection.get(include=["metadatas"])["metadatas"]
    acts = sorted({os.path.basename(m.get("source", "")) for m in all_meta if m})
    return {"acts": acts}



