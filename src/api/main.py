import sys
sys.stdout.reconfigure(line_buffering=True)
print("DEBUG: API File Loaded Successfully")
from contextlib import asynccontextmanager
import os
import re
import time
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import uuid
import string


from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpointEmbeddings

import groq as groq_sdk
import sqlite3
from langchain_tavily import TavilySearch 

import json
from fastapi.responses import StreamingResponse 

from langsmith import traceable

load_dotenv()

DB_SESSION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "session_store.db")

DB_PATH  = os.getenv("DB_PATH", "data_vector_db") 
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
HF_TOKEN   = os.getenv("HF_TOKEN", "")
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN 

if not GROQ_KEY:
    # Fail fast at startup instead of failing confusingly on the first request.
    raise RuntimeError("GROQ_API_KEY is not set. Check your .env / environment variables.")

SIMILARITY_THRESHOLD = 0.35 
MAX_CONTEXT_CHARS = 4000  # ≈1000 tokens — leaves headroom for system prompt + history + question under the TPM cap

class LegalAdvisorState(TypedDict):
    original_question:   str
    session_id:          str
    question_type:       str
    standalone_question: str
    retrieved_docs:      list
    similarity_score:    float
    context:             str
    query_source:        str
    chat_history:        list
    answer:              str
    planned_action:      str

# Groundedness detection
# When similarity search returns a false-positive chunk above the threshold,
# the LLM correctly says "I couldn't find this in the context" — but
# retrieved_docs still holds that chunk, so it was being cited as a source
# even though the answer admits it found nothing. We detect refusal /
# not-found language in the final answer and use that to decide whether
# sources should be attached to the response.

NOT_FOUND_PATTERNS = re.compile(
    r"(couldn't find|could not find|does not contain|doesn't contain|"
    r"no (relevant|verifiable) (information|answer)|"
    r"not (a real|mentioned) in (the )?context|"
    r"context does not contain)",
    re.IGNORECASE,
)
def is_grounded_answer(answer: str) -> bool:
    """Returns False if the answer looks like a refusal / not-found response,
    in which case retrieved_docs should NOT be surfaced as citations."""
    if not answer:
        return False
    return not NOT_FOUND_PATTERNS.search(answer)

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


embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN
)


# LLM 
llm = ChatGroq(
    model=MODEL,
    api_key=GROQ_KEY,
    temperature=0.2,
    max_tokens=1024, 
    model_kwargs={"frequency_penalty": 0.4},  
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
#Topic classifier — prevents off-topic web searches
LEGAL_KEYWORDS = [
    "law", "act", "legal", "court", "judge", "rights", "section",
    "punishment", "crime", "contract", "property", "marriage", "divorce",
    "employment", "worker", "salary", "wage", "compensation", "gratuity",
    "maternity", "consumer", "rti", "constitution", "fundamental", "article",
    "ipc", "judgement", "verdict", "case", "petition", "bail", "warrant",
    "harassment", "discrimination", "termination", "dispute", "police",
    "arrest", "fir", "complaint", "tribunal", "relief", "damages", "penalty",
    "offence", "accused", "defendant", "plaintiff", "advocate", "solicitor",
    "bns", "bnss", "bsa", "gst","tax", "taxation"
]

def is_legal_question(question: str) -> bool:
    """
    Returns True if question is legal in nature.
    Prevents web search from being triggered for
    completely off-topic questions like iPhone prices.
    """
    q = question.lower()
    if any(keyword in q for keyword in LEGAL_KEYWORDS):
        return True
    return any(kw in q for domain_kws in DOMAINS.values() for kw in domain_kws)

def detect_question_type(question: str) -> str:
    """
    Returns one of: 'scenario', 'multi_hop', 'compound', 'simple'
    Used to route the question to the correct prompt template.
    """
    q = question.lower()

    #hypothetical / what-if question
    if any(kw in q for kw in SCENARIO_KEYWORDS):
        return "scenario"

    # 2+ legal domains → needs multi-hop retrieval
    matched = sum(1 for kws in DOMAINS.values() if any(kw in q for kw in kws))
    if matched >= 2:
        return "multi_hop"

    #multiple questions in one sentence
    compound_signals = ["and how", "and what", "also", "additionally", "as well as"]
    if any(s in q for s in compound_signals) and len(question) > 80:
        return "compound"

    # Default: simple factual question
    return "simple"

async def call_with_retry(chain, input_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(chain.invoke, input_data)
        except Exception as e:
            # A 413/rate_limit_exceeded on request size will fail
            # identically on every retry — the payload doesn't shrink
            # between attempts. Fail fast instead of wasting up to
            # 3 seconds retrying a guaranteed failure.
            if "rate_limit_exceeded" in str(e) or "Request too large" in str(e):
                raise
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

#SPECIALIZED PROMPTS
# Different prompt templates for different question types.
SCENARIO_SYSTEM = """You are an expert Indian legal advisor. The user is asking a hypothetical or scenario-based legal question.

Use the context below to:
1. Identify which Indian law(s) apply to this scenario
2. State what the law says about this exact situation
3. Give a practical step-by-step answer of what the person should do
4. Quote specific section numbers where possible
5. Mention any exceptions or conditions that apply

If the context does not contain a real, verifiable answer, explicitly say so.
NEVER invent case names, party names, or judgement years that are not in the context.
Only make claims that are directly supported by the context provided above. If you know additional relevant legal information that is not present in the context, do not include it in your answer — mention only that further research may be needed for aspects not covered here.
If the provided context does not relate to the question being asked, say so explicitly rather than answering from general knowledge, even if you know the answer.


Context:
{context}"""

MULTIHOP_SYSTEM = """You are an expert Indian legal advisor. This question involves multiple areas of Indian law.

Use the context below to:
1. Identify ALL relevant Acts that apply
2. Answer each legal aspect separately with its Act name
3. Explain how these laws interact or overlap in this situation
4. Give a unified practical answer
5. Quote section numbers from each relevant Act

If the context does not contain a real, verifiable answer, explicitly say so.
NEVER invent case names, party names, or judgement years that are not in the context.
Only make claims that are directly supported by the context provided above. If you know additional relevant legal information that is not present in the context, do not include it in your answer — mention only that further research may be needed for aspects not covered here.
If the provided context does not relate to the question being asked, say so explicitly rather than answering from general knowledge, even if you know the answer.

Context:
{context}"""

DEFAULT_SYSTEM = """You are an expert Indian legal advisor with deep knowledge of Indian law.

Use ONLY the context provided below to answer the question.
- Be specific and detailed.
- Quote the relevant section or article number when possible.
- Structure your answer with numbered points if there are multiple conditions or rights.
- If the answer is not in the context, say what related information you found instead.

If the context does not contain a real, verifiable answer, explicitly say so.
NEVER invent case names, party names, or judgement years that are not in the context.
Only make claims that are directly supported by the context provided above. If you know additional relevant legal information that is not present in the context, do not include it in your answer — mention only that further research may be needed for aspects not covered here.
If the provided context does not relate to the question being asked, say so explicitly rather than answering from general knowledge, even if you know the answer.

Context:
{context}"""

SYSTEM_PROMPTS = {
    "scenario":  SCENARIO_SYSTEM,
    "multi_hop": MULTIHOP_SYSTEM,
    "compound":  DEFAULT_SYSTEM,
    "simple":    DEFAULT_SYSTEM,
}

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

PLANNER_SYSTEM = """You are a routing agent for an Indian legal advisor system.
A user's question scored in an ambiguous range against the legal knowledge base —
not clearly answerable, not clearly out of scope.

Decide the single best next action:
- "web_search" — the question is legal in nature and specific enough that a web
  search for current/general legal information would help.
- "clarify" — the question is broad or vague, but IS about law or legal concepts
  in general (even without naming a specific Act, country, or section).
  Example: "what is the law", "tell me about my rights", "how does a court work"
  — these are legal topics, just too broad to answer usefully. Ask ONE short
  question to narrow the scope (e.g. which country's law, which specific topic).
- "refuse" — the question has NOTHING to do with law or legal topics at all
  (e.g. cooking, sports scores, general trivia).

When in doubt between "clarify" and "refuse": if the question mentions law,
rights, courts, legal concepts, or anything justice-adjacent — even vaguely —
choose "clarify", not "refuse".

Respond with EXACTLY one line in this format, nothing else:
ACTION: <web_search|clarify|refuse>
REASON: <one short sentence>
CLARIFYING_QUESTION: <only if ACTION is clarify, otherwise leave blank>
"""

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM),
    ("human", "{question}")
])
planner_chain = planner_prompt | llm | StrOutputParser()

def init_db():
    conn= sqlite3.connect(DB_SESSION_PATH)
    cursor= conn.cursor()
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
        
        #fetch all rows
        rows = cursor.fetchall()
        
        #convert each row to HumanMessage or AIMessage
        history = []
        for _, _, msg_type, msg_content, _ in rows:
            if msg_type == "human":
                history.append(HumanMessage(content=msg_content))
            elif msg_type == "ai":
                history.append(AIMessage(content=msg_content))  

        return history

    finally:
        #close connection (finally ensures this runs even if an error happens)
        conn.close()
    
def save_history(session_id: str, human: str, ai: str):
    conn = sqlite3.connect(DB_SESSION_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""INSERT INTO messages (session_id, msg_type, msg_content, msg_time)
        VALUES (?, ?, ?, ?)""", (session_id, "human", human, time.time()))
        cursor.execute("""INSERT INTO messages (session_id, msg_type, msg_content, msg_time)
        VALUES (?, ?, ?, ?)""", (session_id, "ai", ai, time.time())) 
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

    if isinstance(results, str):
        return f"No relevant results found from Indian legal sources for this query. ({results})"
    
    if "results" not in results or not results["results"]:
        return "No relevant results found from Indian legal sources for this query."

    results_text = "Web Search Results (Indian Legal Sources):\n\n" + "\n\n---\n\n".join(
        f"{res['content']}\nSource: {res['url']}"
        for res in results['results']    )
    return results_text
    
# langraph nodes

def node_load_history(state: LegalAdvisorState)->dict:
    """
    Load the conversation history from sqlite for this session
    """
    session_id = state["session_id"]
    history = get_history(session_id)
    return {"chat_history": history}
def node_classify_question(state: LegalAdvisorState)->dict:
    """
    Classify the question type
    """
    question_type = detect_question_type(state["original_question"])
    return {"question_type": question_type}
async def node_rephrase_query(state: LegalAdvisorState)->dict:
    """
    Rephrase the question using the conversation history
    """
    chat_history=state["chat_history"]
    original_question= state["original_question"]

    chat_history = trim_history_to_fit(chat_history)

    if chat_history:
        standalone = await call_with_retry(rephrase_chain, {
            "input": state["original_question"],
            "chat_history": state["chat_history"],
        })
    else:
        standalone = original_question
    return {"standalone_question": standalone,
            "chat_history": chat_history}

def node_retrieve_chunks(state: LegalAdvisorState) -> dict:
    global vectordb
    
    if vectordb is None:
        raise Exception("VectorDB is not initialized. Cannot retrieve chunks.")
    standalone = state["standalone_question"]  
    docs_with_scores = vectordb.similarity_search_with_relevance_scores(standalone, k=6)    
    docs = [doc for doc, score in docs_with_scores]
    highest_score = max([max(score, 0) for doc, score in docs_with_scores], default=0)
    
    return {"retrieved_docs": docs, "similarity_score": highest_score}

def _is_ambiguous(score:float, question:str)->bool :
    """ Fast , cheap check for whether this case is worth an LLM planning call.
    Clear high/low scores skip straight to the deterministic path- only the genuinely uncertain middle
      , or suspiciously short/vague questions, 
      trigger the more expensive LLM judgment call."""
    AMBIGUOUS_LOW=SIMILARITY_THRESHOLD-0.05
    AMBIGUOUS_HIGH = SIMILARITY_THRESHOLD+0.10
    if AMBIGUOUS_LOW <= score<=AMBIGUOUS_HIGH :
        return True
    cleaned = question.translate(str.maketrans("", "", string.punctuation))
    if len(cleaned.strip().split()) <= 4:
        return True
    return False
async def node_plan_action(state: LegalAdvisorState) -> dict:
    question = state["standalone_question"]
    result = await call_with_retry(planner_chain, {"question": question})
    print(f"DEBUG planner raw output:\n{result}\n---")

    action = "web_search"
    clarifying_q = ""
    for line in result.split("\n"):
        if line.startswith("ACTION"):
            action = line.replace("ACTION:", "").strip().lower()
        elif line.startswith("CLARIFYING_QUESTION"):
            clarifying_q = line.replace("CLARIFYING_QUESTION:", "").strip()
    # everything below is now OUTSIDE the for loop — runs once, after all
    # three lines have been read, not once per line

    if action == "clarify" and clarifying_q:
        return {
            "answer": clarifying_q,
            "query_source": "clarification",
            "context": "",
            "retrieved_docs": [],
        }
    elif action == "refuse":
        return {
            "answer": "I am a Legal Advisor AI specializing in Indian law. "
                      "I can only answer questions related to Indian legal acts, "
                      "rights, and court judgements. Please ask a legal question.",
            "query_source": "out_of_scope",
            "context": "",
            "retrieved_docs": [],
        }
    else:
        context = web_search_fallback(question)
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n\n[context truncated to fit model token limit]"
        return {
            "query_source": "web_search",
            "context": context,
            "retrieved_docs": [],
        }


def node_check_threshold(state: LegalAdvisorState) -> dict:
    """
    Check similarity score and route to correct source:
    - Above threshold → vector_search (ChromaDB)
    - Below threshold + legal → web_search (Tavily)
    - Below threshold + not legal → out_of_scope (refuse)
    """
    similarity_score = state["similarity_score"]
    standalone_question = state["standalone_question"]
    docs = state["retrieved_docs"]

    if similarity_score >= SIMILARITY_THRESHOLD and not _is_ambiguous(similarity_score, standalone_question):
        print(f"DEBUG: score={similarity_score:.4f}, ambiguous={_is_ambiguous(similarity_score, standalone_question)}")
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n\n[context truncated to fit model token limit]"
        return {
            "context": context,
            "query_source": "vector_search",
            "retrieved_docs": docs,
        }
    if _is_ambiguous(similarity_score, standalone_question):
        return{
            "query_source":"needs_planning",
        }

    else:
        # Score too low — check if legal question
        if not is_legal_question(standalone_question) and not is_legal_question(state["original_question"]):
            # Completely off-topic — refuse politely
            return {
                "answer": "I am a Legal Advisor AI specializing in Indian law. "
                          "I can only answer questions related to Indian legal acts, "
                          "rights, and court judgements. Please ask a legal question.",
                "query_source": "out_of_scope",
                "context": "",
                "retrieved_docs": [],
            }

        # Legal but not in DB → web search
        print("Below threshold → web search")
        context = web_search_fallback(standalone_question)
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n\n[context truncated to fit model token limit]"
        return {
            "context": context,
            "query_source": "web_search",
            "retrieved_docs": [],
        }      
    
async def node_generate_answer(state: LegalAdvisorState) -> dict:
    """Generate answer using LLM with correct prompt for question type."""
    question_type     = state["question_type"]
    context           = state["context"]
    chat_history      = state["chat_history"]
    original_question = state["original_question"]

    # prevention from  template injection
    safe_context = context.replace("{", "{{").replace("}", "}}")
    system_msg   = SYSTEM_PROMPTS[question_type].format(context=safe_context)

    #  system + history + current question
    messages = [("system", system_msg)]
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            messages.append(("human", msg.content))
        elif isinstance(msg, AIMessage):
            messages.append(("ai", msg.content))
    messages.append(("human", original_question))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain  = prompt | llm | StrOutputParser()
    answer = await call_with_retry(chain, {})

        #if the answer isn't actually grounded in the retrieved
    # context (refusal / not-found language), drop retrieved_docs so we
    # never cite sources on a response that admits it found nothing.
    result = {"answer": answer}
    if not is_grounded_answer(answer):
        result["retrieved_docs"] = []
    return result
  
def node_save_history(state: LegalAdvisorState)->dict:
    """
    Save the conversation history to sqlite for this session
    """
    session_id = state["session_id"]
    original_question = state["original_question"]
    answer = state["answer"]
    save_history(session_id,original_question, answer)
    return {}

#graph definition
def build_graph():
    graph = StateGraph(LegalAdvisorState)

    # Add all nodes
    graph.add_node("load_history",       node_load_history)
    graph.add_node("classify_question",  node_classify_question)
    graph.add_node("rephrase_query",     node_rephrase_query)
    graph.add_node("retrieve_chunks",    node_retrieve_chunks)
    graph.add_node("check_threshold",    node_check_threshold)
    graph.add_node("plan_action",        node_plan_action)   
    graph.add_node("generate_answer",    node_generate_answer)
    graph.add_node("save_history",       node_save_history)

    #entry point 
    graph.set_entry_point("load_history")

    #linear edges
    graph.add_edge("load_history",      "classify_question")
    graph.add_edge("classify_question", "rephrase_query")
    graph.add_edge("rephrase_query",    "retrieve_chunks")
    graph.add_edge("retrieve_chunks",   "check_threshold")
    graph.add_conditional_edges(
        "check_threshold",
        lambda state: state["query_source"],
        {
            "vector_search":"generate_answer",
            "web_search": "generate_answer",   
            "out_of_scope": "save_history",
            "needs_planning":  "plan_action", 
        }
    )
    graph.add_conditional_edges(
    "plan_action",
    lambda state: state["query_source"],
    {
        "web_search":     "generate_answer",
        "out_of_scope":   "save_history",
        "clarification":  "save_history",           # NEW — clarifying question is the final answer
    }
    )
    graph.add_edge("generate_answer", "save_history")
    graph.add_edge("save_history",    END)

    return graph.compile()
legal_graph = build_graph()

@traceable(name="Legal Advisor Graph")
async def run_legal_advisor(state):
    # This tells LangGraph to use your wrapper as the "parent" 
    # and not create its own separate trace for the inner parts.
    return await legal_graph.ainvoke(state, {"run_name": "Legal Advisor Graph"})

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    cleanup_old_sessions(days=30)

    #  Initialize global resources
    global embeddings, vectordb, retriever

    # Embeddings — environment aware
    if os.getenv("RENDER"):
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=HF_TOKEN   # use module-level constant
        )
        print("Using HuggingFace Endpoint Embeddings (API)")
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Using local HuggingFace Embeddings")

    try:
        vectordb  = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 15}
        )
        print(f"ChromaDB loaded: {vectordb._collection.count()} chunks")
    except Exception as e:
        print(f"CRITICAL: Failed to initialize ChromaDB: {e}")
        vectordb  = None
        retriever = None

    yield   # ← server handles requests here
    print("Server shutting down cleanly.")

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
 
    session_id = req.session_id or str(uuid.uuid4())
 
    try:
        result = await run_legal_advisor({
            "original_question": question,
            "session_id":        session_id,
            "chat_history":      [],
            "question_type":     "",
            "standalone_question": "",
            "retrieved_docs":    [],
            "similarity_score":  0.0,
            "context":           "",
            "query_source":      "",
            "answer":            "",
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Something went wrong processing your question.")

    seen    = set()
    sources = []
    if is_grounded_answer(result["answer"]):
        for doc in result["retrieved_docs"]:
            meta = doc.metadata
            src  = os.path.basename(meta.get("source", "Unknown"))
            page = meta.get("page")
            if (src, page) not in seen:
                seen.add((src, page))
                sources.append(Source(source=src, page=page))
 
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        session_id=session_id,
        question_type=result["question_type"],
        query_source=result["query_source"],
    )

# @app.get("/ask/stream")
# async def ask_stream(question : str, session_id : Optional[str] = None):
#     session_id = session_id or str(uuid.uuid4())
#     chat_history= get_history(session_id)

#     async def generate():
#         question_type= detect_question_type(question)
#         print(f"DEBUG question_type: repr='{repr(question_type)}'")  # ← add this

#         chat_history_trim = trim_history_to_fit(chat_history)
#         if chat_history_trim :
#             standalone= call_with_retry(
#                 rephrase_chain,
#                 {
#                     "input": question,
#                     "chat_history": chat_history_trim,
#                 }
#             )
#         else :
#             standalone = question

#         docs_with_scores = vectordb.similarity_search_with_relevance_scores(standalone, k=6)
#         docs = [doc for doc, _ in docs_with_scores]
#         highest_score = max([max(score, 0) for doc, score in docs_with_scores], default=0)
            
#         if highest_score >= SIMILARITY_THRESHOLD:
#             context= "\n\n---\n\n".join(doc.page_content for doc in docs)
#             query_source = "vector_search"
#         else :
#             if not is_legal_question(standalone):
#                 context = "I am a Legal Advisor AI specializing in Indian law. I can only answer questions related to Indian legal acts, rights, and court judgements. Please ask a legal question."
#                 query_source = "out_of_scope"
#                 docs=[]
#             else:
#                 context = web_search_fallback(standalone)
#                 query_source = "web_search"
#                 docs= []
#         safe_context = context.replace("{", "{{").replace("}", "}}")

#         system_msg   = SYSTEM_PROMPTS[question_type].format(context=safe_context)

#         messages = [("system", system_msg)]
#         for msg in chat_history_trim:
#             if isinstance(msg, HumanMessage):
#                 messages.append(("human", msg.content))
#             elif isinstance(msg, AIMessage):
#                 messages.append(("ai", msg.content))
#         messages.append(("human", question))

#         prompt = ChatPromptTemplate.from_messages(messages)
#         chain  = prompt | llm | StrOutputParser()
#         yield f"data: {json.dumps({'type': 'meta', 'session_id': session_id, 'question_type': question_type, 'query_source': query_source})}\n\n"
#         full_answer = ""
#         for token in chain.stream({}):
#             full_answer += token
#              # send token to browser in SSE format
#             yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
#         seen    = set()
#         sources = []
#         for doc in docs:
#             meta = doc.metadata
#             src  = os.path.basename(meta.get("source", "Unknown"))
#             page = meta.get("page")
#             if (src, page) not in seen:
#                 seen.add((src, page))
#                 sources.append({"source": src, "page": page})
        
#         yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

#         save_history(session_id ,question , full_answer)
#         yield "data: [DONE]\n\n"
       

#     return StreamingResponse(generate(), media_type="text/event-stream")



@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear chat history — called when user clicks New Chat."""
    conn = sqlite3.connect(DB_SESSION_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
        return {"status": "success", "cleared": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/sessions")
async def list_sessions():
    """Returns every session that has at least one message, most recent first,
    with a short preview so the sidebar can show something meaningful."""
    conn = sqlite3.connect(DB_SESSION_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT session_id, MIN(msg_time), MAX(msg_time), COUNT(*)
            FROM messages
            GROUP BY session_id
            ORDER BY MAX(msg_time) DESC
        """)
        rows = cursor.fetchall()

        sessions = []
        for session_id, started, last_active, msg_count in rows:
            cursor.execute("""
                SELECT msg_content FROM messages
                WHERE session_id = ? AND msg_type = 'human'
                ORDER BY msg_time ASC LIMIT 1
            """, (session_id,))
            first_q = cursor.fetchone()
            preview = first_q[0] if first_q else "New conversation"
            if len(preview) > 60:
                preview = preview[:60] + "..."

            sessions.append({
                "session_id": session_id,
                "preview": preview,
                "message_count": msg_count // 2,
                "last_active": last_active,
            })
        return {"sessions": sessions}
    finally:
        conn.close()


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    conn = sqlite3.connect(DB_SESSION_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT msg_type, msg_content FROM messages
            WHERE session_id = ? ORDER BY msg_time ASC
        """, (session_id,))
        rows = cursor.fetchall()
        return {
            "session_id": session_id,
            "messages": [{"role": r[0], "content": r[1]} for r in rows]
        }
    finally:
        conn.close()


@app.get("/ask/stream")
async def ask_stream(question: str, session_id: Optional[str] = None):
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
 
    print(f"DEBUG: Entering ask_stream for question: {question}")
    session_id = session_id or str(uuid.uuid4())
 
    async def generate():
        state = {
            "original_question": question,
            "session_id": session_id,
            "chat_history": [],
            "question_type": "",
            "standalone_question": "",
            "retrieved_docs": [],
            "similarity_score": 0.0,
            "context": "",
            "query_source": "",
            "answer": ""
        }
 
        full_answer = ""
 
        try:
            async for event in legal_graph.astream_events(state, version="v2"):
                kind = event["event"]
                node_name = event.get("name", "")
 
                if kind == "on_chat_model_stream":
                        # Every LLM call in the graph (rephrase_query AND generate_answer)
                        # fires on_chat_model_stream. Without filtering by node, rephrase
                        # tokens leak into the visible answer, sometimes even replacing it
                        # entirely (see: gratuity/landlord test cases). Only stream tokens
                        # that actually came from the final-answer node.
                        node_source = event.get("metadata", {}).get("langgraph_node")
                        if node_source == "generate_answer":
                           content = event["data"]["chunk"].content
                           if content:
                             full_answer += content
                             yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
 
                elif kind == "on_chain_start":
                    yield f"data: {json.dumps({'type': 'status', 'node': node_name})}\n\n"
 
                elif kind == "on_chain_end":
                    output = event["data"].get("output", {})
 
                    if node_name == "classify_question" and "question_type" in output:
                        yield f"data: {json.dumps({'type': 'metadata_update', 'question_type': output['question_type']})}\n\n"
 
                    if node_name == "check_threshold" and "query_source" in output:
                        yield f"data: {json.dumps({'type': 'metadata_update', 'query_source': output['query_source']})}\n\n"
 
                    if isinstance(output, dict):
                        state.update(output)
 
            # out_of_scope sets "answer" directly in node_check_threshold and
            # never calls the LLM, so no on_chat_model_stream events fire and
            # no 'token' SSE event is sent by the loop above. Emit the
            # fallback answer as a token here so the client actually shows it.
            if not full_answer:
                fallback_answer = state.get("answer", "")
                if fallback_answer:
                    yield f"data: {json.dumps({'type': 'token', 'content': fallback_answer})}\n\n"
                    full_answer = fallback_answer
 
            sources = []
            if is_grounded_answer(full_answer):
                seen = set()
                for doc in state.get("retrieved_docs", []):
                    meta = doc.metadata
                    src = os.path.basename(meta.get("source", "Unknown"))
                    page = meta.get("page")
                    if (src, page) not in seen:
                        seen.add((src, page))
                        sources.append({"source": src, "page": page})
 
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
 
        except Exception as e:
            # Previously an exception anywhere in the graph just silently
            # dropped the SSE connection with no signal to the frontend.
            # Log the real error server-side; send only a generic message
            # to the client (never leak internal errors/keys/stack traces).
            print(f"ERROR in /ask/stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Something went wrong generating the answer. Please try again.'})}\n\n"
 
        finally:
            yield "data: [DONE]\n\n"
 
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
async def health():
    docs_count = vectordb._collection.count() if vectordb else 0

    return {
        "status":          "ok" if vectordb else "degraded",
        "docs_in_db":      docs_count,
        "model":           MODEL,
        "active_sessions": get_active_session_count(),
    }

@app.get("/acts")
async def list_acts():
    if not vectordb:
        return {"acts": []}
    all_meta = vectordb._collection.get(include=["metadatas"])["metadatas"]
    acts = sorted({os.path.basename(m.get("source", "")) for m in all_meta if m})
    return {"acts": acts}



