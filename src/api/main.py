import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
 
load_dotenv()
 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
 
import groq as groq_sdk
 
DB_PATH    = os.getenv("DB_PATH", "data_vector_db")
GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
 
# ── Auto-pick a working Groq model ────────────────────────────────────────────
def get_working_model():
    client = groq_sdk.Groq(api_key=GROQ_KEY)
    available = [m.id for m in client.models.list().data]
    preferred = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
    ]
    for m in preferred:
        if m in available:
            print(f"✅ Using model: {m}")
            return m
    fallback = available[0]
    print(f"⚠️  Preferred models not found, falling back to: {fallback}")
    return fallback
 
MODEL = get_working_model()
 
# ── Embeddings + Vector DB ────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
vectordb = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings,
)
 
# k=8 retrieves more chunks = more context = better answers
retriever = vectordb.as_retriever(
    search_type="mmr",          # Maximum Marginal Relevance
    search_kwargs={"k": 8, "fetch_k": 20,"score_threshold": 0.5}
)
 
# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model=MODEL,
    api_key=GROQ_KEY,
    temperature=0.2,
    max_tokens=1024,
)
 
# ── Prompt ────────────────────────────────────────────────────────────────────
prompt = PromptTemplate.from_template("""You are an expert Indian legal advisor with deep knowledge of Indian law.
 
Use ONLY the context provided below to answer the question. 
- Be specific and detailed in your answer.
- Quote the relevant section or article number when possible.
- Structure your answer with numbered points if there are multiple conditions or rights.
- If the exact answer is not in the context, say what related information you found instead of saying nothing.
 
Context:
{context}
 
Question: {question}
 
Detailed Answer:""")
 
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
 
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
 
# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Legal Advisor AI")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
app.mount("/static", StaticFiles(directory="static"), name="static")
 
 
class QueryRequest(BaseModel):
    question: str
 
class Source(BaseModel):
    source: str
    page: int | None = None
 
class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
 
 
@app.get("/")
async def root():
    return FileResponse("static/index.html")
 
 
@app.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        answer = chain.invoke(question)
        docs   = retriever.invoke(question)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
 
    seen = set()
    sources: list[Source] = []
    for doc in docs:
        meta = doc.metadata
        src  = os.path.basename(meta.get("source", "Unknown"))
        page = meta.get("page")
        if (src, page) not in seen:
            seen.add((src, page))
            sources.append(Source(source=src, page=page))
 
    return QueryResponse(answer=answer, sources=sources)
 
 
@app.get("/health")
async def health():
    return {"status": "ok", "docs_in_db": vectordb._collection.count(), "model": MODEL}
 
 
@app.get("/acts")
async def list_acts():
    all_meta = vectordb._collection.get(include=["metadatas"])["metadatas"]
    acts = sorted({os.path.basename(m.get("source", "")) for m in all_meta if m})
    return {"acts": acts}
 
