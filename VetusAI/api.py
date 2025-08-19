import os, sys, time, subprocess
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from enum import Enum
from typing import Optional, List, Dict, Any

# Local modules (now the heavy lifting lives here)
from lookup import search_threads, build_context_from_cards, get_thread_detail as _get_thread_detail, get_email_detail as _get_email_detail
from llm import query_llm

# ---- Configuration ----
OLLAMA_MODEL = "llama3.1-12k:8b"
OLLAMA_URL = "http://localhost:11434"
VECTOR_DB_DIR = "./vectorstore"
SYSTEM_PROMPT = ""
RESULTS_QTY = 15
RAG_THRESHOLD = 25.00  # percentage 1-99


app = FastAPI(title="Vetus AI Assistant", description="Backend API for document ingestion, semantic search, and LLM Q&A.", version="1.1")

# === Context Mode Enum ===
class ContextMode(str, Enum):
    summary = "summary"   # use topic/detail1/detail2/subject
    full = "full"         # use full email data (cleaned/clipped)
    hybrid = "hybrid"     # summary + first email

# === Requests / Responses ===
class AskRequest(BaseModel):
    user_input: str
    model_name: str = "deepseek-r1-12k:14b"
    model_loc: str = "external_api"
    system_prompt: str = SYSTEM_PROMPT
    use_chat: bool = True
    return_meta: bool = False

class SearchRequest(BaseModel):
    query: str
    top_k: int = RESULTS_QTY            # max number of search results
    min_score: float = RAG_THRESHOLD    # minimum score percentage
    include_full_emails: bool = False

class SearchHitCard(BaseModel):
    thread_id: Optional[str] = None
    score: Optional[float] = None
    subject: Optional[str] = None
    topic: Optional[str] = None
    detail1: Optional[str] = None
    detail2: Optional[str] = None
    link: Optional[str] = None
    emails: Optional[List[str]] = None
    emails_count: Optional[int] = None
    full_emails: Optional[List[Dict[str, Any]]] = None    # Only returned when include_full_emails=True
    
    # Fields for document results (all optional)
    kind: Optional[str] = None
    doc_path: Optional[str] = None
    doc_name: Optional[str] = None
    paragraph: Optional[str] = None
    page: Optional[int] = None
    paragraph_index: Optional[int] = None

class SearchResponse(BaseModel):
    results: List[SearchHitCard]

class RagAskRequest(BaseModel):
    user_input: str
    query: Optional[str] = None           # optional explicit search query
    top_k: int = RESULTS_QTY
    min_score: float = RAG_THRESHOLD
    context_mode: ContextMode = ContextMode.summary
    model_name: str = "deepseek-r1-12k:14b"
    model_loc: str = "external_api"
    system_prompt: str = SYSTEM_PROMPT
    
class RagAskResponse(BaseModel):
    answer: str
    sources: List[SearchHitCard]          # same structure; includes link(s)
    used_query: str


# === API Endpoints ===
@app.post("/vectors-rebuild", status_code=201)
def rebuild_vectorstore():
    try:
        result = subprocess.run(["python", "ingest.py"], check=True, capture_output=True, text=True)
        return {"status": "success", "message": "Vectorstore rebuilt successfully.", "log": result.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild vectorstore: {e.stderr}")

@app.post("/ask")
def ask_question(request: AskRequest):
    """Thin pass-through to LLM; no RAG here."""
    try:
        answer = query_llm(request.user_input, model_name=request.model_name, model_loc=request.model_loc, system_prompt=request.system_prompt, use_chat=request.use_chat, return_meta=request.return_meta)
        # Keep response shape stable for existing front-ends
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM query failed: {e}")

@app.post("/lookup", response_model=SearchResponse)
def lookup(req: SearchRequest):
    """RAG search that returns thread 'cards' already organized by lookup.py."""
    cards = search_threads(req.query, top_k=req.top_k, min_score=req.min_score, include_full_emails=req.include_full_emails)
    return SearchResponse(results=[SearchHitCard(**c) for c in cards])

@app.post("/ask-with-rag", response_model=RagAskResponse)
def ask_with_rag(req: RagAskRequest):
    # 1) Choose query (if not provided, reuse the question)
    query = req.query or req.user_input

    # 2) Retrieve threads; include full emails only when needed for the selected context mode
    include_full = req.context_mode in (ContextMode.full, ContextMode.hybrid)
    cards = search_threads(query, top_k=req.top_k, min_score=req.min_score, include_full_emails=include_full)

    # 3) Build context server-side (delegated to lookup.py)
    context = build_context_from_cards(cards, req.context_mode.value)

    # 4) Compose prompt
    system_prompt = req.system_prompt or "You are a very helpful assistent that will answer the user's questions based ONLY on the provided context."
    post_prompt = "\n".join([
        "CONTEXT:",
        context if context else "(no relevant context found)",
        "",
        "USER QUESTION:",
        req.user_input,
        "",
        "INSTRUCTIONS:",
        "- If the context answers the question, answer using it.",
        "- If something is unclear or not in context, say so briefly."
    ])

    answer = query_llm(post_prompt, model_name=req.model_name, model_loc=req.model_loc, system_prompt=system_prompt)
    return RagAskResponse(answer=answer, sources=[SearchHitCard(**c) for c in cards], used_query=query)

@app.get("/threads/{thread_id}")
def get_thread_detail(thread_id: str):
    td = _get_thread_detail(thread_id)
    if not td:
        raise HTTPException(status_code=404, detail="Thread not found")
    return td

@app.get("/emails/{email_id}")
def get_email_detail(email_id: str):
    ed = _get_email_detail(email_id)
    if not ed:
        raise HTTPException(status_code=404, detail="Email not found")
    return ed

@app.get("/vectors-check")
def get_vectorstore_last_updated():
    try:
        timestamp_path = os.path.join(VECTOR_DB_DIR, "last_updated.txt")
        if os.path.exists(timestamp_path):
            with open(timestamp_path, "r") as f:
                timestamp = int(f.read())
            return {"last_updated": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}
        else:
            return {"last_updated": "Never"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read vectorstore timestamp: {e}")
    
@app.get("/doc")
async def doc_page():
    # Serve the static HTML shell; it will fetch the snippet via /doc-snippet
    return FileResponse("static/doc.html")

@app.get("/open-location")
def open_location(path: str):
    """Open the folder containing `path` in the OS file explorer (select file if possible)."""
    try:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Path not found.")

        is_file = os.path.isfile(path)
        folder = os.path.dirname(path) if is_file else path

        try:
            if sys.platform.startswith("win"):
                if is_file:
                    subprocess.Popen(["explorer", "/select,", path])
                else:
                    subprocess.Popen(["explorer", folder])
            elif sys.platform == "darwin":
                if is_file:
                    subprocess.Popen(["open", "-R", path])
                else:
                    subprocess.Popen(["open", folder])
            else:
                # Linux / others
                subprocess.Popen(["xdg-open", folder])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to open location: {e}")

        # Tiny confirmation page (opens in a new tab)
        html = f"<html><body><p>Opened in file explorer:</p><pre>{path}</pre></body></html>"
        return HTMLResponse(content=html, status_code=200)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

# --- Static pages ---
@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/search")
async def search_get():
    return FileResponse("static/search.html")

@app.get("/chat")
async def chat():
    return FileResponse("static/chat.html")

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
