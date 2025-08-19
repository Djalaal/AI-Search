from __future__ import annotations
import os, json
import urllib.parse
from typing import List, Dict, Any, Optional, Iterable
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

VECTORDB_PATH = "./vectorstore"
MODEL_NAME = "nomic-embed-text"
RESULTS_QTY = 15
RAG_THRESHOLD = 25.00

PROCESSED_DIR = "./processed"
EMAILS_JSONL = os.path.join(PROCESSED_DIR, "emails.jsonl")
THREADS_JSONL = os.path.join(PROCESSED_DIR, "threads.jsonl")
SUMMARIES_JSONL = os.path.join(PROCESSED_DIR, "email_summaries.jsonl")

# --- Lazy caches ---
_threads_map: Dict[str, List[str]] = {}
_summaries_map: Dict[str, Dict[str, str]] = {}
_email_cache: Dict[str, dict] = {}

# --- Lazy getters for thread -> email_ids and summaries ---
def _get_threads_map(thread_ids: Iterable[str]) -> Dict[str, List[str]]:
    need = {tid for tid in thread_ids if tid not in _threads_map}
    if not need:
        return {tid: _threads_map.get(tid, []) for tid in thread_ids}
    if not os.path.exists(THREADS_JSONL):
        return {tid: [] for tid in thread_ids}
    with open(THREADS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            tid = rec.get("thread_id")
            if tid in need:
                _threads_map[tid] = rec.get("email_ids", [])
                need.remove(tid)
                if not need:
                    break
    return {tid: _threads_map.get(tid, []) for tid in thread_ids}

def _get_summaries_map(thread_ids: Iterable[str]) -> Dict[str, Dict[str, str]]:
    need = {tid for tid in thread_ids if tid not in _summaries_map}
    if not need:
        return {tid: _summaries_map.get(tid, {}) for tid in thread_ids}
    if not os.path.exists(SUMMARIES_JSONL):
        return {tid: {} for tid in thread_ids}
    with open(SUMMARIES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                s = json.loads(line)
            except Exception:
                continue
            tid = s.get("thread_id")
            if tid in need:
                _summaries_map[tid] = {
                    "subject": s.get("subject", ""),
                    "topic": s.get("topic", ""),
                    "detail1": s.get("detail1", ""),
                    "detail2": s.get("detail2", ""),
                }
                need.remove(tid)
                if not need:
                    break
    return {tid: _summaries_map.get(tid, {}) for tid in thread_ids}

# --- Email body loading (still lazy, one pass over JSONL) ---
def _iter_email_records() -> Iterable[dict]:
    if not os.path.exists(EMAILS_JSONL):
        return []
    with open(EMAILS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue

def _ensure_emails_loaded(email_ids: Iterable[str]) -> None:
    need = {eid for eid in email_ids if eid and eid not in _email_cache}
    if not need:
        return
    for rec in _iter_email_records():
        eid = rec.get("email_id")
        if eid in need:
            _email_cache[eid] = rec.get("data", {})
            need.remove(eid)
            if not need:
                break

def _emails_for_threads(thread_ids: List[str]) -> Dict[str, dict]:
    # Use the lazy thread map to collect only the eids we need
    tmap = _get_threads_map(thread_ids)
    all_eids: List[str] = []
    for tid in thread_ids:
        all_eids.extend(tmap.get(tid, []))
    _ensure_emails_loaded(all_eids)
    return {eid: _email_cache[eid] for eid in all_eids if eid in _email_cache}

# --- Vectorstore / RAG ---
def load_vectorstore(path: str = VECTORDB_PATH, model_name: str = MODEL_NAME):
    embeddings = OllamaEmbeddings(model=model_name)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def run_rag_question(question: str, rag_model: str = MODEL_NAME, top_k: int = RESULTS_QTY, min_score: float = RAG_THRESHOLD):
    db = load_vectorstore(model_name=rag_model)
    similar_docs = db.similarity_search_with_score(question, k=top_k)
    docs = []
    for doc, score in similar_docs:
        score_percentage = (1 - score) * 100.0
        if score_percentage >= min_score:
            doc.metadata["score"] = score_percentage
            docs.append(doc)
    return docs

# --- Helpers for card/meta assembly ---
def _thread_meta(thread_id: str) -> Dict[str, str]:
    m = _get_summaries_map([thread_id]).get(thread_id, {})
    return {
        "subject": m.get("subject", ""),
        "topic": m.get("topic", ""),
        "detail1": m.get("detail1", ""),
        "detail2": m.get("detail2", ""),
    }

def docs_to_cards(docs, include_full_emails: bool = False) -> List[Dict[str, Any]]:
    # Collect unique thread IDs from RAG hits
    seen = set()
    tids: List[str] = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        tid = md.get("thread_id")
        if tid and tid not in seen:
            seen.add(tid)
            tids.append(tid)

    # Load only the needed summaries
    sums = _get_summaries_map(tids)
    
    # Always get the thread->email_ids map so we can count emails (no bodies loaded)
    thread_email_map: Dict[str, List[str]] = _get_threads_map(tids) if tids else {}

    # Optionally load only the needed thread->email_ids and their bodies
    emails_by_id: Dict[str, dict] = {}
    if include_full_emails and tids:
        thread_email_map = _get_threads_map(tids)
        emails_by_id = _emails_for_threads(tids)

    # Build unique cards per thread
    cards: List[Dict[str, Any]] = []
    consumed = set()
    
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        tid = md.get("thread_id")
        
        if tid:
            # Existing thread card flow (dedupe per thread)
            if tid in consumed:
                continue
            consumed.add(tid)

            meta = sums.get(tid, {})
            card: Dict[str, Any] = {
                "thread_id": tid,
                "score": md.get("score"),
                "subject": meta.get("subject", ""),
                "topic": meta.get("topic", ""),
                "detail1": meta.get("detail1", ""),
                "detail2": meta.get("detail2", ""),
                "link": f"/threads/{tid}",
                "emails_count": len(thread_email_map.get(tid, [])),
            }
            if include_full_emails:
                email_ids = thread_email_map.get(tid, [])
                fulls = []
                for eid in email_ids:
                    data = emails_by_id.get(eid) or {}
                    fulls.append({
                        "email_id": eid,
                        "subject": data.get("subject", ""),
                        "sender": data.get("sender", ""),
                        "receiver": data.get("receiver", ""),
                        "date": data.get("date", ""),
                        "clean_body": data.get("body", ""),
                    })
                card["emails"] = email_ids
                card["full_emails"] = fulls
            cards.append(card)
            continue
    
        # Document paragraph hit (no thread_id)
        file_path = md.get("file_path") or md.get("source")
        if not file_path:
            continue

        cards.append({
            "kind": "doc",
            "score": md.get("score"),
            "doc_path": file_path,
            "doc_name": os.path.basename(file_path),
            "page": md.get("page"),
            "paragraph_index": md.get("paragraph_index"),
            "paragraph": (d.page_content or "").strip(),
        })
        
    return cards

def build_context_from_cards(cards: List[Dict[str, Any]], mode: str = "summary") -> str:
    blocks: List[str] = []
    for c in cards:
        tid = c.get("thread_id", "")
        if tid:
            if mode == "summary":
                part = "\n".join([
                    f"Thread ID: {tid}",
                    *(f"{k.capitalize()}: {v}" for k, v in (
                        ("subject", c.get("subject")),
                        ("topic", c.get("topic")),
                        ("detail1", c.get("detail1")),
                        ("detail2", c.get("detail2"))
                    ) if v)
                ])
                blocks.append(part)
            elif mode == "full":
                fulls = c.get("full_emails") or []
                if not fulls:
                    part = "\n".join([
                        f"Thread ID: {tid}",
                        *(f"{k.capitalize()}: {v}" for k, v in (
                            ("subject", c.get("subject")),
                            ("topic", c.get("topic")),
                            ("detail1", c.get("detail1")),
                            ("detail2", c.get("detail2"))
                        ) if v)
                    ])
                    blocks.append(part)
                    continue
                email_blocks = []
                for i, em in enumerate(fulls, 1):
                    header = "\n".join([
                        f"Thread ID: {tid}",
                        f"Email {i}/{len(fulls)}",
                        f"Subject: {em.get('subject','')}",
                        f"From: {em.get('sender','')}",
                        f"To: {em.get('receiver','')}",
                        f"Date: {em.get('date','')}",
                    ])
                    body = em.get("clean_body", "") or ""
                    email_blocks.append(f"{header}\n\n{body}")
                blocks.append("\n\n---\n\n".join(email_blocks))
            elif mode == "hybrid":
                head = "\n".join([
                    f"Thread ID: {tid}",
                    *(f"{k.capitalize()}: {v}" for k, v in (
                        ("subject", c.get("subject")),
                        ("topic", c.get("topic")),
                        ("detail1", c.get("detail1")),
                        ("detail2", c.get("detail2"))
                    ) if v)
                ])
                tail = ""
                fulls = c.get("full_emails") or []
                if fulls:
                    em = fulls[0]
                    tail = "\n\n" + "\n".join([
                        f"Email 1/{len(fulls)}",
                        f"From: {em.get('sender','')}",
                        f"Date: {em.get('date','')}",
                        "",
                        em.get("clean_body","")
                    ])
                blocks.append(head + tail)
            continue
            
        # Document paragraph context
        if c.get("doc_path"):
            part = "\n".join(filter(None, [
                f"Document: {c.get('doc_name','')}",
                f"Page: {c.get('page')}" if c.get('page') is not None else "",
                "",
                c.get("paragraph","")
            ]))
            blocks.append(part)
    
    return "\n\n======\n\n".join([b for b in blocks if b])

def search_threads(query: str, top_k: int = RESULTS_QTY, min_score: float = RAG_THRESHOLD, include_full_emails: bool = False) -> List[Dict[str, Any]]:
    docs = run_rag_question(query, top_k=top_k, min_score=min_score)
    if not docs:
        return []
    return docs_to_cards(docs, include_full_emails=include_full_emails)

def get_thread_detail(thread_id: str) -> Optional[Dict[str, Any]]:
    # Lazy-load just the requested thread's email_ids and summary
    email_ids = _get_threads_map([thread_id]).get(thread_id, [])
    meta = _thread_meta(thread_id)
    # If we truly have nothing for this thread, treat as not found
    if not email_ids and not any(meta.values()):
        return None
    return {
        "thread_id": thread_id,
        "topic": meta.get("topic", ""),
        "detail1": meta.get("detail1", ""),
        "detail2": meta.get("detail2", ""),
        "email_ids": email_ids,
        "subject": meta.get("subject", ""),
    }

def get_email_detail(email_id: str) -> Optional[Dict[str, Any]]:
    _ensure_emails_loaded([email_id])
    data = _email_cache.get(email_id)
    if not data:
        return None
    return {"email_id": email_id, "data": data}
