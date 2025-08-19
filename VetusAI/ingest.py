import os
import time
import json
import re
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from ollama_check import check_ollama
from chunking import chunk_by_semantics

# Activate the virtual python environment using: .venv\Scripts\activate
# The purpose of this code is the scan, process and embed raw documents, and store them in a vector database

# === Configuration ===
OLLAMA_MODEL = "nomic-embed-text"
DOCUMENTS_DIR = "C:/Users/DLont/Desktop/AI Tools/VetusAI/documents"
SUMMARIES_FILE = "./processed/email_summaries.jsonl"
VECTOR_DB_DIR = "./vectorstore"
CHUNKS_FILE = os.path.join(VECTOR_DB_DIR, "email_chunks.jsonl")


    
def save_chunks_jsonl(chunks: List[Document], output_path: str = CHUNKS_FILE):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in chunks:
            json.dump({
                "content": doc.page_content,
                "metadata": {k: str(v) for k, v in doc.metadata.items()}
            }, f, ensure_ascii=False)
            f.write("\n")
    print(f"✅ Saved {len(chunks)} chunks to {output_path}")
    
def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; normalize whitespace; drop empties."""
    if not text:
        return []
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = re.split(r"\n\s*\n", t)
    return [p.strip() for p in paras if p.strip()]

def _split_lines(string: str) -> list[str]:
    """Split on newline and trim; drop empties."""
    if not string:
        return []
    return [line.strip() for line in string.replace("\r\n", "\n").split("\n") if line.strip()]

def _add_line_docs(docs, lines, segment, metadata, subject, topic):
    """Create one embedding per line, with short context for recall."""
    n = len(lines)
    for i, line in enumerate(lines, start=1):
        # Put the bullet text first (most weight), then short context
        page = f"{line}\nSubject: {subject}\nTopic: {topic}\nSegment: {segment}"
        docs.append(
            Document(
                page_content=page,
                metadata={**metadata,
                          "segment": f"{segment}_line",
                          "line_index": i,
                          "line_total": n}
            )
        )
    return n

# === Step 1: Load all supported documents ===
def load_documents() -> List[Document]:
    docs = []

    # === Load standard files ===
    for root, _, files in os.walk(DOCUMENTS_DIR):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                ext = filename.lower()
                if ext.endswith(".pdf"):
                    # CHANGED: split PDF pages into paragraphs
                    loader = PyPDFLoader(filepath)
                    loaded_pages = loader.load()
                    count = 0
                    for page_doc in loaded_pages:
                        page_num = page_doc.metadata.get("page")
                        for i, para in enumerate(_split_paragraphs(page_doc.page_content), start=1):
                            docs.append(
                                Document(
                                    page_content=para,
                                    metadata={
                                        **page_doc.metadata,
                                        "type": "file",
                                        "source": filepath,            # ensure file path present
                                        "file_path": filepath,
                                        "file_name": os.path.basename(filepath),
                                        "page": page_num,
                                        "paragraph_index": i,
                                    }
                                )
                            )
                            count += 1
                    print(f"✅ Loaded: {filepath} ({count} paragraphs doc)")
                    continue  # IMPORTANT: skip the old `docs.extend(loaded)` for PDFs

                elif ext.endswith(".txt"):
                    loader = TextLoader(filepath)
                    loaded = loader.load()
                    docs.extend(loaded)
                    print(f"✅ Loaded: {filepath} ({len(loaded)} docs)")
                elif ext.endswith((".docx", ".doc")):
                    loader = UnstructuredWordDocumentLoader(filepath)
                    loaded = loader.load()
                    docs.extend(loaded)
                    print(f"✅ Loaded: {filepath} ({len(loaded)} docs)")
                elif ext.endswith((".xlsx", ".xls")):
                    loader = UnstructuredExcelLoader(filepath)
                    loaded = loader.load()
                    docs.extend(loaded)
                    print(f"✅ Loaded: {filepath} ({len(loaded)} docs)")
                else:
                    continue
            except Exception as e:
                print(f"❌ Failed to load {filepath}: {e}")

    # === Load email summaries (only subject, topic, detail1, detail2) ===
    total_emails = 0
    if os.path.exists(SUMMARIES_FILE):
        with open(SUMMARIES_FILE, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                try:
                    data = json.loads(line)
                    thread_id = data.get("thread_id", f"Thread-{i}")
                    subject = data.get("subject", "").strip()
                    topic = data.get("topic", "").strip()
                    detail1 = data.get("detail1", "").strip()
                    detail2 = data.get("detail2", "").strip()

                    metadata = {
                        "type": "email_summary",
                        "source": os.path.basename(SUMMARIES_FILE),
                        "thread_id": thread_id
                    }
                    total_emails += 1

                    # Embed each field as its own document
                    if subject:
                        docs.append(Document(page_content=subject, metadata={**metadata, "segment": "subject"}))
                    if topic and topic != "...":
                        docs.append(Document(page_content=topic, metadata={**metadata, "segment": "topic"}))
                    if detail1 and detail1 != "...":
                        docs.append(Document(page_content=detail1, metadata={**metadata, "segment": "detail1"}))
                        _add_line_docs(docs, _split_lines(detail1), "detail1", metadata, subject, topic)
                    if detail2 and detail2 != "...":
                        docs.append(Document(page_content=detail2, metadata={**metadata, "segment": "detail2"}))
                        _add_line_docs(docs, _split_lines(detail2), "detail2", metadata, subject, topic)


                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping malformed JSON on line {i}: {e}")


    # Save the total number of chunks
    total_chunks = len(docs)
    with open(os.path.join(VECTOR_DB_DIR, "total_chunks.txt"), "w", encoding="utf-8") as f:
        f.write(str(int(total_chunks)))

    print(f"✅ Loaded {total_chunks} total chunks from {total_emails} email summaries and files")
    return docs


# === Step 2: Embed chunks using Ollama ===
def embed_documents(docs: List[Document], model: str = OLLAMA_MODEL):
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    embeddings = OllamaEmbeddings(model=model)
    db = FAISS.from_documents(docs, embeddings)
    print(f"✅ Embedding {len(docs)} documents complete")
    return db


# === Step 3: Store vector DB locally ===
def save_vectorstore(db: FAISS, db_dir: str = VECTOR_DB_DIR):
    db.save_local(db_dir)
    print(f"✅ Vectorstore saved to {db_dir}")
    with open(os.path.join(VECTOR_DB_DIR, "last_updated.txt"), "w") as f:
        f.write(str(int(time.time())))


# === Pipeline Trigger ===
if __name__ == "__main__":
    print("--- Checking requirements ---")
    check_ollama(OLLAMA_MODEL)

    print("--- Loading documents ---")
    docs = load_documents()

    print("--- Embedding documents ---")
    vector_db = embed_documents(docs)

    print("--- Saving vector database ---")
    save_vectorstore(vector_db)

    print("--- Ingestion is done ---")