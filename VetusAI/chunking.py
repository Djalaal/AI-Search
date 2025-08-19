import tiktoken
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

MODEL_NAME = "nomic-embed-text"
IDEAL_TOKENS = 1000
IDEAL_OVERLAP = 100
   
    
# Token estimation using tiktoken
def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# Chunk text by paragraph boundaries
def chunk_by_paragraph(
    text: str,
    header: str = "",
    size: int = IDEAL_TOKENS,
    overlap: int = IDEAL_OVERLAP,
    seperators: list[str] = ["\n\n\n", "\n\n", "\n", ".", " ", ""],
    with_header: bool = True) -> list[str]:
    
    paragraph_chunker = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=seperators,  # prioritize paragraph-level splits
    )
    
    docs = paragraph_chunker.create_documents([text])
    
    chunks = []
    for i, doc in enumerate(docs):
        if with_header and i > 0:
            chunks.append(header + doc.page_content)
        else:
            chunks.append(doc.page_content)
    else:
        return chunks


# Chunk text using semantic similarity
def chunk_by_semantics(
    text: str,
    header: str = "",
    model_name: str = MODEL_NAME,
    with_header: bool = True) -> list[str]:
    
    semantic_chunker = SemanticChunker(
        embeddings=OllamaEmbeddings(model=model_name),
        breakpoint_threshold_amount=0.5,
    )
    
    docs = semantic_chunker.create_documents([text])
    
    chunks = []
    for i, doc in enumerate(docs):
        if with_header and i > 0:
            chunks.append(header + doc.page_content)
        else:
            chunks.append(doc.page_content)
    
    return chunks