"""
Core RAG module - contains all the RAG logic
Embedding-agnostic: embeddings MUST be provided by caller
"""

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import shutil
import urllib.parse
from pathlib import Path
from typing import Optional, List, Tuple

# Progress bar support
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Install tqdm for progress bars: pip install tqdm")

# ---------------- Paths & constants
SCRIPT_DIR = Path(__file__).parent.absolute()
CHROMA_DB_PATH = SCRIPT_DIR / "chroma_db"
DATA_PATH = SCRIPT_DIR / "Data"
FACT_SHEET_PATH = DATA_PATH / "fact_sheet.txt"
STRUCTURAL_CONTEXT_PATH = DATA_PATH / "project_annex"  # No space in folder name

# Chat history configuration
MAX_HISTORY_MESSAGES = 6

# Chunking configuration
CHUNK_SIZE = 800  # Good balance of context and granularity
CHUNK_OVERLAP = 150  # More overlap to preserve continuity
RETRIEVAL_K = 14  # Total chunks: 4 structural + 10 regular
RETRIEVAL_K_EXTENDED = 20  # Retrieve more candidates for prioritization

# Embedding batch size for progress tracking
EMBEDDING_BATCH_SIZE = 100


# ---------------- Fact Sheet Loader

def load_fact_sheet() -> str:
    """Load fact sheet from text file"""
    try:
        if FACT_SHEET_PATH.exists():
            print(f"Loading fact sheet from {FACT_SHEET_PATH}")
            content = FACT_SHEET_PATH.read_text(encoding="utf-8")
            print(f"Fact sheet loaded: {len(content)} characters")
            return content
        else:
            print(f"Warning: Fact sheet not found at {FACT_SHEET_PATH}")
            return "No fact sheet available."
    except Exception as e:
        print(f"Error loading fact sheet: {e}")
        return "No fact sheet available."

FACT_SHEET = load_fact_sheet()


# ---------------- Loaders

class UTF8TextLoader(TextLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoding = "utf-8"


# ---------------- Utilities

def extract_url_from_document(text: str) -> Optional[str]:
    """Extract URL from document text (expected to be in format 'URL: https://...')"""
    for line in text.split("\n"):
        if line.startswith("URL:"):
            url = line.replace("URL:", "").strip()
            return url if url else None
    return None


def build_local_pdf_url(source_path: str) -> Optional[str]:
    """
    Build URL for PDFs using hybrid approach:
    1. Check for public URL in .url.txt file (for Zenodo articles)
    2. Fall back to API server
    """
    try:
        path = Path(source_path).resolve()
        rel = path.relative_to(DATA_PATH.resolve())
        
        # Check for public URL file (common pattern: filename.pdf.url.txt)
        url_file = Path(str(path) + ".url.txt")
        if url_file.exists():
            try:
                with open(url_file, 'r', encoding='utf-8') as f:
                    public_url = f.read().strip()
                    if public_url and public_url.startswith('http'):
                        return public_url
            except Exception:
                pass
        
        # Fall back to API server (use env var for flexibility)
        api_base = os.getenv("API_BASE_URL", "http://localhost:5000")
        return f"{api_base}/api/pdf/{urllib.parse.quote(rel.as_posix())}"
    except Exception:
        return None


def format_chat_history(chat_history: List) -> str:
    if not chat_history:
        return "No previous conversation."

    lines = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines)


# ---------------- Vector DB

def setup_vector_db(*, embeddings, rebuild: bool) -> Chroma:
    """
    Create or load Chroma DB using ONLY the provided embeddings.
    Attaches URL metadata to documents for clickable references.
    Marks structural context PDFs with special metadata for prioritization.
    """

    if embeddings is None:
        raise ValueError("Embeddings must be explicitly provided")

    # Hard reset on rebuild
    if rebuild and CHROMA_DB_PATH.exists():
        print("Rebuild requested → deleting existing Chroma DB")
        shutil.rmtree(CHROMA_DB_PATH)

    # Try loading existing DB
    if CHROMA_DB_PATH.exists() and not rebuild:
        print("Loading existing vector database...")
        vectordb = Chroma(
            persist_directory=str(CHROMA_DB_PATH),
            embedding_function=embeddings
        )

        try:
            count = vectordb._collection.count()
            print(f"Found {count} vectors in database")
            if count > 0:
                return vectordb
        except Exception:
            pass

        print("Existing DB invalid or empty → rebuilding")
        shutil.rmtree(CHROMA_DB_PATH)

    # ---------------- Build from scratch
    print("Creating new vector database...")

    documents = []

    if not DATA_PATH.exists():
        raise RuntimeError("Data directory does not exist")

    # Text files - with URL extraction
    print("Loading text documents...")
    text_docs = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=UTF8TextLoader
    ).load()

    # Filter out .url.txt files and fact_sheet (it's loaded separately)
    text_docs = [doc for doc in text_docs 
                if not doc.metadata.get('source', '').endswith('.url.txt')
                and 'fact_sheet.txt' not in doc.metadata.get('source', '')]

    # Extract URLs and add to metadata
    for doc in text_docs:
        url = extract_url_from_document(doc.page_content)
        if url:
            doc.metadata["url"] = url
        doc.metadata["document_type"] = "regular"

    documents.extend(text_docs)
    print(f"Loaded {len(text_docs)} text documents")

    # PDFs - with hybrid URL generation (excluding fact_sheet)
    print("Loading PDF documents...")
    pdf_docs = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    ).load()

    # Filter out fact_sheet.pdf (it's loaded separately as authoritative source)
    pdf_docs = [doc for doc in pdf_docs 
                if 'fact_sheet' not in doc.metadata.get('source', '').lower()]

    # Build URLs and add to metadata
    structural_pdfs_found = set()
    for doc in pdf_docs:
        source_path = doc.metadata.get("source")
        if source_path:
            url = build_local_pdf_url(source_path)
            if url:
                doc.metadata["url"] = url
            
            # Mark structural context PDFs
            if STRUCTURAL_CONTEXT_PATH.exists() and str(STRUCTURAL_CONTEXT_PATH) in source_path:
                doc.metadata["document_type"] = "structural_context"
                doc.metadata["priority"] = "high"
                # Only print once per unique PDF file
                pdf_name = Path(source_path).name
                if pdf_name not in structural_pdfs_found:
                    structural_pdfs_found.add(pdf_name)
                    print(f"  → Marked as STRUCTURAL CONTEXT: {pdf_name}")
            else:
                doc.metadata["document_type"] = "regular"

    documents.extend(pdf_docs)
    print(f"Loaded {len(pdf_docs)} PDF documents")

    if not documents:
        raise RuntimeError("No documents found to index")

    print(f"Total documents: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Embed documents in batches with progress bar
    print(f"\nGenerating embeddings for {len(chunks)} chunks...")
    print("This may take 10-30 minutes depending on your CPU/GPU...")
    
    vectordb = None
    
    if HAS_TQDM:
        # With progress bar
        batch_iterator = tqdm(
            range(0, len(chunks), EMBEDDING_BATCH_SIZE),
            desc="Embedding chunks",
            unit="batch"
        )
    else:
        # Without progress bar - just print updates
        batch_iterator = range(0, len(chunks), EMBEDDING_BATCH_SIZE)
        total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    
    for i, batch_start in enumerate(batch_iterator):
        batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, len(chunks))
        batch = chunks[batch_start:batch_end]
        
        if vectordb is None:
            # Create initial DB with first batch
            vectordb = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(CHROMA_DB_PATH)
            )
        else:
            # Add subsequent batches
            vectordb.add_documents(batch)
        
        # Print progress if no tqdm
        if not HAS_TQDM:
            print(f"  Processed batch {i+1}/{total_batches} ({batch_end}/{len(chunks)} chunks)")

    print(f"\n✓ Vector DB created with {len(chunks)} vectors")
    return vectordb


# ---------------- RAG chain

def create_rag_chain(*, llm, embeddings, rebuild: bool = False):
    """
    Create RAG chain using caller-provided LLM and embeddings.
    Returns documents with URL metadata for clickable references.
    Prioritizes structural context documents in retrieval.
    """

    vectordb = setup_vector_db(
        embeddings=embeddings,
        rebuild=rebuild
    )

    # Retrieve more documents to allow prioritization
    retriever = vectordb.as_retriever(
        search_kwargs={"k": RETRIEVAL_K_EXTENDED}
    )

    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are analyzing excerpts from GÉANT network project documentation.

DOCUMENTS YOU ARE ANALYZING:
- GN5-1 Technical Annex (GÉANT flagship project phase 1)
- GN5-2 Technical Annex (GÉANT flagship project phase 2)
- GÉANT Project Fact Sheet

YOUR TASK: Answer questions using ONLY the document excerpts provided below. These excerpts are real text extracted from the actual project documents.

FACT SHEET:
{fact_sheet}

PREVIOUS MESSAGES:
{chat_history}

===== DOCUMENT EXCERPTS FROM: {source_list} =====
{context}
===== END OF DOCUMENT EXCERPTS =====

When asked about GN5-1 or GN5-2, use the excerpts above. DO NOT claim these are unknown projects."""
    ),
    ("human", "{question}")
])

    def rag_chain(question: str, chat_history: List = None) -> Tuple[str, List]:
        chat_history = chat_history or []

        # Retrieve extended set of documents
        all_docs = retriever.invoke(question)

        # Separate structural vs regular docs
        structural_docs = [d for d in all_docs if d.metadata.get("document_type") == "structural_context"]
        regular_docs = [d for d in all_docs if d.metadata.get("document_type") != "structural_context"]

        # Prioritize structural context - 4 structural + 10 regular
        max_structural = min(len(structural_docs), 4)  # Up to 4 structural chunks
        remaining_slots = RETRIEVAL_K - max_structural
        
        docs = structural_docs[:max_structural] + regular_docs[:remaining_slots]
        
        if structural_docs:
            print(f"→ Retrieved {len(structural_docs[:max_structural])} structural context chunks + {len(regular_docs[:remaining_slots])} regular chunks")

        sources = []
        seen = set()
        for doc in docs:
            src = doc.metadata.get("source", "Unknown").replace("\\", "/").split("/")[-1]
            if src not in seen:
                # Mark structural sources
                if doc.metadata.get("document_type") == "structural_context":
                    sources.append(f"{src} [STRUCTURAL]")
                else:
                    sources.append(src)
                seen.add(src)

        source_list = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sources))

        context = "\n\n".join(
            f"--- Source: {doc.metadata.get('source', 'Unknown')} [{doc.metadata.get('document_type', 'regular').upper()}] ---\n{doc.page_content}"
            for doc in docs
        )

        messages = prompt.format_messages(
            question=question,
            fact_sheet=FACT_SHEET,
            context=context,
            chat_history=format_chat_history(chat_history),
            source_list=source_list
        )

        result = llm.invoke(messages)
        answer = result.content if hasattr(result, "content") else str(result)

        return answer, docs

    return rag_chain