"""
Core RAG module - contains all the RAG logic
Updated to use Groq API instead of local Ollama
"""

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import urllib.parse
from pathlib import Path
from typing import Optional, List, Tuple

# ---------------- Paths & constants
SCRIPT_DIR = Path(__file__).parent.absolute()
CHROMA_DB_PATH = SCRIPT_DIR / "chroma_db"
DATA_PATH = SCRIPT_DIR / "Data"

# Groq configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"  # Fast and good quality

# Chat history configuration
MAX_HISTORY_MESSAGES = 6

# Chunking configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
RETRIEVAL_K = 8

# Custom TextLoader with UTF-8 encoding
class UTF8TextLoader(TextLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoding = "utf-8"


def extract_url_from_document(text: str) -> Optional[str]:
    """Extract URL from document text if present"""
    lines = text.split('\n')
    for line in lines:
        if line.startswith('URL:'):
            url = line.replace('URL:', '').strip()
            return url if url else None
    return None


def build_local_pdf_url(source_path: str) -> Optional[str]:
    """Build local URL for PDF files"""
    try:
        path = Path(source_path).resolve()
        rel = path.relative_to(DATA_PATH.resolve())
    except Exception:
        return None
    return f"https://geant-rag.onrender.com/api/pdf/{urllib.parse.quote(rel.as_posix())}"


def format_chat_history(chat_history: List) -> str:
    """Format chat history into a readable string for context"""
    if not chat_history:
        return "No previous conversation."

    formatted = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
    return "\n".join(formatted)


def setup_vector_db(embeddings, rebuild: bool = False):
    """Load documents from Data folder and create/load vector database"""

    if os.path.exists(CHROMA_DB_PATH) and not rebuild:
        print("Loading existing vector database...")
        vectordb = Chroma(
            persist_directory=str(CHROMA_DB_PATH),
            embedding_function=embeddings
        )
        try:
            collection_count = vectordb._collection.count() if hasattr(vectordb, '_collection') else 0
            print(f"Found {collection_count} documents in database")
            if collection_count > 0:
                return vectordb
        except:
            pass
        print("Database is empty, rebuilding...")

    if rebuild and os.path.exists(CHROMA_DB_PATH):
        print("Rebuild flag set - deleting existing database...")
        import shutil
        shutil.rmtree(CHROMA_DB_PATH)

    print("Creating new vector database...")
    documents = []

    # Load text files
    if os.path.exists(DATA_PATH):
        text_loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=UTF8TextLoader)
        text_docs = text_loader.load()
        for doc in text_docs:
            url = extract_url_from_document(doc.page_content)
            if url:
                doc.metadata['url'] = url
        documents.extend(text_docs)
        print(f"Loaded {len(text_docs)} text documents")

        # Load PDF files
        pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
        pdf_docs = pdf_loader.load()
        for doc in pdf_docs:
            source_path = doc.metadata.get('source')
            if source_path:
                url = build_local_pdf_url(source_path)
                if url:
                    doc.metadata['url'] = url
        documents.extend(pdf_docs)
        print(f"Loaded {len(pdf_docs)} PDF documents")

    print(f"Total: {len(documents)} documents loaded")

    if not documents:
        print("ERROR: No documents found in Data folder!")
        return None

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Create and persist vector database
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DB_PATH)
    )
    print(f"Created vector database with {len(chunks)} document chunks")
    return vectordb


def create_rag_chain(llm, embeddings, rebuild: bool = False):
    """
    Create RAG chain with provided LLM and embeddings
    """
    vectordb = setup_vector_db(
        embeddings=embeddings,
        rebuild=rebuild
    )

    if vectordb is None:
        raise ValueError("Failed to initialize vector database")

    retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context.
Use the context below to answer the question. If you cannot find the answer in the context, say so.

Previous conversation:
{chat_history}

Context from documents:
{context}

Available sources:
{source_list}"""),
        ("human", "{question}")
    ])

    def rag_chain(question: str, chat_history: List = None) -> Tuple[str, List]:
        if chat_history is None:
            chat_history = []

        # Retrieve relevant documents
        docs = retriever.invoke(question)

        # Build sources list
        unique_sources = []
        seen = set()
        for doc in docs:
            source_file = doc.metadata.get('source', 'Unknown').replace('\\', '/').split('/')[-1]
            if source_file not in seen:
                unique_sources.append(source_file)
                seen.add(source_file)

        source_list = "\n".join([f"{i+1}. {source}" for i, source in enumerate(unique_sources)])

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source_file = doc.metadata.get('source', 'Unknown').replace('\\', '/').split('/')[-1]
            context_parts.append(f"--- Document {i} (Source: {source_file}) ---\n{doc.page_content}\n")
        context = "\n".join(context_parts)

        # Format chat history
        formatted_history = format_chat_history(chat_history)

        # Create the prompt
        messages = prompt_template.format_messages(
            context=context,
            question=question,
            chat_history=formatted_history,
            source_list=source_list
        )

        # Call LLM
        result = llm.invoke(messages)

        # Extract answer text
        answer_text = result.content if hasattr(result, "content") else str(result)
        
        return answer_text, docs

    return rag_chain