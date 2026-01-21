from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
import logging
from dotenv import load_dotenv
from pathlib import Path
import urllib.parse
from typing import Optional  # Add this import

load_dotenv()

# Import from rag_module
from rag_module import create_rag_chain, DATA_PATH, MAX_HISTORY_MESSAGES

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Flask app
# -------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# -------------------------------------------------------------------
# Global state
# -------------------------------------------------------------------
chat_histories = {}
rag_chain = None

# -------------------------------------------------------------------
# URL utility functions
# -------------------------------------------------------------------
def extract_url_from_document(text: str) -> Optional[str]:  # Changed from str | None
    """Extract URL from document text (expected to be in format 'URL: https://...')"""
    lines = text.split('\n')
    for line in lines:
        if line.startswith('URL:'):
            url = line.replace('URL:', '').strip()
            return url if url else None
    return None

def build_local_pdf_url(source_path: str) -> Optional[str]:  # Changed from str | None
    """Build a local PDF URL for files in the Data/ directory"""
    try:
        path = Path(source_path).resolve()
        rel = path.relative_to(Path(DATA_PATH).resolve())
    except Exception:
        return None
    return f"http://localhost:5000/api/pdf/{urllib.parse.quote(rel.as_posix())}"

# -------------------------------------------------------------------
# RAG initialization (LOCAL)
# -------------------------------------------------------------------
def get_rag_chain():
    global rag_chain
    if rag_chain is None:
        logger.info("Initializing LOCAL RAG chain (Ollama + HF embeddings)...")

        # Local LLM via Ollama
        llm = ChatOllama(
            model="llama3.2:latest",
            temperature=0.7
        )

        # ✅ Stable local embeddings (NO Ollama dependency)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )

        rag_chain = create_rag_chain(
            llm=llm,
            embeddings=embeddings,
            rebuild=False
        )

        logger.info("Local RAG chain ready")

    return rag_chain

# -------------------------------------------------------------------
# PDF serving endpoint
# -------------------------------------------------------------------
@app.route('/api/pdf/<path:filename>', methods=['GET'])
def serve_pdf(filename):
    """Serve local PDF files from the Data directory."""
    if not filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Not a PDF'}), 400
    try:
        return send_from_directory(DATA_PATH, filename)
    except FileNotFoundError:
        return jsonify({'error': 'PDF not found'}), 404

# -------------------------------------------------------------------
# Chat endpoint
# -------------------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
@cross_origin()
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not question:
        return jsonify({"error": "No message provided"}), 400

    chain = get_rag_chain()

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    chat_history = chat_histories[session_id]

    logger.info(f"Question: {question[:80]}")

    answer, docs = chain(question, chat_history)

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    if len(chat_history) > MAX_HISTORY_MESSAGES:
        chat_histories[session_id] = chat_history[-MAX_HISTORY_MESSAGES:]

    # Format sources with URLs
    sources = []
    if docs:
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            source = source.replace("\\", "/").split("/")[-1]
            url = doc.metadata.get("url")
            sources.append({"name": source, "url": url})

    return jsonify({
        "answer": answer,
        "sources": sources,
        "session_id": session_id
    }), 200

# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mode": "local"}), 200

# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting LOCAL Flask server...")
    get_rag_chain()  # warm-up
    app.run(host="0.0.0.0", port=5000, debug=True)