"""
Cloud Run compatible API (drop-in)
File name: api_gc.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
import os
import re
import logging
from dotenv import load_dotenv

load_dotenv()

# Import from rag_module
from rag_module import create_rag_chain, DATA_PATH, MAX_HISTORY_MESSAGES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# CORS - only allow Vercel and localhost (for local dev)
CORS(
    app,
    resources={r"/*": {"origins": ["https://geantai-prod.vercel.app", "https://www.geantai-prod.vercel.app"]}},
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)

# Global state
chat_histories = {}
rag_chain = None

def get_rag_chain():
    global rag_chain
    if rag_chain is None:
        try:
            logger.info("Initializing RAG chain with Groq (Cloud Run)...")

            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise RuntimeError("GROQ_API_KEY not set")

            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                groq_api_key=groq_api_key
            )
            logger.info("LLM initialized")

            # Use local HuggingFace embeddings
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name="./model_cache/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embeddings initialized")

            # Create the RAG chain
            logger.info("Creating RAG chain...")
            rag_chain = create_rag_chain(
                llm=llm,
                embeddings=embeddings,
                rebuild=False
            )
            logger.info(f"RAG chain created: {type(rag_chain)}")

            if rag_chain is None:
                raise RuntimeError("create_rag_chain returned None!")

            logger.info("RAG chain ready")
        except Exception as e:
            logger.error(f"Failed to initialize RAG chain: {e}", exc_info=True)
            raise
    
    return rag_chain



@app.route("/api/chat", methods=["POST"])
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

    logger.info(f"Question: {question[:60]}")

    answer, docs = chain(question, chat_history)

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    if len(chat_history) > MAX_HISTORY_MESSAGES:
        chat_histories[session_id] = chat_history[-MAX_HISTORY_MESSAGES:]

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
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "groq"}), 200


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "online",
        "message": "GEANT RAG API is running",
        "version": "1.0",
        "endpoints": ["/api/health", "/api/chat", "/api/pdf/<filename>"]
    }), 200


@app.route("/api/pdf/<path:filename>", methods=["GET"])
def serve_pdf(filename):
    try:
        return send_from_directory(DATA_PATH, filename)
    except Exception as e:
        logger.error(f"PDF error: {e}")
        return jsonify({"error": "PDF not found"}), 404


# Local dev only (Cloud Run ignores this)
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

