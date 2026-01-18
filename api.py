"""
API for deployment (Render/Vercel) - Uses Groq instead of Ollama
Simplified CORS configuration to avoid conflicts
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Import from rag_module
from rag_module import create_rag_chain, DATA_PATH, MAX_HISTORY_MESSAGES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Disable strict slashes - allows /api/chat and /api/chat/ to work the same
app.url_map.strict_slashes = False

# SIMPLIFIED CORS - Let Flask-CORS handle everything
CORS(app)

# Session management
chat_histories = {}
rag_chain = None

def get_rag_chain():
    global rag_chain
    if rag_chain is None:
        logger.info("Initializing RAG chain with Groq (deployed)...")

        # Check for API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables!")

        # Use Groq for LLM (works remotely)
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            groq_api_key=groq_api_key
        )

        # Use HuggingFace embeddings (works remotely)
        logger.info("Loading HuggingFace embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        rag_chain = create_rag_chain(
            llm=llm,
            embeddings=embeddings,
            rebuild=False
        )

        logger.info("RAG chain ready!")
    return rag_chain


@app.route('/', methods=['GET'])
def root():
    """Root endpoint to verify server is running"""
    return jsonify({
        'status': 'online',
        'message': 'GEANT RAG API is running',
        'version': '1.0',
        'endpoints': ['/api/health', '/api/chat', '/api/pdf/<filename>']
    }), 200


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'groq'}), 200


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')

        if not question:
            return jsonify({'error': 'No message provided'}), 400

        chain = get_rag_chain()
        if chain is None:
            return jsonify({'error': 'RAG chain not initialized'}), 500

        if session_id not in chat_histories:
            chat_histories[session_id] = []

        chat_history = chat_histories[session_id]

        logger.info(f"Processing question: {question[:50]}...")
        answer, docs = chain(question, chat_history)

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

        if len(chat_history) > MAX_HISTORY_MESSAGES:
            chat_histories[session_id] = chat_history[-MAX_HISTORY_MESSAGES:]

        sources = []
        if docs:
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                source = source.replace('\\', '/').split('/')[-1]
                url = doc.metadata.get('url', None)
                sources.append({'name': source, 'url': url})

        return jsonify({
            'answer': answer,
            'sources': sources,
            'session_id': session_id
        }), 200
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/pdf/<path:filename>', methods=['GET'])
def serve_pdf(filename):
    """Serve PDF files from the Data directory"""
    try:
        return send_from_directory(DATA_PATH, filename)
    except Exception as e:
        logger.error(f"Error serving PDF: {str(e)}")
        return jsonify({'error': 'PDF not found'}), 404


if __name__ == '__main__':
    logger.info("Starting Flask server (deployment mode)...")
    
    # Initialize the RAG chain on startup
    try:
        get_rag_chain()
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {str(e)}")
    
    # Get port from environment variable (Render sets this)
    port = int(os.getenv('PORT', 5000))
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False  # Set to False in production
    )