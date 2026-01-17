from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import urllib.parse
from pathlib import Path
from typing import Optional
import logging

# Import from rag_module
from rag_module import create_rag_chain, DATA_PATH, MAX_HISTORY_MESSAGES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# SIMPLE CORS - as recommended in Flask-CORS docs
CORS(app)
cors = CORS(app, resources={
    r"/*": {
        "origins": "*"
    }
})

# Session management
chat_histories = {}
rag_chain = None

def get_rag_chain():
    """Lazy load the RAG chain only when first needed"""
    global rag_chain
    if rag_chain is None:
        logger.info("Initializing RAG chain for first time...")
        rag_chain = create_rag_chain(rebuild=False)
        logger.info("RAG chain ready!")
    return rag_chain

# ---------------- API ENDPOINTS

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'GEANT RAG API is running',
        'endpoints': ['/api/health', '/api/chat', '/api/clear_history'],
        'cors': 'enabled'
    }), 200

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint - returns immediately without loading RAG"""
    logger.info("Health check requested")
    return jsonify({'status': 'healthy', 'message': 'API is running'}), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.json
        question = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not question:
            logger.warning("Empty message received")
            return jsonify({'error': 'No message provided'}), 400

        logger.info(f"[Session: {session_id}] Question: {question}")
        
        # Lazy load RAG chain
        chain = get_rag_chain()
        
        # Get or create chat history for this session
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        
        chat_history = chat_histories[session_id]
        
        # Get answer
        answer, docs = chain(question, chat_history)
        
        # Update chat history
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
        
        # Keep only last N messages
        if len(chat_history) > MAX_HISTORY_MESSAGES:
            chat_histories[session_id] = chat_history[-MAX_HISTORY_MESSAGES:]
        
        # Format sources
        sources = []
        if docs:
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                source = source.replace('\\', '/').split('/')[-1]
                url = doc.metadata.get('url', None)
                sources.append({'name': source, 'url': url})
        
        logger.info(f"Returning answer with {len(sources)} sources")
        return jsonify({
            'answer': answer,
            'sources': sources,
            'session_id': session_id
        }), 200

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear chat history for a session"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id in chat_histories:
            chat_histories[session_id] = []
        
        logger.info(f"Cleared history for session: {session_id}")
        return jsonify({'message': 'History cleared', 'session_id': session_id}), 200
        
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pdf/<path:filename>', methods=['GET'])
def serve_pdf(filename):
    """Serve PDF files"""
    if not filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Not a PDF'}), 400
    
    logger.info(f"Serving PDF: {filename}")
    return send_from_directory(DATA_PATH, filename)

# This block only runs with 'python api.py', not with gunicorn
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask app on port {port}...")
    logger.info("Note: RAG chain will initialize on first chat request")
    app.run(debug=False, host='0.0.0.0', port=port)

# Log initialization
logger.info("Flask app initialized with Flask-CORS")