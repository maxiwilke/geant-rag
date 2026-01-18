from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
import os
import logging
from dotenv import load_dotenv

load_dotenv()  # Load .env file

# Embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

# Import from rag_module
from rag_module import create_rag_chain, DATA_PATH, MAX_HISTORY_MESSAGES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS logging
logging.getLogger('flask_cors').level = logging.DEBUG

# Create Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Session management
chat_histories = {}
rag_chain = None

def get_rag_chain():
    global rag_chain
    if rag_chain is None:
        logger.info("Initializing RAG chain (local Ollama)...")

        llm = ChatOllama(
            model="llama3.2:latest",
            temperature=0.7
        )

        rag_chain = create_rag_chain(
            llm=llm,
            embeddings=embeddings,
            rebuild=False
        )

        logger.info("RAG chain ready!")
    return rag_chain

# --- endpoints unchanged ---
@app.route('/api/chat', methods=['POST'])
@cross_origin()
def chat():
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

    return jsonify({'answer': answer, 'sources': sources, 'session_id': session_id}), 200

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    get_rag_chain()  # Initialize RAG on startup
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )