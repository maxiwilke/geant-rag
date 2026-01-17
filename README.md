# GEANT RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for querying GEANT project documents using LangChain. Includes a Flask API and a React web UI, plus a CLI and Streamlit UI.

## Prerequisites

- **Python**: 3.9 or higher
- **Ollama**: Required for running local LLMs
  - Download from: https://ollama.ai
  - Pull required models: `ollama pull llama3.2` and `ollama pull nomic-embed-text`
- **Node.js**: Required for the React UI (install from https://nodejs.org)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd geant-rag
```

2. Create a virtual environment:
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure Ollama is running:
```bash
# Start Ollama service (if not already running)
ollama serve
```

## Usage

### Option 1: Command-line Chatbot (main.py)
```bash
python main.py
```

### Option 2: Flask API (api.py)
```bash
python api.py
```

API endpoints:
- `POST http://localhost:5000/api/chat` with JSON `{"message":"your question"}`
- `GET http://localhost:5000/api/health`

### Option 3: React Web UI (react-chatbot)
Start the API first, then:
```bash
cd react-chatbot
npm install
npm run dev
```
Open the URL shown by Vite (usually `http://localhost:5173`).

### Option 4: Streamlit Web Interface (app.py)
```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501`

## Project Structure

```
geant-rag/
main.py                 # CLI chatbot interface
api.py                  # Flask API (React UI connects here)
app.py                  # Streamlit web interface
react-chatbot/          # React UI
requirements.txt        # Python dependencies
Data/                   # Document storage
rag/                    # RAG pipeline notebooks
chroma_db/              # Vector database (auto-created)
assets/                 # UI assets
```

## Configuration

Edit the following variables in `api.py`, `main.py`, or `app.py` to customize:

- `LLM_MODEL`: LLM model to use (default: "llama3.2")
- `EMBEDDING_MODEL`: Embedding model (default: "nomic-embed-text")
- `DATA_PATH`: Path to document directory (default: "Data")
- `CHROMA_DB_PATH`: Vector database location (default: "./chroma_db")

## Troubleshooting

### Issue: Connection refused (Ollama not running)
**Solution**: Start Ollama service
```bash
ollama serve
```

### Issue: Models not found
**Solution**: Pull required models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Issue: Method Not Allowed at /api/chat
**Cause**: The endpoint only accepts POST, not GET.
**Solution**: Use the React UI or send a POST request with JSON.

### Issue: No documents found in Data folder
**Solution**: Ensure text files (.txt) or PDFs are in the `Data/` directory

### Issue: Chroma database errors
**Solution**: Delete the `chroma_db/` directory and re-run to rebuild
```bash
python main.py
```

## Development

The project includes Jupyter notebooks for development and preprocessing:
- `rag/rag_preprocessing.ipynb` - Document preprocessing pipeline
- `rag/rag_chunking_indexing.ipynb` - Text chunking and indexing

## License

Please refer to the LICENSE file for details.

## Notes

- The vector database is persisted in `chroma_db/` for performance
- First run will take longer as it processes all documents
- Ensure sufficient disk space for the vector database
