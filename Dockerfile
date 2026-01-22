FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api_gc.py .
COPY rag_module.py .

# Copy the pre-built vector database
COPY chroma_db/ ./chroma_db/

# Copy data folder (for PDFs)
COPY Data/ ./Data/

# ✅ Copy the downloaded model
COPY model_cache/ ./model_cache/

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HOME=/app/model_cache

# Use better gunicorn settings
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 300 api_gc:app