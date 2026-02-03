# GÉANT RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for querying GÉANT project documents using LangChain. Supports multiple interfaces (CLI, Flask API, Streamlit, React UI) and demonstrates full end-to-end deployment for organizational knowledge access.

## Purpose

This project accompanies a Master’s thesis investigating user-centred RAG chatbots for enterprise knowledge retrieval. It provides a fully working prototype for querying heterogeneous GÉANT documents, showing improvements in efficiency, effectiveness, and user satisfaction. The code illustrates key steps from data collection to local and cloud deployment.

## Features

- **Document scraping**: Extracts content from GÉANT websites using BeautifulSoup.  
- **Vector embeddings**: MiniL6 model used to generate embeddings for documents. Enables deployment using Groq API.  
- **Retrieval-Augmented Generation**: RAG pipeline built with LangChain, grounding answers in retrieved documents.  
- **Multiple interfaces**:
  - CLI interface (`main.py`)
  - Flask API for local hosting (`api.py`)
  - Streamlit web interface (`app.py`)
  - React frontend (TypeScript) hosted on Vercel
- **Deployment**:
  - Backend containerized with Docker and deployed on Google Cloud
  - Frontend deployed on Vercel
- **Local and cloud API**:
  - Local API connects to local Chroma vector database
  - Deployed API currently limited (database is local)

## Prerequisites

- **Python 3.9+**
- **Ollama** (https://ollama.ai) for running local LLMs
  - Pull models: `ollama pull llama3.2` and `ollama pull nomic-embed-text`
- **Node.js** (for React UI) https://nodejs.org
- **Docker** (for cloud backend deployment)

## Installation

1. Clone repository:
```bash
git clone <repository-url>
cd geant-rag