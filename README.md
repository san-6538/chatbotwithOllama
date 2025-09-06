# Ollama RAG Chatbot - RAG + Agent Chatbot

## 🚀 Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot with agent capabilities.  
It has:
- **FastAPI backend**
- **Streamlit frontend**
- **RAG pipeline with embeddings + vector search**
- **Agent integration**

---

## 🐳 Docker Setup

### Build Image
```bash
docker build -t ollama-chatbot .
```

### Run Container
```bash
docker run -p 8000:8000  ollama-chatbot
```

---

## 🔎 Endpoints
- FastAPI backend: `http://localhost:8000`
- Streamlit UI: `streamlit run ui.py`
