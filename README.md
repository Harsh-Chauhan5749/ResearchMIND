# 🧠 ResearchMind AI

> **Intelligent Research Paper Analysis & Knowledge Retrieval System**  
> Powered by Retrieval-Augmented Generation (RAG), Hybrid Vector Search, and Free/Local LLMs.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚀 Overview

ResearchMind AI is a production-patterned RAG application that turns academic PDFs into a **queryable, comparable, and searchable local knowledge base**. It is engineered to run completely offline (via Ollama) or in hybrid cloud mode (via Groq/Gemini APIs) with no paid dependencies.

---

## 🏆 Key Engineering Highlights (Interview Talking Points)

If presenting this project in a **placement or technical internship interview**, highlight these architectural decisions:

* **Hybrid Dense/Sparse Search (RRF):** Fuses dense semantic vector search (ChromaDB) with exact keyword matching (BM25) using **Reciprocal Rank Fusion (RRF)**. This mirrors production search systems (like Elasticsearch) and improves precision on technical/mathematical terms.
* **Fail-Safe Abstraction (Graceful Fallback):** Features a custom [core/llm_handler.py](file:///c:/Users/hpscu/OneDrive/Documents/Research_Mind/core/llm_handler.py) wrapper that handles API rate-limits or internet outages by **automatically falling back to local Ollama (Llama 3.2)** dynamically without interrupting the session.
* **Semantic Section-Aware Chunking:** Rather than splitting PDFs arbitrarily, the processor reads section headers (Abstract, Intro, Methods) and preserves paragraph boundaries with sliding-window overlaps to maintain context cohesion.
* **Strict Schema Integrity:** Mitigates common PyArrow serialization type conflicts in Streamlit dataframes by enforcing uniform string-casting on mixed database entries (like publication years).
* **Production DevOps Patterns:** Containerized using Docker/Docker-Compose for easy scaling and isolated frontend/backend deployments. Included FastAPI endpoints alongside the Streamlit UI.

---

## ✨ Core Features

* **Ingestion Pipeline:** Automatic text, table, and metadata extraction from uploaded PDFs or direct import via arXiv ID.
* **Streaming RAG Chat:** Conversational interface with source citations, relevance ratings, and confidence scoring.
* **Cross-Paper Comparison:** Side-by-side matrices comparing research methodologies, datasets, and performance metrics.
* **Hybrid Search Panel:** Combined keyword and semantic search query filters with CSV/JSON metadata exporting.
* **System Analytics:** Graphical dashboards representing system health, chunk statistics, and database ingestion timelines.

---

## 🏗️ Architecture Flow

```
┌─────────────────────────────────────────────────────────┐
│               Streamlit Frontend (5 pages)              │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │       RAG Pipeline          │
         │   Multi-query expansion     │
         │   Hybrid retrieval (RRF)    │
         │   Context assembly          │
         │   LLM generation            │
         │   Confidence scoring        │
         └──┬──────────────────┬───────┘
            │                  │
  ┌*********▼******┐  ┌******▼**********┐
  │   VectorStore   │  │   LLM Handler    │
  │  ChromaDB dense │  │  Groq  (free)    │
  │  BM25 keyword   │  │  Gemini (free)   │
  │  RRF fusion     │  │  Ollama (offline) │
  └─────────┬───────┘  └──────────────────┘
            │
  ┌*********▼***********┐
  │    PDF Processor     │
  │  PyMuPDF + PLumber   │
  │  Semantic chunking   │
  │  Section detection   │
  └─────────────────────┘
```

---

## ⚡ Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/Harsh-Chauhan5749/ResearchMIND.git
cd ResearchMIND
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory using the template:
```bash
cp .env.example .env
```
Open `.env` and choose your provider:
* **Groq (Cloud):** Set `LLM_PROVIDER=groq` and add your `GROQ_API_KEY`.
* **Gemini (Cloud):** Set `LLM_PROVIDER=gemini` and add your `GEMINI_API_KEY`.
* **Ollama (Local Offline):** Set `LLM_PROVIDER=ollama`. Run `ollama pull llama3.2` locally.

### 3. Run the App
* **Streamlit UI:**
  ```bash
  streamlit run streamlit_app.py
  ```
  Open **[http://localhost:8501](http://localhost:8501)** in your browser.
* **FastAPI Backend (Optional):**
  ```bash
  uvicorn api.main:app --reload
  ```

---

## 📁 Repository Structure

```
.
├── streamlit_app.py          # Streamlit UI home page
├── pages/
│   ├── 01_Upload_Papers.py       # PDF parsing & indexing
│   ├── 02_Chat_with_Papers.py     # Streaming Q&A Chat
│   ├── 03_Compare_Papers.py       # Comparison matrix
│   ├── 04_Knowledge_Search.py     # RRF search panel
│   └── 05_Analytics_Dashboard.py  # System graphs & stats
├── core/
│   ├── config.py             # Global constants & variables
│   ├── database.py           # Metadata SQLite schema
│   ├── pdf_processor.py      # PyMuPDF processing pipeline
│   ├── embeddings.py         # Local SentenceTransformer embeddings
│   ├── vector_store.py       # ChromaDB + BM25 search
│   ├── llm_handler.py        # Groq/Gemini/Ollama router
│   ├── rag_pipeline.py       # Query-expansion RAG manager
│   └── summarizer.py         # Summary generation
├── api/
│   ├── main.py               # FastAPI entrypoint
│   └── routes.py             # REST API endpoint definitions
├── tests/
│   └── test_api.py           # Pytest unit test coverage
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔮 Project Roadmap & Status

- [x] Arxiv API integration for automatic paper fetching
- [x] FastAPI backend for REST deployment
- [x] Docker containerization
- [x] Unit tests with pytest
- [ ] Cross-encoder reranking (improves result quality ~10–15%)
- [ ] Citation network visualization

---

## 📄 License

MIT License — free for personal, academic, and commercial use.
