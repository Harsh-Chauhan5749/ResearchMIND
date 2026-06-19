# 🧠 ResearchMind AI

> **Intelligent Research Paper Analysis & Knowledge Retrieval System**  
> Powered by Retrieval-Augmented Generation, Hybrid Vector Search, and Free LLMs.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚀 What Is This?

ResearchMind AI is a full-stack AI system that turns a folder of research PDFs into a **queryable, searchable, comparable knowledge base** — with no paid APIs required.

Unlike simple PDF summarizers, it builds a **persistent semantic memory** you can:
- **Chat with** in natural language
- **Search across** using hybrid retrieval
- **Compare** papers side-by-side
- **Visualize** with analytics

---

## ✨ Key Features

| Feature | Details |
|---------|---------|
| 📄 **PDF Ingestion** | PyMuPDF + PDFPlumber — text, tables, metadata, sections |
| 🔢 **Local Embeddings** | `all-MiniLM-L6-v2` (free, runs offline) |
| 🔍 **Hybrid Retrieval** | ChromaDB (dense cosine) + BM25 (keyword) fused via RRF |
| 🧠 **Streaming Chat** | Token-by-token streaming with source citations |
| 📊 **Paper Comparison** | LLM deep-compare: methods, datasets, results, strengths |
| 📈 **Analytics** | Upload timeline, chunk distribution, system health |
| 💯 **Confidence Scoring** | Per-answer confidence based on retrieval similarity |
| 🆓 **100% Free LLMs** | Groq (Llama 70B) · Gemini Flash · Ollama (offline) |

---

## 🏗️ Architecture

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
  ┌─────────▼──────┐  ┌───────▼──────────┐
  │   VectorStore   │  │   LLM Handler    │
  │  ChromaDB dense │  │  Groq  (free)    │
  │  BM25 keyword   │  │  Gemini (free)   │
  │  RRF fusion     │  │  Ollama (offline) │
  └─────────┬───────┘  └──────────────────┘
            │
  ┌─────────▼───────────┐
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
git clone https://github.com/yourusername/researchmind-ai.git
cd researchmind-ai

pip install -r requirements.txt
```

### 2. Get a Free LLM API Key (pick one)

**Option A — Groq (Recommended: fastest, Llama 3.1 70B)**
```
1. Go to https://console.groq.com
2. Sign up free → API Keys → Create API Key
3. Free tier: 30 requests/min, 6000 tokens/min
```

**Option B — Google Gemini (Gemini 1.5 Flash)**
```
1. Go to https://ai.google.dev
2. Get API Key → Create API Key
3. Free tier: 15 requests/min
```

**Option C — Ollama (fully offline, no key needed)**
```bash
# Install from https://ollama.ai
ollama pull llama3.2   # ~2GB download
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your key:
# GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
# LLM_PROVIDER=groq
```

### 4. Run

```bash
streamlit run streamlit_app.py
```

Visit `http://localhost:8501` 🎉

---

## 📁 Project Structure

```
researchmind-ai/
├── streamlit_app.py          # Home page + setup guide
├── pages/
│   ├── 01_Upload_Papers.py       # PDF ingestion pipeline
│   ├── 02_Chat_with_Papers.py     # Streaming RAG chat
│   ├── 03_Compare_Papers.py       # Cross-paper comparison
│   ├── 04_Knowledge_Search.py     # Hybrid semantic search
│   └── 05_Analytics_Dashboard.py  # System analytics
├── core/
│   ├── config.py             # Central configuration
│   ├── database.py           # SQLite metadata store
│   ├── pdf_processor.py      # PDF → chunks pipeline
│   ├── embeddings.py         # Sentence transformer service
│   ├── vector_store.py       # ChromaDB + BM25 hybrid search
│   ├── llm_handler.py        # Groq / Gemini / Ollama unified API
│   ├── rag_pipeline.py       # Full RAG orchestration
│   └── summarizer.py         # Structured summarization
├── data/
│   ├── uploads/              # Stored PDF files
│   ├── chroma_db/            # Persistent vector index
│   └── exports/              # Exported results
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔬 Technical Deep-Dive

### Hybrid Retrieval (the core innovation)

Most RAG systems use only dense vector search. ResearchMind AI uses **Reciprocal Rank Fusion (RRF)** to combine:

- **Dense retrieval** (ChromaDB, cosine similarity) — finds semantically similar content
- **BM25 keyword retrieval** — finds exact term matches, especially good for technical terms

```
RRF score = 0.60 × (1/(k + dense_rank)) + 0.40 × (1/(k + bm25_rank))
```

This is the same technique used by Elasticsearch and production RAG systems at FAANG companies.

### Multi-Query Expansion

Before retrieval, the query is expanded into variants to improve recall:
1. Original query
2. Content-word-focused version (stopwords removed)

### Chunking Strategy

Rather than fixed-size chunks, the processor:
- Detects section boundaries (Abstract, Introduction, Methodology, etc.)
- Splits at paragraph → sentence → word boundaries
- Maintains overlap for context continuity

### Confidence Scoring

Each answer comes with a confidence score computed from:
- Average cosine similarity of top-3 retrieved chunks
- Down-weighted if the LLM signals missing context ("not found", "not mentioned", etc.)

---

## 💬 Example Queries

```
Summarize the methodology of this paper.
What datasets were used for training and evaluation?
Compare BERT and GPT approaches to language modeling.
What are the key limitations mentioned across all papers?
Which paper achieves the highest BLEU score?
What future work directions are suggested?
Find all mentions of transformer architecture.
```

---

## 🏆 FAANG Resume Talking Points

This project demonstrates:

1. **RAG System Design** — end-to-end retrieval-augmented generation pipeline
2. **Vector Databases** — ChromaDB with persistent HNSW index
3. **Hybrid Search** — BM25 + dense retrieval with RRF fusion
4. **Embedding Models** — sentence-transformers, normalized cosine similarity
5. **LLM Integration** — streaming, multi-provider abstraction, prompt engineering
6. **Full-Stack Dev** — Streamlit multi-page app with custom CSS
7. **Data Engineering** — PDF parsing, semantic chunking, metadata extraction
8. **System Architecture** — modular design, caching, error handling
9. **SQL** — SQLite for metadata with proper indexing
10. **Production Patterns** — singleton clients, batch processing, graceful degradation

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

---

*Built as a portfolio project demonstrating production-level AI system design.*
