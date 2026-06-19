# 🧠 ResearchMind AI

> **Intelligent Research Paper Analysis & Knowledge Retrieval System**  
> Powered by Retrieval-Augmented Generation (RAG), Hybrid Vector Search, and Local/Cloud LLM Fallbacks.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚀 Overview

ResearchMind AI is an advanced, production-patterned RAG application designed to parse, index, and query academic research papers. It creates a **persistent semantic memory** from uploaded PDFs, allowing researchers to chat with their library, run side-by-side paper comparisons, and search across thousands of pages. It features an automated hybrid retrieval engine and operates either completely offline (via Ollama) or through high-performance APIs (via Groq or Google Gemini).

---

## ✨ Key Features

| Feature | Technical Strategy & Implementation Details |
|---------|---------------------------------------------|
| 📄 **PDF Ingestion** | Extracts text structure, tabular structures, and document metadata via PyMuPDF and PDFPlumber. |
| 🔢 **Local Embeddings** | Runs the `all-MiniLM-L6-v2` transformer model locally to map text to a dense 384-dimensional space. |
| 🔍 **Hybrid Retrieval** | Pairs dense vector search (ChromaDB) with sparse keyword matching (BM25) fused via Reciprocal Rank Fusion. |
| 🧠 **Streaming Chat** | Streams responses token-by-token using SSE (Server-Sent Events) with automatic page-level citations. |
| 📊 **Paper Comparison** | Runs multi-document context assembly to generate cross-comparison matrices of methods, datasets, and outcomes. |
| 📈 **Analytics** | Visualizes vector space density, page counts, publication years, and system database ingestion metrics. |
| 💯 **Confidence Scoring** | Computes a self-verification score based on cosine similarity and LLM feedback checking. |
| 🆓 **Flexible Providers** | Unified API router connecting Groq (Llama 70B), Gemini Flash, and local Ollama. |

---

## 🏗️ System Architecture

The layout below highlights the data flow between document ingestion, vector databases, search fusion, and LLM processing:

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

## 🔬 Technical Deep-Dive & Strategies

For developers and technical interviewers, here is the detailed breakdown of the engineering strategies implemented in this project:

### 1. Hybrid Retrieval & Reciprocal Rank Fusion (RRF)
RAG systems that rely solely on dense vector embeddings often struggle with keyword-specific lookups, product codes, equations, or specialized terms. ResearchMind AI addresses this by running a **two-pronged retrieval strategy**:
* **Dense Retrieval (Semantic Search):** Generates normalized 384-dimensional embeddings of text chunks using `all-MiniLM-L6-v2`. It queries ChromaDB using cosine similarity to find conceptually related information (e.g., matching "deep neural net" with "multilayer perceptron").
* **Sparse Retrieval (Keyword Search):** Computes BM25 scores (best-matching TF-IDF variants) on the exact vocabulary of the query. This captures critical terms, metrics, values, and specific terminology that vector embeddings might dilute.
* **Reciprocal Rank Fusion (RRF) Fusion:** To combine these two disparate scoring lists fairly, the system uses RRF, which rates documents based on their positional rank rather than their raw scores:

$$\text{RRF Score} = w_{\text{dense}} \times \left( \frac{1}{k + \text{rank}_{\text{dense}}} \right) + w_{\text{sparse}} \times \left( \frac{1}{k + \text{rank}_{\text{sparse}}} \right)$$

*Here, $k$ is set to $60$ (a standard smoothing constant to avoid over-weighting outlier rankings), and the weights are configured at $60\%$ dense and $40\%$ sparse.*

### 2. Multi-Query Query Expansion
Natural language queries are often brief or poorly phrased. To improve recall, the system runs the query through an expansion step:
1. **Stopword Elimination:** Strips filler words (like "the", "a", "explain") to isolate core content words.
2. **Alternative Queries:** Evaluates synonyms and semantic variants.
3. **Multi-Vector Queries:** Searches ChromaDB and BM25 with both the original query and the stripped query, merging the results. This prevents a poorly worded user question from missing critical matches.

### 3. Context-Aware Semantic Chunking
Standard RAG systems split documents using rigid, character-based boundaries (e.g., every 500 characters). In scientific literature, this breaks tables, splits single numbers from their units, and disrupts sentences. ResearchMind AI implements a custom parser:
* **Section Header Registration:** Parses the PDF text layout to detect changes in font size or spacing that represent section boundaries (e.g., "Methodology", "Results").
* **Semantic boundaries:** Prefers splitting text at paragraphs or sentence completions (`.`, `?`, `!`) rather than splitting in the middle of a line.
* **Overlapping Sliding Window:** Incorporates a 200-character overlap buffer. This ensures that keywords located at the tail-end of one chunk are carried over into the next chunk, preserving context continuity.

### 4. Self-Verification & Answer Confidence Scoring
To combat LLM hallucinations and give the user feedback on the reliability of an answer, the system calculates a dynamic confidence score:
1. **Retrieval Score:** Averages the cosine similarity of the top-3 chunks retrieved from the databases.
2. **Negative Feedback Penalty:** Checks the generated LLM text for negative-assurance keywords (e.g., "not mentioned", "not found in context", "I do not know"). If these patterns are detected, the confidence score is severely penalized (set to 0% or reduced by 50%).
3. **Final Metric:** Renders a visual color-coded indicator (Green: High, Orange: Moderate, Red: Low) representing the final confidence level.

### 5. High-Availability LLM Routing & Offline Fallback
To provide a reliable experience, the application features an interface that decouples the UI from the LLM endpoint. It supports:
* **Cloud Mode (Groq / Gemini):** Runs high-performance cloud models with larger parameter sizes for complex reasoning.
* **Automatic Exception Interceptor:** If the application detects a connection failure (due to lack of internet) or an API error (due to rate-limits), the handler intercepts the exception and automatically reroutes the prompt to the local **Ollama** server running locally (`http://localhost:11434`). This ensures the app remains operational at all times.

---

## ⚡ Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/Harsh-Chauhan5749/ResearchMIND.git
cd ResearchMIND
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```
Open `.env` and set your preferred provider:
* **Groq (Cloud):** Set `LLM_PROVIDER=groq` and add your `GROQ_API_KEY`.
* **Gemini (Cloud):** Set `LLM_PROVIDER=gemini` and add your `GEMINI_API_KEY`.
* **Ollama (Local Offline):** Set `LLM_PROVIDER=ollama`. Make sure to download the model locally using `ollama pull llama3.2`.

### 3. Run the Application
* **Streamlit UI:**
  ```bash
  streamlit run streamlit_app.py
  ```
  Access the app at **[http://localhost:8501](http://localhost:8501)**.
* **FastAPI Backend (REST API):**
  ```bash
  uvicorn api.main:app --reload
  ```
  Access the API documentation at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

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

---

*Built as a portfolio project demonstrating production-level AI system design.*
