"""
ResearchMind AI – Home Page (streamlit_app.py)
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
from core.database import init_db, get_paper_stats
from core.vector_store import VectorStore
from core.config import LLM_PROVIDER, GROQ_MODEL, GEMINI_MODEL, OLLAMA_MODEL, APP_VERSION

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
.stDeployButton {display:none;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4c1d95 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid #4338ca44;
}
.hero h1 { font-size: 2.8rem; margin: 0; }
.hero p  { color: #c7d2fe; margin: 0.4rem 0 0; font-size: 1.05rem; }

/* Stat card */
.stat-card {
    background: #1e2035;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    border: 1px solid #374151;
    margin-bottom: 0.8rem;
}
.stat-card .val { font-size: 2rem; font-weight: 700; color: #a78bfa; }
.stat-card .lbl { font-size: 0.85rem; color: #9ca3af; margin-top: 0.2rem; }

/* Feature card */
.feat-card {
    background: #161824;
    border-radius: 10px;
    padding: 1.2rem;
    border: 1px solid #2d3748;
    height: 100%;
}
.feat-card h3 { margin-top: 0; color: #e2e8f0; }
.feat-card p  { color: #9ca3af; font-size: 0.9rem; margin: 0; }

/* Status badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-green  { background: #064e3b; color: #6ee7b7; }
.badge-yellow { background: #78350f; color: #fcd34d; }
.badge-blue   { background: #1e3a5f; color: #93c5fd; }
</style>
""", unsafe_allow_html=True)

# ─── Init DB ──────────────────────────────────────────────────────────────────
init_db()

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <h1>🧠 ResearchMind AI</h1>
    <p>Intelligent Research Paper Analysis &amp; Knowledge Retrieval</p>
    <p><small>RAG · Hybrid Vector Search · Free LLMs · v{APP_VERSION}</small></p>
</div>
""", unsafe_allow_html=True)

# ─── Stats ────────────────────────────────────────────────────────────────────
try:
    stats  = get_paper_stats()
    vs_cnt = VectorStore().count()

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, val, lbl in zip(
        [col1, col2, col3, col4, col5],
        [stats["total_papers"], stats["total_pages"],
         vs_cnt, stats["total_chat_sessions"], stats["total_messages"]],
        ["📄 Papers", "📃 Pages", "🧩 Chunks", "💬 Sessions", "✉️ Messages"],
    ):
        with col:
            st.markdown(
                f'<div class="stat-card"><div class="val">{val}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )
except Exception:
    pass

st.divider()

# ─── LLM Status ───────────────────────────────────────────────────────────────
_provider_label = {
    "groq":   ("🟢", "Groq",   GROQ_MODEL,   "badge-green"),
    "gemini": ("🟡", "Gemini", GEMINI_MODEL,  "badge-yellow"),
    "ollama": ("🔵", "Ollama", OLLAMA_MODEL,  "badge-blue"),
}
icon, name, model, cls = _provider_label.get(
    LLM_PROVIDER, ("⚪", LLM_PROVIDER, "unknown", "badge-blue")
)
st.markdown(
    f'<b>Active LLM:</b> &nbsp;'
    f'<span class="badge {cls}">{icon} {name} · {model}</span>',
    unsafe_allow_html=True,
)
st.write("")

# ─── Features ─────────────────────────────────────────────────────────────────
st.markdown("## 🚀 Features")

features = [
    ("📄 Upload & Index",
     "Upload PDF papers. Auto-extract text, tables, metadata, and index with embeddings."),
    ("🧠 Chat with Papers",
     "Ask any question. Hybrid RAG retrieves relevant chunks and generates grounded answers."),
    ("📊 Compare Papers",
     "Side-by-side LLM comparison: objectives, methods, datasets, results, and strengths."),
    ("🔍 Knowledge Search",
     "Semantic + keyword search across your entire library. Filter by paper, export results."),
    ("📈 Analytics",
     "Upload timeline, top papers by size, section coverage, and vector-space stats."),
]

cols = st.columns(len(features))
for col, (title, desc) in zip(cols, features):
    with col:
        st.markdown(
            f'<div class="feat-card"><h3>{title}</h3><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

st.divider()

# ─── Setup Guide ──────────────────────────────────────────────────────────────
with st.expander("⚙️ First-Time Setup Guide", expanded=stats.get("total_papers", 0) == 0):
    st.markdown("""
### 1. Get a free API key (choose one)

| Provider | Speed | Quality | Setup |
|----------|-------|---------|-------|
| **Groq** ⭐ | Ultra-fast | Llama 3.1 70B | [console.groq.com](https://console.groq.com) → API Keys → Create |
| **Gemini** | Fast | Gemini 1.5 Flash | [ai.google.dev](https://ai.google.dev) → Get API Key |
| **Ollama** | Offline | Any local model | Install [ollama.ai](https://ollama.ai), run `ollama pull llama3.2` |

### 2. Create your `.env` file

```bash
cp .env.example .env
# Open .env and paste your key:
# GROQ_API_KEY=gsk_xxxxxxxxxxxx
# LLM_PROVIDER=groq
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run streamlit_app.py
```

### 5. Upload your first paper

Navigate to **📄 Upload Papers** in the sidebar and drag-drop a PDF.
""")

# ─── Architecture Diagram ─────────────────────────────────────────────────────
with st.expander("🏗️ System Architecture"):
    st.markdown("""
```
┌─────────────────────────────────────────────────────────────────┐
│                        ResearchMind AI                          │
│                      Streamlit Frontend                         │
└───────────────┬─────────────────────────────────────────────────┘
                │
    ┌───────────▼────────────┐
    │    PDF Processor        │  PyMuPDF + PDFPlumber
    │  Extract · Clean · Chunk│  Smart overlapping chunks
    └───────────┬────────────┘
                │
    ┌───────────▼────────────┐
    │   Embedding Service     │  sentence-transformers
    │  all-MiniLM-L6-v2 (free)│  Normalized 384-dim vectors
    └──────┬────────┬─────────┘
           │        │
    ┌──────▼──┐  ┌──▼──────┐
    │ ChromaDB│  │  BM25   │  Hybrid Search
    │  Dense  │  │ Keyword │  ──────────────
    └──────┬──┘  └──┬──────┘   RRF Fusion
           │        │
    ┌──────▼────────▼─────────┐
    │    RAG Pipeline          │
    │ Multi-query → Retrieve   │
    │ → Context → LLM → Cite  │
    └──────────┬──────────────┘
               │
    ┌──────────▼──────────────┐
    │   LLM (Free Providers)  │
    │  Groq · Gemini · Ollama │
    └─────────────────────────┘
```
""")

st.caption("ResearchMind AI · Built for FAANG-level portfolio · All LLMs are free-tier")
