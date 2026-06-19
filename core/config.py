"""
ResearchMind AI – Configuration Module
All settings pulled from environment variables with sensible defaults.
"""

import os
from pathlib import Path
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
EXPORT_DIR = DATA_DIR / "exports"
DB_PATH    = str(DATA_DIR / "researchmind.db")

# Create directories on import
for _dir in [UPLOAD_DIR, CHROMA_DIR, EXPORT_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── LLM Provider ─────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Groq (free: console.groq.com)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

# Gemini (free: ai.google.dev)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Ollama (offline: ollama.ai)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")

# ─── Embedding ────────────────────────────────────────────────────────────────
# Free local model — downloads ~90MB on first run
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# ─── RAG / Chunking ──────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000   # characters per chunk
CHUNK_OVERLAP = 200    # overlap between consecutive chunks
TOP_K         = 6      # chunks to retrieve per query
RRF_K         = 60     # reciprocal rank fusion constant

# ─── App ──────────────────────────────────────────────────────────────────────
APP_NAME    = "ResearchMind AI"
APP_VERSION = "2.0.0"
