"""
ResearchMind AI – Vector Store
Hybrid retrieval: ChromaDB (dense) + BM25 (keyword) fused with Reciprocal Rank Fusion.
This is the core retrieval engine powering the RAG pipeline.
"""

from typing import List, Dict, Optional, Any
import json

import chromadb
from rank_bm25 import BM25Okapi
import numpy as np

from core.config import CHROMA_DIR, TOP_K, RRF_K
from core.embeddings import EmbeddingService
from core.pdf_processor import TextChunk


# ChromaDB client – one instance per process
_chroma_client: Optional[chromadb.PersistentClient] = None

# CrossEncoder – loaded lazily
_cross_encoder = None

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except ImportError:
            pass
    return _cross_encoder


def _get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


class VectorStore:
    """
    Manages vector storage and hybrid retrieval.

    Architecture:
    ┌─────────────┐   ┌─────────────────┐
    │ Dense (cos) │   │  BM25 (TF-IDF)  │
    │  ChromaDB   │   │    In-memory     │
    └──────┬──────┘   └────────┬────────┘
           │                   │
           └──────┬────────────┘
              RRF Fusion
                  │
             Ranked Results
    """

    COLLECTION = "research_papers"

    def __init__(self):
        self.client     = _get_chroma_client()
        self.embedder   = EmbeddingService()
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        # BM25 index (rebuilt from ChromaDB when needed)
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[str]   = []
        self._bm25_meta: List[Dict]  = []
        self._bm25_ids:  List[str]   = []
        self._build_bm25()

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[TextChunk], paper_id: int, filename: str):
        """Embed and store chunks; rebuild BM25 index afterwards."""
        if not chunks:
            return

        texts      = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts, batch_size=64, show_progress=False)

        ids       = [f"p{paper_id}_c{c.chunk_index}" for c in chunks]
        metadatas = [
            {
                "paper_id":     paper_id,
                "filename":     filename,
                "page_num":     c.page_num,
                "chunk_index":  c.chunk_index,
                "section":      c.section or "",
                "text_preview": c.text[:200],
            }
            for c in chunks
        ]

        # Upsert in batches of 128
        batch = 128
        for i in range(0, len(ids), batch):
            self.collection.upsert(
                ids=ids[i : i + batch],
                embeddings=embeddings[i : i + batch],
                documents=texts[i : i + batch],
                metadatas=metadatas[i : i + batch],
            )

        self._build_bm25()

    def delete_paper(self, paper_id: int):
        """Remove all chunks for a given paper."""
        try:
            result = self.collection.get(
                where={"paper_id": paper_id}, include=[]
            )
            if result["ids"]:
                self.collection.delete(ids=result["ids"])
            self._build_bm25()
        except Exception:
            pass

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        k: int = TOP_K,
        paper_ids: Optional[List[int]] = None,
    ) -> List[Dict]:
        """
        Hybrid search: dense + BM25 fused via Reciprocal Rank Fusion (RRF).
        Returns up to k results sorted by fused score descending.
        """
        total = self.collection.count()
        if total == 0:
            return []

        n = min(k * 4, total)
        where = self._build_where(paper_ids)

        # ── Dense (ChromaDB) ──────────────────────────────────────────────────
        query_emb = self.embedder.embed_query(query)
        dense_kw: Dict[str, Any] = {
            "query_embeddings": [query_emb],
            "n_results":        n,
            "include":          ["documents", "metadatas", "distances"],
        }
        if where:
            dense_kw["where"] = where

        dr = self.collection.query(**dense_kw)
        dense_docs  = dr["documents"][0]
        dense_metas = dr["metadatas"][0]
        dense_dists = dr["distances"][0]

        # ── Build RRF score table ─────────────────────────────────────────────
        rrf: Dict[str, Dict] = {}

        for rank, (doc, meta, dist) in enumerate(zip(dense_docs, dense_metas, dense_dists)):
            uid = f"p{meta['paper_id']}_c{meta['chunk_index']}"
            if uid not in rrf:
                rrf[uid] = {
                    "text":       doc,
                    "metadata":   meta,
                    "similarity": 1.0 - dist,   # cosine similarity
                }
            rrf[uid]["dense_score"] = 1.0 / (RRF_K + rank + 1)

        # ── BM25 ──────────────────────────────────────────────────────────────
        if self._bm25 and self._bm25_docs:
            tokens = query.lower().split()
            scores = self._bm25.get_scores(tokens)
            top_idx = np.argsort(scores)[::-1][:n]

            for rank, idx in enumerate(top_idx):
                meta = self._bm25_meta[idx]

                # Respect paper filter
                if paper_ids and meta.get("paper_id") not in paper_ids:
                    continue

                uid = self._bm25_ids[idx]
                if uid not in rrf:
                    rrf[uid] = {
                        "text":       self._bm25_docs[idx],
                        "metadata":   meta,
                        "similarity": 0.0,
                    }
                rrf[uid]["bm25_score"] = 1.0 / (RRF_K + rank + 1)

        # ── Fuse & rank ───────────────────────────────────────────────────────
        results = []
        for uid, data in rrf.items():
            data["final_score"] = (
                0.60 * data.get("dense_score", 0.0)
                + 0.40 * data.get("bm25_score", 0.0)
            )
            results.append(data)

        results.sort(key=lambda x: x["final_score"], reverse=True)
        top_results = results[:k * 2]  # take top 2k for reranking

        # ── Cross-encoder Reranking ───────────────────────────────────────────
        ce = _get_cross_encoder()
        if ce and top_results:
            pairs = [[query, res["text"]] for res in top_results]
            ce_scores = ce.predict(pairs)
            for res, score in zip(top_results, ce_scores):
                res["final_score"] = float(score)  # override score with CE score
            top_results.sort(key=lambda x: x["final_score"], reverse=True)

        return top_results[:k]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self.collection.count()

    def get_paper_chunk_count(self, paper_id: int) -> int:
        try:
            r = self.collection.get(where={"paper_id": paper_id}, include=[])
            return len(r["ids"])
        except Exception:
            return 0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_bm25(self):
        """Rebuild BM25 index from all documents currently in ChromaDB."""
        try:
            all_data = self.collection.get(include=["documents", "metadatas"])
            if not all_data["documents"]:
                self._bm25 = None
                return

            self._bm25_docs  = all_data["documents"]
            self._bm25_meta  = all_data["metadatas"]
            self._bm25_ids   = all_data["ids"]
            tokenized        = [d.lower().split() for d in self._bm25_docs]
            self._bm25       = BM25Okapi(tokenized)
        except Exception:
            self._bm25 = None

    @staticmethod
    def _build_where(paper_ids: Optional[List[int]]) -> Optional[Dict]:
        if not paper_ids:
            return None
        if len(paper_ids) == 1:
            return {"paper_id": paper_ids[0]}
        return {"paper_id": {"$in": paper_ids}}
