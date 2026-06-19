"""
ResearchMind AI – RAG Pipeline
Full Retrieval-Augmented Generation pipeline:

  1. Multi-query expansion  → better recall
  2. Hybrid retrieval       → dense + BM25 via RRF
  3. Context assembly       → formatted with source labels
  4. LLM generation         → grounded answer (streaming or blocking)
  5. Confidence scoring     → based on retrieval scores + answer uncertainty
"""

from typing import List, Dict, Optional, Generator, Tuple

from core.vector_store import VectorStore
from core.llm_handler import LLMHandler
from core.config import TOP_K


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are ResearchMind AI, an expert academic research assistant.

Your role:
- Answer questions based STRICTLY on retrieved context from research papers
- Cite every factual claim using [Source N, Page X] notation
- If the context does not contain the answer, say so clearly rather than guessing
- Use clear academic language while remaining accessible
- Be thorough but concise; prefer bullet points for lists of facts

Never fabricate citations or information not present in the context."""


# ─── RAG Pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Orchestrates the full retrieve → augment → generate cycle.
    Instantiate once and reuse across requests.
    """

    def __init__(self):
        self.vector_store = VectorStore()
        self.llm          = LLMHandler()

    # ── Public API ────────────────────────────────────────────────────────────

    def answer(
        self,
        query: str,
        paper_ids: Optional[List[int]] = None,
        chat_history: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Blocking answer.
        Returns: {answer, sources, confidence, context_chunks}
        """
        chunks = self._retrieve(query, paper_ids)

        if not chunks:
            return {
                "answer":         _NO_CONTEXT_MSG,
                "sources":        [],
                "confidence":     0.0,
                "context_chunks": [],
            }

        context  = _build_context(chunks)
        history  = _format_history(chat_history)
        prompt   = _build_prompt(query, context, history)
        answer   = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
        sources  = _extract_sources(chunks)
        conf     = _confidence(chunks, answer)

        return {
            "answer":         answer,
            "sources":        sources,
            "confidence":     conf,
            "context_chunks": chunks,
        }

    def stream_answer(
        self,
        query: str,
        paper_ids: Optional[List[int]] = None,
        chat_history: Optional[List[Dict]] = None,
    ) -> Generator[Tuple[str, any], None, None]:
        """
        Streaming answer generator.
        Yields ("chunk", token_str) and finally ("done", metadata_dict).
        """
        chunks = self._retrieve(query, paper_ids)

        if not chunks:
            yield ("chunk", _NO_CONTEXT_MSG)
            yield ("done", {"sources": [], "confidence": 0.0, "answer": _NO_CONTEXT_MSG})
            return

        context  = _build_context(chunks)
        history  = _format_history(chat_history)
        prompt   = _build_prompt(query, context, history)

        full_answer = ""
        for token in self.llm.stream(prompt, system_prompt=SYSTEM_PROMPT):
            full_answer += token
            yield ("chunk", token)

        sources = _extract_sources(chunks)
        conf    = _confidence(chunks, full_answer)

        yield ("done", {
            "answer":     full_answer,
            "sources":    sources,
            "confidence": conf,
        })

    def search_only(
        self,
        query: str,
        paper_ids: Optional[List[int]] = None,
        k: int = TOP_K,
    ) -> List[Dict]:
        """Return raw retrieved chunks without LLM generation (for Search page)."""
        return self._retrieve(query, paper_ids, k=k)

    # ── Private ───────────────────────────────────────────────────────────────

    def _retrieve(
        self, query: str, paper_ids: Optional[List[int]], k: int = TOP_K
    ) -> List[Dict]:
        """Multi-query retrieval with deduplication."""
        expanded = _expand_query(query)

        seen: set  = set()
        results: List[Dict] = []

        for q in expanded:
            for item in self.vector_store.hybrid_search(q, k=k, paper_ids=paper_ids):
                uid = f"{item['metadata']['paper_id']}_{item['metadata']['chunk_index']}"
                if uid not in seen:
                    results.append(item)
                    seen.add(uid)

        results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return results[:k]


# ─── Helpers (pure functions) ─────────────────────────────────────────────────

_NO_CONTEXT_MSG = (
    "I couldn't find relevant information in the indexed papers. "
    "Please upload papers first, or try rephrasing your question."
)


def _expand_query(query: str) -> List[str]:
    """
    Simple multi-query expansion:
      1. Original query
      2. Noun-phrase focused version (content words only)
    More sophisticated expansion would use an LLM call — skipped here
    to avoid extra latency on free-tier APIs.
    """
    queries = [query]
    stop = {"what", "is", "the", "a", "an", "of", "in", "on", "at",
            "to", "for", "with", "how", "does", "do", "are", "was",
            "were", "and", "or", "but", "if", "this", "that", "which"}
    keywords = [w for w in query.lower().split() if w not in stop and len(w) > 3]
    if keywords:
        queries.append(" ".join(keywords))
    return queries


def _build_context(chunks: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        m    = c["metadata"]
        text = c["text"]
        parts.append(
            f"[Source {i} | File: {m['filename']} | Page: {m['page_num']}"
            + (f" | Section: {m.get('section','')}" if m.get("section") else "")
            + f"]\n{text}"
        )
    return "\n\n{'─'*60}\n\n".join(parts)


def _format_history(history: Optional[List[Dict]], last_n: int = 4) -> str:
    if not history:
        return ""
    snippet = history[-last_n:]
    lines = []
    for msg in snippet:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n\n".join(lines)


def _build_prompt(query: str, context: str, history: str) -> str:
    prompt = f"CONTEXT FROM RESEARCH PAPERS:\n\n{context}\n\n"
    if history:
        prompt += f"CONVERSATION HISTORY:\n{history}\n\n"
    prompt += (
        f"QUESTION: {query}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer ONLY from the provided context\n"
        "- Cite every claim as [Source N, Page X]\n"
        "- If the context doesn't answer the question, state that explicitly\n"
        "- Be comprehensive but avoid repetition\n\n"
        "ANSWER:"
    )
    return prompt


def _extract_sources(chunks: List[Dict]) -> List[Dict]:
    seen: set = set()
    sources   = []
    for c in chunks:
        m   = c["metadata"]
        key = f"{m['filename']}_p{m['page_num']}"
        if key not in seen:
            sources.append({
                "filename":       m["filename"],
                "page_num":       m["page_num"],
                "paper_id":       m["paper_id"],
                "section":        m.get("section", ""),
                "relevance":      round(c.get("final_score", 0), 4),
                "similarity":     round(c.get("similarity", 0), 4),
                "preview":        m.get("text_preview", "")[:150],
            })
            seen.add(key)
    return sources


def _confidence(chunks: List[Dict], answer: str) -> float:
    if not chunks:
        return 0.0
    top3   = [c.get("similarity", 0) for c in chunks[:3]]
    avg    = sum(top3) / len(top3)

    # Down-weight if answer signals missing context
    hedge  = any(p in answer.lower() for p in [
        "not found", "no information", "cannot find",
        "not mentioned", "not available", "don't have",
        "not in the context", "no relevant",
    ])
    return round(avg * (0.4 if hedge else 1.0), 3)
