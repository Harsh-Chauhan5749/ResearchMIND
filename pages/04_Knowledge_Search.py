"""
ResearchMind AI – Page 4: Knowledge Search
Semantic + BM25 hybrid search across the full paper library.
Supports paper filtering, relevance sorting, and result export.
"""

import json
import time
import pandas as pd
import streamlit as st

from core.database import get_all_papers
from core.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="Knowledge Search – ResearchMind AI", page_icon="🔍", layout="wide"
)

st.markdown("""
<style>
/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
.stDeployButton {display:none;}
footer {visibility: hidden;}
header {visibility: hidden;}

.search-header { background: linear-gradient(90deg,#1c1917,#44403c);
                 padding:1rem 1.5rem; border-radius:10px; margin-bottom:1rem; }
.result-card   { background:#1e2035; border-radius:10px; padding:1rem 1.2rem;
                 border:1px solid #374151; margin-bottom:0.8rem; }
.result-title  { font-weight:600; color:#e2e8f0; }
.result-meta   { color:#6b7280; font-size:0.8rem; margin:0.2rem 0 0.5rem; }
.result-text   { color:#d1d5db; font-size:0.88rem; line-height:1.6; }
.score-badge   { background:#312e81; color:#a5b4fc; padding:2px 8px;
                 border-radius:999px; font-size:0.75rem; float:right; }
.highlight     { background:#3730a3; border-radius:3px; padding:0 2px; color:#e0e7ff; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="search-header">
    <h2 style="margin:0">🔍 Knowledge Search</h2>
    <p style="margin:0;color:#d6d3d1;font-size:0.9rem">
        Hybrid semantic + keyword search across your entire research library
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Load papers ──────────────────────────────────────────────────────────────
papers = get_all_papers()

if not papers:
    st.info("No papers indexed. Go to **Upload Papers** first.")
    st.stop()

# ─── Search Controls ──────────────────────────────────────────────────────────
col_search, col_k = st.columns([4, 1])

with col_search:
    query = st.text_input(
        "Search query",
        placeholder="e.g. 'attention mechanism', 'transformer architecture', 'BLEU score'…",
        label_visibility="collapsed",
    )

with col_k:
    top_k = st.selectbox("Results", [5, 10, 15, 20], index=0)

# Filters
with st.expander("🎛️ Filters", expanded=False):
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        paper_filter = st.multiselect(
            "Limit to papers",
            options  = [p["id"] for p in papers],
            format_func = lambda pid: next(
                (p.get("title") or p["filename"])[:50]
                for p in papers if p["id"] == pid
            ),
            default  = [],
        )

    with col_f2:
        min_relevance = st.slider("Minimum relevance score", 0.0, 1.0, 0.0, 0.05)

# ─── Run Search ───────────────────────────────────────────────────────────────
if query:
    rag = RAGPipeline()

    with st.spinner("Searching…"):
        t0      = time.time()
        results = rag.search_only(
            query     = query,
            paper_ids = paper_filter or None,
            k         = top_k,
        )
        elapsed = time.time() - t0

    # Apply relevance filter
    results = [r for r in results if r.get("final_score", 0) >= min_relevance]

    st.caption(
        f"Found **{len(results)}** relevant chunks in **{elapsed:.2f}s** · "
        f"Hybrid search (Dense 60% + BM25 40%)"
    )

    if not results:
        st.info("No results match your query and filters. Try different keywords.")
    else:
        # ── Keyword highlight helper ─────────────────────────────────────────
        def highlight(text: str, terms: list) -> str:
            for term in terms:
                if term.lower() in text.lower():
                    # simple case-insensitive replace (first occurrence)
                    idx = text.lower().find(term.lower())
                    text = (
                        text[:idx]
                        + f'<span class="highlight">{text[idx:idx+len(term)]}</span>'
                        + text[idx + len(term):]
                    )
            return text

        query_terms = [w for w in query.split() if len(w) > 2]

        # ── Result cards ─────────────────────────────────────────────────────
        for i, res in enumerate(results, 1):
            meta  = res["metadata"]
            text  = res["text"][:600]
            score = res.get("final_score", 0)
            sim   = res.get("similarity", 0)

            text_hl = highlight(text, query_terms)

            st.markdown(
                f'<div class="result-card">'
                f'<span class="score-badge">Relevance {int(score*100)}%</span>'
                f'<div class="result-title">Result {i} &nbsp;·&nbsp; {meta["filename"]}</div>'
                f'<div class="result-meta">'
                f'Page {meta["page_num"]}'
                + (f' · {meta.get("section","")}' if meta.get("section") else "")
                + f'&nbsp;|&nbsp; Dense sim: {sim:.3f}'
                f'</div>'
                f'<div class="result-text">{text_hl}{"…" if len(res["text"]) > 600 else ""}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Full text expander
            if len(res["text"]) > 600:
                with st.expander("Show full chunk"):
                    st.markdown(res["text"])

        st.divider()

        # ── Export ───────────────────────────────────────────────────────────
        st.subheader("📥 Export Results")

        export_data = [
            {
                "rank":      i + 1,
                "filename":  r["metadata"]["filename"],
                "page_num":  r["metadata"]["page_num"],
                "section":   r["metadata"].get("section", ""),
                "relevance": round(r.get("final_score", 0), 4),
                "text":      r["text"],
            }
            for i, r in enumerate(results)
        ]

        df = pd.DataFrame(export_data)
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.download_button(
                "📊 Export as CSV",
                data      = df.to_csv(index=False),
                file_name = f"search_{query[:30].replace(' ','_')}.csv",
                mime      = "text/csv",
                use_container_width=True,
            )
        with col_e2:
            st.download_button(
                "📄 Export as JSON",
                data      = json.dumps(export_data, indent=2),
                file_name = f"search_{query[:30].replace(' ','_')}.json",
                mime      = "application/json",
                use_container_width=True,
            )

# ─── Example Queries ──────────────────────────────────────────────────────────
if not query:
    st.divider()
    st.subheader("💡 Example Queries")
    examples = [
        "attention mechanism self-attention",
        "BLEU score evaluation metrics",
        "dataset training validation split",
        "limitations future work",
        "transformer architecture encoder decoder",
        "gradient descent optimization",
        "overfitting regularization dropout",
        "benchmark comparison state of the art",
    ]
    cols = st.columns(4)
    for i, ex in enumerate(examples):
        if cols[i % 4].button(ex, use_container_width=True):
            st.query_params["q"] = ex
            st.rerun()
