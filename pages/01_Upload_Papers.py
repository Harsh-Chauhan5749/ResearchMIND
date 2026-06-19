"""
ResearchMind AI – Page 1: Upload & Index Papers
"""

import os
import shutil
import time
from pathlib import Path

import streamlit as st

from core.config import UPLOAD_DIR
from core.database import (
    add_paper, update_paper, delete_paper,
    get_all_papers, get_paper_by_id,
)
from core.pdf_processor import PDFProcessor
from core.embeddings import EmbeddingService
from core.vector_store import VectorStore
from core.summarizer import Summarizer
import arxiv

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Upload Papers – ResearchMind AI", page_icon="📄", layout="wide")

st.markdown("""
<style>
/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
.stDeployButton {display:none;}
footer {visibility: hidden;}
header {visibility: hidden;}

.upload-zone { border: 2px dashed #4338ca; border-radius: 12px; padding: 2rem;
               text-align: center; background: #161824; margin-bottom: 1rem; }
.paper-card  { background: #1e2035; border-radius: 10px; padding: 1rem 1.2rem;
               border: 1px solid #2d3748; margin-bottom: 0.7rem; }
.paper-title { font-weight: 600; color: #e2e8f0; font-size: 1rem; }
.paper-meta  { color: #6b7280; font-size: 0.82rem; margin-top: 0.2rem; }
.badge-ready { background:#064e3b; color:#6ee7b7; padding:2px 8px;
               border-radius:999px; font-size:0.75rem; }
.badge-proc  { background:#7c2d12; color:#fcd34d; padding:2px 8px;
               border-radius:999px; font-size:0.75rem; }
.step-label  { color: #a78bfa; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("📄 Upload & Index Research Papers")
st.caption("Upload PDFs → extract text → embed with vectors → generate AI summary")

st.divider()

def process_paper(dest_path: str, filename: str, gen_summary: bool):
    processor  = PDFProcessor()
    embedder   = EmbeddingService()
    vs         = VectorStore()
    summarizer = Summarizer() if gen_summary else None

    st.markdown(f"---\n#### Processing: `{filename}`")

    # ── DB entry (status = processing) ────────────────────────────────────
    paper_id = add_paper(
        filename=filename,
        file_path=dest_path,
        file_size=os.path.getsize(dest_path),
    )

    progress = st.progress(0, text="Starting…")

    try:
        # Step 1 – Extract
        progress.progress(10, text="📖 Extracting text from PDF…")
        content = processor.process(dest_path)
        meta    = content.metadata

        progress.progress(30, text="✂️ Chunking document…")
        chunks = processor.create_chunks(content.pages, paper_id, filename)

        progress.progress(50, text=f"🔢 Embedding {len(chunks)} chunks…")
        vs.add_chunks(chunks, paper_id, filename)

        # Step 2 – Summary (optional)
        summary = ""
        if gen_summary and summarizer:
            progress.progress(70, text="🧠 Generating AI summary (may take ~15 s)…")
            try:
                summary = summarizer.summarize(content.pages)
            except Exception as e:
                summary = f"Summary generation failed: {e}"

        # Step 3 – Persist metadata
        progress.progress(90, text="💾 Saving metadata…")
        update_paper(
            paper_id    = paper_id,
            total_chunks= len(chunks),
            summary     = summary,
            status      = "ready",
            title       = meta.get("title", ""),
        )

        # Update initial DB row with extracted metadata
        from core.database import get_connection
        import json
        conn = get_connection()
        conn.execute(
            """UPDATE papers SET authors=?, abstract=?, total_pages=?, year=?, citations=?
               WHERE id=?""",
            (meta.get("authors",""), meta.get("abstract",""),
             meta.get("total_pages", len(content.pages)),
             meta.get("year"), json.dumps(meta.get("citations", [])), paper_id),
        )
        conn.commit()
        conn.close()

        progress.progress(100, text="✅ Done!")
        time.sleep(0.4)
        progress.empty()

        # Success summary
        cols = st.columns(4)
        cols[0].metric("Pages",  meta.get("total_pages", len(content.pages)))
        cols[1].metric("Chunks", len(chunks))
        cols[2].metric("Year",   meta.get("year") or "–")
        cols[3].metric("Tables", len(content.tables))

        if meta.get("title"):
            st.success(f"**{meta['title']}** indexed successfully!")
        else:
            st.success(f"`{filename}` indexed successfully!")

    except Exception as e:
        progress.empty()
        update_paper(paper_id, status="error")
        st.error(f"❌ Error processing `{filename}`: {e}")
        st.exception(e)


# ─── Upload Section ───────────────────────────────────────────────────────────
st.subheader("Add New Papers")

gen_summary = st.checkbox("Generate AI summary after indexing", value=True,
                          help="Uses your LLM API. Uncheck to save API calls.")

tab_upload, tab_arxiv = st.tabs(["📤 Upload PDF", "🌐 Import from ArXiv"])

with tab_upload:
    uploaded_files = st.file_uploader(
        "Drop PDF files here (up to 200 MB each)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("🚀 Process & Index", type="primary", use_container_width=True):
        for uploaded in uploaded_files:
            filename = uploaded.name
            dest_path = str(UPLOAD_DIR / filename)
            with open(dest_path, "wb") as f:
                f.write(uploaded.read())
            
            process_paper(dest_path, filename, gen_summary)
        st.rerun()

with tab_arxiv:
    st.markdown("Fetch a paper directly from arXiv using its ID (e.g., `2303.08774`) or a search query.")
    arxiv_query = st.text_input("ArXiv ID or Query", placeholder="2303.08774")
    
    if st.button("📥 Fetch & Import from ArXiv", type="primary", use_container_width=True):
        if arxiv_query:
            try:
                # If it looks like an ID
                client = arxiv.Client()
                if arxiv_query.replace(".", "").isdigit():
                    search = arxiv.Search(id_list=[arxiv_query])
                else:
                    search = arxiv.Search(query=arxiv_query, max_results=1)
                
                paper = next(client.results(search))
                filename = f"{paper.title[:50].replace(' ', '_').replace('/', '_')}.pdf"
                dest_path = str(UPLOAD_DIR / filename)
                
                st.info(f"Downloading: {paper.title}...")
                
                import requests
                # Many arxiv pdf_urls don't append .pdf, adding it ensures the direct file download
                pdf_url = paper.pdf_url
                if not pdf_url.endswith(".pdf"):
                    pdf_url += ".pdf"
                    
                response = requests.get(pdf_url, stream=True)
                response.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                process_paper(dest_path, filename, gen_summary)
                st.rerun()
            except StopIteration:
                st.error("❌ No paper found on ArXiv for this query.")
            except Exception as e:
                st.error(f"❌ Error fetching from ArXiv: {e}")

st.divider()

# ─── Paper Library ────────────────────────────────────────────────────────────
st.subheader("📚 Paper Library")

papers = get_all_papers()

if not papers:
    st.info("No papers indexed yet. Upload your first PDF above.")
    st.stop()

# Filter / search
search_q = st.text_input("🔎 Filter papers", placeholder="Search by title or filename…")
if search_q:
    q = search_q.lower()
    papers = [p for p in papers if q in (p.get("title") or "").lower()
              or q in p["filename"].lower()]

st.caption(f"Showing {len(papers)} paper(s)")

for paper in papers:
    with st.container():
        col_info, col_btns = st.columns([5, 1])

        with col_info:
            title    = paper.get("title") or paper["filename"]
            authors  = paper.get("authors") or "Unknown authors"
            badge    = ('badge-ready' if paper['status'] == 'ready' else 'badge-proc')
            badge_lbl= ('✅ Ready' if paper['status'] == 'ready' else '⏳ Processing')
            st.markdown(
                f'<div class="paper-card">'
                f'<div class="paper-title">{title} '
                f'<span class="{badge}">{badge_lbl}</span></div>'
                f'<div class="paper-meta">'
                f'👤 {authors[:80]} &nbsp;|&nbsp; 📄 {paper["total_pages"]} pages &nbsp;|&nbsp; '
                f'🧩 {paper["total_chunks"]} chunks &nbsp;|&nbsp; 📅 {paper["upload_date"][:10]}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with col_btns:
            st.write("")  # spacing
            if st.button("🗑️", key=f"del_{paper['id']}",
                         help="Delete paper and its vectors"):
                # Remove from vector store
                try:
                    VectorStore().delete_paper(paper["id"])
                except Exception:
                    pass
                # Remove file
                try:
                    if paper.get("file_path") and Path(paper["file_path"]).exists():
                        os.remove(paper["file_path"])
                except Exception:
                    pass
                delete_paper(paper["id"])
                st.rerun()

        # Expandable summary
        if paper.get("summary"):
            with st.expander(f"📝 View Summary — {paper['filename']}"):
                st.markdown(paper["summary"])
                if paper.get("abstract"):
                    st.markdown("**Abstract:**")
                    st.markdown(f"> {paper['abstract'][:600]}…")
