"""
ResearchMind AI – Page 2: Chat with Papers
Streaming RAG-powered Q&A with source citations and confidence scores.
"""

import uuid
import time

import streamlit as st

from core.database import (
    get_all_papers, save_message, get_chat_history, ensure_session,
)
from core.rag_pipeline import RAGPipeline

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat – ResearchMind AI", page_icon="🧠", layout="wide"
)

st.markdown("""
<style>
/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
.stDeployButton {display:none;}
footer {visibility: hidden;}
header {visibility: hidden;}

.chat-header { background: linear-gradient(90deg,#1e1b4b,#312e81);
               padding: 1rem 1.5rem; border-radius: 10px; margin-bottom: 1rem; }
.source-box  { background: #1a1a2e; border-left: 3px solid #6366f1;
               padding: 0.6rem 0.9rem; border-radius: 6px; margin: 0.3rem 0;
               font-size: 0.82rem; color: #9ca3af; }
.conf-bar-wrap { display:flex; align-items:center; gap:0.5rem; margin:0.4rem 0; }
.conf-label  { font-size:0.78rem; color:#9ca3af; width:80px; }
.conf-bar    { height:6px; border-radius:3px; background:#6366f1; }
.conf-bg     { flex:1; background:#2d3748; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─── Session Init ─────────────────────────────────────────────────────────────
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = str(uuid.uuid4())
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "selected_paper_ids" not in st.session_state:
    st.session_state.selected_paper_ids = []

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📚 Paper Scope")
    papers = get_all_papers()

    if not papers:
        st.warning("No papers indexed. Go to **Upload Papers** first.")
        scope_ids = []
    else:
        scope_mode = st.radio("Search across", ["All papers", "Selected papers"], index=0)

        scope_ids = []
        if scope_mode == "Selected papers":
            for p in papers:
                label = (p.get("title") or p["filename"])[:50]
                if st.checkbox(label, key=f"scope_{p['id']}"):
                    scope_ids.append(p["id"])
        else:
            scope_ids = None  # None = all papers

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_session_id = str(uuid.uuid4())
        st.session_state.chat_messages   = []
        st.rerun()

    st.caption("**Tips:**")
    st.caption("• *Summarize this paper*")
    st.caption("• *What datasets were used?*")
    st.caption("• *Compare the methodologies*")
    st.caption("• *What are the limitations?*")

# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <h2 style="margin:0">🧠 Chat with Your Research Papers</h2>
    <p style="margin:0;color:#c7d2fe;font-size:0.9rem">
        Hybrid RAG · Streaming · Source Citations · Confidence Scoring
    </p>
</div>
""", unsafe_allow_html=True)

if not papers:
    st.info("📄 Upload papers first using the **Upload Papers** page.")
    st.stop()

# ─── Chat History Display ─────────────────────────────────────────────────────
rag = RAGPipeline()

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s) cited"):
                for i, src in enumerate(msg["sources"], 1):
                    conf_pct = int(src.get("relevance", 0) * 100)
                    st.markdown(
                        f'<div class="source-box">'
                        f'<b>[Source {i}]</b> {src["filename"]} · Page {src["page_num"]}'
                        + (f' · {src["section"]}' if src.get("section") else "")
                        + f'<div class="conf-bar-wrap">'
                        f'<span class="conf-label">Relevance</span>'
                        f'<div class="conf-bg"><div class="conf-bar" style="width:{conf_pct}%"></div></div>'
                        f'<span style="font-size:0.75rem;color:#a78bfa">{conf_pct}%</span>'
                        f'</div>'
                        + (f'<div style="margin-top:0.3rem;font-style:italic">{src["preview"]}</div>'
                           if src.get("preview") else "")
                        + '</div>',
                        unsafe_allow_html=True,
                    )

        # Confidence indicator
        if msg["role"] == "assistant" and msg.get("confidence") is not None:
            conf = msg["confidence"]
            color = "#22c55e" if conf > 0.7 else "#f59e0b" if conf > 0.4 else "#ef4444"
            st.markdown(
                f'<span style="font-size:0.75rem;color:{color}">●</span> '
                f'<span style="font-size:0.75rem;color:#6b7280">'
                f'Confidence: {int(conf*100)}%</span>',
                unsafe_allow_html=True,
            )

# ─── Input ────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about your research papers…")

if user_input:
    # Add user message
    st.session_state.chat_messages.append({
        "role": "user", "content": user_input
    })
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Save to DB
    ensure_session(st.session_state.chat_session_id, paper_ids=scope_ids)
    save_message(st.session_state.chat_session_id, "user", user_input)

    # Stream assistant response
    with st.chat_message("assistant", avatar="🧠"):
        placeholder = st.empty()
        full_text   = ""
        metadata    = {}

        with st.spinner("Retrieving relevant context…"):
            gen = rag.stream_answer(
                query      = user_input,
                paper_ids  = scope_ids,
                chat_history = st.session_state.chat_messages[:-1],
            )
            for event_type, payload in gen:
                if event_type == "chunk":
                    full_text += payload
                    placeholder.markdown(full_text + "▌")
                elif event_type == "done":
                    metadata = payload
                    placeholder.markdown(full_text)

        sources    = metadata.get("sources", [])
        confidence = metadata.get("confidence", 0.0)

        # Show sources
        if sources:
            with st.expander(f"📎 {len(sources)} source(s) cited"):
                for i, src in enumerate(sources, 1):
                    conf_pct = int(src.get("relevance", 0) * 100)
                    st.markdown(
                        f'<div class="source-box">'
                        f'<b>[Source {i}]</b> {src["filename"]} · Page {src["page_num"]}'
                        + (f' · {src["section"]}' if src.get("section") else "")
                        + f'<div class="conf-bar-wrap">'
                        f'<span class="conf-label">Relevance</span>'
                        f'<div class="conf-bg"><div class="conf-bar" style="width:{conf_pct}%"></div></div>'
                        f'<span style="font-size:0.75rem;color:#a78bfa">{conf_pct}%</span>'
                        f'</div>'
                        + (f'<div style="margin-top:0.3rem;font-style:italic">{src["preview"]}</div>'
                           if src.get("preview") else "")
                        + '</div>',
                        unsafe_allow_html=True,
                    )

        # Confidence
        color = "#22c55e" if confidence > 0.7 else "#f59e0b" if confidence > 0.4 else "#ef4444"
        st.markdown(
            f'<span style="font-size:0.75rem;color:{color}">●</span> '
            f'<span style="font-size:0.75rem;color:#6b7280">'
            f'Confidence: {int(confidence*100)}%</span>',
            unsafe_allow_html=True,
        )

    # Persist assistant message
    st.session_state.chat_messages.append({
        "role":       "assistant",
        "content":    full_text,
        "sources":    sources,
        "confidence": confidence,
    })
    save_message(
        st.session_state.chat_session_id, "assistant", full_text,
        sources=sources, confidence=confidence,
    )
