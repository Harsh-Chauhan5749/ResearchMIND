"""
ResearchMind AI – Page 5: Analytics Dashboard
System stats, upload timeline, top papers, vector store health.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

from core.database import get_all_papers, get_paper_stats, get_all_sessions
from core.vector_store import VectorStore
from core.config import LLM_PROVIDER, GROQ_MODEL, GEMINI_MODEL, OLLAMA_MODEL, APP_VERSION

st.set_page_config(
    page_title="Analytics – ResearchMind AI", page_icon="📈", layout="wide"
)

PURPLE = "#7C3AED"
INDIGO = "#4F46E5"
TEAL   = "#0D9488"
AMBER  = "#D97706"
ROSE   = "#E11D48"

st.markdown("""
<style>
/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
.stDeployButton {display:none;}
footer {visibility: hidden;}
header {visibility: hidden;}

.analytics-header { background:linear-gradient(135deg,#0f172a,#1e1b4b);
                    padding:1rem 1.5rem; border-radius:10px; margin-bottom:1rem;
                    border:1px solid #312e81; }
.kpi-card         { background:#1e2035; border-radius:10px; padding:1.2rem;
                    text-align:center; border:1px solid #374151; }
.kpi-val          { font-size:2.2rem; font-weight:700; color:#a78bfa; }
.kpi-lbl          { font-size:0.82rem; color:#6b7280; margin-top:0.2rem; }
.health-row       { background:#1a1a2e; border-radius:8px; padding:0.6rem 1rem;
                    margin:0.3rem 0; display:flex; justify-content:space-between;
                    align-items:center; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="analytics-header">
    <h2 style="margin:0">📈 Analytics Dashboard</h2>
    <p style="margin:0;color:#c7d2fe;font-size:0.9rem">
        ResearchMind AI v{APP_VERSION} · LLM: {LLM_PROVIDER.capitalize()}
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
papers  = get_all_papers()
stats   = get_paper_stats()
vs      = VectorStore()

# ─── KPI Row ──────────────────────────────────────────────────────────────────
kpis = [
    (stats["total_papers"],        "📄 Papers Indexed"),
    (stats["total_pages"],         "📃 Total Pages"),
    (vs.count(),                   "🧩 Vector Chunks"),
    (stats["total_chat_sessions"], "💬 Chat Sessions"),
    (stats["total_messages"],      "✉️ Messages"),
]

cols = st.columns(len(kpis))
for col, (val, lbl) in zip(cols, kpis):
    with col:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
            f'<div class="kpi-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

if not papers:
    st.info("No papers indexed yet. Go to **Upload Papers** to get started.")
    st.stop()

st.divider()

# ─── Row 1: Upload Timeline + Pages Distribution ──────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📅 Upload Timeline")
    df_time = pd.DataFrame(papers)
    df_time["upload_date"] = pd.to_datetime(df_time["upload_date"])
    df_time["date"]        = df_time["upload_date"].dt.date
    daily = df_time.groupby("date").size().reset_index(name="count")

    fig = go.Figure(go.Bar(
        x=daily["date"], y=daily["count"],
        marker_color=INDIGO, opacity=0.85,
    ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Papers Uploaded",
        paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A",
        font=dict(color="#e2e8f0"),
        height=300, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Pages per Paper")
    df_pages = pd.DataFrame(papers).sort_values("total_pages", ascending=False).head(15)
    labels   = [
        ((p.get("title") or p["filename"])[:30])
        for _, p in df_pages.iterrows()
    ]
    fig = go.Figure(go.Bar(
        y=labels, x=df_pages["total_pages"],
        orientation="h",
        marker_color=TEAL, opacity=0.85,
    ))
    fig.update_layout(
        xaxis_title="Pages", yaxis_title="",
        paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A",
        font=dict(color="#e2e8f0"),
        height=300, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── Row 2: Chunks per Paper + Year Distribution ──────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("🧩 Chunks per Paper")
    df_ch = pd.DataFrame(papers).sort_values("total_chunks", ascending=False).head(10)
    labels = [(p.get("title") or p["filename"])[:30] for _, p in df_ch.iterrows()]

    fig = go.Figure(go.Bar(
        x=labels, y=df_ch["total_chunks"],
        marker_color=PURPLE, opacity=0.85,
    ))
    fig.update_layout(
        xaxis_tickangle=-25,
        xaxis_title="", yaxis_title="Chunks",
        paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A",
        font=dict(color="#e2e8f0"),
        height=300, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("📆 Publication Year Distribution")
    years = [p.get("year") for p in papers if p.get("year")]
    if years:
        year_df = pd.Series(years).value_counts().sort_index().reset_index()
        year_df.columns = ["year", "count"]
        fig = go.Figure(go.Bar(
            x=year_df["year"], y=year_df["count"],
            marker_color=AMBER, opacity=0.85,
        ))
        fig.update_layout(
            xaxis_title="Year", yaxis_title="Papers",
            paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A",
            font=dict(color="#e2e8f0"),
            height=300, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No year information extracted from papers yet.")

st.divider()

# ─── Paper Details Table ──────────────────────────────────────────────────────
st.subheader("📋 Paper Index")

df_table = pd.DataFrame([
    {
        "Title":   (p.get("title") or p["filename"])[:50],
        "Authors": (p.get("authors") or "–")[:35],
        "Year":    str(p.get("year") or "–"),
        "Pages":   p.get("total_pages", 0),
        "Chunks":  p.get("total_chunks", 0),
        "Summary": "✅" if p.get("summary") else "❌",
        "Uploaded":p.get("upload_date", "")[:10],
    }
    for p in papers
])

st.dataframe(df_table, use_container_width=True, hide_index=True)

st.download_button(
    "📥 Export Paper Index (CSV)",
    data      = df_table.to_csv(index=False),
    file_name = "researchmind_index.csv",
    mime      = "text/csv",
)

st.divider()

# ─── System Health ────────────────────────────────────────────────────────────
st.subheader("⚙️ System Health")

_model_map = {
    "groq":   GROQ_MODEL,
    "gemini": GEMINI_MODEL,
    "ollama": OLLAMA_MODEL,
}
health_items = [
    ("LLM Provider",        f"{LLM_PROVIDER.capitalize()} · {_model_map.get(LLM_PROVIDER,'–')}",  "🟢"),
    ("Embedding Model",     "all-MiniLM-L6-v2 (local)",                                            "🟢"),
    ("Vector DB",           f"ChromaDB · {vs.count()} chunks",                                     "🟢"),
    ("Hybrid Search",       "Dense (ChromaDB cosine) + BM25 with RRF fusion",                      "🟢"),
    ("Papers Indexed",      str(stats["total_papers"]),                                             "🟢"),
    ("App Version",         APP_VERSION,                                                            "🔵"),
]

for label, value, icon in health_items:
    st.markdown(
        f'<div class="health-row">'
        f'<span style="color:#d1d5db">{label}</span>'
        f'<span style="color:#9ca3af">{icon} {value}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
