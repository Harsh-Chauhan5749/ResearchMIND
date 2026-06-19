"""
ResearchMind AI – Page 3: Compare Papers
Select 2-5 papers for side-by-side LLM comparison + radar visualizations.
"""

import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from core.database import get_all_papers
from core.summarizer import Summarizer

st.set_page_config(
    page_title="Compare Papers – ResearchMind AI", page_icon="📊", layout="wide"
)

st.markdown("""
<style>
/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
.stDeployButton {display:none;}
footer {visibility: hidden;}
header {visibility: hidden;}

.compare-header { background: linear-gradient(90deg,#14532d,#166534);
                  padding: 1rem 1.5rem; border-radius: 10px; margin-bottom:1rem; }
.paper-col      { background:#1e2035; border-radius:10px; padding:1rem;
                  border:1px solid #374151; height:100%; }
.section-header { color:#86efac; font-weight:700; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="compare-header">
    <h2 style="margin:0">📊 Compare Research Papers</h2>
    <p style="margin:0;color:#bbf7d0;font-size:0.9rem">
        Select 2–5 papers for LLM-powered deep comparison
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Paper Selection ──────────────────────────────────────────────────────────
papers = get_all_papers()

if len(papers) < 2:
    st.warning("You need at least 2 indexed papers to compare. Go to **Upload Papers**.")
    st.stop()

st.subheader("Select Papers to Compare")

selected = []
cols = st.columns(min(len(papers), 3))
for i, paper in enumerate(papers):
    with cols[i % 3]:
        label = (paper.get("title") or paper["filename"])[:60]
        if st.checkbox(f"**{label}**\n\n{paper['filename']}", key=f"cmp_{paper['id']}"):
            selected.append(paper)

if len(selected) < 2:
    st.info("☝️ Select at least 2 papers above.")
    st.stop()

if len(selected) > 5:
    st.warning("Maximum 5 papers can be compared at once. Please deselect some.")
    st.stop()

st.success(f"**{len(selected)} papers selected** — ready to compare")

# ─── Check if all have summaries ──────────────────────────────────────────────
missing = [p for p in selected if not p.get("summary")]
if missing:
    st.warning(
        f"{len(missing)} paper(s) don't have summaries yet. "
        "Generating comparison from paper titles only — results may be less detailed. "
        "For best results, re-upload with 'Generate AI summary' checked."
    )

# ─── Action Buttons ───────────────────────────────────────────────────────────
col_btn1, col_btn2 = st.columns([1, 3])
run_compare = col_btn1.button("🔍 Run Comparison", type="primary", use_container_width=True)

# ─── Side-by-Side Quick View ──────────────────────────────────────────────────
st.divider()
st.subheader("📋 Side-by-Side Overview")

overview_cols = st.columns(len(selected))
for col, paper in zip(overview_cols, selected):
    with col:
        st.markdown(
            f'<div class="paper-col">'
            f'<b>{paper.get("title") or paper["filename"]}</b><br>'
            f'<small style="color:#6b7280">👤 {(paper.get("authors") or "Unknown")[:40]}</small><br>'
            f'<small style="color:#6b7280">📄 {paper["total_pages"]} pages · {paper["total_chunks"]} chunks</small>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if paper.get("abstract"):
            with st.expander("Abstract"):
                st.markdown(paper["abstract"][:500])

# ─── Radar Chart ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("📡 Structural Comparison")

# Compute simple heuristic scores from metadata
def score_paper(p):
    return {
        "Length":       min(p.get("total_pages", 0) / 20, 1.0),
        "Depth":        min(p.get("total_chunks", 0) / 200, 1.0),
        "Has Abstract": 1.0 if p.get("abstract") else 0.0,
        "Has Summary":  1.0 if p.get("summary") else 0.0,
        "Recent":       1.0 if (p.get("year") or 0) >= 2022 else
                        0.6 if (p.get("year") or 0) >= 2019 else 0.3,
    }

categories = ["Length", "Depth", "Has Abstract", "Has Summary", "Recent"]
fig = go.Figure()

for paper in selected:
    scores = score_paper(paper)
    vals   = [scores[c] for c in categories]
    vals   += [vals[0]]  # close radar
    fig.add_trace(go.Scatterpolar(
        r     = vals,
        theta = categories + [categories[0]],
        fill  = 'toself',
        name  = (paper.get("title") or paper["filename"])[:35],
        opacity=0.75,
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True,
    paper_bgcolor="#0F0F1A",
    plot_bgcolor ="#0F0F1A",
    font=dict(color="#e2e8f0"),
    height=420,
    margin=dict(l=40, r=40, t=30, b=30),
)
st.plotly_chart(fig, use_container_width=True)

# ─── Metadata Comparison Table ────────────────────────────────────────────────
st.subheader("📊 Metadata Comparison")

rows = {
    "Title":      [(p.get("title") or p["filename"])[:50] for p in selected],
    "Authors":    [(p.get("authors") or "–")[:40] for p in selected],
    "Year":       [str(p.get("year") or "–") for p in selected],
    "Pages":      [str(p.get("total_pages", "–")) for p in selected],
    "Chunks":     [str(p.get("total_chunks", "–")) for p in selected],
    "Has Summary":[("✅" if p.get("summary") else "❌") for p in selected],
}

df = pd.DataFrame(
    rows,
    index=[f"Paper {i+1}" for i in range(len(selected))]
).T
st.dataframe(df, use_container_width=True)

# ─── LLM Deep Comparison ──────────────────────────────────────────────────────
if run_compare:
    st.divider()
    st.subheader("🧠 AI Deep Comparison")

    paper_data = [
        {
            "filename": p["filename"],
            "summary":  p.get("summary") or (
                f"Title: {p.get('title','')}\nAbstract: {p.get('abstract','')[:500]}"
            ),
        }
        for p in selected
    ]

    with st.spinner(f"Comparing {len(selected)} papers with AI… (20–40 s)"):
        try:
            summarizer  = Summarizer()
            comparison  = summarizer.compare(paper_data)
            st.markdown(comparison)

            # Save to session state for export
            st.session_state["last_comparison"] = comparison

        except Exception as e:
            st.error(f"Comparison failed: {e}")
            st.exception(e)

    # Export
    if st.session_state.get("last_comparison"):
        st.download_button(
            "📥 Download Comparison (MD)",
            data    = st.session_state["last_comparison"],
            file_name="comparison.md",
            mime    ="text/markdown",
        )

