"""
Adaptive RAG — Streamlit demo

Run:
    streamlit run app.py
"""

import json
import sys
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Adaptive RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* Pipeline step cards */
.step-card {
    border-left: 4px solid #4ade80;
    padding: 10px 14px;
    margin: 6px 0;
    background: rgba(74,222,128,0.06);
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
}
.step-card.warning {
    border-left-color: #facc15;
    background: rgba(250,204,21,0.06);
}
.step-card.info {
    border-left-color: #60a5fa;
    background: rgba(96,165,250,0.06);
}

/* Badge styles */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-local { background: #dcfce7; color: #166534; }
.badge-web   { background: #fef3c7; color: #92400e; }

/* Answer box */
.answer-box {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    color: #f1f5f9;
    line-height: 1.7;
    margin-top: 8px;
}

/* Metric pill */
.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading pipeline (first run may take ~30s)…")
def load_pipeline():
    from src.ingestion.vectorstore import collection_size, get_vectorstore
    from src.pipeline.graph import build_graph

    get_vectorstore()   # warm up ChromaDB + embeddings
    graph = build_graph()
    size = collection_size()
    return graph, size


def make_relevance_chart(documents: list[dict], threshold: float) -> go.Figure:
    if not documents:
        return None

    names = [f"Doc {i + 1}" for i in range(len(documents))]
    scores = [d.get("relevance_score") or 0.0 for d in documents]
    colors = ["#4ade80" if s >= threshold else "#f87171" for s in scores]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=scores,
            marker_color=colors,
            text=[f"{s:.2f}" for s in scores],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#facc15",
        annotation_text=f"Threshold ({threshold})",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Document Relevance Scores",
        yaxis=dict(range=[0, 1.15], title="Relevance"),
        xaxis_title="Retrieved Documents",
        showlegend=False,
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
    )
    return fig


def run_pipeline(graph, query: str, threshold: float, top_k: int) -> dict:
    from src.config import settings
    from src.pipeline.graph import make_initial_state

    # Temporarily override settings for this request
    _orig_threshold = settings.relevance_threshold
    _orig_top_k = settings.top_k
    settings.relevance_threshold = threshold
    settings.top_k = top_k

    try:
        state_parts = make_initial_state(query).copy()
        state_parts["pipeline_log"] = []

        for chunk in graph.stream(state_parts, stream_mode="updates"):
            for node_name, update in chunk.items():
                for k, v in update.items():
                    if k == "pipeline_log":
                        state_parts[k] = state_parts.get(k, []) + v
                    elif v is not None:
                        state_parts[k] = v
    finally:
        settings.relevance_threshold = _orig_threshold
        settings.top_k = _orig_top_k

    return state_parts


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Pipeline Settings")

    threshold = st.slider(
        "Relevance threshold",
        min_value=0.1, max_value=0.9, value=0.6, step=0.05,
        help="If avg doc score < threshold, web search is triggered",
    )
    top_k = st.slider(
        "Documents to retrieve (k)",
        min_value=1, max_value=10, value=5,
    )

    st.divider()
    st.markdown("## 📂 Ingest Documents")
    uploaded = st.file_uploader(
        "Upload .txt or .pdf files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Ingest uploaded files", type="primary"):
        from src.ingestion.loader import split_documents
        from src.ingestion.vectorstore import add_documents
        from langchain_core.documents import Document

        raw = []
        for f in uploaded:
            text = f.read().decode("utf-8", errors="ignore")
            raw.append(Document(page_content=text, metadata={"filename": f.name}))

        from src.ingestion.loader import split_documents
        chunks = split_documents(raw)
        ids = add_documents([c.page_content for c in chunks], [c.metadata for c in chunks])
        st.success(f"Indexed {len(ids)} chunks from {len(uploaded)} file(s).")

    st.divider()
    st.markdown("## ℹ️ About")
    st.caption(
        "Corrective RAG pipeline built with LangGraph + Claude. "
        "Automatically falls back to Tavily web search when local context confidence is low."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🔍 Adaptive RAG")
st.markdown(
    "**Corrective Retrieval-Augmented Generation** — evaluates local context quality "
    "before answering, and falls back to live web search when needed."
)

# Load pipeline (cached)
try:
    graph, doc_count = load_pipeline()
except Exception as e:
    st.error(f"Failed to load pipeline: {e}\n\nCheck your `.env` file and API keys.")
    st.stop()

# Collection status banner
col_a, col_b, col_c = st.columns(3)
col_a.metric("Indexed documents", doc_count)
col_b.metric("Relevance threshold", threshold)
col_c.metric("Retrieve top-k", top_k)

if doc_count == 0:
    st.warning(
        "Vector store is empty. Run `python data/ingest.py` to load sample documents, "
        "or upload files in the sidebar."
    )

st.divider()

# Query input
query = st.text_area(
    "Ask a question",
    placeholder="e.g. What is the decompose-and-recompose technique in CRAG?",
    height=90,
)

run_btn = st.button("▶ Run Pipeline", type="primary", disabled=not query.strip())

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution + results
# ─────────────────────────────────────────────────────────────────────────────

if run_btn and query.strip():
    t_start = time.perf_counter()

    left, right = st.columns([3, 2])

    with left:
        st.markdown("### Pipeline Execution")

        node_labels = {
            "retrieve": "Node 1 · Retrieve",
            "evaluate_relevance": "Node 2 · Evaluate Relevance",
            "web_search": "Node 3 · Web Search (Tavily)",
            "filter_context": "Node 4 · Decompose & Recompose",
            "generate": "Node 5 · Generate",
        }

        step_placeholders = {k: st.empty() for k in node_labels}
        for k, label in node_labels.items():
            step_placeholders[k].markdown(
                f'<div class="step-card info">⏳ {label} <span style="color:#60a5fa">pending…</span></div>',
                unsafe_allow_html=True,
            )

        from src.pipeline.graph import make_initial_state
        from src.config import settings

        _orig_threshold = settings.relevance_threshold
        _orig_top_k = settings.top_k
        settings.relevance_threshold = threshold
        settings.top_k = top_k

        state = make_initial_state(query)
        state["pipeline_log"] = []

        try:
            for chunk in graph.stream(state, stream_mode="updates"):
                for node_name, update in chunk.items():
                    for k, v in update.items():
                        if k == "pipeline_log":
                            state[k] = state.get(k, []) + (v or [])
                        elif v is not None:
                            state[k] = v

                    # Update the corresponding step card
                    logs = update.get("pipeline_log", [])
                    msg = logs[0] if logs else f"{node_name} complete"
                    label = node_labels.get(node_name, node_name)

                    if "web_search" in node_name and update.get("web_search_triggered"):
                        css = "warning"
                    else:
                        css = "step-card"

                    step_placeholders[node_name].markdown(
                        f'<div class="step-card {css}">✓ {label}<br>'
                        f'<small style="color:#94a3b8">{msg}</small></div>',
                        unsafe_allow_html=True,
                    )
        finally:
            settings.relevance_threshold = _orig_threshold
            settings.top_k = _orig_top_k

        elapsed = time.perf_counter() - t_start

    # Right panel — metrics + relevance chart
    with right:
        st.markdown("### Retrieval Metrics")

        web_triggered = state.get("web_search_triggered", False)
        avg_rel = state.get("avg_relevance", 0.0)

        routing_badge = (
            '<span class="badge badge-web">🌐 WEB FALLBACK</span>'
            if web_triggered
            else '<span class="badge badge-local">💾 LOCAL ONLY</span>'
        )
        st.markdown(f"**Routing decision:** {routing_badge}", unsafe_allow_html=True)
        st.metric("Avg relevance score", f"{avg_rel:.2f}", delta=f"{avg_rel - threshold:+.2f} vs threshold")
        st.metric("Pipeline latency", f"{elapsed:.1f}s")
        st.metric("Sources used", f"{len(state.get('documents', []))} local + {len(state.get('web_documents', []))} web")

        fig = make_relevance_chart(state.get("documents", []), threshold)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Answer
    st.markdown("### Answer")
    answer = state.get("answer", "No answer generated.")
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    # Pipeline log
    with st.expander("🔎 Pipeline log", expanded=False):
        for entry in state.get("pipeline_log", []):
            st.code(entry, language=None)

    # Sources
    sources = state.get("sources", [])
    if sources:
        with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
            for i, src in enumerate(sources, 1):
                badge = "🌐 Web" if src.get("source") == "web" else "💾 Local"
                score_str = f" · score={src['score']:.2f}" if src.get("score") else ""
                header = f"**{badge} Source {i}**{score_str}"
                if src.get("url"):
                    header += f" — [{src.get('title', src['url'])}]({src['url']})"
                st.markdown(header)
                st.caption(src.get("content_preview", "")[:400])
                if i < len(sources):
                    st.divider()

    # Pipeline log for copy
    st.divider()
    st.markdown("### Full Pipeline State (JSON)")
    with st.expander("View raw state", expanded=False):
        safe_state = {
            k: v for k, v in state.items()
            if k not in ("filtered_context",) and not (isinstance(v, list) and len(str(v)) > 3000)
        }
        st.json(safe_state)
