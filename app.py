"""
Adaptive RAG — Streamlit demo (v2: chat interface + reranker + LangSmith)

Run:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Adaptive RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.step-done  { border-left:4px solid #4ade80; background:rgba(74,222,128,.06);
              padding:8px 14px; margin:4px 0; border-radius:0 8px 8px 0; font-size:.88rem; }
.step-web   { border-left:4px solid #facc15; background:rgba(250,204,21,.08);
              padding:8px 14px; margin:4px 0; border-radius:0 8px 8px 0; font-size:.88rem; }
.step-pend  { border-left:4px solid #475569; background:rgba(71,85,105,.06);
              padding:8px 14px; margin:4px 0; border-radius:0 8px 8px 0;
              font-size:.88rem; color:#64748b; }
.badge-local{ display:inline-block; padding:2px 10px; border-radius:12px;
              background:#dcfce7; color:#166534; font-size:.78rem; font-weight:600; }
.badge-web  { display:inline-block; padding:2px 10px; border-radius:12px;
              background:#fef3c7; color:#92400e; font-size:.78rem; font-weight:600; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []          # displayed chat bubbles
if "pipeline_history" not in st.session_state:
    st.session_state.pipeline_history = []  # RAG chat_history for the pipeline
if "last_pipeline_state" not in st.session_state:
    st.session_state.last_pipeline_state = None

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    threshold = st.slider("Relevance threshold", 0.1, 0.9, 0.6, 0.05,
                          help="Below this → Tavily web search triggered")
    top_k = st.slider("Retrieve top-k", 1, 10, 5)

    show_pipeline = st.toggle("Show pipeline details", value=True)

    st.divider()
    st.markdown("## 📂 Ingest Documents")
    uploaded = st.file_uploader("Upload .txt / .pdf", type=["txt", "pdf"],
                                accept_multiple_files=True)
    if uploaded and st.button("Ingest", type="primary"):
        from langchain_core.documents import Document
        from src.ingestion.loader import split_documents
        from src.ingestion.vectorstore import add_documents

        raw = [
            Document(page_content=f.read().decode("utf-8", errors="ignore"),
                     metadata={"filename": f.name})
            for f in uploaded
        ]
        chunks = split_documents(raw)
        ids = add_documents([c.page_content for c in chunks],
                            [c.metadata for c in chunks])
        st.success(f"Indexed {len(ids)} chunks.")

    st.divider()
    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        st.session_state.pipeline_history = []
        st.session_state.last_pipeline_state = None
        st.rerun()

    st.markdown("## 🔭 Observability")
    from src.observability import langsmith_enabled
    if langsmith_enabled():
        st.success("LangSmith tracing **enabled**")
        st.caption("Every query is logged with metadata to your LangSmith project.")
    else:
        st.warning("LangSmith **disabled**")
        st.caption("Set `LANGCHAIN_TRACING_V2=true` in `.env` to enable.")

# ─────────────────────────────────────────────────────────────────────────────
# Load pipeline (cached)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading pipeline…")
def load_pipeline():
    from src.ingestion.vectorstore import collection_size, get_vectorstore
    from src.pipeline.graph import build_graph
    from src.pipeline.nodes import _get_reranker  # warm up cross-encoder

    get_vectorstore()
    graph = build_graph()
    _get_reranker()  # download / load model on startup
    return graph, collection_size()


try:
    graph, doc_count = load_pipeline()
except Exception as e:
    st.error(f"Failed to load pipeline: {e}\n\nCheck your `.env` API keys.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🔍 Adaptive RAG")
st.caption(
    "Corrective RAG · LangGraph · ChromaDB · Cross-encoder reranker · "
    "Tavily web fallback · Claude · LangSmith"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Indexed docs", doc_count)
col2.metric("Threshold", threshold)
col3.metric("Retrieve k", top_k)
col4.metric("History turns", len(st.session_state.pipeline_history) // 2)

if doc_count == 0:
    st.warning("Vector store is empty. Run `python data/ingest.py` or upload files in the sidebar.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline detail renderer (defined before first use)
# ─────────────────────────────────────────────────────────────────────────────


def _render_pipeline_details(ps: dict, thr: float) -> None:
    with st.expander("📊 Pipeline details", expanded=False):
        # Routing badge
        web = ps.get("web_search_triggered", False)
        badge = (
            '<span class="badge-web">🌐 WEB FALLBACK</span>'
            if web
            else '<span class="badge-local">💾 LOCAL ONLY</span>'
        )
        avg = ps.get("avg_relevance", 0.0)
        st.markdown(
            f"{badge} &nbsp; Avg relevance: **{avg:.2f}** (threshold: {thr})",
            unsafe_allow_html=True,
        )

        # Relevance + rerank chart
        docs = ps.get("documents", [])
        if docs:
            names = [f"Doc {i+1}" for i in range(len(docs))]
            rel_scores = [d.get("relevance_score") or 0.0 for d in docs]
            rerank_scores = [d.get("rerank_score") or 0.0 for d in docs]
            colors = ["#4ade80" if s >= thr else "#f87171" for s in rel_scores]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="LLM relevance", x=names, y=rel_scores,
                marker_color=colors,
                text=[f"{s:.2f}" for s in rel_scores],
                textposition="outside",
            ))
            fig.add_trace(go.Scatter(
                name="Rerank score", x=names, y=rerank_scores,
                mode="markers+lines",
                marker=dict(size=10, color="#60a5fa"),
                line=dict(color="#60a5fa", dash="dot"),
            ))
            fig.add_hline(y=thr, line_dash="dash", line_color="#facc15",
                          annotation_text=f"Threshold ({thr})")
            fig.update_layout(
                height=260, showlegend=True,
                legend=dict(orientation="h", y=1.15),
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                yaxis=dict(range=[0, 1.2]),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Sources
        sources = ps.get("sources", [])
        if sources:
            st.markdown("**Sources:**")
            for i, src in enumerate(sources, 1):
                icon = "🌐" if src.get("source") == "web" else "💾"
                score_str = f"score={src['score']:.2f}" if src.get("score") else ""
                rerank_str = f"rerank={src['rerank_score']:.2f}" if src.get("rerank_score") else ""
                meta = " · ".join(filter(None, [score_str, rerank_str]))
                header = f"{icon} **Source {i}** {meta}"
                if src.get("url"):
                    header += f" — [{src.get('title', src['url'])}]({src['url']})"
                st.caption(header)
                st.caption(src.get("content_preview", "")[:300])

        # Pipeline log
        logs = ps.get("pipeline_log", [])
        if logs:
            st.markdown("**Pipeline log:**")
            for entry in logs:
                st.code(entry, language=None)


# ─────────────────────────────────────────────────────────────────────────────
# Chat history display
# ─────────────────────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_pipeline and msg.get("pipeline_state"):
            _render_pipeline_details(msg["pipeline_state"], threshold)

# ─────────────────────────────────────────────────────────────────────────────
# Chat input + pipeline execution
# ─────────────────────────────────────────────────────────────────────────────

if query := st.chat_input("Ask a question…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run pipeline with streaming step cards
    with st.chat_message("assistant"):
        from src.config import settings
        from src.observability import make_run_config
        from src.pipeline.graph import make_initial_state

        node_labels = {
            "retrieve": "Node 1 · Retrieve (ChromaDB)",
            "rerank": "Node 2 · Cross-encoder Rerank",
            "evaluate_relevance": "Node 3 · LLM Relevance Scoring",
            "web_search": "Node 4 · Web Search (Tavily)",
            "filter_context": "Node 5 · Decompose & Recompose",
            "generate": "Node 6 · Generate (Claude)",
        }

        if show_pipeline:
            step_slots = {k: st.empty() for k in node_labels}
            for k, label in node_labels.items():
                step_slots[k].markdown(
                    f'<div class="step-pend">⏳ {label}</div>',
                    unsafe_allow_html=True,
                )

        answer_slot = st.empty()

        _orig_thr = settings.relevance_threshold
        _orig_k = settings.top_k
        settings.relevance_threshold = threshold
        settings.top_k = top_k

        state = make_initial_state(query, chat_history=st.session_state.pipeline_history)
        state["pipeline_log"] = []

        run_cfg = make_run_config(
            query,
            threshold=threshold,
            source="streamlit",
            history_turns=len(st.session_state.pipeline_history) // 2,
        )

        t0 = time.perf_counter()
        try:
            for chunk in graph.stream(state, stream_mode="updates", config=run_cfg):
                for node_name, update in chunk.items():
                    # Merge into local state
                    for k, v in update.items():
                        if k == "pipeline_log":
                            state[k] = state.get(k, []) + (v or [])
                        elif v is not None:
                            state[k] = v

                    if show_pipeline and node_name in step_slots:
                        logs = update.get("pipeline_log", [])
                        msg = logs[0] if logs else f"{node_name} complete"
                        css = "step-web" if (node_name == "web_search") else "step-done"
                        step_slots[node_name].markdown(
                            f'<div class="{css}">✓ {node_labels.get(node_name, node_name)}'
                            f'<br><small style="color:#94a3b8">{msg}</small></div>',
                            unsafe_allow_html=True,
                        )
        finally:
            settings.relevance_threshold = _orig_thr
            settings.top_k = _orig_k

        elapsed = time.perf_counter() - t0
        answer = state.get("answer", "No answer generated.")
        answer_slot.markdown(answer)

        if show_pipeline:
            _render_pipeline_details(state, threshold)

        st.caption(f"⏱ {elapsed:.1f}s · avg relevance {state.get('avg_relevance', 0):.2f}")

    # Persist to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "pipeline_state": state,
    })

    # Update pipeline conversation history for next turn
    st.session_state.pipeline_history.append({"role": "user", "content": query})
    st.session_state.pipeline_history.append({"role": "assistant", "content": answer})
    st.session_state.last_pipeline_state = state
