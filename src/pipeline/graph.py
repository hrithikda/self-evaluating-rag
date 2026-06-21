"""
LangGraph workflow assembly for the Adaptive RAG pipeline.

  START
    │
    ▼
  retrieve          (Node 1 — ChromaDB top-k)
    │
    ▼
  rerank            (Node 2 — cross-encoder, local, no API cost)
    │
    ▼
  evaluate_relevance (Node 3 — LLM scorer, parallel)
    │
    ├─── avg ≥ threshold ──► filter_context ─┐
    │                                         │
    └─── avg < threshold ──► web_search ──────┘
                                              │
                                              ▼
                                        filter_context  (Node 5 — decompose-recompose)
                                              │
                                              ▼
                                           generate     (Node 6 — Claude + chat history)
                                              │
                                             END
"""

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.pipeline.nodes import (
    evaluate_relevance_node,
    filter_context_node,
    generate_node,
    rerank_node,
    retrieve_node,
    route_after_evaluation,
    web_search_node,
)
from src.pipeline.state import ChatMessage, RAGState


def build_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("evaluate_relevance", evaluate_relevance_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("filter_context", filter_context_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "evaluate_relevance")

    workflow.add_conditional_edges(
        "evaluate_relevance",
        route_after_evaluation,
        path_map={
            "web_search": "web_search",
            "filter_context": "filter_context",
        },
    )

    workflow.add_edge("web_search", "filter_context")
    workflow.add_edge("filter_context", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def make_initial_state(
    query: str,
    chat_history: list[ChatMessage] | None = None,
) -> RAGState:
    return RAGState(
        query=query,
        chat_history=chat_history or [],
        documents=[],
        relevance_scores=[],
        avg_relevance=0.0,
        web_search_triggered=False,
        web_documents=[],
        filtered_context="",
        answer="",
        sources=[],
        pipeline_log=[],
    )
