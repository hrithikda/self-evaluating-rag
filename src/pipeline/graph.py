"""
LangGraph workflow assembly for the Adaptive RAG pipeline.

  START
    │
    ▼
  retrieve          (Node 1 — ChromaDB)
    │
    ▼
  evaluate_relevance (Node 2 — LLM scorer, parallel)
    │
    ├─── avg ≥ threshold ──► filter_context ─┐
    │                                         │
    └─── avg < threshold ──► web_search ──────┘
                                              │
                                              ▼
                                        filter_context  (Node 4 — decompose-recompose)
                                              │
                                              ▼
                                           generate     (Node 5 — Claude)
                                              │
                                             END
"""

from langgraph.graph import END, START, StateGraph

from src.pipeline.nodes import (
    evaluate_relevance_node,
    filter_context_node,
    generate_node,
    retrieve_node,
    route_after_evaluation,
    web_search_node,
)
from src.pipeline.state import RAGState


def build_graph() -> StateGraph:
    workflow = StateGraph(RAGState)

    # Register nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("evaluate_relevance", evaluate_relevance_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("filter_context", filter_context_node)
    workflow.add_node("generate", generate_node)

    # Static edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "evaluate_relevance")

    # Conditional routing after evaluation
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


def make_initial_state(query: str) -> RAGState:
    return RAGState(
        query=query,
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
