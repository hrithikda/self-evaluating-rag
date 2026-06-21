import operator
from typing import Annotated, TypedDict


class RAGState(TypedDict):
    """Full state passed between LangGraph nodes."""

    # Input
    query: str

    # Retrieval (ChromaDB)
    documents: list[dict]
    relevance_scores: list[float]
    avg_relevance: float

    # Web fallback (Tavily)
    web_search_triggered: bool
    web_documents: list[dict]

    # Context processing
    filtered_context: str

    # Output
    answer: str
    sources: list[dict]

    # Observability — accumulated across all nodes via operator.add
    pipeline_log: Annotated[list[str], operator.add]
