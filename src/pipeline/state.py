import operator
from typing import Annotated, TypedDict


class ChatMessage(TypedDict):
    role: str     # "user" | "assistant"
    content: str


class RAGState(TypedDict):
    """Full state passed between LangGraph nodes."""

    # Input
    query: str
    chat_history: list[ChatMessage]   # previous turns, passed in by caller

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
