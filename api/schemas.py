from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str    # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation turns for multi-turn context",
    )
    relevance_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Override default threshold (0.6). Lower = more web search.",
    )
    top_k: int | None = Field(
        default=None, ge=1, le=20,
        description="Documents to retrieve from ChromaDB",
    )


class SourceDoc(BaseModel):
    source: str
    content_preview: str
    score: float | None = None
    rerank_score: float | None = None
    reason: str | None = None
    url: str | None = None
    title: str | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    web_search_triggered: bool
    avg_relevance_score: float
    sources: list[SourceDoc]
    pipeline_log: list[str]
    # Return updated history so stateless clients can carry it forward
    updated_history: list[ChatMessage]


class IngestRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    metadatas: list[dict] | None = None


class IngestResponse(BaseModel):
    added: int
    collection_size: int


class HealthResponse(BaseModel):
    status: str
    collection_size: int
    model: str
    threshold: float
    langsmith_enabled: bool
