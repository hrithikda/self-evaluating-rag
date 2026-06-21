from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Natural language question")
    relevance_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Override default threshold (0.6). Lower = more likely to trigger web search.",
    )
    top_k: int | None = Field(
        default=None, ge=1, le=20,
        description="Number of documents to retrieve from ChromaDB",
    )


class SourceDoc(BaseModel):
    source: str                  # "local" | "web"
    content_preview: str
    score: float | None = None
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


class IngestRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Raw text chunks to add")
    metadatas: list[dict] | None = Field(default=None, description="Optional metadata per chunk")


class IngestResponse(BaseModel):
    added: int
    collection_size: int


class HealthResponse(BaseModel):
    status: str
    collection_size: int
    model: str
    threshold: float
